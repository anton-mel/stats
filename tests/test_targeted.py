from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pie_evals.node import build as pb
from pie_evals.orchestrate import dashboard, flops
from pie_evals.orchestrate.matrix import Matrix
from pie_evals.orchestrate.store import Store
from pie_evals.schema import CellStatus, PerfMetrics, Provenance, Record, Tier

ROOT = Path(__file__).resolve().parents[1]

QWEN35_08B = {
    "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
    "intermediate_size": 3584, "vocab_size": 248320, "attn_output_gate": True,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
    "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


@pytest.fixture(scope="module")
def matrix():
    return Matrix.load(ROOT / "matrix")


def _cell(matrix, platform, workload, artifact="gemma-4-e4b-bf16"):
    return next(c for c in matrix.runnable(Tier.TARGETED)
                if c.platform.id == platform and c.workload.id == workload and c.artifact.id == artifact)


def _rec(cell, value, when, *, commit, status=CellStatus.PASS, run="r"):
    prov = Provenance(harness_commit="h", pie_commit=commit, engine_version=commit, checkpoint_revision="rev", hardware_fingerprint={"gpu_cores": "40"})
    return Record(run_id=run, job_id="j", tier=Tier.TARGETED, cell_id=cell.cell_id, cell_key=cell.cell_key, cell=cell, status=status,
                  perf=PerfMetrics(primary="decode_tok_s", decode_tok_s=value, prefill_tok_s=value * 30, rounds=[value], cov=0.002),
                  provenance=prov, started_at=when, duration_s=30.0)


def _node_out(tmp_path, records):
    d = tmp_path / "out"
    d.mkdir()
    (d / "records.jsonl").write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    return d


def test_targeted_is_small_and_baselines_run_on_their_copies(matrix):
    cells = matrix.runnable(Tier.TARGETED)
    assert cells and all(c.mode.tp == 1 for c in cells)
    assert {c.engine.value for c in cells} == {"pie", "ollama"}
    assert all(c.engine.value == "pie" or c.workload.id != "control-aa" for c in cells)
    assert all(c.artifact.baseline_of for c in cells if c.engine.value == "ollama")
    assert not any(c.artifact.baseline_of for c in cells if c.engine.value == "pie")
    assert {c.workload.id for c in cells} == {"control-aa", "ob-story-200", "ob-512-200", "ob-2048-200", "ss-128-64", "lc-1k-128", "lc-2k-128", "c8", "c32"}
    assert not matrix.check_budget(Tier.TARGETED)


def test_metal_build_is_cached_without_a_wheel(tmp_path):
    d = pb.cache_dir("a" * 40, ["metal"], tmp_path)
    (d / "wasm").mkdir(parents=True)
    (d / "pie").write_text("")
    (d / "wasm" / "x.wasm").write_text("")
    assert pb.is_cached("a" * 40, ["metal"], tmp_path)
    assert not pb.is_cached("a" * 40, ["cuda"], tmp_path)


def test_flops_scale_with_the_rate():
    one = flops.tflops({"decode_tok_s": 100.0, "prefill_tok_s": 1000.0}, {"prefill": 128, "decode": 64}, QWEN35_08B)
    two = flops.tflops({"decode_tok_s": 200.0, "prefill_tok_s": 2000.0}, {"prefill": 128, "decode": 64}, QWEN35_08B)
    assert two["decode_tflops"] == pytest.approx(2 * one["decode_tflops"])
    assert two["prefill_tflops"] == pytest.approx(2 * one["prefill_tflops"])
    per_token = flops.decode_flops_per_token(QWEN35_08B, 160)
    assert 1.4e9 < per_token < 1.8e9
    batched = flops.tflops({"output_tok_s": 1000.0, "prefill_tok_s": 5.0}, {"prefill": 128, "decode": 128, "concurrency": 8}, QWEN35_08B)
    assert batched["prefill_tflops"] is None and batched["decode_tflops"] > 0
    assert flops.tflops({"decode_tok_s": 1.0}, {"prefill": 1}, None) == {"prefill_tflops": None, "decode_tflops": None}


def test_site_has_pushes_pool_and_people(tmp_path, matrix, monkeypatch):
    monkeypatch.setattr(flops, "model_config", lambda repo: QWEN35_08B)
    st = Store(tmp_path / "store")
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for i, wl in enumerate(["ss-128-64", "lc-2k-128"]):
        cell = _cell(matrix, "m5-max-48g", wl)
        for j in range(2):
            st.write_run([_rec(cell, 400 + j, t0 + timedelta(days=j), commit=f"c{j}", run=f"r{i}{j}")], tier=Tier.TARGETED, run_id=f"r{i}{j}")
    ollama = _cell(matrix, "m5-max-48g", "ss-128-64", artifact="gemma-4-e4b-ollama-bf16")
    ro = _rec(ollama, 300, t0 + timedelta(days=1), commit="c1", run="o1")
    ro.provenance.engine_version = "0.34.4"
    st.write_run([ro], tier=Tier.TARGETED, run_id="o1")
    live = [{"name": "a-mac", "status": "online", "busy": True, "labels": [{"name": "self-hosted"}, {"name": "macOS"}, {"name": "m5-max-48g"}]},
            {"name": "b-mac", "status": "online", "busy": False, "labels": [{"name": "self-hosted"}, {"name": "macOS"}, {"name": "m1-max-32g"}]},
            {"name": "stranger", "status": "online", "busy": False, "labels": [{"name": "self-hosted"}, {"name": "Linux"}]}]
    data = dashboard.build(st, matrix, live, repo="o/evals", pie_repo="o/pie", lookup_commits=False)
    assert [c["sha"] for c in data["commits"]] == ["c0", "c1"]
    assert [(m["id"], m["kind"], m["status"]) for m in data["pool"]] == [("m1-max-32g", "self-hosted", "idle"), ("m5-max-48g", "self-hosted", "busy")]
    by_test = data["results"]["m5-max-48g"]["models"]["gemma-4-e4b-bf16"]
    assert set(by_test) == {"ss-128-64", "lc-2k-128"}
    assert by_test["ss-128-64"]["c1"]["decode"] == 401 and by_test["ss-128-64"]["c1"]["prefill"] == 401 * 30
    assert "gemma-4-e4b-ollama-bf16" not in data["results"]["m5-max-48g"]["models"]
    arm = data["baselines"]["m5-max-48g"]["gemma-4-e4b-bf16"]["ss-128-64"][ollama.artifact.baseline_label]
    assert arm["decode"] == 300 and arm["version"] == "0.34.4"
    assert [b["id"] for b in data["benchmarks"]] == ["ob-story-200", "ss-128-64", "ob-512-200", "lc-1k-128", "ob-2048-200", "lc-2k-128", "c8", "c32"]
    assert data["people"] == []
    assert any(m["id"] == "gemma-4-e4b-bf16" and m["has_results"] for m in data["models"])
    assert not any(m["id"].endswith("ollama-bf16") for m in data["models"])
    assert dashboard.render(st, matrix, tmp_path / "site", live, repo="o/evals", lookup_commits=False) == 2
    assert "openRuns" in (tmp_path / "site" / "index.html").read_text()


def test_people_access_and_last_active(tmp_path, monkeypatch):
    runs = [{"workflow_runs": [
        {"display_title": "pie eval " + "a" * 40, "event": "repository_dispatch", "triggering_actor": {"login": "bot"}, "status": "completed",
         "run_started_at": "2026-09-21T10:00:00Z", "updated_at": "2026-09-21T10:06:00Z", "created_at": "2026-09-21T10:00:00Z"},
        {"display_title": "pie eval " + "b" * 40, "event": "workflow_dispatch", "triggering_actor": {"login": "friend"}, "status": "completed",
         "run_started_at": "2026-09-01T10:00:00Z", "updated_at": "2026-09-01T10:30:00Z", "created_at": "2026-09-01T10:00:00Z"},
        {"display_title": "pie eval " + "b" * 40, "event": "workflow_dispatch", "triggering_actor": {"login": "friend"}, "status": "completed",
         "run_started_at": "2026-07-01T10:00:00Z", "updated_at": "2026-07-01T11:00:00Z", "created_at": "2026-07-01T10:00:00Z"},
    ]}]
    collaborators = [[{"login": "author", "role_name": "write"}, {"login": "friend", "role_name": "admin"}, {"login": "idle", "role_name": "write"}]]
    monkeypatch.setattr(dashboard, "_paginate", lambda path: runs if "runs" in path else collaborators)
    used = dashboard.usage("o/evals", {"a" * 40: "author"})
    assert used == {"author": {"last": "2026-09-21T10:00:00Z"}, "friend": {"last": "2026-09-01T10:00:00Z"}}
    rows = dashboard.people("o/evals", {"a" * 40: "author"})
    assert [(r["login"], r["role"]) for r in rows] == [("author", "write"), ("friend", "admin"), ("idle", "write")]
