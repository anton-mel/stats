
from pathlib import Path

import pytest

from pie_evals.node import runner as runner_mod
from pie_evals.node.engines.base import BenchResult, Engine, EngineLaunchError
from pie_evals.orchestrate.jobs import make_jobs
from pie_evals.orchestrate.matrix import Matrix
from pie_evals.orchestrate.store import Store
from pie_evals.schema import CellStatus, ErrorClass, PerfMetrics, Record, Tier

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine(Engine):
    name = "pie"
    script = "scripts/bench/fake.py"
    calls: list[str] = []
    common: dict[str, list[str]] = {}
    behaviour: dict[str, object] = {}

    def default_python(self):
        return "python3"

    def model_arg(self):
        return self.artifact.base_model

    def engine_args(self, workload):
        return []

    def version(self):
        return "deadbeef"

    def run(self, workload, common_args, out_dir, timeout_s):
        FakeEngine.calls.append(workload.id)
        FakeEngine.common[workload.id] = list(common_args)
        b = FakeEngine.behaviour.get(workload.id)
        if isinstance(b, EngineLaunchError):
            raise b
        values = b if isinstance(b, list) else [float(b or 100.0)]
        v = values[min(len([c for c in FakeEngine.calls if c == workload.id]) - 1, len(values) - 1)]
        perf = PerfMetrics(primary="decode_tok_s", decode_tok_s=v, output_tok_s=v * 8, prompt_tokens=128, output_tokens=64, requests=1, failed=0)
        return BenchResult(perf=perf, summary={}, requests=[{"ok": True, "prompt_tokens": 128, "output_token_ids": [1, 2, 3]}], argv=[], output_token_ids=[[1, 2, 3]])


@pytest.fixture
def job(tmp_path):
    m = Matrix.load(ROOT / "matrix")
    cells = [c for c in m.runnable(Tier.TARGETED) if c.platform.id == "m4-pro-48g" and c.artifact.id == "gemma-4-e4b-bf16" and str(c.engine) == "pie"]
    j = make_jobs(m, Tier.TARGETED, pie_commit="deadbeef", store=Store(tmp_path / "store"), platforms=["m4-pro-48g"], cells=cells)[0]
    assert {c.workload.id for c in j.cells} >= {"control-aa", "ss-128-64", "c8"}
    return j


@pytest.fixture
def patched(monkeypatch, tmp_path):
    FakeEngine.calls = []
    FakeEngine.common = {}
    FakeEngine.behaviour = {}
    monkeypatch.setattr(runner_mod, "get_engine", lambda name: FakeEngine)
    monkeypatch.setattr(runner_mod, "load_recipe", lambda *a, **k: {})
    snap = tmp_path / "hf" / "models--google--gemma-4-E4B-it" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"num_hidden_layers": 28}')
    (snap / "model.safetensors").write_bytes(b"x")

    class P:
        def before_job(self, plat):
            return {"state": "ok"}

        def between_engines(self, plat, names):
            return {}

        def after_model(self, plat, before):
            return []

    monkeypatch.setattr(runner_mod.pf, "Preflight", P)
    monkeypatch.setattr(runner_mod.prov, "hardware_fingerprint", lambda: {"gpu": "fake"})
    monkeypatch.setattr(runner_mod.prov, "build_provenance", lambda *a, **k: runner_mod.prov.Provenance(harness_commit="h", engine_version="deadbeef", checkpoint_revision="abc", hardware_fingerprint={"gpu": "fake"}))
    monkeypatch.setattr("pie_evals.node.reclaim.reclaim", lambda *a, **k: 0.0)
    return tmp_path / "hf"


def _run(job, hf, tmp_path):
    r = runner_mod.NodeRunner(job, pie_root=tmp_path / "pie", out_dir=tmp_path / "out", hf_cache=hf, build=False)
    recs = r.run()
    lines = (tmp_path / "out" / "records.jsonl").read_text().splitlines()
    assert len(lines) >= len(recs)
    return recs, [Record.model_validate_json(ln) for ln in lines]


def test_happy_path_one_process_all_cells(job, patched, tmp_path):
    recs, _ = _run(job, patched, tmp_path)
    assert {r.status for r in recs} == {CellStatus.PASS}
    assert FakeEngine.calls[0] == "control-aa"
    assert all(r.provenance.pie_commit == "deadbeef" for r in recs)
    ss = next(r for r in recs if r.cell.workload.id == "ss-128-64")
    assert len(ss.perf.rounds) == 1 + job.repetition.confirm_rounds


def test_history_within_band_skips_confirmation(job, patched, tmp_path):
    cid = next(c.cell_id for c in job.cells if c.workload.id == "ss-128-64")
    job = job.model_copy(update={"history": {cid: [100.0, 100.5, 99.5, 100.2]}})
    recs, _ = _run(job, patched, tmp_path)
    ss = next(r for r in recs if r.cell.workload.id == "ss-128-64")
    assert len(ss.perf.rounds) == 1


def test_noisy_rounds_mark_cell_noisy(job, patched, tmp_path):
    FakeEngine.behaviour["ss-128-64"] = [100.0, 90.0, 110.0]
    recs, _ = _run(job, patched, tmp_path)
    ss = next(r for r in recs if r.cell.workload.id == "ss-128-64")
    assert ss.status == CellStatus.NOISY and "cov" in ss.invalid_reason


def test_control_failure_invalidates_process(job, patched, tmp_path):
    FakeEngine.behaviour["control-aa"] = [100.0, 130.0, 130.0, 100.0, 130.0, 130.0]
    recs, _ = _run(job, patched, tmp_path)
    ctrl = next(r for r in recs if r.cell.workload.id == "control-aa")
    assert ctrl.status == CellStatus.NOISY
    others = [r for r in recs if r.cell.workload.id != "control-aa"]
    assert others and all(r.status == CellStatus.NOISY and r.error_class == ErrorClass.HARNESS_INVALID for r in others)
    assert FakeEngine.calls.count("control-aa") == 6


def test_noisy_control_gets_one_more_look(job, patched, tmp_path):
    FakeEngine.behaviour["control-aa"] = [100.0, 130.0]
    recs, _ = _run(job, patched, tmp_path)
    ctrl = next(r for r in recs if r.cell.workload.id == "control-aa")
    assert ctrl.status == CellStatus.PASS and FakeEngine.calls.count("control-aa") == 6
    assert all(r.status == CellStatus.PASS for r in recs if r.cell.workload.id != "control-aa")


def test_engine_failure_is_classified(job, patched, tmp_path):
    FakeEngine.behaviour["c8"] = EngineLaunchError(ErrorClass.OOM, "device memory exhausted")
    recs, _ = _run(job, patched, tmp_path)
    c8 = next(r for r in recs if r.cell.workload.id == "c8")
    assert c8.status == CellStatus.FAIL and c8.error_class == ErrorClass.OOM


def test_missing_checkpoint_fails_load(job, patched, tmp_path):
    recs, _ = _run(job, tmp_path / "empty-hf", tmp_path)
    assert recs == []
    lines = [Record.model_validate_json(ln) for ln in (tmp_path / "out" / "records.jsonl").read_text().splitlines()]
    assert lines and all(r.status == CellStatus.FAIL and r.error_class == ErrorClass.LOAD_FAIL for r in lines)


def test_soft_budget_records_unreached_cells_as_not_run(job, patched, tmp_path):
    job = job.model_copy(update={"budget_s": 0, "kill_s": 600})
    recs, lines = _run(job, patched, tmp_path)
    assert recs == []
    assert lines and all(r.status == CellStatus.NOT_RUN and "budget" in (r.invalid_reason or "") for r in lines)
    assert {r.cell_id for r in lines} == {c.cell_id for c in job.cells}


def test_per_cell_timeout_is_bounded_by_kill_deadline(job, patched, tmp_path):
    seen = []
    orig = FakeEngine.run

    def spy(self, workload, common_args, out_dir, timeout_s):
        seen.append(timeout_s)
        return orig(self, workload, common_args, out_dir, timeout_s)

    FakeEngine.run = spy
    try:
        job = job.model_copy(update={"budget_s": 3600, "kill_s": 120, "per_cell_timeout_s": 1800})
        _run(job, patched, tmp_path)
    finally:
        FakeEngine.run = orig
    assert seen and all(t <= 120 for t in seen)


def test_out_dir_is_absolute_for_subprocesses(job, patched, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner_mod.NodeRunner(job, pie_root=tmp_path / "pie", out_dir=Path("out/rel"), hf_cache=patched, build=False)
    assert r.out.is_absolute() and r.out == (tmp_path / "out/rel").resolve()


def test_cell_priority_runs_control_then_cheap_shapes_first():
    from types import SimpleNamespace as NS

    from pie_evals.node.runner import cell_priority

    def cell(kind, est):
        return NS(workload=NS(kind=NS(value=kind), est_minutes=est))

    cells = [cell("closed_loop", 3.0), cell("single_stream", 0.5), cell("control_aa", 0.5), cell("long_context", 1.5)]
    got = sorted(cells, key=cell_priority)
    assert [c.workload.kind.value for c in got] == ["control_aa", "single_stream", "long_context", "closed_loop"]


def test_heavy_model_on_a_small_mac_boots_short_and_long_shapes_apart():
    from pie_evals.node.engines.shape import context_tokens_for
    from pie_evals.node.runner import SHORT_CONTEXT_TOKENS, split_heavy_processes
    from pie_evals.orchestrate.jobs import process_key
    from pie_evals.schema import WorkloadSpec

    m = Matrix.load(ROOT / "matrix")
    lc16k = WorkloadSpec(id="lc-16k-128", kind="long_context", params={"prefill": 16384, "decode": 128, "concurrency": 1})
    for aid, split in (("gemma-4-26b-a4b-mlx4", True), ("gemma-4-e4b-bf16", False)):
        cells = [c for c in m.runnable(Tier.TARGETED) if c.artifact.id == aid and c.platform.id == "m1-max-32g" and str(c.engine) == "pie"]
        cells.append(cells[0].model_copy(update={"workload": lc16k}))
        groups = {process_key(cells[0]): cells}
        out = split_heavy_processes(groups)
        keys = sorted(k[2] for k in out)
        if split:
            assert keys == ["tp1-long", "tp1-short"], keys
            for k, cs in out.items():
                longs = [context_tokens_for(c.workload) > SHORT_CONTEXT_TOKENS for c in cs]
                assert all(longs) if k[2].endswith("-long") else not any(longs)
            assert sum(len(cs) for cs in out.values()) == len(cells)
        else:
            assert out == groups
        assert split_heavy_processes({process_key(cells[0]): cells[:-1]}) == {process_key(cells[0]): cells[:-1]}


def test_recipe_follows_the_cell_and_keeps_what_the_boot_added(job, monkeypatch):
    from pie_evals.node.engines.shape import workload_concurrency

    monkeypatch.setattr(runner_mod, "load_recipe", lambda engine, name, plat, wl, family=None: {"slots": workload_concurrency(wl), "_recipe": name})
    ss = next(c for c in job.cells if c.workload.id == "ss-128-64")
    c8 = next(c for c in job.cells if c.workload.id == "c8")
    base = runner_mod.load_recipe("pie", "default", ss.platform, ss.workload)
    base["snapshot_dir"], base["program_path"] = "/snap", "examples/x"
    fresh = runner_mod.recipe_for_cell(base, "default", c8)
    assert (base["slots"], fresh["slots"]) == (1, 8)
    assert fresh["snapshot_dir"] == "/snap" and fresh["program_path"] == "examples/x"


def test_recipe_keeps_the_knobs_the_server_booted_with(job, monkeypatch):
    monkeypatch.setattr(runner_mod, "load_recipe", lambda engine, name, plat, wl, family=None: {"num_parallel": "$serve_concurrency", "_recipe": name})
    c8 = next(c for c in job.cells if c.workload.id == "c8")
    assert runner_mod.recipe_for_cell({"num_parallel": 8}, "competitive", c8)["num_parallel"] == 8
    assert runner_mod.recipe_for_cell({}, "competitive", c8)["num_parallel"] == "$serve_concurrency"


def test_warmup_follows_the_workload(job, patched, tmp_path):
    _run(job, patched, tmp_path)

    def warm(wl):
        args = FakeEngine.common[wl]
        return args[args.index("--warmup") + 1]

    assert warm("ob-512-200") == "1" and warm("ss-128-64") == "2"


def test_a_slow_first_round_is_confirmed_away_not_withheld(job, patched, tmp_path):
    job = job.model_copy(update={"repetition": job.repetition.model_copy(update={"min_rounds": 3, "confirm_rounds": 2, "max_rounds": 5})})
    FakeEngine.behaviour["ss-128-64"] = [90.0, 100.0, 101.0, 100.0, 100.0]
    FakeEngine.behaviour["c8"] = [100.0, 80.0, 120.0, 85.0, 118.0]
    recs, _ = _run(job, patched, tmp_path)
    ss = next(r for r in recs if r.cell.workload.id == "ss-128-64")
    assert ss.status == CellStatus.PASS and len(ss.perf.rounds) == 5 and ss.perf.decode_tok_s == 100.0
    c8 = next(r for r in recs if r.cell.workload.id == "c8")
    assert c8.status == CellStatus.NOISY and len(c8.perf.rounds) == 4
