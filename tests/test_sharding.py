from pathlib import Path

import pytest

from pie_evals.orchestrate.jobs import cell_minutes, make_jobs, process_key, shard_groups
from pie_evals.orchestrate.matrix import Matrix
from pie_evals.orchestrate.store import Store
from pie_evals.schema import Tier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def matrix():
    return Matrix.load(ROOT / "matrix")


def test_shards_respect_budget_and_keep_groups_whole(matrix):
    cells = [c for c in matrix.runnable(Tier.TARGETED) if c.platform.id == "m4-pro-48g"]
    budget = 10.0
    shards = shard_groups(cells, budget)
    assert len(shards) > 1
    assert sum(len(s) for s in shards) == len(cells)
    placement = {}
    for i, sh in enumerate(shards):
        for c in sh:
            placement.setdefault(process_key(c), set()).add(i)
    groups = {}
    for c in cells:
        groups.setdefault(process_key(c), []).append(c)
    for k, idxs in placement.items():
        if len(idxs) > 1:
            assert sum(cell_minutes(c) for c in groups[k]) > budget, f"group {k} split without need"


def test_jobs_carry_time_policy(matrix, tmp_path):
    jobs = make_jobs(matrix, Tier.TARGETED, pie_commit="abc", store=Store(tmp_path), platforms=["m4-pro-48g"])
    assert jobs
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids))
    for j in jobs:
        assert j.budget_s == matrix.job_budget_minutes * 60
        assert j.kill_s == int(matrix.job_budget_minutes * matrix.kill_factor * 60)
        assert j.per_cell_timeout_s <= j.kill_s
        assert j.pie_build_features == ["metal"]
        if any(str(c.engine) == "ollama" for c in j.cells):
            assert j.baseline_versions.get("ollama") == matrix.engines["ollama"].pin


def test_every_tier_within_job_caps(matrix):
    cells = matrix.expand()
    for t in Tier:
        assert matrix.check_budget(t, cells) == [], matrix.check_budget(t, cells)


def test_available_platforms_keeps_only_online_macs(matrix, monkeypatch):
    import json

    from pie_evals.orchestrate import jobs as J

    runners = {"runners": [{"status": "online", "labels": [{"name": "self-hosted"}, {"name": "macos"}, {"name": "m1-max-32g"}]},
                           {"status": "offline", "labels": [{"name": "m4-pro-48g"}]}]}
    monkeypatch.setattr(J.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": json.dumps(runners)})())
    ok, skipped = J.available_platforms(matrix, "o/r")
    assert ok == ["m1-max-32g"]
    assert "m4-pro-48g" in skipped and "m2-max" in skipped
