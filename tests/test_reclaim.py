from pathlib import Path

from pie_evals.node import reclaim
from pie_evals.orchestrate.jobs import make_jobs
from pie_evals.orchestrate.matrix import Matrix
from pie_evals.schema import Tier

ROOT = Path(__file__).resolve().parents[1]


def test_reclaim_keeps_this_jobs_imports_and_builds(tmp_path: Path, monkeypatch):
    m = Matrix.load(ROOT / "matrix")
    job = make_jobs(m, Tier.TARGETED, pie_commit="c" * 40, platforms=["m4-pro-48g"], engines=["pie"], cells=[c for c in m.expand() if c.artifact.id == "gemma-4-e4b-bf16"])[0]
    root = tmp_path / "cache"
    for d in ("artifacts/gemma-4-e4b-bf16-cccccccc", "artifacts/gemma-4-e4b-bf16-01234567", "artifacts/qwen3.6-27b-mlx4-cccccccc", "artifacts/.tmp-x", "builds/cccccccccccc-metal", "builds/012345678901-metal"):
        (root / d).mkdir(parents=True)
        (root / d / "f").write_bytes(b"x" * 10)
    monkeypatch.setenv("PIE_EVALS_CACHE", str(root))
    monkeypatch.delenv("PIE_EVALS_RECLAIM_HF", raising=False)
    dropped = []
    monkeypatch.setattr(reclaim, "_drop_hf_repos", lambda *a, **k: dropped.append(a))
    monkeypatch.setattr(reclaim, "free_gib", lambda p: 500.0)
    assert reclaim.reclaim(job, tmp_path / "hf", log=lambda *_: None) == 0.0
    assert (root / "artifacts/qwen3.6-27b-mlx4-cccccccc").exists()
    monkeypatch.setattr(reclaim, "free_gib", lambda p: 10.0)
    reclaim.reclaim(job, tmp_path / "hf", log=lambda *_: None)
    assert (root / "artifacts/gemma-4-e4b-bf16-cccccccc").exists()
    assert (root / "builds/cccccccccccc-metal").exists()
    for gone in ("artifacts/gemma-4-e4b-bf16-01234567", "artifacts/qwen3.6-27b-mlx4-cccccccc", "artifacts/.tmp-x", "builds/012345678901-metal"):
        assert not (root / gone).exists(), gone
    assert dropped == []
    monkeypatch.setenv("PIE_EVALS_RECLAIM_HF", "1")
    reclaim.reclaim(job, tmp_path / "hf", log=lambda *_: None)
    assert dropped and dropped[0][1] == {c.artifact.base_model for c in job.cells}
