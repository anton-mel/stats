"""Disk room on the Mac between jobs.

Every run lands on the same few Macs, which hold every checkpoint, every
imported ``.zt`` artifact and the pie builds. Before a job starts, when the disk
is short, drop what this job does not need: imports and builds of other
artifacts/commits first, then HF repos this job's cells never reference. The
next job that needs them fetches again — slower than a cache hit, never a full
disk. Ollama's own store is left alone: ``ollama rm`` is the user's call.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pie_evals.schema import JobSpec

LOW_WATER_GIB = 80.0  # start reclaiming below this much free space


def free_gib(path: Path) -> float:
    try:
        st = os.statvfs(path)
    except OSError:
        return float("inf")
    return st.f_bavail * st.f_frsize / 2**30


def cache_root() -> Path | None:
    from .build import DEFAULT_CACHE

    return Path(os.environ.get("PIE_EVALS_CACHE", DEFAULT_CACHE))


def reclaim(job: JobSpec, hf_cache: Path, *, low_water_gib: float = LOW_WATER_GIB, log=print) -> float:
    """Free space for ``job`` when the disk is short; returns GiB freed."""
    root = cache_root()
    probe = root if root and root.exists() else hf_cache
    before = free_gib(probe)
    if before >= low_water_gib:
        return 0.0
    commit = (job.pie_commit or "")[:8]
    wanted_ids = {c.artifact.id for c in job.cells}
    wanted_models = {c.artifact.base_model for c in job.cells}
    log(f"disk: {before:.0f} GiB free under {low_water_gib:.0f}; reclaiming what this job does not need")
    if root and root.exists():
        for d in sorted((root / "artifacts").glob("*")) if (root / "artifacts").exists() else []:
            aid, _, tail = d.name.rpartition("-")
            if d.name.startswith(".tmp-") or aid not in wanted_ids or (commit and tail != commit):
                _rm(d, log)
        for d in sorted((root / "builds").glob("*")) if (root / "builds").exists() else []:
            if commit and not d.name.startswith(commit):
                _rm(d, log)
    if free_gib(probe) < low_water_gib:
        if os.environ.get("PIE_EVALS_RECLAIM_HF") == "1":
            _drop_hf_repos(hf_cache, wanted_models, log)
        else:
            log(f"disk still short; HF repos are not dropped unless PIE_EVALS_RECLAIM_HF=1 (the cache {hf_cache} is shared)")
    after = free_gib(probe)
    log(f"disk: {after:.0f} GiB free after reclaiming")
    return max(0.0, after - before)


def _drop_hf_repos(hf_cache: Path, keep_models: set[str], log) -> None:
    """Delete every cached HF repo this job does not use, through huggingface_hub
    so the shared blob store is cleaned with the refs."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return
    try:
        info = scan_cache_dir(hf_cache)
    except Exception as e:  # noqa: BLE001 — an unreadable cache is left alone
        log(f"hf cache scan failed ({str(e)[:80]}); not touched")
        return
    revisions = []
    for repo in info.repos:
        if repo.repo_id in keep_models:
            continue
        revisions += [rev.commit_hash for rev in repo.revisions]
        log(f"reclaim: hf repo {repo.repo_id} ({repo.size_on_disk / 2**30:.1f} GiB)")
    if revisions:
        info.delete_revisions(*revisions).execute()


def _rm(path: Path, log) -> None:
    size = _du(path)
    shutil.rmtree(path, ignore_errors=True)
    log(f"reclaim: {path} ({size / 2**30:.1f} GiB)")


def _du(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total
