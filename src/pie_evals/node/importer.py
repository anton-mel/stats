"""pie artifacts (``.zt``) for checkpoints pie will not serve directly.

pie serves an HF bf16 snapshot as it is, but a quantized MLX checkpoint's
weight planes must be relaid for the kernels, and pie refuses to do that at
boot ("this load would relayout a weight plane ... Run `pie model import`").
``pie model import <snapshot> --out <dir>`` writes ``<dir>/<name>.zt`` once;
the bench then takes that file as ``--model``. Artifacts are kept per
(artifact, pie commit) on the shared cache: the stamp inside a .zt is
checked against the build that reads it.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from pie_evals.schema import ArtifactSpec, SourceFormat

RELAYOUT_MARK = "would relayout a weight plane"


def needs_import(artifact: ArtifactSpec) -> bool:
    # pie serve reads safetensors and .zt only; mlx and gguf checkpoints go
    # through `pie model import` (nightly 35967154121: the GGUF snapshot dir
    # "holds no .safetensors and no .zt container")
    return artifact.source_format in (SourceFormat.MLX, SourceFormat.GGUF)


def artifact_root(root: Path | None = None) -> Path:
    from .build import DEFAULT_CACHE

    return root or Path(os.environ.get("PIE_EVALS_CACHE", DEFAULT_CACHE)) / "artifacts"


def artifact_dir(artifact: ArtifactSpec, commit: str, root: Path | None = None) -> Path:
    return artifact_root(root) / f"{artifact.id}-{commit[:8]}"


def find_zt(d: Path) -> Path | None:
    zts = sorted(d.glob("*.zt")) if d.exists() else []
    return zts[-1] if zts else None


def ensure_artifact(artifact: ArtifactSpec, snapshot: Path, pie_bin: Path, commit: str, *, root: Path | None = None, timeout_s: int = 3600, log=print) -> Path:
    """Return the .zt for ``artifact`` at ``commit``, importing if absent.
    Concurrent pods: import into a private temp dir, then rename into place."""
    d = artifact_dir(artifact, commit, root)
    have = find_zt(d)
    if have:
        return have
    tmp = d.parent / f".tmp-{d.name}-{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:6]}"  # pods on one volume share pids (pie #650: a shared spool zeroes the loser)
    tmp.mkdir(parents=True, exist_ok=True)
    source = snapshot / artifact.gguf_file if artifact.gguf_file else snapshot  # a GGUF repo is one file plus the base model's config.json
    argv = [str(pie_bin), "model", "import", str(source), "--out", str(tmp), "--keep-source"]
    if artifact.pie_sku:
        argv += ["--sku", artifact.pie_sku]
    env = {**os.environ}
    log(f"import: {' '.join(argv)}")
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
    zt = find_zt(tmp)
    if proc.returncode != 0 or zt is None:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError(f"pie model import failed for {artifact.id} (exit {proc.returncode}):\n" + "\n".join(tail))
    if find_zt(d):  # someone else finished first
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        return find_zt(d)  # type: ignore[return-value]
    d.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, d)
    return find_zt(d)  # type: ignore[return-value]
