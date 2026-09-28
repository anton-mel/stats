"""What exactly was measured. A record without this is not a record.

``quant_block_hash`` hashes the checkpoint's *entire* quantization block —
including per-tensor overrides. Reading only the top-level ``bits`` /
``group_size`` was exactly the defect that produced wrong tokens for a week:
two checkpoints with the same headline scheme and different per-tensor
entries hashed the same, and a regression hid behind an "identical"
artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pie_evals.schema import Provenance

ENV_PREFIXES = ("PIE_", "OLLAMA_")
ENV_EXACT = ("HF_HUB_CACHE", "HF_HOME", "OMP_NUM_THREADS", "METAL_DEVICE_WRAPPER_TYPE", "MTL_DEBUG_LAYER")


def _run(argv: list[str], timeout: float = 30.0, cwd: str | Path | None = None) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


# ---- hardware ------------------------------------------------------------------


def _mac_fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {"os": "macos", "arch": platform.machine()}
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if brand:
        fp["cpu"] = brand.strip()
    mem = _run(["sysctl", "-n", "hw.memsize"])
    if mem and mem.strip().isdigit():
        fp["memory_gib"] = round(int(mem.strip()) / 1024**3, 1)
    prof = _run(["system_profiler", "SPHardwareDataType"], timeout=60)
    if prof:
        for line in prof.splitlines():
            s = line.strip()
            for key, label in (("chip", "Chip:"), ("cores", "Total Number of Cores:"),
                               ("model", "Model Identifier:"), ("memory", "Memory:")):
                if s.startswith(label):
                    fp[key] = s[len(label):].strip()
    gpu = _run(["system_profiler", "SPDisplaysDataType"], timeout=60)
    if gpu:
        m = re.search(r"Total Number of Cores:\s*(\d+)", gpu)
        if m:
            fp["gpu_cores"] = m.group(1)
    sw = _run(["sw_vers", "-productVersion"])
    if sw:
        fp["macos"] = sw.strip()
    return fp


def hardware_fingerprint() -> dict[str, Any]:
    fp = _mac_fingerprint()
    fp["hostname"] = platform.node()
    return fp


# ---- versions -------------------------------------------------------------------


def versions() -> dict[str, Any]:
    v: dict[str, Any] = {"os_version": platform.platform(), "python": platform.python_version()}
    sw = _run(["sw_vers", "-productVersion"])
    if sw:
        v["os_version"] = f"macOS {sw.strip()}"
    return v


def git_commit(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    out = _run(["git", "-C", str(p), "rev-parse", "HEAD"])
    if not out:
        return None
    sha = out.strip()
    dirty = _run(["git", "-C", str(p), "status", "--porcelain", "--untracked-files=no"])
    if dirty and dirty.strip():
        sha += "-dirty"
    return sha


# ---- checkpoint -----------------------------------------------------------------


def checkpoint_revision(snapshot_dir: str | Path) -> str:
    """The snapshot directory name under the HF cache is the commit; for a
    miniature it is the recipe tag (``miniature_snapshot_dir`` puts it
    there). A directory not laid out that way is identified by its name."""
    p = Path(snapshot_dir).resolve()
    return p.name


def _load_config(snapshot_dir: str | Path) -> dict[str, Any] | None:
    cfg = Path(snapshot_dir) / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text())
    except (OSError, ValueError):
        return None


def quant_block(snapshot_dir: str | Path) -> dict[str, Any] | None:
    """The checkpoint's whole quantization block — ``quantization`` (MLX,
    with its per-tensor overrides) or ``quantization_config`` (HF)."""
    cfg = _load_config(snapshot_dir)
    if not cfg:
        return None
    block = cfg.get("quantization")
    if block is None:
        block = cfg.get("quantization_config")
    if block is None and isinstance(cfg.get("text_config"), dict):
        tc = cfg["text_config"]
        block = tc.get("quantization") or tc.get("quantization_config")
    return block if isinstance(block, dict) else None


def quant_block_hash(snapshot_dir: str | Path) -> str | None:
    block = quant_block(snapshot_dir)
    if block is None:
        return None
    return hashlib.sha256(json.dumps(block, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---- env -------------------------------------------------------------------------


def relevant_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for k, v in env.items():
        if k.startswith(ENV_PREFIXES) or k in ENV_EXACT:
            out[k] = v
    return dict(sorted(out.items()))


# ---- assembly --------------------------------------------------------------------


def harness_root() -> Path | None:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / ".git").exists():
            return parent
    return None


def build_provenance(
    pie_root: str | Path,
    snapshot_dir: str | Path,
    engine_version: str | None,
    engine_config: dict[str, Any],
    recipe_name: str | None,
    runner: str | None,
    *,
    pie_build_features: list[str] | None = None,
    machine_state: dict[str, Any] | None = None,
    dataset_hashes: dict[str, str] | None = None,
    harness_path: str | Path | None = None,
) -> Provenance:
    v = versions()
    hroot = Path(harness_path) if harness_path else harness_root()
    return Provenance(
        harness_commit=git_commit(hroot) if hroot else None,
        pie_commit=git_commit(pie_root),
        pie_build_features=list(pie_build_features or []),
        engine_version=engine_version,
        engine_config=dict(engine_config or {}),
        engine_config_recipe=recipe_name,
        checkpoint_revision=checkpoint_revision(snapshot_dir),
        quant_block_hash=quant_block_hash(snapshot_dir),
        dataset_hashes=dict(dataset_hashes or {}),
        hardware_fingerprint=hardware_fingerprint(),
        driver_version=v.get("driver_version"),
        cuda_version=v.get("cuda_version"),
        os_version=v.get("os_version"),
        env=relevant_env(),
        runner=runner,
        machine_state=dict(machine_state or {}),
    )
