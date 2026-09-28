"""Checkpoints on the node.

pie's ``resolve_local_model`` refuses to touch the network, so pie-evals
fetches checkpoints itself through ``huggingface_hub`` (``HF_TOKEN`` honoured
for gated repos). An Ollama artifact's weights live in Ollama's own store; its
HF repo only supplies the tokenizer and chat template the shared client renders
prompts with, so only those files are fetched for it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pie_evals.schema import ArtifactSpec

WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*", "*.py", "*.jinja"]


def num_layers_of(snapshot_dir: str | Path) -> int | None:
    cfg_path = Path(snapshot_dir) / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return None
    n = cfg.get("num_hidden_layers")
    if n is None and isinstance(cfg.get("text_config"), dict):
        n = cfg["text_config"].get("num_hidden_layers")
    return int(n) if n is not None else None


def hf_cache_dir() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else Path.home() / ".cache/huggingface/hub"))


def snapshot_dir_if_present(artifact: ArtifactSpec, hf_cache: Path) -> Path | None:
    org, _, name = artifact.base_model.partition("/")
    base = hf_cache / f"models--{org}--{name}" / "snapshots"
    marker = artifact.gguf_file or "config.json"  # a GGUF repo carries no config.json
    if artifact.revision and (base / artifact.revision / marker).exists() and has_weights(base / artifact.revision, artifact):
        return base / artifact.revision
    snaps = sorted(d for d in base.glob("*") if (d / marker).exists() and has_weights(d, artifact)) if base.exists() else []
    return snaps[-1] if snaps else None


def has_weights(snapshot: Path, artifact: ArtifactSpec) -> bool:
    """A snapshot whose weight files are gone (the blobs behind the symlinks
    reclaimed; gemma-4-E4B on the L40S pod, nightly 36099816046 s07) still
    carries its config.json and would be served: vLLM then boots into
    "Cannot find any model weights". ``exists`` follows the symlink. An
    Ollama artifact needs only the tokenizer."""
    if artifact.gguf_file:
        return (snapshot / artifact.gguf_file).exists()
    return any(f.exists() for f in snapshot.glob("*.safetensors"))


class SnapshotMissing(FileNotFoundError):
    pass


def ensure_snapshot(artifact: ArtifactSpec, hf_cache: Path, *, download: bool = True, log=print) -> Path:
    """Return the artifact's snapshot dir, downloading it only when
    ``download`` is set."""
    have = snapshot_dir_if_present(artifact, hf_cache)
    if have:
        return _with_gguf_config(artifact, have, log)
    if not download:
        raise SnapshotMissing(f"checkpoint {artifact.base_model} not in HF cache {hf_cache} (run `pie-evals-node prepare`, or `run --download`; pie's resolve_local_model refuses network)")
    from huggingface_hub import snapshot_download

    log(f"download: {artifact.base_model}@{artifact.revision or 'main'} -> {hf_cache}")
    kw = {"revision": artifact.revision} if artifact.revision else {}
    if artifact.gguf_file:
        kw["allow_patterns"] = ["*.json", artifact.gguf_file]
    else:
        kw["allow_patterns"] = WEIGHT_PATTERNS
    path = snapshot_download(artifact.base_model, cache_dir=str(hf_cache), token=os.environ.get("HF_TOKEN") or None, **kw)  # an empty secret must not become "Bearer "
    return _with_gguf_config(artifact, Path(path), log)


def _with_gguf_config(artifact: ArtifactSpec, path: Path, log=print) -> Path:
    """pie reads a snapshot's encoding from config.json ("a snapshot must carry
    the config.json its encoding is read from"); a GGUF repo ships none, so the
    base model's is copied in — also into a snapshot fetched before this existed."""
    if artifact.gguf_config_from and not (path / "config.json").exists():
        from huggingface_hub import hf_hub_download

        log(f"download: config.json of {artifact.gguf_config_from} -> {path}")
        src = hf_hub_download(artifact.gguf_config_from, "config.json", token=os.environ.get("HF_TOKEN") or None)
        (path / "config.json").write_bytes(Path(src).read_bytes())
    return path
