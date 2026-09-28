"""Workload-shape helpers shared by every adapter.

The bench scripts take the *engine* side of a shape (admission cap, context
length) as flags, while ``pie_evals.node.workloads.synthetic`` produces the
*client* side (prompt, max-tokens, concurrency). Both derive from the same
``WorkloadSpec`` so that pie and every baseline are handed the same numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

from pie_evals.schema import WorkloadSpec

#: pie's KV page is 16 tokens and vLLM's default block is 16; every engine
#: here rounds its context to something coarser than that, so the cross-engine
#: ``--max-model-len`` is quantised to 2048 and never sits on a shape boundary.
MAX_MODEL_LEN_QUANTUM = 2048

#: common.py wraps every prompt in a system+user chat template; the wrapper
#: costs a few dozen tokens on the Qwen/Llama/Gemma templates. Budgeted before
#: rounding so a shape that lands exactly on a quantum is not truncated.
CHAT_TEMPLATE_MARGIN_TOKENS = 64


def workload_concurrency(workload: WorkloadSpec) -> int:
    """The engine-side admission cap this shape runs at (1 for latency shapes).

    Mirrors ``base.workload_mode_args``: the ``latency`` subcommand always
    means one request in flight, and the ``tput`` subcommand takes the shape's
    ``concurrency`` parameter.
    """
    p = workload.params
    kind = str(workload.kind)
    if kind in ("single_stream", "control_aa"):
        return 1
    if kind == "long_context":
        return max(1, int(p.get("concurrency", 1)))
    return max(1, int(p.get("concurrency", 8)))


def workload_num_requests(workload: WorkloadSpec) -> int:
    p = workload.params
    kind = str(workload.kind)
    if kind in ("single_stream", "control_aa"):
        return int(p.get("requests", 8))
    if kind == "long_context":
        conc = workload_concurrency(workload)
        if conc <= 1:
            return int(p.get("requests", 4))
        return int(p.get("num_requests", conc * 2))
    conc = workload_concurrency(workload)
    return int(p.get("num_requests", p.get("variants", conc * 4)))


def context_tokens_for(workload: WorkloadSpec) -> int:
    """Longest prompt + output any single request of this shape needs."""
    p = workload.params
    kind = str(workload.kind)
    if kind in ("single_stream", "control_aa", "long_context", "concurrency"):
        return int(p.get("prefill", 128)) + int(p.get("decode", 64))
    if kind == "prefix_shared":
        return int(p.get("shared_prefix", 1024)) + int(p.get("unique_suffix", 64)) + int(p.get("decode", 64))
    if kind == "mixed_length":
        return int(p.get("max_prompt", 4096)) + int(p.get("max_output", 1024))
    if kind == "replay":
        # synthetic.py realises replay as a 512-token prompt / 256-token output
        # until common.py grows a --trace flag.
        return 512 + 256
    raise ValueError(f"unknown workload kind {kind}")


def max_model_len_for(workload: WorkloadSpec) -> int:
    """``--max-model-len`` for this shape: prefill + decode (+ template margin)
    rounded up to a multiple of ``MAX_MODEL_LEN_QUANTUM``."""
    need = context_tokens_for(workload) + CHAT_TEMPLATE_MARGIN_TOKENS
    q = MAX_MODEL_LEN_QUANTUM
    return ((need + q - 1) // q) * q


def is_prefix_workload(workload: WorkloadSpec) -> bool:
    return str(workload.kind) == "prefix_shared"


# ---- HF cache lookups (huggingface_hub-free, mirrors scripts/bench/common.py) -------


def hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if cache := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(cache).expanduser())
    if home := os.environ.get("HF_HOME"):
        roots.append(Path(home).expanduser() / "hub")
    roots.append(Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface" / "hub")
    return roots


def hf_snapshot_dir(repo_id: str, revision: str | None = None) -> Path | None:
    """The local snapshot directory of ``repo_id`` or None when not downloaded.

    A path that exists is returned as-is (a checkpoint may be given by path).
    Otherwise the cache is searched the way ``common.resolve_local_model``
    does (``refs/main`` -> ``snapshots/<rev>``), with ``revision`` winning when
    given and the newest snapshot as a last resort.
    """
    candidate = Path(repo_id).expanduser()
    if candidate.exists():
        return candidate.resolve()
    encoded = f"models--{repo_id.replace('/', '--')}"
    for root in hf_cache_roots():
        repo = root / encoded
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        if revision and (snapshots / revision).is_dir():
            return (snapshots / revision).resolve()
        for ref_name in ("main", "master"):
            ref = repo / "refs" / ref_name
            if ref.is_file():
                rev = ref.read_text().strip()
                if (snapshots / rev).is_dir():
                    return (snapshots / rev).resolve()
        dirs = sorted((d for d in snapshots.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime_ns)
        if dirs:
            return dirs[-1].resolve()
    return None


def hf_snapshot_file(repo_id: str, file_name: str, revision: str | None = None) -> Path | None:
    """``file_name`` inside the local snapshot of ``repo_id`` (GGUF arms), or None."""
    snap = hf_snapshot_dir(repo_id, revision)
    if snap is None:
        return None
    direct = snap / file_name
    if direct.is_file():
        return direct.resolve()
    # GGUF repos sometimes nest quant files one directory down (Q4_K_M/...).
    hits = sorted(snap.rglob(Path(file_name).name))
    return hits[0].resolve() if hits else None


def serve_envelope(workloads: list[WorkloadSpec]) -> WorkloadSpec:
    """The shape one server must admit for every workload of a process: the
    longest context of any of them and the highest concurrency of any of them.
    Picking the single "widest" workload by concurrency × context let a
    long-context latency shape win over c8 in the smoke tier, so the server
    booted with an admission cap of 1 and every smoke c8 measured single-stream
    (4090 smoke c8 422 tok/s against 6716 in the nightly)."""
    by_ctx = max(workloads, key=context_tokens_for)
    conc = max(workload_concurrency(w) for w in workloads)
    reqs = max(workload_num_requests(w) for w in workloads)
    if workload_concurrency(by_ctx) >= conc:
        return by_ctx
    # keep the kind: a long_context shape passes its prompt as a shared prefix,
    # and recast as `concurrency` it would put 8k-32k words on the argv (E2BIG,
    # nightly 35971360265); only the latency kinds ignore a concurrency param
    kind = "concurrency" if str(by_ctx.kind) in ("single_stream", "control_aa") else by_ctx.kind
    return by_ctx.model_copy(update={"kind": kind, "params": {**by_ctx.params, "concurrency": conc, "num_requests": max(reqs, conc * 2)}})
