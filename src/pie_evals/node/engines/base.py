"""Engine adapter contract.

Every adapter drives a bench script built on the pinned pie checkout's
``scripts/bench/common.py`` — one client, one prompt construction, one JSON
envelope — which is what makes a cross-engine number a comparison at all:
pie through ``scripts/bench/pie_bench.py``, Ollama through this repo's
``ollama_bench.py`` (which imports the same ``common.py``).
The adapter's job is therefore: build the argv for a (artifact, mode,
workload, recipe), run it under a timeout, classify failure, and lift the
JSON into ``PerfMetrics``.

An adapter that needs to hold an engine process across workloads (one model
per process, workloads interleaved inside it) uses the ``serve``/``attach``
pair; the bench scripts support that through ``PIE_BENCH_SERVE_ONLY`` /
``PIE_BENCH_SERVER_URL`` for pie and a spawned ``ollama serve`` for Ollama.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pie_evals.schema import (
    ArtifactSpec,
    EngineName,
    ErrorClass,
    Mode,
    PerfMetrics,
    PlatformSpec,
    WorkloadSpec,
)


class EngineLaunchError(RuntimeError):
    def __init__(self, error_class: ErrorClass, message: str):
        super().__init__(message)
        self.error_class = error_class


@dataclass
class BenchResult:
    perf: PerfMetrics
    summary: dict[str, Any]
    requests: list[dict[str, Any]]
    argv: list[str]
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0
    output_token_ids: list[list[int]] = field(default_factory=list)


# Patterns that classify a failed run. Order matters: the first match wins.
_ERROR_PATTERNS: list[tuple[ErrorClass, re.Pattern[str]]] = [
    (ErrorClass.OOM, re.compile(r"out of memory|OutOfMemoryError|kIOGPUCommandBufferCallbackErrorOutOfMemory|MTLResourceOptions.*failed|insufficient memory", re.I)),
    (ErrorClass.DOESNT_FIT, re.compile(r"does not fit|refus\w+ .*fit|exceeds .*working set|recommendedMaxWorkingSetSize|host fit check", re.I)),
    (ErrorClass.LOAD_FAIL, re.compile(r"failed to load|load plan|missing quant metadata|unknown variant|No such file|safetensors|model import|failed to import|not found in cache|resolve_local_model|matches no SKU|no SKU this build|missing .*\.wasm|not in HF cache|pull model manifest|model .* not found", re.I)),
    (ErrorClass.CRASH, re.compile(r"Traceback|panicked at|Segmentation fault|SIGSEGV|SIGABRT|invalid resource handle|invalid argument", re.I)),
]


def classify_failure(returncode: int, stderr: str, stdout: str, timed_out: bool) -> tuple[ErrorClass, str]:
    if timed_out:
        return ErrorClass.HANG, "timed out (hang is only ever observed via timeout)"
    text = stderr + "\n" + stdout
    # A Python traceback is described by its exception line (the last
    # "SomeError: ..." line), never by a code line that happens to contain a
    # keyword ("resolved_model = resolve_local_model(args.model)" is not a message).
    if "Traceback (most recent call last)" in text:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        exc = next((ln.strip() for ln in reversed(lines) if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Exit|Interrupt)\b", ln.strip())), lines[-1].strip() if lines else "Traceback")
        for cls, pat in _ERROR_PATTERNS[:3]:
            if pat.search(exc):
                return cls, exc[:400]
        return ErrorClass.CRASH, exc[:400]
    for cls, pat in _ERROR_PATTERNS:
        m = pat.search(text)
        if m:
            line = next((ln for ln in text.splitlines() if m.group(0) in ln), m.group(0))
            return cls, line.strip()[:400]
    if returncode < 0:
        return ErrorClass.CRASH, f"killed by signal {-returncode}"
    return ErrorClass.CRASH, f"exit {returncode}: " + (stderr.strip().splitlines() or ["(no stderr)"])[-1][:400]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def perf_from_common_json(summary: dict[str, Any], requests: list[dict[str, Any]], num_layers: int | None, concurrency: int | None = None) -> PerfMetrics:
    """Lift ``scripts/bench/common.py``'s BenchSummary + RequestResult rows."""
    completed = [r for r in requests if r.get("ok")]
    out_tok = int(summary.get("output_tokens") or sum(r.get("output_tokens", 0) for r in completed))
    prompt_tok = int(summary.get("prompt_tokens") or sum(r.get("prompt_tokens", 0) for r in completed))
    wall = float(summary.get("wall_s") or 0.0)
    output_tok_s = float(summary.get("output_tok_per_s") or (out_tok / wall if wall else 0.0))
    ttft = [r["ttft_s"] * 1000.0 for r in completed if r.get("ttft_s") is not None]
    # decode tok/s for a lane = output tokens / (latency - ttft)
    decode_lane = []
    prefill_lane = []
    itl_all: list[float] = []
    for r in completed:
        lat = r.get("latency_s")
        tt = r.get("ttft_s")
        n = r.get("output_tokens") or 0
        if lat and tt is not None and n > 1 and lat > tt:
            decode_lane.append((n - 1) / (lat - tt))
        if tt and r.get("prompt_tokens"):
            prefill_lane.append(r["prompt_tokens"] / tt)
        it = r.get("intertoken_us") or []
        if isinstance(it, list):
            itl_all.extend(float(x) / 1000.0 for x in it)
    counters = {}
    for k, v in (summary.get("config") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            counters[str(k)] = float(v)
    # derived counters (device_idle_pct, host_us_per_step, ...) are filled by the pie adapter
    concurrency = concurrency or int((summary.get("config") or {}).get("concurrency") or 0) or None
    decode_tok_s = float(np.median(decode_lane)) if decode_lane else None
    # single-stream: the lane's decode rate is the aggregate; concurrency: aggregate output tok/s is the headline
    primary = "decode_tok_s" if (concurrency in (None, 1)) and decode_tok_s else "output_tok_s"
    ms_per_tok_per_layer = None
    if num_layers and decode_tok_s:
        ms_per_tok_per_layer = 1000.0 / decode_tok_s / num_layers
    return PerfMetrics(
        primary=primary,
        decode_tok_s=decode_tok_s,
        prefill_tok_s=float(np.median(prefill_lane)) if prefill_lane else None,
        output_tok_s=output_tok_s or None,
        ttft_ms_p50=_pct(ttft, 50) if ttft else (summary.get("ttft_p50_ms") or None),
        ttft_ms_p99=_pct(ttft, 99) if ttft else (summary.get("ttft_p99_ms") or None),
        itl_ms_p50=_pct(itl_all, 50) if itl_all else (summary.get("intertoken_p50_ms") or None),
        itl_ms_p99=_pct(itl_all, 99) if itl_all else (summary.get("intertoken_p99_ms") or None),
        latency_ms_p50=summary.get("latency_p50_ms"),
        latency_ms_p99=summary.get("latency_p99_ms"),
        prompt_tokens=prompt_tok,
        output_tokens=out_tok,
        requests=int(summary.get("requests") or len(requests)),
        failed=int(summary.get("failed") or sum(1 for r in requests if not r.get("ok"))),
        ms_per_token_per_layer=ms_per_tok_per_layer,
        counters=counters,
    )


class Engine(ABC):
    """One adapter per engine. Instances are per (artifact, mode, platform)."""

    name: EngineName
    #: bench script under the pie checkout, e.g. "scripts/bench/pie_bench.py"
    script: str
    #: environment the engine's processes get unless the caller set the key
    env_defaults: dict[str, str] = {}

    def __init__(
        self,
        *,
        pie_root: Path,
        artifact: ArtifactSpec,
        platform: PlatformSpec,
        mode: Mode,
        recipe: dict[str, Any],
        python: str | None = None,
        env: dict[str, str] | None = None,
        num_layers: int | None = None,
    ):
        self.pie_root = Path(pie_root)
        self.artifact = artifact
        self.platform = platform
        self.mode = mode
        self.recipe = recipe  # engine knobs for this (engine, platform, artifact) — see recipes/
        self.python = python or self.default_python()
        self.env = dict(env or {})
        for k, v in {**self.env_defaults, **self.default_env()}.items():
            self.env.setdefault(k, v)
        self.num_layers = num_layers
        self.server_url: str | None = None
        self._server_proc: subprocess.Popen | None = None

    def default_env(self) -> dict[str, str]:
        """Runtime-computed environment defaults (paths under the cache root);
        static ones go in ``env_defaults``. The caller's env wins over both."""
        return {}

    # ---- to implement per engine ---------------------------------------------
    @abstractmethod
    def default_python(self) -> str: ...

    @abstractmethod
    def model_arg(self) -> str:
        """What ``--model`` receives for this engine (HF id, GGUF path, snapshot dir...)."""

    @abstractmethod
    def engine_args(self, workload: WorkloadSpec) -> list[str]:
        """Engine-specific flags: memory, admission, recipe knobs."""

    def version(self) -> str:
        return "unknown"

    def counters_from_summary(self, summary: dict[str, Any]) -> dict[str, float]:
        return {}

    def resident_gib(self) -> float | None:
        return None

    # ---- optional long-lived server (one model per process) ------------------
    def serve(self, workload: WorkloadSpec, log_path: Path, timeout_s: int) -> None:
        """Default: no persistent server; each ``run`` boots the engine."""
        return None

    def stop(self) -> None:
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def leftover_process_names(self) -> list[str]:
        """Process names that may outlive the parent and hold the GPU."""
        return []

    # ---- the run -------------------------------------------------------------
    def script_path(self) -> Path:
        """The bench script this adapter runs (a path under the pie checkout by default)."""
        return self.pie_root / self.script

    def build_argv(self, workload: WorkloadSpec, common_args: list[str], json_out: Path) -> list[str]:
        mode_sub, mode_args = workload_mode_args(workload)
        argv = [self.python, str(self.script_path()), mode_sub]
        argv += ["--model", self.model_arg()]
        argv += mode_args
        argv += common_args
        argv += self.engine_args(workload)
        argv += ["--json-out", str(json_out), "--dump-all-token-ids"]
        return argv

    def run(self, workload: WorkloadSpec, common_args: list[str], out_dir: Path, timeout_s: int) -> BenchResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_out = out_dir / "bench.json"
        argv = self.build_argv(workload, common_args, json_out)
        env = {**os.environ, **self.env}
        if self.server_url:
            env["PIE_BENCH_SERVER_URL"] = self.server_url
        (out_dir / "argv.txt").write_text(shlex.join(argv) + "\n")
        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                argv, cwd=str(self.pie_root / "scripts/bench"), env=env, capture_output=True, text=True, timeout=timeout_s
            )
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc, stdout, stderr = -1, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        dur = time.monotonic() - t0
        (out_dir / "stdout.txt").write_text(stdout)
        (out_dir / "stderr.txt").write_text(stderr)
        if rc != 0 or not json_out.exists():
            cls, msg = classify_failure(rc, stderr, stdout, timed_out)
            if rc == 0 and not timed_out:
                lines = [ln.strip() for ln in (stderr + "\n" + stdout).splitlines() if ln.strip() and not set(ln.strip()) <= set("─│╭╮╰╯═║╔╗╚╝ ")]
                msg = "bench exited 0 without writing bench.json; last output: " + " | ".join(lines[-3:])[:400]
                cls = ErrorClass.HARNESS_INVALID
            raise EngineLaunchError(cls, msg)
        data = json.loads(json_out.read_text())
        summary = data.get("summary", {})
        requests = data.get("requests", [])
        failed = [r for r in requests if not r.get("ok")]
        if requests and len(failed) == len(requests):
            err = next((str(r.get("error") or "") for r in failed if r.get("error")), "(no error text)")
            cls = ErrorClass.HARNESS_INVALID if re.search(r"temperature must be|Failed to parse JSON input", err) else ErrorClass.INCOMPATIBLE if re.search(r"forward-hybrid|not valid on|interface", err) else ErrorClass.CRASH
            raise EngineLaunchError(cls, f"all {len(requests)} requests failed: {err[:300]}")
        perf = perf_from_common_json(summary, requests, self.num_layers, concurrency=int(workload.params.get("concurrency", 1)))
        perf.counters.update(self.counters_from_summary(summary))
        perf.resident_gib = self.resident_gib()
        ids = [r.get("output_token_ids") or [] for r in requests]
        return BenchResult(
            perf=perf, summary=summary, requests=requests, argv=argv,
            stdout_tail=stdout[-4000:], stderr_tail=stderr[-4000:], duration_s=dur, output_token_ids=ids,
        )


def workload_mode_args(workload: WorkloadSpec) -> tuple[str, list[str]]:
    """Map a WorkloadSpec onto common.py's ``latency`` / ``tput`` subcommands."""
    p = workload.params
    kind = str(workload.kind)
    if kind in ("single_stream", "control_aa"):
        return "latency", ["--requests", str(p.get("requests", 8))]
    if kind == "long_context":
        conc = int(p.get("concurrency", 1))
        if conc <= 1:
            return "latency", ["--requests", str(p.get("requests", 4))]
        return "tput", ["--num-requests", str(p.get("num_requests", conc * 2)), "--concurrency", str(conc)]
    if kind in ("concurrency", "prefix_shared", "mixed_length", "replay"):
        conc = int(p.get("concurrency", 8))
        n = int(p.get("num_requests", p.get("variants", conc * 4)))
        return "tput", ["--num-requests", str(n), "--concurrency", str(conc)]
    raise ValueError(f"unknown workload kind {kind}")


_REGISTRY: dict[str, type[Engine]] = {}


def register(cls: type[Engine]) -> type[Engine]:
    _REGISTRY[str(cls.name)] = cls
    return cls


def get_engine(name: str | EngineName) -> type[Engine]:
    # import side effects register the adapters
    from . import pie  # noqa: F401

    return _REGISTRY[str(name)]
