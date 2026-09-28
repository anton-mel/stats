"""pie adapter: drives ``scripts/bench/pie_bench.py``.

Everything emitted here is a flag ``pie_bench.py`` (or ``scripts/bench/common.py``)
parses; knobs the script has no flag for are noted with ``TODO(pie/benches)``
rather than invented. The bench inferlet (``text-completion-bench`` by
default) is pointed at through ``PIE_BENCH_INFERLET_DIR`` and ``--inferlet-dir``.
"""

from __future__ import annotations

import collections
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pie_evals.schema import Backend, EngineName, ErrorClass, WorkloadSpec

from .base import Engine, EngineLaunchError, classify_failure, register
from .shape import max_model_len_for

#: platform backend -> pie_bench ``--engine``
_BACKEND_TO_ENGINE: dict[Backend, str] = {
    Backend.METAL: "metal",
}

#: recipe knob -> flag, value passed through as ``str(value)``.
_DIRECT_FLAGS: tuple[tuple[str, str], ...] = (
    ("total_pages", "--total-pages"),
    ("swap_pool_size", "--swap-pool-size"),
    ("frame_size", "--frame-size"),
    ("frame_submit_depth", "--frame-submit-depth"),
    ("frame_dispatch_depth", "--frame-dispatch-depth"),
    ("submit_deadline", "--submit-deadline"),
    ("run_ahead_frames", "--run-ahead-frames"),
    ("max_forward_tokens", "--max-forward-tokens"),
    ("max_forward_requests", "--max-forward-requests"),
    ("worker_threads", "--worker-threads"),
    ("wasm_warm_slots", "--wasm-warm-slots"),
    ("wasm_warm_memory_mb", "--wasm-warm-memory-mb"),
    ("default_token_limit", "--default-token-limit"),
    ("default_endowment_pages", "--default-endowment-pages"),
    ("admission_oversubscription_factor", "--admission-oversubscription-factor"),
    ("device_weight_budget", "--device-weight-budget"),
    ("host_weight_budget", "--host-weight-budget"),
    ("pie_bin", "--pie-bin"),
    ("server_startup_timeout", "--server-startup-timeout"),
    ("cpu_mem_budget", "--cpu-mem-budget"),
    ("cpu_affinity", "--cpu-affinity"),
    ("request_timeout", "--request-timeout"),
    ("wasm_delay_us", "--wasm-delay-us"),
    ("venv", "--venv"),
    # NOTE: ``--kv-pages`` exists but pie_bench documents it as DEAD for
    # cuda_native (never reaches engine_options); ``total_pages`` is the cap.
)

#: recipe knob -> BooleanOptionalAction flag (True -> --x, False -> --no-x).
_BOOL_OPTIONAL_FLAGS: tuple[tuple[str, str], ...] = (
    ("pretokenized_prompts", "--pretokenized-prompts"),
    ("single_process_batch", "--single-process-batch"),
    ("defer_start", "--defer-start"),
    ("system_speculation", "--system-speculation"),
)

#: recipe knob -> store_true flag.
_STORE_TRUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("host_first_token", "--host-first-token"),
    ("report_arrivals", "--report-arrivals"),
    ("report_wall_clock", "--report-wall-clock"),
)


def _scalar(v: Any) -> str:
    """Format a value the way pie_bench's ``--engine-option`` parser reads it back."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _num(cfg: dict[str, Any], *keys: str) -> float | None:
    """First numeric value among ``keys`` in ``cfg`` (labelled or raw status key)."""
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


@register
class PieEngine(Engine):
    name = EngineName.PIE
    script = "scripts/bench/pie_bench.py"

    #: path under the pie tree of the bench inferlet project (a built
    #: ``Pie.toml`` + ``target/wasm32-wasip2/release/*.wasm``).
    DEFAULT_PROGRAM_PATH = "examples/text-completion-bench"

    def __init__(self, *args: Any, program_path: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._program_path = ""
        self.program_path = program_path or self.recipe.get("program_path") or self.DEFAULT_PROGRAM_PATH
        sdk_paths = [
            str(self.pie_root / "python" / "client" / "src"),
            str(self.pie_root / "python" / "server" / "python"),
        ]
        existing = self.env.get("PYTHONPATH") or os.environ.get("PYTHONPATH")
        self.env["PYTHONPATH"] = ":".join(sdk_paths + ([existing] if existing else []))
        self._server_log = None

    # ---- program (inferlet) ------------------------------------------------
    @property
    def program_path(self) -> str:
        return self._program_path

    @program_path.setter
    def program_path(self, value: str) -> None:
        self._program_path = str(value)
        self.env["PIE_BENCH_INFERLET_DIR"] = str(self.inferlet_dir)

    @property
    def inferlet_dir(self) -> Path:
        p = Path(self._program_path)
        return p if p.is_absolute() else self.pie_root / p

    # ---- contract ----------------------------------------------------------
    def default_python(self) -> str:
        return os.environ.get("PIE_PY") or sys.executable

    def model_arg(self) -> str:
        # pie_bench resolves an HF id through common.resolve_local_model (HF
        # cache, then ~/.pie/programs); a local path is taken as-is. The runner
        # resolves the checkpoint (or its imported .zt) and passes the path
        # through the recipe; a bare HF id is the fallback.
        return str(self.recipe.get("snapshot_dir") or self.artifact.base_model)

    def pie_engine_name(self) -> str:
        return _BACKEND_TO_ENGINE[self.platform.backend]

    def engine_args(self, workload: WorkloadSpec) -> list[str]:
        r = self.recipe
        engine = self.pie_engine_name()
        args: list[str] = ["--engine", engine, "--inferlet-dir", str(self.inferlet_dir)]
        if r.get("device"):
            args += ["--device", str(r["device"])]

        args += ["--max-model-len", str(max_model_len_for(workload))]

        if r.get("report_timing", True):
            args += ["--report-timing"]

        for knob, flag in _DIRECT_FLAGS:
            v = r.get(knob)
            if v is not None:
                args += [flag, str(v)]
        for knob, flag in _BOOL_OPTIONAL_FLAGS:
            v = r.get(knob)
            if v is True:
                args.append(flag)
            elif v is False:
                args.append("--no-" + flag[2:])
        for knob, flag in _STORE_TRUE_FLAGS:
            if r.get(knob):
                args.append(flag)

        # TODO(pie/benches): the worker takes the quantization SKU as
        # `[model] sku`, but pie_bench.py builds ModelConfig without it and has
        # no --sku flag, so `artifact.pie_sku` cannot be passed today. The
        # recipe's `engine_option` may carry `"$pie_sku"` as a value, which is
        # substituted here - it only helps once the worker accepts such a key
        # under [model.engine.options] (unknown keys refuse to boot).
        for k, v in (r.get("engine_option") or {}).items():
            if v == "$pie_sku":
                if not self.artifact.pie_sku:
                    continue
                v = self.artifact.pie_sku
            args += ["--engine-option", f"{k}={_scalar(v)}"]
        for k, v in (r.get("extra_input") or {}).items():
            args += ["--extra-input", f"{k}={_scalar(v)}"]
        return args

    def version(self) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(self.pie_root), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            return out or "unknown"
        except Exception:
            return "unknown"

    def counters_from_summary(self, summary: dict[str, Any]) -> dict[str, float]:
        """Engine-internal counters from ``model_status`` as pie_bench labels
        them in ``summary.config`` (raw ``default.*`` keys are accepted too)."""
        cfg = summary.get("config") or {}
        wall_s = summary.get("wall_s")
        out: dict[str, float] = {}

        batches = _num(cfg, "total batches", "default.total_batches")
        if batches is not None:
            out["batches"] = batches

        idle_us = _num(cfg, "device idle us", "default.fire.quorum.device_idle_us")
        if idle_us is not None:
            out["device_idle_us"] = idle_us
            if isinstance(wall_s, (int, float)) and wall_s > 0:
                out["device_idle_pct"] = 100.0 * idle_us / (float(wall_s) * 1e6)

        cum_batch_us = _num(cfg, "cumulative_batch_latency_us", "default.cumulative_batch_latency_us")
        avg_batch_us = _num(cfg, "avg batch latency us", "default.avg_batch_latency_us")
        if avg_batch_us is None and cum_batch_us is not None and batches:
            avg_batch_us = cum_batch_us / batches
        if avg_batch_us is not None:
            out["batch_latency_us"] = avg_batch_us
        engine_fire_us = _num(cfg, "fire.execute.engine_fire_us", "default.fire.execute.engine_fire_us")
        if engine_fire_us is None:
            fire_sum = _num(cfg, "default.fire.execute.engine_fire_us_sum")
            if fire_sum is not None and batches:
                engine_fire_us = fire_sum / batches
        if engine_fire_us is not None:
            out["engine_fire_us_per_step"] = engine_fire_us
            if avg_batch_us is not None:
                # what a step costs beyond the device: batch build, dispatch,
                # post-dispatch bookkeeping. Only meaningful on profile builds
                # where engine_fire_us is non-zero.
                out["host_us_per_step"] = max(0.0, avg_batch_us - engine_fire_us)

        ta_sum = _num(cfg, "turnaround sum us", "default.fire.quorum.turnaround_sum_us")
        ta_n = _num(cfg, "turnaround n", "default.fire.quorum.turnaround_n")
        if ta_sum is not None and ta_n:
            out["guest_turnaround_us"] = ta_sum / ta_n
        ta_max = _num(cfg, "turnaround max us", "default.fire.quorum.turnaround_max_us")
        if ta_max is not None:
            out["guest_turnaround_max_us"] = ta_max

        for label, key in (
            ("wave_fires", "wave fires"),
            ("wave_avg_active_pipelines", "wave avg active pipelines"),
            ("wave_avg_missing_pipelines", "wave avg missing pipelines"),
            ("max_forward_requests_observed", "max forward requests"),
            ("process_avg_admission_wait_us", "process avg admission wait us"),
            ("process_avg_instantiate_us", "process avg instantiate us"),
        ):
            v = _num(cfg, key)
            if v is not None:
                out[label] = v
        return out

    def leftover_process_names(self) -> list[str]:
        return ["pie"]

    # ---- persistent server (PIE_BENCH_SERVE_ONLY) -------------------------
    def serve(self, workload: WorkloadSpec, log_path: Path, timeout_s: int) -> None:
        """Boot one pie server with this cell's config and keep it for every
        workload that follows (``run`` then connects through
        ``PIE_BENCH_SERVER_URL``).

        pie_bench honours ``PIE_BENCH_SERVE_ONLY`` only on its embedded-python
        path (``cuda_native``); the Metal engine boots ``pie serve`` per run, so
        this is a no-op here until pie_bench serves Metal the same way.
        """
        if self.pie_engine_name() != "cuda_native":
            return
        if self._server_proc is not None:
            return
        from pie_evals.node.workloads import common_args_for

        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        json_out = log_path.with_suffix(".serve.json")
        argv = self.build_argv(workload, common_args_for(workload), json_out)
        env = {**os.environ, **self.env, "PIE_BENCH_SERVE_ONLY": "1"}
        env.pop("PIE_BENCH_SERVER_URL", None)
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            argv, cwd=str(self.pie_root / "scripts/bench"), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        lines: queue.Queue[str | None] = queue.Queue()
        announced = threading.Event()

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if not announced.is_set():
                    lines.put(line)
            lines.put(None)

        threading.Thread(target=pump, name="pie-serve-log", daemon=True).start()
        tail: collections.deque[str] = collections.deque(maxlen=200)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill(proc, log)
                raise EngineLaunchError(ErrorClass.HANG, f"pie serve did not announce PIE_BENCH_SERVER_URL within {timeout_s}s")
            try:
                line = lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                rc = proc.wait()
                log.close()
                cls, msg = classify_failure(rc, "", "".join(tail), False)
                raise EngineLaunchError(cls, f"pie serve exited before ready: {msg}")
            tail.append(line)
            m = re.search(r"PIE_BENCH_SERVER_URL=(\S+)", line)
            if m:
                announced.set()
                self.server_url = m.group(1)
                self._server_proc = proc
                self._server_log = log
                return

    @staticmethod
    def _kill(proc: subprocess.Popen, log: Any) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log.close()
        except Exception:
            pass

    def stop(self) -> None:
        super().stop()
        self.server_url = None
        if self._server_log is not None:
            try:
                self._server_log.close()
            finally:
                self._server_log = None
