"""Execute a JobSpec on this machine.

Order of operations, per job:

1. preflight: hardware fingerprint, machine state (thermal, power, low power
   mode, displays), the pie build for the pinned commit.
2. group cells into processes: one model per process; within a process the
   workloads/programs are interleaved (ABBA when repeating), and every cell
   in the group shares the same loaded weights.
3. per cell: run once; consult history (adaptive repetition); confirm if
   outside the band; compute CoV; NOISY if the spread is above policy.
4. write ``records.jsonl`` incrementally so a killed job still leaves what
   it finished.

Every failure is classified into ``ErrorClass``; a hang is a timeout.
"""

from __future__ import annotations

import os
import random
import signal
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pie_evals.schema import (
    Cell,
    CellStatus,
    ErrorClass,
    JobSpec,
    PerfMetrics,
    Record,
)

from . import preflight as pf
from . import provenance as prov
from .engines import EngineLaunchError, get_engine
from .engines.recipes import load_recipe
from .engines.shape import context_tokens_for, serve_envelope
from .metrics.stats import cov, decide_repetition, median
from .snapshots import num_layers_of
from .workloads import common_args_for

# the long-context shapes (lc-8k, lc-32k) size the server's arena and KV rows; when
# that envelope does not fit the card the short shapes still can (gemma-4-26b on a 4090)
SHORT_CONTEXT_TOKENS = 8192
MEMORY_BOUND_MARKS = ("device memory exhausted", "weight residency", "does not hold this deployment")


HEAVY_FRACTION = 0.35  # weights past this share of the card: the 32k envelope leaves the pool a seat or two


def split_heavy_processes(groups: dict[tuple, list]) -> dict[tuple, list]:
    """One boot per model serves every cell at the envelope of the longest shape.
    On a card the weights nearly fill, that envelope starves the concurrency
    cells: Qwen3.6-27B (14 GiB) on the RTX 5090 booted for lc-32k got a pool of
    "1 sequences at the declared context", c64 ran serially (p50 latency 152 s)
    and every request timed out (nightly 36081616837 s13). A heavy model boots
    twice instead: the shapes under ``SHORT_CONTEXT_TOKENS`` on their own
    envelope, then the long ones. Light models keep the single boot."""
    out: dict[tuple, list] = {}
    for key, cells in groups.items():
        first = cells[0]
        per_gpu = float(first.platform.memory_gib or 0) * max(1, int(first.mode.tp))
        heavy = per_gpu > 0 and float(first.artifact.expected_gib or 0) / per_gpu > HEAVY_FRACTION
        short = [c for c in cells if context_tokens_for(c.workload) <= SHORT_CONTEXT_TOKENS]
        if not heavy or not short or len(short) == len(cells):
            out[key] = cells
            continue
        engine, artifact, mode = key
        out[(engine, artifact, f"{mode}-short")] = short
        out[(engine, artifact, f"{mode}-long")] = [c for c in cells if c not in short]
    return out


def recipe_for_cell(recipe: dict, recipe_name: str, cell: Cell) -> dict:
    """The process's recipe resolved for ``cell``'s own shape. Until nightly
    36099862659 the recipe was resolved once, for the process's first cell:
    SGLang then captured graphs to that cell's concurrency (32) and ran every
    wider shape eager, and vLLM's prefix cache followed the first shape. The
    keys the boot added (snapshot_dir, pin, program_path) are kept."""
    fresh = load_recipe(str(cell.engine), recipe_name, cell.platform, cell.workload, family=cell.artifact.family)
    for k, v in recipe.items():
        fresh.setdefault(k, v)
    return fresh


def cell_priority(cell) -> tuple[int, float]:
    """Order of the cells one engine process serves: the control A/A first (if the
    harness cannot agree with itself nothing else is read), then the cheap shapes
    before the heavy ones. A 27B on the H100 spent 15 and 19 min on c256 and
    kv-oversub (estimated 3 each) and the soft budget then dropped its
    single-stream and long-context cells, the numbers a row is read for (nightly
    36081635037 s20). The sort is stable, so the interleave shuffle survives
    within one estimate tier."""
    return (0 if cell.workload.kind.value == "control_aa" else 1, float(cell.workload.est_minutes))


class NodeRunner:
    def __init__(self, job: JobSpec, *, pie_root: Path, out_dir: Path, hf_cache: Path | None = None, runner_name: str | None = None, build: bool = True, download: bool = False):
        self.job = job
        self.download = download
        self.pie_root = Path(pie_root).resolve()
        # absolute: cell output paths are handed to bench subprocesses that run
        # with their own cwd (scripts/bench), and a relative --out made them
        # write bench.json somewhere the runner never looked
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.hf_cache = Path(hf_cache or os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache/huggingface/hub")
        self.runner_name = runner_name or os.environ.get("RUNNER_NAME") or os.uname().nodename
        self.run_id = f"{job.job_id}-{uuid.uuid4().hex[:6]}"
        self.records_path = self.out / "records.jsonl"
        self.build = build
        self.platform = job.cells[0].platform if job.cells else None
        self._log = open(self.out / "runner.log", "a")
        (self.out / "run_id.txt").write_text(self.run_id + "\n")
        self.t_start = time.monotonic()
        self._killed = threading.Event()
        self._done: set[str] = set()
        self._last_state: dict = {}
        self._start_watchdog()

    # ------------------------------------------------------------------ time policy
    def elapsed_s(self) -> float:
        return time.monotonic() - self.t_start

    def over_budget(self) -> bool:
        """Soft budget: no new cell starts after ``job.budget_s``."""
        return self.elapsed_s() >= self.job.budget_s

    def remaining_to_kill_s(self) -> float:
        return max(1.0, self.job.kill_s - self.elapsed_s())

    def cell_timeout_s(self, cell: Cell) -> int:
        """A round may take at most ten times the workload's estimate (floor
        ten minutes), under the job-wide cap and the kill deadline. Nightly
        35959142812 lost a whole 60-minute shard to one hung 30-second cell
        (pie #649): the flat 5400 s cap let it run to the client's timeout."""
        scaled = max(600.0, float(cell.workload.est_minutes) * 60.0 * 10.0)
        return int(min(self.job.per_cell_timeout_s, scaled, self.remaining_to_kill_s()))

    def _start_watchdog(self) -> None:
        """Hard deadline: at ``job.kill_s`` (= budget × kill_factor) kill every
        child in our process group and exit. The workflow's timeout-minutes
        is the outer layer of the same
        limit; this is the one that leaves a record behind."""

        def _fire() -> None:
            self._killed.set()
            try:
                self.log(f"WATCHDOG: hard deadline {self.job.kill_s}s reached; killing children and exiting")
                self._emit_unreached("hard deadline reached (job.kill_s); force-killed")
                self._log.flush()
            finally:
                # kill children (not ourselves first): every subprocess we spawned shares our pgid
                try:
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                    time.sleep(3)
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                except OSError:
                    pass
                os._exit(124)

        t = threading.Timer(self.job.kill_s, _fire)
        t.daemon = True
        t.start()
        self._watchdog = t

    def _emit_unreached(self, reason: str) -> None:
        """Record every cell that has not been reached as NOT_RUN with the
        reason. Called once by the runner when the soft budget stops it, or by
        the watchdog. Idempotent per cell via ``_done``."""
        for c in self.job.cells:
            if c.cell_id in self._done:
                continue
            self._done.add(c.cell_id)
            self.emit(self._not_run(c, reason))

    # ------------------------------------------------------------------ logging
    def log(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def emit(self, rec: Record) -> None:
        with open(self.records_path, "a") as f:
            f.write(rec.model_dump_json() + "\n")

    # ------------------------------------------------------------------ setup
    def ensure_pie(self) -> None:
        """Pin the checkout to the job's commit and put the build artifacts in
        place: cache hit → restore; miss → build here and cache."""
        if not self.job.pie_commit:
            return
        from . import build as pb

        mirror = Path(os.environ["PIE_MIRROR"]) if os.environ.get("PIE_MIRROR") else None
        if not self.build:
            # --no-build: use the tree as it is; only pin it if it is a real checkout
            if (self.pie_root / ".git").exists():
                pb.ensure_checkout(self.pie_root, self.job.pie_commit, mirror=mirror)
            else:
                self.log(f"--no-build and {self.pie_root} is not a git checkout; using it as-is")
            return
        pb.ensure_checkout(self.pie_root, self.job.pie_commit, mirror=mirror)
        pb.ensure(self.pie_root, self.job.pie_commit, self.job.pie_build_features,
                  target_dir=Path(os.environ["CARGO_TARGET_DIR"]) if os.environ.get("CARGO_TARGET_DIR") else None,
                  python=os.environ.get("PIE_PY", "python3"), log=self.log)
        self.log(f"pie interpreter for the bench: {os.environ.get('PIE_PY', 'python3')}")

    def snapshot_dir(self, cell: Cell) -> Path:
        from .snapshots import ensure_snapshot

        try:
            # a download mid-shard can outgrow what the job-start reclaim left
            from .reclaim import LOW_WATER_GIB, reclaim

            reclaim(self.job, self.hf_cache, low_water_gib=max(LOW_WATER_GIB, float(cell.artifact.expected_gib or 0) * 1.2 + 10), log=self.log)
            return ensure_snapshot(cell.artifact, self.hf_cache, download=self.download, log=self.log)
        except EngineLaunchError:
            raise
        except Exception as e:  # download / shrink failure: the model, not the harness
            from huggingface_hub.errors import LocalEntryNotFoundError

            # the hub unreachable, or a full disk, says nothing of the model: a re-dispatch measures it
            unreachable = isinstance(e, (LocalEntryNotFoundError, ConnectionError, TimeoutError)) or type(e).__module__.split(".")[0] in ("requests", "httpx", "urllib3") or (isinstance(e, OSError) and e.errno == 28)
            raise EngineLaunchError(ErrorClass.HARNESS_INVALID if unreachable else ErrorClass.LOAD_FAIL, f"checkpoint unavailable for {cell.artifact.id}: {str(e).strip().splitlines()[-1][:300] if str(e).strip() else e!r}") from e

    # ------------------------------------------------------------------ run
    def run(self) -> list[Record]:
        job = self.job
        try:
            os.setpgrp()  # our own process group, so the watchdog's killpg reaches only our children
        except OSError:
            pass
        self.log(f"job {job.job_id} tier={job.tier} platform={job.platform_id} cells={len(job.cells)} run_id={self.run_id}")
        try:
            self.ensure_pie()
        except Exception as e:  # a build failure fails every pie cell identically
            self.log(f"pie build failed: {e}")
            for c in job.cells:
                if str(c.engine) == "pie":
                    self.emit(self._failed(c, ErrorClass.LOAD_FAIL, f"pie build failed: {e}", None))
            job = job.model_copy(update={"cells": [c for c in job.cells if str(c.engine) != "pie"]})
        pre = pf.Preflight()
        machine_before = pre.before_job(self.platform.os if self.platform else None)
        self._last_state = machine_before  # failure records carry the state the machine was in
        try:
            from .reclaim import reclaim

            reclaim(job, self.hf_cache, log=self.log)  # a Mac that hosts every run fills its disk otherwise
        except Exception as e:  # noqa: BLE001 — room-making must never fail the shard
            self.log(f"disk reclaim skipped: {str(e)[:120]}")
        fingerprint = prov.hardware_fingerprint()
        records: list[Record] = []
        groups = split_heavy_processes(job.cells_by_process())
        self.log(f"{len(groups)} engine processes to run")
        for gkey, cells in groups.items():
            engine_name, artifact_key, mode_key = gkey
            if self.over_budget():
                self.log(f"soft budget {job.budget_s}s exhausted after {self.elapsed_s():.0f}s; not starting {engine_name} {artifact_key}")
                continue
            self.log(f"--- process {engine_name} {artifact_key} {mode_key} ({len(cells)} cells) t+{self.elapsed_s():.0f}s")
            engine = None
            snapshot = None
            try:
                snapshot = self.snapshot_dir(cells[0])
                if cells[0].artifact.ollama_tag and self.download:
                    from .engines.ollama import ensure_ollama_model

                    ensure_ollama_model(cells[0].artifact.ollama_tag, pull=True, log=self.log)
            except EngineLaunchError as e:
                for c in cells:
                    self.emit(self._failed(c, e.error_class, str(e), fingerprint))
                continue
            except Exception as e:  # nothing about one model may take the job down
                self.log(f"process setup crashed: {e}\n{traceback.format_exc()}")
                for c in cells:
                    self.emit(self._failed(c, ErrorClass.HARNESS_INVALID, f"process setup: {e}", fingerprint))
                continue
            num_layers = num_layers_of(snapshot)
            first = cells[0]
            model_path = snapshot
            if str(first.engine) == "pie" and self.job.pie_commit:
                from .importer import ensure_artifact, needs_import

                if needs_import(first.artifact):
                    try:
                        model_path = ensure_artifact(first.artifact, snapshot, self.pie_root / "target/release/pie", self.job.pie_commit, log=self.log)
                    except Exception as e:
                        for c in cells:
                            self.emit(self._failed(c, ErrorClass.LOAD_FAIL, f"artifact import: {str(e).strip().splitlines()[-1][:900]}", fingerprint))
                        continue
            recipe_name = "competitive" if str(first.engine) != "pie" else "default"
            try:
                pre.between_engines(self.platform.os if self.platform else None, [])
                cls = get_engine(first.engine)
                recipe = load_recipe(str(first.engine), recipe_name, first.platform, first.workload, family=first.artifact.family)
                recipe["program_path"] = first.program.path
                recipe["snapshot_dir"] = str(model_path)
                if str(first.engine) in self.job.baseline_versions:
                    recipe["pin"] = self.job.baseline_versions[str(first.engine)]
                engine = cls(pie_root=self.pie_root, artifact=first.artifact, platform=first.platform, mode=first.mode, recipe=recipe, num_layers=num_layers)
                engine_version = engine.version()
                # one boot per model: the bench then attaches to this server for every
                # cell and round instead of reloading the weights each time
                serve_log = self.out / "serve" / f"{artifact_key.replace('/', '_')}-{mode_key}.log"

                try:
                    engine, model_path = self._boot(engine, serve_envelope([c.workload for c in cells]), serve_log, cls=cls, first=first, recipe=recipe, num_layers=num_layers, snapshot=snapshot, model_path=model_path)
                except EngineLaunchError as e:
                    # a long-context envelope may not fit beside the weights: give the
                    # long shapes up, keep the short ones
                    short = [c for c in cells if context_tokens_for(c.workload) <= SHORT_CONTEXT_TOKENS]
                    if not any(m in str(e) for m in MEMORY_BOUND_MARKS) or not short or len(short) == len(cells):
                        raise
                    self.log(f"engine boot at the full envelope failed for memory; retrying with the shapes under {SHORT_CONTEXT_TOKENS} tokens ({len(short)} of {len(cells)} cells)")
                    for c in cells:
                        if c not in short:
                            self.emit(self._failed(c, e.error_class, f"does not fit beside the short shapes: {e}", fingerprint))
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    engine = cls(pie_root=self.pie_root, artifact=first.artifact, platform=first.platform, mode=first.mode, recipe=recipe, num_layers=num_layers)
                    cells = short
                    engine, model_path = self._boot(engine, serve_envelope([c.workload for c in cells]), serve_log, cls=cls, first=first, recipe=recipe, num_layers=num_layers, snapshot=snapshot, model_path=model_path)
                if getattr(engine, "server_url", None):
                    self.log(f"serving {artifact_key} at {engine.server_url}")
            except EngineLaunchError as e:
                self.log(f"engine boot failed: {e.error_class} {e}")
                for c in cells:
                    self.emit(self._failed(c, e.error_class, str(e), fingerprint))
                if engine:
                    engine.stop()
                continue
            except Exception as e:
                self.log(f"engine setup failed: {e}\n{traceback.format_exc()}")
                for c in cells:
                    self.emit(self._failed(c, ErrorClass.LOAD_FAIL, f"engine setup: {e}", fingerprint))
                continue
            order = list(cells)
            if job.repetition.interleave:
                random.Random(self.run_id).shuffle(order)
            order.sort(key=cell_priority)
            control_ok = True
            model_state = pre.before_model(self.platform) if hasattr(pre, "before_model") else machine_before
            self._last_state = model_state
            for cell in order:
                if cell.cell_id in self._done:
                    continue
                if self.over_budget():
                    self.log(f"soft budget exhausted at t+{self.elapsed_s():.0f}s; remaining cells of this process not started")
                    break
                if not control_ok and job.control_required:
                    rec = self._invalid(cell, "control A/A failed on this process; numbers not read", fingerprint)
                    records.append(rec)
                    self.emit(rec)
                    continue
                if hasattr(engine, "program_path"):
                    engine.program_path = cell.program.path  # the cell's own inferlet, not the process's first
                recipe = recipe_for_cell(recipe, recipe_name, cell)
                engine.recipe = recipe
                rec = self._run_cell(cell, engine, snapshot, engine_version, recipe, recipe_name, fingerprint, machine_before)
                if cell.workload.kind.value == "control_aa" and rec.status == CellStatus.NOISY:
                    # a spread a hair past the bound on a freshly booted process withholds every
                    # number of the process (gemma-4-E4B tp2: 5.6 % against 5 %); one more look
                    # before condemning it — a process that really drifts fails twice
                    self.log(f"control A/A noisy ({rec.invalid_reason}); running the control once more")
                    again = self._run_cell(cell, engine, snapshot, engine_version, recipe, recipe_name, fingerprint, machine_before)
                    if again.status == CellStatus.PASS:
                        rec = again
                if cell.workload.kind.value == "control_aa" and rec.status != CellStatus.PASS:
                    control_ok = False
                    self.log("control A/A failed; remaining cells in this process are HARNESS_INVALID")
                records.append(rec)
                self._done.add(cell.cell_id)
                self.emit(rec)
            try:
                if engine:
                    engine.stop()
                    pre.between_engines(self.platform.os if self.platform else None, engine.leftover_process_names())
            except Exception as e:
                self.log(f"teardown: {e}")
            invalid = pre.after_model(self.platform.os if self.platform else None, model_state)
            if invalid:
                self.log(f"model run invalidated: {invalid}")
                for r in [r for r in records if r.cell.artifact.artifact_key == artifact_key and r.status == CellStatus.PASS]:
                    r.status = CellStatus.NOISY
                    r.invalid_reason = "; ".join(invalid)
                    self.emit(r)
                # a machine that was not in the expected state fails the engine at boot:
                # that is the machine's fault, not the engine's, so it is harness-invalid
                for r in [r for r in records if r.cell.artifact.artifact_key == artifact_key and r.status == CellStatus.FAIL]:
                    r.error_class = ErrorClass.HARNESS_INVALID
                    r.invalid_reason = "; ".join(invalid)
                    self.emit(r)
        self._emit_unreached(f"job budget ({job.budget_s}s) exhausted before this cell was reached" if self.over_budget() else "not reached (earlier failure)")
        self._watchdog.cancel()
        self.log(f"done in {self.elapsed_s():.0f}s (budget {job.budget_s}s, kill {job.kill_s}s)")
        self._log.close()
        return records

    def _boot(self, engine, shape, serve_log: Path, *, cls, first: Cell, recipe: dict, num_layers, snapshot: Path, model_path: Path):
        """Serve ``shape``; a pie checkpoint it refuses to read directly is
        imported as an artifact once and served from there. Returns the
        (possibly rebuilt) engine and the path it serves."""
        try:
            engine.serve(shape, serve_log, int(min(self.job.load_timeout_s, self.remaining_to_kill_s())))
        except EngineLaunchError as e:
            from .importer import RELAYOUT_MARK, ensure_artifact

            if RELAYOUT_MARK not in str(e) or str(first.engine) != "pie" or not self.job.pie_commit or model_path != snapshot:
                raise
            self.log("pie refuses to serve this checkpoint directly; importing it as an artifact and retrying")
            model_path = ensure_artifact(first.artifact, snapshot, self.pie_root / "target/release/pie", self.job.pie_commit, log=self.log)
            recipe["snapshot_dir"] = str(model_path)
            engine = cls(pie_root=self.pie_root, artifact=first.artifact, platform=first.platform, mode=first.mode, recipe=recipe, num_layers=num_layers)
            engine.serve(shape, serve_log, int(min(self.job.load_timeout_s, self.remaining_to_kill_s())))
        return engine, model_path

    # ------------------------------------------------------------------ one cell
    def _noise_bound(self, cell: Cell) -> float:
        policy = self.job.repetition
        tight = policy.cov_noisy_threshold if int(cell.mode.tp) <= 1 else policy.cov_noisy_threshold_concurrent
        return tight if cell.workload.kind.value in ("single_stream", "control_aa", "long_context") else policy.cov_noisy_threshold_concurrent

    def _run_cell(self, cell: Cell, engine, snapshot: Path, engine_version: str, recipe: dict, recipe_name: str, fingerprint: dict, machine_state: dict) -> Record:
        t0 = time.monotonic()
        cell = cell.model_copy(update={"engine_version": engine_version})
        cell_out = self.out / "cells" / cell.cell_id
        common = common_args_for(cell.workload, warmup=int(cell.workload.params.get("warmup", 2))) + list(cell.program.bench_args)
        policy = self.job.repetition
        # a concurrent shape's round is minutes long (kv-oversub, mixed, c256: 2/3 of a nightly's cell time):
        # two rounds that agree within its bound are the record; a third comes only through confirmation
        need = min(policy.min_rounds, 2) if cell.workload.kind.value not in ("single_stream", "control_aa", "long_context") else policy.min_rounds
        rounds: list[float] = []
        results = []
        try:
            for i in range(need):
                res = engine.run(cell.workload, common, cell_out / f"r{i}", self.cell_timeout_s(cell))
                results.append(res)
                rounds.append(self._primary(res.perf))
            hist = self.job.history.get(cell.cell_id, [])
            if need == 1:
                d = decide_repetition(rounds[0], hist, sigma=policy.history_sigma, confirm_rounds=policy.confirm_rounds)
                self.log(f"{cell.workload.id}/{cell.program.id}: {rounds[0]:.4g} — {d.reason}")
                for _ in range(d.more_rounds):
                    if len(rounds) >= policy.max_rounds:
                        break
                    res = engine.run(cell.workload, common, cell_out / f"r{len(rounds)}", self.cell_timeout_s(cell))
                    results.append(res)
                    rounds.append(self._primary(res.perf))
            elif cov(rounds) > self._noise_bound(cell):
                # the policy's confirmation rounds: a first round still warming settles in the next ones
                for _ in range(min(policy.confirm_rounds, policy.max_rounds - len(rounds))):
                    res = engine.run(cell.workload, common, cell_out / f"r{len(rounds)}", self.cell_timeout_s(cell))
                    results.append(res)
                    rounds.append(self._primary(res.perf))
        except EngineLaunchError as e:
            self.log(f"{cell.workload.id}/{cell.program.id}: {e.error_class} {e}")
            return self._failed(cell, e.error_class, str(e), fingerprint, duration=time.monotonic() - t0)
        except Exception as e:
            self.log(f"{cell.workload.id}/{cell.program.id}: harness exception {e}\n{traceback.format_exc()}")
            return self._failed(cell, ErrorClass.HARNESS_INVALID, f"harness: {e}", fingerprint, duration=time.monotonic() - t0)
        # median of the steady rounds (the last min_rounds) is the record; every round travels with it
        steady = list(range(len(rounds)))[-need:] if need > 1 else list(range(len(rounds)))
        med_idx = sorted(steady, key=lambda i: rounds[i])[len(steady) // 2]
        perf: PerfMetrics = results[med_idx].perf
        perf.rounds = rounds
        perf.cov = cov([rounds[i] for i in steady])
        perf.load_s = results[0].duration_s - (results[med_idx].duration_s if len(results) > 1 else 0) if len(results) > 1 else None
        status = CellStatus.PASS
        invalid = None
        # a tensor-parallel process adds collective latency to every step, so its
        # single-stream rounds spread like a concurrent shape's (gemma-4-E4B tp2 on
        # L40S x2: A/A spread 2.1 % against the 2 % bound, every cell withheld)
        tight = policy.cov_noisy_threshold if int(cell.mode.tp) <= 1 else policy.cov_noisy_threshold_concurrent
        thr = self._noise_bound(cell)
        if perf.failed:
            status, invalid = CellStatus.FAIL, f"{perf.failed} of {perf.requests} requests failed"
        elif len(rounds) >= 2 and perf.cov > thr:
            status, invalid = CellStatus.NOISY, f"cov {perf.cov:.3%} > {thr:.1%}"
        if cell.workload.kind.value == "control_aa" and len(rounds) >= 2:
            spread = abs(rounds[0] - rounds[-1]) / median(rounds)
            if spread > tight:
                status, invalid = CellStatus.NOISY, f"control A/A spread {spread:.3%} > {tight:.1%}"
        provenance = prov.build_provenance(self.pie_root, snapshot, engine_version, recipe, recipe_name, self.runner_name)
        provenance.hardware_fingerprint = fingerprint
        provenance.machine_state = machine_state
        provenance.pie_commit = self.job.pie_commit or provenance.pie_commit
        provenance.pie_build_features = self.job.pie_build_features
        return Record(
            run_id=self.run_id, job_id=self.job.job_id, tier=self.job.tier, cell_id=cell.cell_id, cell_key=cell.cell_key, cell=cell,
            status=status, error_class=None,
            error_message=None, invalid_reason=invalid, perf=perf, provenance=provenance,
            duration_s=time.monotonic() - t0,
        )

    @staticmethod
    def _primary(perf: PerfMetrics) -> float:
        v = getattr(perf, perf.primary, None)
        if v is None:
            raise EngineLaunchError(ErrorClass.HARNESS_INVALID, f"no {perf.primary} in bench output")
        return float(v)

    def _not_run(self, cell: Cell, reason: str) -> Record:
        p = prov.Provenance(runner=self.runner_name, pie_commit=self.job.pie_commit)
        return Record(run_id=self.run_id, job_id=self.job.job_id, tier=self.job.tier, cell_id=cell.cell_id, cell_key=cell.cell_key, cell=cell,
                      status=CellStatus.NOT_RUN, invalid_reason=reason, provenance=p)

    def _failed(self, cell: Cell, cls: ErrorClass, msg: str, fingerprint: dict | None, duration: float | None = None) -> Record:
        self._done.add(cell.cell_id)
        p = prov.Provenance(hardware_fingerprint=fingerprint or {}, runner=self.runner_name, pie_commit=self.job.pie_commit, harness_commit=prov.git_commit(Path(__file__).resolve().parents[3]))
        p.machine_state = self._last_state
        return Record(run_id=self.run_id, job_id=self.job.job_id, tier=self.job.tier, cell_id=cell.cell_id, cell_key=cell.cell_key, cell=cell,
                      status=CellStatus.FAIL, error_class=cls, error_message=msg[:1000], provenance=p, duration_s=duration)

    def _invalid(self, cell: Cell, reason: str, fingerprint: dict | None) -> Record:
        r = self._failed(cell, ErrorClass.HARNESS_INVALID, reason, fingerprint)
        r.status = CellStatus.NOISY
        r.invalid_reason = reason
        return r
