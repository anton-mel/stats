"""The result record. One per (cell, run). Provenance fields are mandatory:
``Record.validate_for_store`` refuses a record that cannot say what it
measured."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, ClassVar

import pyarrow as pa
from pydantic import BaseModel, Field

from .cell import AccuracyStatus, Cell, CellStatus, ErrorClass, Tier


class Provenance(BaseModel):
    harness_commit: str | None = None
    pie_commit: str | None = None
    pie_build_features: list[str] = Field(default_factory=list)
    engine_version: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict, description="the exact knobs the engine ran with")
    engine_config_recipe: str | None = Field(default=None, description="default | recipe:<name> | sweep-best")
    checkpoint_revision: str | None = None
    quant_block_hash: str | None = Field(default=None, description="sha256 of the checkpoint's full quantization block")
    dataset_hashes: dict[str, str] = Field(default_factory=dict)
    hardware_fingerprint: dict[str, Any] = Field(default_factory=dict)
    driver_version: str | None = None
    cuda_version: str | None = None
    os_version: str | None = None
    env: dict[str, str] = Field(default_factory=dict, description="NCCL_*, PIE_*, LD_LIBRARY_PATH ... that affect results")
    runner: str | None = None
    machine_state: dict[str, Any] = Field(default_factory=dict, description="thermal/power/other-load checks (mac), gpu drain wait (linux)")

    REQUIRED: ClassVar[tuple[str, ...]] = ("harness_commit", "engine_version", "checkpoint_revision", "hardware_fingerprint")


class PerfMetrics(BaseModel):
    primary: str = Field(default="decode_tok_s", description="which field regression detection keys on")
    decode_tok_s: float | None = None
    prefill_tok_s: float | None = None
    output_tok_s: float | None = None  # aggregate throughput incl. all concurrent lanes
    ttft_ms_p50: float | None = None
    ttft_ms_p99: float | None = None
    itl_ms_p50: float | None = None
    itl_ms_p99: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p99: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    requests: int | None = None
    failed: int | None = None
    resident_gib: float | None = None
    load_s: float | None = None
    ms_per_token_per_layer: float | None = Field(default=None, description="normalized so miniature cells can be eyeballed against full ones")
    counters: dict[str, float] = Field(default_factory=dict, description="engine-internal: device_idle_pct, host_us_per_step, guest_turnaround_us, eta_compile_ms ...")
    rounds: list[float] = Field(default_factory=list, description="primary metric per round")
    cov: float | None = None


class AccuracyMetrics(BaseModel):
    status: AccuracyStatus = AccuracyStatus.NOT_RUN
    reference: str | None = Field(default=None, description="hf | mlxlm | llamacpp | vllm | pie:<backend> | bf16-self")
    t0_first_divergence: list[int | None] = Field(default_factory=list, description="per prompt; None = identical")
    t0_max_logit_gap: float | None = None
    t0_benign: bool | None = None
    t1_mean_kl: float | None = None
    t1_topk_agreement: float | None = None
    t1_ppl: float | None = None
    t1_ppl_reference: float | None = None
    t2_scores: dict[str, float] = Field(default_factory=dict)
    t2_reference_scores: dict[str, float] = Field(default_factory=dict)
    t2_ci: dict[str, tuple[float, float]] = Field(default_factory=dict)
    acceptance_rate: float | None = None  # speculative programs
    cosine: float | None = None  # diffusion programs
    detail: dict[str, Any] = Field(default_factory=dict)


class Record(BaseModel):
    run_id: str
    job_id: str
    tier: Tier
    cell_id: str
    cell_key: str
    cell: Cell
    status: CellStatus
    error_class: ErrorClass | None = None
    error_message: str | None = None
    invalid_reason: str | None = None
    perf: PerfMetrics | None = None
    accuracy: AccuracyMetrics = Field(default_factory=AccuracyMetrics)
    provenance: Provenance = Field(default_factory=Provenance)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_s: float | None = None

    def validate_for_store(self) -> list[str]:
        """Return the list of missing mandatory provenance fields (empty = ok)."""
        missing = []
        if self.status in (CellStatus.PASS, CellStatus.NOISY) and self.perf is not None:
            for f in Provenance.REQUIRED:
                v = getattr(self.provenance, f)
                if v in (None, "", {}, []):
                    missing.append(f)
            if self.cell.engine.value == "pie" and not self.provenance.pie_commit:
                missing.append("pie_commit")
        return missing

    # ---- flat row for parquet -------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        c = self.cell
        p = self.perf or PerfMetrics()
        a = self.accuracy
        pv = self.provenance
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "tier": str(self.tier),
            "cell_id": self.cell_id,
            "cell_key": self.cell_key,
            "engine": str(c.engine),
            "engine_version": c.engine_version,
            "platform": c.platform.id,
            "backend": str(c.platform.backend),
            "accelerator": c.platform.accelerator,
            "arch": c.platform.arch,
            "artifact": c.artifact.id,
            "artifact_key": c.artifact.artifact_key,
            "family": c.artifact.family,
            "scheme": str(c.artifact.scheme),
            "artifact_kind": str(c.artifact.kind),
            "workload": c.workload.id,
            "workload_kind": str(c.workload.kind),
            "program": c.program.id,
            "mode": c.mode.key,
            "tp": c.mode.tp,
            "status": str(self.status),
            "error_class": str(self.error_class) if self.error_class else None,
            "error_message": self.error_message,
            "invalid_reason": self.invalid_reason,
            "primary_metric": p.primary,
            "primary_value": getattr(p, p.primary, None),
            "decode_tok_s": p.decode_tok_s,
            "prefill_tok_s": p.prefill_tok_s,
            "output_tok_s": p.output_tok_s,
            "ttft_ms_p50": p.ttft_ms_p50,
            "ttft_ms_p99": p.ttft_ms_p99,
            "itl_ms_p50": p.itl_ms_p50,
            "itl_ms_p99": p.itl_ms_p99,
            "latency_ms_p50": p.latency_ms_p50,
            "latency_ms_p99": p.latency_ms_p99,
            "prompt_tokens": p.prompt_tokens,
            "output_tokens": p.output_tokens,
            "requests": p.requests,
            "failed": p.failed,
            "resident_gib": p.resident_gib,
            "load_s": p.load_s,
            "ms_per_token_per_layer": p.ms_per_token_per_layer,
            "counters_json": json.dumps(p.counters, sort_keys=True),
            "rounds_json": json.dumps(p.rounds),
            "cov": p.cov,
            "accuracy_status": str(a.status),
            "accuracy_reference": a.reference,
            "t0_first_divergence_json": json.dumps(a.t0_first_divergence),
            "t0_max_logit_gap": a.t0_max_logit_gap,
            "t0_benign": a.t0_benign,
            "t1_mean_kl": a.t1_mean_kl,
            "t1_topk_agreement": a.t1_topk_agreement,
            "t1_ppl": a.t1_ppl,
            "t1_ppl_reference": a.t1_ppl_reference,
            "t2_scores_json": json.dumps(a.t2_scores, sort_keys=True),
            "t2_reference_scores_json": json.dumps(a.t2_reference_scores, sort_keys=True),
            "acceptance_rate": a.acceptance_rate,
            "cosine": a.cosine,
            "harness_commit": pv.harness_commit,
            "pie_commit": pv.pie_commit,
            "pie_build_features": ",".join(pv.pie_build_features),
            "engine_config_json": json.dumps(pv.engine_config, sort_keys=True, default=str),
            "engine_config_recipe": pv.engine_config_recipe,
            "checkpoint_revision": pv.checkpoint_revision,
            "quant_block_hash": pv.quant_block_hash,
            "dataset_hashes_json": json.dumps(pv.dataset_hashes, sort_keys=True),
            "hardware_fingerprint_json": json.dumps(pv.hardware_fingerprint, sort_keys=True, default=str),
            "driver_version": pv.driver_version,
            "cuda_version": pv.cuda_version,
            "os_version": pv.os_version,
            "env_json": json.dumps(pv.env, sort_keys=True),
            "runner": pv.runner,
            "machine_state_json": json.dumps(pv.machine_state, sort_keys=True, default=str),
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "record_json": self.model_dump_json(),
        }


ARROW_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("job_id", pa.string()),
        ("tier", pa.string()),
        ("cell_id", pa.string()),
        ("cell_key", pa.string()),
        ("engine", pa.string()),
        ("engine_version", pa.string()),
        ("platform", pa.string()),
        ("backend", pa.string()),
        ("accelerator", pa.string()),
        ("arch", pa.string()),
        ("artifact", pa.string()),
        ("artifact_key", pa.string()),
        ("family", pa.string()),
        ("scheme", pa.string()),
        ("artifact_kind", pa.string()),
        ("workload", pa.string()),
        ("workload_kind", pa.string()),
        ("program", pa.string()),
        ("mode", pa.string()),
        ("tp", pa.int32()),
        ("status", pa.string()),
        ("error_class", pa.string()),
        ("error_message", pa.string()),
        ("invalid_reason", pa.string()),
        ("primary_metric", pa.string()),
        ("primary_value", pa.float64()),
        ("decode_tok_s", pa.float64()),
        ("prefill_tok_s", pa.float64()),
        ("output_tok_s", pa.float64()),
        ("ttft_ms_p50", pa.float64()),
        ("ttft_ms_p99", pa.float64()),
        ("itl_ms_p50", pa.float64()),
        ("itl_ms_p99", pa.float64()),
        ("latency_ms_p50", pa.float64()),
        ("latency_ms_p99", pa.float64()),
        ("prompt_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("requests", pa.int64()),
        ("failed", pa.int64()),
        ("resident_gib", pa.float64()),
        ("load_s", pa.float64()),
        ("ms_per_token_per_layer", pa.float64()),
        ("counters_json", pa.string()),
        ("rounds_json", pa.string()),
        ("cov", pa.float64()),
        ("accuracy_status", pa.string()),
        ("accuracy_reference", pa.string()),
        ("t0_first_divergence_json", pa.string()),
        ("t0_max_logit_gap", pa.float64()),
        ("t0_benign", pa.bool_()),
        ("t1_mean_kl", pa.float64()),
        ("t1_topk_agreement", pa.float64()),
        ("t1_ppl", pa.float64()),
        ("t1_ppl_reference", pa.float64()),
        ("t2_scores_json", pa.string()),
        ("t2_reference_scores_json", pa.string()),
        ("acceptance_rate", pa.float64()),
        ("cosine", pa.float64()),
        ("harness_commit", pa.string()),
        ("pie_commit", pa.string()),
        ("pie_build_features", pa.string()),
        ("engine_config_json", pa.string()),
        ("engine_config_recipe", pa.string()),
        ("checkpoint_revision", pa.string()),
        ("quant_block_hash", pa.string()),
        ("dataset_hashes_json", pa.string()),
        ("hardware_fingerprint_json", pa.string()),
        ("driver_version", pa.string()),
        ("cuda_version", pa.string()),
        ("os_version", pa.string()),
        ("env_json", pa.string()),
        ("runner", pa.string()),
        ("machine_state_json", pa.string()),
        ("started_at", pa.timestamp("us", tz="UTC")),
        ("duration_s", pa.float64()),
        ("record_json", pa.string()),
    ]
)


def records_to_table(records: list[Record]) -> pa.Table:
    rows = [r.to_row() for r in records]
    cols = {name: [row.get(name) for row in rows] for name in ARROW_SCHEMA.names}
    return pa.table(cols, schema=ARROW_SCHEMA)
