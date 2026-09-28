"""The cell: the unit everything in pie-evals hangs off.

    cell = (engine@version, platform, artifact, workload, program, mode)

Every measurement is a record attached to one cell. Every cell is always in
exactly one status (see ``CellStatus``); a cell that was never run shows up as
``not_run`` rather than silently disappearing — that is how missing coverage
is surfaced.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StrEnumBase(StrEnum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class EngineName(StrEnumBase):
    PIE = "pie"
    OLLAMA = "ollama"


class Backend(StrEnumBase):
    """pie driver / device backend."""

    METAL = "metal"


class QuantScheme(StrEnumBase):
    BF16 = "bf16"
    MXFP4 = "mxfp4"
    AFFINE_U4_G64 = "affine_u4_g64"  # mlx-community 4-bit, group 64
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q8_0 = "gguf_q8_0"
    NVFP4 = "nvfp4"


class SourceFormat(StrEnumBase):
    HF_SAFETENSORS = "hf_safetensors"
    GGUF = "gguf"
    MLX = "mlx"
    OLLAMA = "ollama"


class ArtifactKind(StrEnumBase):
    FULL = "full"


class WorkloadKind(StrEnumBase):
    SINGLE_STREAM = "single_stream"
    CONCURRENCY = "concurrency"
    LONG_CONTEXT = "long_context"
    PREFIX_SHARED = "prefix_shared"
    MIXED_LENGTH = "mixed_length"
    REPLAY = "replay"
    CONTROL_AA = "control_aa"  # harness self-check: engine vs itself


class Tier(StrEnumBase):
    TARGETED = "targeted"


class CellStatus(StrEnumBase):
    PASS = "pass"
    FAIL = "fail"
    DECLARED_UNSUPPORTED = "declared_unsupported"
    NOT_RUN = "not_run"
    NOISY = "noisy"  # ran, but the harness refuses to read the numbers


class ErrorClass(StrEnumBase):
    LOAD_FAIL = "load_fail"
    CRASH = "crash"
    HANG = "hang"  # only ever observed via timeout
    OOM = "oom"
    DOESNT_FIT = "doesnt_fit"  # refused before load by the fit check
    GATE_FAIL = "gate_fail"  # ran, produced wrong output
    HARNESS_INVALID = "harness_invalid"  # preflight/control failed; not the engine's fault
    INPUT_MISMATCH = "input_mismatch"  # prompt/output token parity across engines broke
    INCOMPATIBLE = "incompatible"  # the program's contract does not fit the model (e.g. attention-only inferlet on a hybrid model)


class AccuracyStatus(StrEnumBase):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED_NO_REFERENCE = "skipped_no_reference"
    NOT_RUN = "not_run"


class ArtifactSpec(BaseModel):
    id: str
    base_model: str = Field(description="HF repo id or local path of the checkpoint")
    revision: str | None = Field(default=None, description="checkpoint commit; resolved at run time if None")
    family: str = Field(description="model family key (qwen3, gemma4, deepseek_v4, ...)")
    scheme: QuantScheme
    kv_dtype: str = "bf16"
    source_format: SourceFormat = SourceFormat.HF_SAFETENSORS
    kind: ArtifactKind = ArtifactKind.FULL
    expected_gib: float | None = Field(default=None, description="LM-only resident weight size")
    degradation_budget: dict[str, float] | None = Field(
        default=None,
        description="for quant schemes without a same-weights reference: allowed ppl ratio / score drop vs bf16",
    )
    chat_template: bool = False
    pie_sku: str | None = Field(default=None, description="pie SKU row name carrying the quantization (e.g. gptoss-20b-dflash-u4g64-mxfp4-kv-bf16)")
    max_context: int | None = Field(default=None, description="tokens one sequence may hold on this artifact as pie ships it (the SKU's max_context), when smaller than the HF config's")
    gguf_file: str | None = Field(default=None, description="file name inside a GGUF repo (the arm must be named, never the quant tag)")
    gguf_config_from: str | None = Field(default=None, description="HF repo whose config.json is copied next to the GGUF (pie reads the encoding from config.json; GGUF repos ship none)")
    display_name: str | None = Field(default=None, description="name the site shows for this model (its Ollama tag)")
    ollama_tag: str | None = Field(default=None, description="Ollama model tag this artifact is served as (``gemma4:26b``); source_format ollama")
    baseline_of: str | None = Field(default=None, description="the pie artifact this baseline copy is compared with (same model, the baseline's own weights)")
    baseline_label: str | None = Field(default=None, description="column name for this baseline arm on the site (``Ollama``, ``Ollama MLX``)")
    tiers: list[Tier] = Field(default_factory=lambda: [Tier.TARGETED])

    @model_validator(mode="after")
    def _ollama_needs_tag(self) -> ArtifactSpec:
        if self.source_format == SourceFormat.OLLAMA and not self.ollama_tag:
            raise ValueError(f"artifact {self.id}: an ollama artifact needs ollama_tag")
        return self

    @property
    def artifact_key(self) -> str:
        key = f"{self.base_model}@{self.scheme}/{self.source_format}"
        if self.ollama_tag:
            key += f"#{self.ollama_tag}"
        return key


class PlatformSpec(BaseModel):
    id: str
    os: str  # linux | macos
    backend: Backend
    accelerator: str = Field(description="human name, e.g. 'NVIDIA L40S', 'Apple M1 Max'")
    arch: str = Field(description="ada | blackwell | hopper | ampere | apple7 | apple8 | apple9 ...")
    count: int = 1
    memory_gib: float
    interconnect: str = "pcie"  # pcie | nvlink | uma
    runner_labels: list[str] = Field(default_factory=list, description="GitHub self-hosted runner labels")
    tiers: list[Tier] = Field(default_factory=lambda: [Tier.TARGETED])


class WorkloadSpec(BaseModel):
    id: str
    kind: WorkloadKind
    params: dict[str, Any] = Field(default_factory=dict)
    tiers: list[Tier] = Field(default_factory=lambda: [Tier.TARGETED])
    est_minutes: float = 1.0


class ProgramSpec(BaseModel):
    """An inferlet (pie guest program). ``text-completion-bench`` is the
    serving-path default; every other entry exercises a different ETA
    program and is a coverage cell of its own."""

    id: str
    path: str = Field(description="path under the pie tree, e.g. examples/text-completion-bench")
    category: str  # serving | speculative | kv_policy | adapter | diffusion
    baseline_equivalents: dict[str, str] = Field(
        default_factory=dict, description="engine -> mode flag that is the comparable feature, if any"
    )
    accuracy_gate: str = "token_parity"  # token_parity | cosine | acceptance_rate | none
    pie_only: bool = True
    spec_dec: str | None = Field(default=None, description="speculative mode this program requires")
    bench_args: list[str] = Field(default_factory=list, description="extra bench flags this program needs (e.g. a sampling inferlet rejects temperature 0)")
    workloads: list[str] | None = Field(default=None, description="restrict to these workload ids (None = all)")
    families: list[str] | None = Field(default=None, description="restrict to these model families (None = all)")
    tiers: list[Tier] = Field(default_factory=lambda: [Tier.TARGETED])


class Mode(BaseModel):
    id: str = "tp1"
    tp: int = 1
    spec_dec: str | None = None  # dflash2 | dspark | mtp | ngram | eagle | None
    extra: dict[str, Any] = Field(default_factory=dict)
    tiers: list[Tier] = Field(default_factory=lambda: [Tier.TARGETED])

    @property
    def key(self) -> str:
        parts = [f"tp{self.tp}"]
        if self.spec_dec:
            parts.append(self.spec_dec)
        for k in sorted(self.extra):
            parts.append(f"{k}={self.extra[k]}")
        return ",".join(parts)


class Cell(BaseModel):
    engine: EngineName
    engine_version: str | None = Field(default=None, description="baseline version or pie commit; filled at run time")
    platform: PlatformSpec
    artifact: ArtifactSpec
    workload: WorkloadSpec
    program: ProgramSpec
    mode: Mode = Field(default_factory=Mode)
    tiers: list[Tier] = Field(default_factory=list)
    declared_unsupported_reason: str | None = None

    @property
    def cell_key(self) -> str:
        """Identity independent of engine version — used to line up history."""
        return "|".join(
            [
                str(self.engine),
                self.platform.id,
                self.artifact.artifact_key,
                self.workload.id,
                self.program.id,
                self.mode.key,
            ]
        )

    @property
    def cell_id(self) -> str:
        return hashlib.sha256(self.cell_key.encode()).hexdigest()[:16]

    @property
    def is_baseline(self) -> bool:
        return self.engine != EngineName.PIE

    def accuracy_applicable(self) -> bool:
        return self.artifact.kind == ArtifactKind.FULL and self.program.accuracy_gate != "none"
