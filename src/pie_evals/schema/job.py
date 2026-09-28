"""A job is what one node executes: a list of cells that share a platform,
plus the pins (pie commit, baseline versions) and the policies that the node
must follow. The node knows nothing about who launched it."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .cell import Cell, Tier


class RepetitionPolicy(BaseModel):
    """Adaptive repetition: run once, compare with the cell's history; only
    when the result is outside ``history_sigma`` × the historical spread run
    ``confirm_rounds`` more and take the median."""

    min_rounds: int = 1
    confirm_rounds: int = 2
    max_rounds: int = 5
    history_sigma: float = 3.0
    cov_noisy_threshold: float = 0.02  # a cell whose rounds spread more than 2% is NOISY
    cov_noisy_threshold_concurrent: float = 0.05  # concurrency / mixed / prefix shapes: scheduling variance is real
    interleave: bool = True  # ABBA ordering across cells that share an engine process


class JobSpec(BaseModel):
    job_id: str
    tier: Tier
    platform_id: str
    shard: int = 0
    pie_commit: str | None = Field(default=None, description="pie commit to build/use; None for baseline-only jobs")
    pie_build_features: list[str] = Field(default_factory=list)
    baseline_versions: dict[str, str] = Field(default_factory=dict, description="engine -> pinned version")
    cells: list[Cell]
    repetition: RepetitionPolicy = Field(default_factory=RepetitionPolicy)
    per_cell_timeout_s: int = 1800
    load_timeout_s: int = 1200
    history: dict[str, list[float]] = Field(
        default_factory=dict,
        description="cell_id -> recent primary-metric values, shipped with the job so the node can decide repetitions offline",
    )
    control_required: bool = True
    output_dir: str = "out"
    budget_s: int = Field(default=3600, description="soft budget: no new cell starts after this")
    kill_s: int = Field(default=5400, description="hard deadline: the runner's watchdog kills everything at this point")
    est_minutes: float = 0.0

    def cells_by_process(self) -> dict[tuple, list[Cell]]:
        """Cells that can share one engine process: same engine, artifact, mode.
        One model per process; workloads/programs are interleaved within it."""
        groups: dict[tuple, list[Cell]] = {}
        for c in self.cells:
            key = (str(c.engine), c.artifact.artifact_key, c.mode.key)
            groups.setdefault(key, []).append(c)
        return groups
