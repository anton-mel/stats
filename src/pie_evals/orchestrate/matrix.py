"""Load the matrix YAMLs and expand them into cells.

Expansion is the product engine × platform × artifact × workload × program ×
mode, pruned in this order:

1. *inapplicable* combinations are not cells at all (an engine that cannot
   load the artifact's format, an engine that does not run on the OS, a
   program restricted to other workloads/families, a TP mode on a platform
   with fewer devices, a pie-only program on a baseline engine);
2. *declared unsupported* combinations (support.yaml, and the fit rule) are
   kept as cells with status ``declared_unsupported`` so the coverage report
   shows them;
3. tiers are the intersection of every component's tiers, then suites.yaml
   ``include``/``exclude`` are applied.

Everything that survives with an empty tier list is still a cell (it shows
as never scheduled) — silence is not allowed.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pie_evals.schema import (
    ArtifactSpec,
    Cell,
    EngineName,
    Mode,
    PlatformSpec,
    ProgramSpec,
    Tier,
    WorkloadSpec,
)


@dataclass
class EngineDecl:
    id: str
    os: list[str]
    formats: list[str]
    tiers: list[Tier]
    recipe: str = "default"
    pin: str | None = None
    release_source: str | None = None


@dataclass
class SupportRule:
    when: dict[str, Any]
    reason: str


@dataclass
class SuiteDecl:
    max_jobs_per_platform: int = 1
    enforce_budget: bool = True
    exclude: list[dict[str, Any]] = field(default_factory=list)
    include: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Matrix:
    engines: dict[str, EngineDecl]
    platforms: dict[str, PlatformSpec]
    artifacts: dict[str, ArtifactSpec]
    workloads: dict[str, WorkloadSpec]
    programs: dict[str, ProgramSpec]
    modes: dict[str, Mode]
    unsupported: list[SupportRule]
    fit_factor: float
    suites: dict[str, SuiteDecl]
    root: Path
    global_exclude: list[dict[str, Any]] = field(default_factory=list)
    job_budget_minutes: float = 60.0
    kill_factor: float = 1.5

    @property
    def kill_minutes(self) -> int:
        return int(round(self.job_budget_minutes * self.kill_factor))

    # ---- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, root: Path | str = "matrix") -> Matrix:
        root = Path(root)

        def read(name: str) -> dict:
            with open(root / name) as f:
                return yaml.safe_load(f) or {}

        eng = {e["id"]: EngineDecl(**e) for e in read("engines.yaml")["engines"]}
        plats = {p["id"]: PlatformSpec(**p) for p in read("platforms.yaml")["platforms"]}
        arts = {a["id"]: ArtifactSpec(**a) for a in read("models.yaml")["artifacts"]}
        wls = {w["id"]: WorkloadSpec(**w) for w in read("workloads.yaml")["workloads"]}
        progs = {p["id"]: ProgramSpec(**p) for p in read("programs.yaml")["programs"]}
        modes = {m["id"]: Mode(**m) for m in read("modes.yaml")["modes"]}
        sup = read("support.yaml")
        rules = [SupportRule(r["when"], r["reason"]) for r in sup.get("unsupported", [])]
        suites_raw = read("suites.yaml")
        suites = {k: SuiteDecl(**v) for k, v in suites_raw.get("suites", {}).items()}
        return cls(
            engines=eng, platforms=plats, artifacts=arts, workloads=wls, programs=progs, modes=modes,
            unsupported=rules, fit_factor=float(sup.get("fit_factor", 1.25)), suites=suites,
            root=root,
            global_exclude=list(suites_raw.get("global_exclude", []) or []),
            job_budget_minutes=float(suites_raw.get("job_budget_minutes", 60)),
            kill_factor=float(suites_raw.get("kill_factor", 1.5)),
        )

    # ---- expansion ---------------------------------------------------------------
    def expand(self) -> list[Cell]:
        cells: list[Cell] = []
        for eng in self.engines.values():
            for plat in self.platforms.values():
                if plat.os not in eng.os:
                    continue
                for art in self.artifacts.values():
                    if str(art.source_format) not in eng.formats:
                        continue
                    for mode in self.modes.values():
                        if mode.tp > 1 and plat.count < mode.tp:
                            continue
                        for prog in self.programs.values():
                            if prog.pie_only and eng.id != "pie":
                                continue
                            if prog.families is not None and art.family not in prog.families:
                                continue
                            for wl in self.workloads.values():
                                if prog.workloads is not None and wl.id not in prog.workloads:
                                    continue
                                if wl.kind.value == "control_aa" and eng.id != "pie":
                                    continue
                                cell_mode = mode.model_copy(update={"spec_dec": prog.spec_dec}) if prog.spec_dec else mode
                                cell = Cell(
                                    engine=EngineName(eng.id), engine_version=eng.pin, platform=plat,
                                    artifact=art, workload=wl, program=prog, mode=cell_mode,
                                )
                                cell.tiers = self._tiers_for(eng, plat, art, wl, prog, mode)
                                reason = self._unsupported_reason(cell)
                                if reason:
                                    cell.declared_unsupported_reason = reason
                                cells.append(cell)
        self._apply_suites(cells)
        return cells

    def _tiers_for(self, eng: EngineDecl, plat: PlatformSpec, art: ArtifactSpec, wl: WorkloadSpec, prog: ProgramSpec, mode: Mode) -> list[Tier]:
        s = set(eng.tiers) & set(plat.tiers) & set(art.tiers) & set(wl.tiers) & set(prog.tiers) & set(mode.tiers)
        return [t for t in Tier if t in s]

    def _unsupported_reason(self, cell: Cell) -> str | None:
        # the physical rules (context, fit) are stated before the declared ones: a cell
        # that cannot fit says so even when a family/engine rule would also exclude it
        physical = self._physical_reason(cell)
        if physical:
            return physical
        for rule in self.unsupported:
            if selector_matches(rule.when, cell):
                return rule.reason
        return None

    def _physical_reason(self, cell: Cell) -> str | None:
        # context rule: a shape longer than the artifact's max_context cannot run on it
        if cell.artifact.max_context:
            need = int(cell.workload.params.get("prefill", 0)) + int(cell.workload.params.get("decode", 0))
            if cell.workload.kind.value == "prefix_shared":
                need = int(cell.workload.params.get("shared_prefix", 0)) + int(cell.workload.params.get("unique_suffix", 0)) + int(cell.workload.params.get("decode", 0))
            if need > cell.artifact.max_context:
                return f"shape needs {need} tokens > the artifact's max_context {cell.artifact.max_context}"
        # fit rule: weights per TP group must fit with headroom
        if cell.artifact.expected_gib is not None:
            per_device = cell.artifact.expected_gib * self.fit_factor / max(1, cell.mode.tp)
            available = cell.platform.memory_gib / max(1, cell.platform.count) if cell.platform.interconnect != "uma" else cell.platform.memory_gib
            if per_device > available:
                return f"does not fit: {per_device:.1f} GiB per device > {available:.1f} GiB"
        return None

    def _apply_suites(self, cells: list[Cell]) -> None:
        for c in cells:
            if any(selector_matches(sel, c) for sel in self.global_exclude):
                c.tiers = []
        for tier_name, suite in self.suites.items():
            tier = Tier(tier_name)
            for c in cells:
                if any(selector_matches(sel, c) for sel in suite.exclude) and tier in c.tiers:
                    c.tiers.remove(tier)
                if any(selector_matches(sel, c) for sel in suite.include) and tier not in c.tiers:
                    c.tiers.append(tier)

    # ---- queries -----------------------------------------------------------------
    def cells_for(self, tier: Tier, cells: list[Cell] | None = None) -> list[Cell]:
        cells = cells if cells is not None else self.expand()
        return [c for c in cells if tier in c.tiers]

    def runnable(self, tier: Tier, cells: list[Cell] | None = None) -> list[Cell]:
        return [c for c in self.cells_for(tier, cells) if c.declared_unsupported_reason is None]

    def budget_report(self, tier: Tier, cells: list[Cell] | None = None) -> dict[str, dict[str, float]]:
        """platform -> {minutes, shards}: estimated GPU-minutes and the number of
        ~job_budget_minutes jobs they shard into (process groups kept whole)."""
        from .jobs import shard_groups

        out: dict[str, dict[str, float]] = {}
        by_plat: dict[str, list[Cell]] = {}
        for c in self.runnable(tier, cells):
            by_plat.setdefault(c.platform.id, []).append(c)
        for plat, pc in by_plat.items():
            minutes = sum(c.workload.est_minutes for c in pc)
            shards = shard_groups(pc, self.job_budget_minutes)
            out[plat] = {"minutes": minutes, "shards": len(shards), "max_shard_minutes": max((sum(c.workload.est_minutes for c in sh) for sh in shards), default=0.0)}
        return out

    def check_budget(self, tier: Tier, cells: list[Cell] | None = None, *, enforced_only: bool = False) -> list[str]:
        suite = self.suites.get(tier.value)
        if not suite or (enforced_only and not suite.enforce_budget):
            return []
        problems = []
        for plat, r in self.budget_report(tier, cells).items():
            if r["shards"] > suite.max_jobs_per_platform:
                problems.append(f"{tier}: platform {plat} needs {int(r['shards'])} jobs of ~{self.job_budget_minutes:.0f} min ({r['minutes']:.0f} min total) > max_jobs_per_platform {suite.max_jobs_per_platform}")
            if r["max_shard_minutes"] > self.job_budget_minutes * self.kill_factor:
                problems.append(f"{tier}: platform {plat} has a single process group of {r['max_shard_minutes']:.0f} min that cannot fit a {self.job_budget_minutes:.0f}-min job")
        return problems


# ---- selectors -------------------------------------------------------------------

_CMP = re.compile(r"^(>=|<=|>|<|!=|==)?\s*(-?\d+)$")
_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "!=": operator.ne, "==": operator.eq, None: operator.eq}


def _cell_field(cell: Cell, key: str) -> Any:
    return {
        "engine": str(cell.engine),
        "backend": str(cell.platform.backend),
        "os": cell.platform.os,
        "arch": cell.platform.arch,
        "platform": cell.platform.id,
        "interconnect": cell.platform.interconnect,
        "family": cell.artifact.family,
        "artifact": cell.artifact.id,
        "scheme": str(cell.artifact.scheme),
        "source_format": str(cell.artifact.source_format),
        "artifact_kind": str(cell.artifact.kind),
        "program": cell.program.id,
        "program_category": cell.program.category,
        "workload": cell.workload.id,
        "workload_kind": str(cell.workload.kind),
        "tp": cell.mode.tp,
        "spec_dec": cell.mode.spec_dec,
    }[key]


def selector_matches(sel: dict[str, Any], cell: Cell) -> bool:
    """All listed fields must match. A key prefixed ``not_`` inverts that
    field's test (``not_workload: [a, b]`` = workload is neither a nor b)."""
    for key, want in sel.items():
        negate = key.startswith("not_")
        have = _cell_field(cell, key[4:] if negate else key)
        if negate:
            if _field_matches(have, want):
                return False
            continue
        if not _field_matches(have, want):
            return False
    return True


def _field_matches(have: Any, want: Any) -> bool:
    if True:
        if isinstance(want, list):
            return have in want
        if isinstance(want, str) and isinstance(have, int) and _CMP.match(want.strip()):
            op, num = _CMP.match(want.strip()).groups()  # type: ignore[union-attr]
            return bool(_OPS[op](have, int(num)))
        return have == want


def summarize(cells: Iterable[Cell]) -> dict[str, int]:
    out: dict[str, int] = {"total": 0, "declared_unsupported": 0}
    for c in cells:
        out["total"] += 1
        if c.declared_unsupported_reason:
            out["declared_unsupported"] += 1
        for t in c.tiers:
            out[f"tier:{t}"] = out.get(f"tier:{t}", 0) + 1
    return out
