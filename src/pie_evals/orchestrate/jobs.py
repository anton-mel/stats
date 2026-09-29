"""Turn cells into JobSpecs: one job per platform per tier, carrying the
pins and the history each cell needs for adaptive repetition."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pie_evals.schema import Cell, JobSpec, RepetitionPolicy, Tier

from .store import Store

if TYPE_CHECKING:
    from .matrix import Matrix


def process_key(c: Cell) -> tuple:
    return (str(c.engine), c.artifact.artifact_key, c.mode.key)



ENGINE_CELL_MINUTES = {"ollama": 0.5}


def cell_minutes(c: Cell, measured: dict[str, float] | None = None) -> float:
    """The cell's median recorded duration where the store has one (the static
    estimate left s00 shards' last cells not_run on every platform), else the estimate."""
    if measured and c.cell_id in measured:
        return measured[c.cell_id]
    return float(c.workload.est_minutes) + ENGINE_CELL_MINUTES.get(str(c.engine), 0.0)

def shard_groups(cells: list[Cell], budget_minutes: float, measured: dict[str, float] | None = None) -> list[list[Cell]]:
    """Bin-pack cells into jobs of ~budget_minutes. A process group (same
    engine, artifact, mode = one loaded model) is never split; groups are
    placed largest-first into the first shard with room (first-fit
    decreasing), so a shard holds a few whole models and nothing else."""
    groups: dict[tuple, list[Cell]] = {}
    for c in cells:
        groups.setdefault(process_key(c), []).append(c)
    # a group larger than the budget is split into budget-sized pieces (the
    # model loads once per piece; pieces after the first carry no control-aa
    # cell, so they run without the A/A gate — recorded in the job's notes)
    pieces: list[list[Cell]] = []
    for g in groups.values():
        if sum(cell_minutes(c, measured) for c in g) <= budget_minutes:
            pieces.append(g)
            continue
        g = sorted(g, key=lambda c: (0 if c.workload.kind.value == "control_aa" else 1, -cell_minutes(c, measured)))
        cur: list[Cell] = []
        load = 0.0
        for c in g:
            if cur and load + cell_minutes(c, measured) > budget_minutes:
                pieces.append(cur)
                cur, load = [], 0.0
            cur.append(c)
            load += cell_minutes(c, measured)
        if cur:
            pieces.append(cur)
    ordered = sorted(pieces, key=lambda g: -sum(cell_minutes(c, measured) for c in g))
    shards: list[list[Cell]] = []
    loads: list[float] = []
    for g in ordered:
        need = sum(cell_minutes(c, measured) for c in g)
        for i, load in enumerate(loads):
            if load + need <= budget_minutes:
                shards[i].extend(g)
                loads[i] += need
                break
        else:
            shards.append(list(g))
            loads.append(need)
    return shards


def make_jobs(
    matrix: Matrix,
    tier: Tier,
    *,
    pie_commit: str | None,
    store: Store | None = None,
    platforms: list[str] | None = None,
    engines: list[str] | None = None,
    cells: list[Cell] | None = None,
    label: str | None = None,
) -> list[JobSpec]:
    cells = matrix.runnable(tier, cells)
    if platforms:
        cells = [c for c in cells if c.platform.id in platforms]
    if engines:
        cells = [c for c in cells if str(c.engine) in engines]
    history = store.history(limit=20) if store else {}
    measured = store.measured_minutes() if store else {}
    by_plat: dict[str, list[Cell]] = {}
    for c in cells:
        by_plat.setdefault(c.platform.id, []).append(c)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    budget = matrix.job_budget_minutes
    kill_s = int(budget * matrix.kill_factor * 60)
    jobs = []
    for plat, plat_cells in sorted(by_plat.items()):
        for idx, shard in enumerate(shard_groups(plat_cells, budget, measured)):
            # one model per process: order by (engine, artifact, mode) so the node loads each model once
            shard.sort(key=lambda c: (str(c.engine), c.artifact.id, c.mode.key, c.program.id, c.workload.id))
            pins = {e.id: e.pin for e in matrix.engines.values() if e.pin and any(str(c.engine) == e.id for c in shard)}
            seed = f"{tier}|{plat}|{idx}|{pie_commit}|{stamp}|{label or ''}"
            job_id = f"{tier}-{plat}-s{idx:02d}-{stamp}-" + hashlib.sha256(seed.encode()).hexdigest()[:6]
            est = sum(cell_minutes(c, measured) for c in shard)
            jobs.append(
                JobSpec(
                    job_id=job_id,
                    tier=tier,
                    platform_id=plat,
                    shard=idx,
                    pie_commit=pie_commit,
                    pie_build_features=["metal"],
                    baseline_versions=pins,
                    cells=shard,
                    repetition=RepetitionPolicy(min_rounds=1),
                    history={c.cell_id: history.get(c.cell_id, []) for c in shard if c.cell_id in history},
                    per_cell_timeout_s=min(1800, kill_s),
                    budget_s=int(budget * 60),
                    kill_s=kill_s,
                    est_minutes=est,
                )
            )
    # baseline comparisons first: a capped dispatch (--max-jobs) must not end up pie-only
    jobs.sort(key=lambda j: (-len({str(c.engine) for c in j.cells}), -j.est_minutes))
    return jobs


def available_platforms(matrix: Matrix, repo: str, token: str | None = None) -> tuple[list[str] | None, dict[str, str]]:
    """Macs a job can actually land on right now: those with an *online*
    registered runner carrying the platform's label. Everything else is
    skipped with a reason — a job queued for a runner that never comes sits
    in GitHub's queue for 24 h and holds the workflow's concurrency group."""
    env = {**os.environ, **({"GH_TOKEN": token} if token else {})}
    try:
        out = subprocess.run(["gh", "api", f"repos/{repo}/actions/runners?per_page=100"], capture_output=True, text=True, env=env, check=True).stdout
        runners = json.loads(out).get("runners", [])
    except Exception as e:  # noqa: BLE001
        return None, {"*": f"runner list unavailable ({str(e).strip()[:120]})"}
    note = ""
    online = {lab["name"] for r in runners if r.get("status") == "online" for lab in r.get("labels", [])}
    ok, skipped = [], {}
    for p in matrix.platforms.values():
        if p.id in online:
            ok.append(p.id)
        else:
            skipped[p.id] = note or f"no online runner with label '{p.id}'"
    return ok, skipped


def recorded_cell_keys(store: Store, tier: Tier, pie_commit: str, baseline_pins: dict[str, str] | None = None) -> set[str]:
    """Cells the store already holds: pie cells at this pie commit, and Ollama
    cells at the Ollama version this Mac runs (an Ollama number does not change
    with pie's commit). A re-dispatch after lost launches, or a new pie commit,
    then repeats only what never ran."""
    t = store.table(tier)
    if t.num_rows == 0:
        return set()
    latest = {}
    for r in t.select(["run_id", "cell_id", "cell_key", "pie_commit", "status", "engine", "engine_version", "error_class"]).to_pylist():
        latest[(r["run_id"], r["cell_id"] or r["cell_key"])] = r
    pins = baseline_pins or {}
    done = set()
    for r in latest.values():
        k, c, s, e, v, cls = r["cell_key"], r["pie_commit"], r["status"], r["engine"], r["engine_version"], r["error_class"]
        if s in ("not_run", "noisy"):  # never reached, or measured on a card that could not agree with itself: a re-dispatch measures again
            continue
        if str(cls or "") == "harness_invalid":
            continue  # the harness, not the engine, failed (dirty card, preflight, budget): no verdict on the cell
        if e != "pie" and s != "pass":
            continue  # a baseline that failed is a recipe or install fault, fixed on the harness side: measure again
        if (e == "pie" and c == pie_commit) or (e != "pie" and v and v == pins.get(str(e))):
            done.add(k)
    return done
