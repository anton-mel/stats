"""Reports: coverage matrix, regression diff, baseline ratios. Markdown out,
rendered by the collect workflow into the repo and GitHub Pages."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from pie_evals.node.metrics.stats import regression_verdict
from pie_evals.schema import Cell, CellStatus, RepetitionPolicy, Tier

from .matrix import Matrix
from .store import Store


@dataclass
class CellView:
    cell: Cell
    status: str
    latest: dict | None
    history: list[float]


def coverage(matrix: Matrix, store: Store, tier: Tier) -> list[CellView]:
    cells = matrix.cells_for(tier)
    latest = store.latest_by_cell()
    hist = store.history()
    views = []
    for c in cells:
        if c.declared_unsupported_reason:
            st = str(CellStatus.DECLARED_UNSUPPORTED)
            row = None
        else:
            row = latest.get(c.cell_id)
            st = row["status"] if row else str(CellStatus.NOT_RUN)
        views.append(CellView(c, st, row, hist.get(c.cell_id, [])))
    return views


def coverage_markdown(views: list[CellView], tier: Tier) -> str:
    counts = defaultdict(int)
    for v in views:
        counts[v.status] += 1
    lines = [f"# Coverage — {tier}", "", "| status | cells |", "|---|---|"]
    for k in [s.value for s in CellStatus]:
        lines.append(f"| {k} | {counts.get(k, 0)} |")
    lines += ["", "## Gaps (expected supported, but not passing)", "", "| engine | platform | artifact | workload | program | mode | status | error | message |", "|---|---|---|---|---|---|---|---|---|"]
    gaps = [v for v in views if v.status not in (str(CellStatus.PASS), str(CellStatus.DECLARED_UNSUPPORTED))]
    for v in sorted(gaps, key=lambda v: (v.status, v.cell.platform.id, v.cell.artifact.id)):
        c = v.cell
        err = (v.latest or {}).get("error_class") or ""
        msg = ((v.latest or {}).get("error_message") or "").replace("|", "\\|")[:120]
        lines.append(f"| {c.engine} | {c.platform.id} | {c.artifact.id} | {c.workload.id} | {c.program.id} | {c.mode.key} | {v.status} | {err} | {msg} |")
    lines += ["", "## Declared unsupported", "", "| reason | cells |", "|---|---|"]
    reasons = defaultdict(int)
    for v in views:
        if v.cell.declared_unsupported_reason:
            reasons[v.cell.declared_unsupported_reason] += 1
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {r} | {n} |")
    # per-family × platform grid of pie pass rate
    lines += ["", "## pie pass rate by family × platform", ""]
    grid: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    plats = sorted({v.cell.platform.id for v in views})
    for v in views:
        if str(v.cell.engine) != "pie" or v.status == str(CellStatus.DECLARED_UNSUPPORTED):
            continue
        g = grid[(v.cell.artifact.family, v.cell.platform.id)]
        g[1] += 1
        if v.status == str(CellStatus.PASS):
            g[0] += 1
    fams = sorted({k[0] for k in grid})
    lines.append("| family | " + " | ".join(plats) + " |")
    lines.append("|---|" + "---|" * len(plats))
    for f in fams:
        cells = []
        for p in plats:
            ok, n = grid.get((f, p), [0, 0])
            cells.append(f"{ok}/{n}" if n else "·")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def regressions(matrix: Matrix, store: Store, tier: Tier, *, sigma: float = 3.0) -> list[dict]:
    """Compare each cell's latest PASS value against its history (noise-aware)."""
    latest = store.latest_by_cell(tier)
    hist = store.history(tier, limit=20)
    out = []
    for cid, row in latest.items():
        if row["status"] != str(CellStatus.PASS) or row["primary_value"] is None:
            continue
        h = hist.get(cid, [])
        prior = h[:-1] if h and abs(h[-1] - float(row["primary_value"])) < 1e-12 else h
        if len(prior) < 3:
            continue
        # a change smaller than the spread a single measurement is allowed (the node's NOISY bound)
        # is not a regression: a 1 % floor failed nightlies on -1.2 % single-stream drifts
        wl = matrix.workloads.get(str(row["workload"]))
        steady = wl is not None and str(wl.kind) in ("single_stream", "control_aa", "long_context")
        policy = RepetitionPolicy()
        floor = policy.cov_noisy_threshold if steady else policy.cov_noisy_threshold_concurrent
        v = regression_verdict(float(row["primary_value"]), prior, sigma=sigma, floor_rel=floor)
        if v.regressed or v.improved:
            out.append(
                {
                    "cell_id": cid, "cell_key": row["cell_key"], "engine": row["engine"], "platform": row["platform"],
                    "artifact": row["artifact"], "workload": row["workload"], "program": row["program"], "mode": row["mode"],
                    "metric": row["primary_metric"], "value": float(row["primary_value"]), "delta_rel": v.delta_rel,
                    "threshold_rel": v.threshold_rel, "kind": "regression" if v.regressed else "improvement",
                    "pie_commit": row["pie_commit"], "engine_version": row["engine_version"], "family": row["family"],
                }
            )
    return sorted(out, key=lambda r: r["delta_rel"])


def regressions_markdown(rows: list[dict]) -> str:
    lines = ["# Regressions / improvements", "", "| kind | engine | platform | artifact | workload | program | mode | metric | value | Δ | threshold | commit/version |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['kind']} | {r['engine']} | {r['platform']} | {r['artifact']} | {r['workload']} | {r['program']} | {r['mode']} | {r['metric']} | {r['value']:.4g} | {r['delta_rel']*100:+.1f}% | ±{r['threshold_rel']*100:.1f}% | {r['pie_commit'] or r['engine_version'] or ''} |"
        )
    if len(lines) == 4:
        lines.append("| (none) | | | | | | | | | | | |")
    return "\n".join(lines) + "\n"


def baseline_ratios(matrix: Matrix, store: Store, tier: Tier) -> list[dict]:
    """pie vs each baseline arm on the same (platform, model, workload, mode). A
    baseline arm is its own artifact (Ollama serves its own copy of the weights)
    paired with pie's through ``baseline_of``."""
    latest = store.latest_by_cell(tier)
    baseline_of = {a.id: a.baseline_of for a in matrix.artifacts.values() if a.baseline_of}
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in latest.values():
        if row["status"] != str(CellStatus.PASS) or row["primary_value"] is None:
            continue
        model = baseline_of.get(row["artifact"], row["artifact"])
        key = (row["platform"], model, row["workload"], row["mode"])
        by_key[key]["pie" if row["engine"] == "pie" else row["artifact"]] = row
    out = []
    for key, arms in by_key.items():
        pie = arms.get("pie")
        if not pie:
            continue
        for arm, row in arms.items():
            if arm == "pie":
                continue
            eng = row["engine"]
            # the same measurement on both sides: aggregate output tok/s whenever
            # both rows carry it, else the primary
            metric = "output_tok_s" if row.get("output_tok_s") and pie.get("output_tok_s") else row["primary_metric"]
            pv = float(pie[metric]) if metric in pie and pie[metric] is not None else float(pie["primary_value"])
            bv = float(row[metric]) if metric in row and row[metric] is not None else float(row["primary_value"])
            ratio = bv / pv if pv else None
            out.append(
                {
                    "platform": key[0], "artifact": pie["artifact"], "workload": key[2], "program": pie["program"], "mode": key[3],
                    "baseline": eng, "baseline_artifact": arm, "baseline_version": row["engine_version"], "metric": metric, "pie_value": pv,
                    "baseline_value": bv, "baseline_over_pie": ratio,
                    "pie_leads": (ratio is not None and ratio < 1.0), "pie_cov": pie["cov"], "baseline_cov": row["cov"],
                    "input_ok": pie["prompt_tokens"] == row["prompt_tokens"] and pie["output_tokens"] == row["output_tokens"],
                    "baseline_recipe": row["engine_config_recipe"],
                }
            )
    return sorted(out, key=lambda r: -(r["baseline_over_pie"] or 0))


def baseline_markdown(rows: list[dict]) -> str:
    lines = ["# pie vs baselines (latest, best-of-recipe)", "", "| platform | artifact | workload | program | mode | baseline | metric | pie | baseline | baseline/pie | inputs match | recipe |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = "" if r["pie_leads"] else " **pie trails**"
        lines.append(
            f"| {r['platform']} | {r['artifact']} | {r['workload']} | {r['program']} | {r['mode']} | {r['baseline_artifact']} {r['baseline_version'] or ''} | {r.get('metric') or ''} | {r['pie_value']:.4g} | {r['baseline_value']:.4g} | {r['baseline_over_pie']:.3f}×{flag} | {'yes' if r['input_ok'] else '**NO**'} | {r['baseline_recipe'] or ''} |"
        )
    return "\n".join(lines) + "\n"



def to_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=1, default=str)
