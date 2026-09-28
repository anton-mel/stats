"""``pie-evals`` — orchestration CLI. Thin, so that a workflow step and a
future coordinator service call the same functions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from pie_evals.schema import Tier

from . import report as rp
from .jobs import make_jobs
from .matrix import Matrix, summarize
from .store import Store


@click.group()
@click.option("--matrix", "matrix_dir", default="matrix", show_default=True)
@click.option("--store", "store_dir", default="store", show_default=True)
@click.pass_context
def main(ctx, matrix_dir, store_dir):
    ctx.obj = {"matrix": Matrix.load(matrix_dir), "store": Store(store_dir)}


@main.command("expand")
@click.option("--tier", type=click.Choice([t.value for t in Tier]), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def expand(obj, tier, as_json):
    """Expand the matrix; print a summary (or the cells as JSON)."""
    m: Matrix = obj["matrix"]
    cells = m.expand()
    if tier:
        cells = m.cells_for(Tier(tier), cells)
    if as_json:
        click.echo(json.dumps([c.model_dump(mode="json") | {"cell_id": c.cell_id, "cell_key": c.cell_key} for c in cells]))
    else:
        click.echo(json.dumps(summarize(cells), indent=1))
        for t in Tier:
            click.echo(f"{t}: budget {json.dumps(m.budget_report(t, cells))}")
        for p in [p for t in Tier for p in m.check_budget(t, cells)]:
            click.echo(f"BUDGET: {p}", err=True)


@main.command("check")
@click.pass_obj
def check(obj):
    """Validate the matrix: budgets per tier. Exit 1 on a violation in an enforced tier."""
    m: Matrix = obj["matrix"]
    cells = m.expand()
    for t in Tier:
        for p in m.check_budget(t, cells):
            click.echo(("ERROR " if m.suites[t.value].enforce_budget else "warn  ") + p, err=True)
    problems = [p for t in Tier for p in m.check_budget(t, cells, enforced_only=True)]
    click.echo(json.dumps(summarize(cells)))
    sys.exit(1 if problems else 0)


@main.command("jobs")
@click.option("--tier", type=click.Choice([t.value for t in Tier]), required=True)
@click.option("--pie-commit", default=None)
@click.option("--platform", "platforms", multiple=True)
@click.option("--engine", "engines", multiple=True)
@click.option("--program", "programs", multiple=True, help="restrict to these program ids")
@click.option("--artifact", "artifacts", multiple=True, help="restrict to these artifact ids")
@click.option("--workload", "workloads", multiple=True, help="restrict to these workload ids (the control A/A always stays)")
@click.option("--max-jobs", type=int, default=None, help="cap the number of jobs (shards) written, baseline-bearing first")
@click.option("--max-jobs-per-platform", type=int, default=None, help="cap per platform, so a capped dispatch spans every platform")
@click.option("--out", type=click.Path(), default="jobs")
@click.option("--label", default=None)
@click.option("--skip-unavailable", is_flag=True, help="drop Macs with no online runner (needs gh + a token with actions:read)")
@click.option("--repo", default=None, help="owner/name for --skip-unavailable (default: $GITHUB_REPOSITORY)")
@click.option("--skip-recorded", is_flag=True, help="drop cells the store already holds at this pie commit / Ollama pin")
@click.pass_obj
def jobs(obj, tier, pie_commit, platforms, engines, programs, artifacts, workloads, max_jobs, max_jobs_per_platform, out, label, skip_unavailable, repo, skip_recorded):
    """Write one JobSpec JSON per Mac shard, plus a GitHub Actions matrix file."""
    from .jobs import available_platforms, recorded_cell_keys

    m: Matrix = obj["matrix"]
    st: Store = obj["store"]
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    plats = list(platforms) or None
    if skip_unavailable:
        ok, skipped = available_platforms(m, repo or os.environ.get("GITHUB_REPOSITORY", "anton-mel/stats"))
        plats = [p for p in (plats or list(m.platforms))] if plats else list(m.platforms)
        plats = [p for p in plats if p in ok]
        (outp / "skipped-platforms.json").write_text(json.dumps(skipped, indent=1))
        for pid, why in skipped.items():
            click.echo(f"skip platform {pid}: {why}", err=True)
    cells = m.expand()
    if programs:
        cells = [c for c in cells if c.program.id in programs]
    if artifacts:
        cells = [c for c in cells if c.artifact.id in artifacts]
    if workloads:
        cells = [c for c in cells if c.workload.id in workloads or str(c.workload.kind) == "control_aa"]
    if skip_recorded and pie_commit:
        done = recorded_cell_keys(st, Tier(tier), pie_commit, {e.id: e.pin for e in m.engines.values() if getattr(e, "pin", None)})
        before = len(cells)
        cells = [c for c in cells if c.cell_key not in done]
        click.echo(f"skip {before - len(cells)} cells already recorded at {pie_commit[:8]}", err=True)
    js = make_jobs(m, Tier(tier), pie_commit=pie_commit, store=st, platforms=plats, engines=list(engines) or None, cells=cells, label=label)
    if max_jobs_per_platform is not None:
        seen: dict[str, int] = {}
        kept = []
        for j in js:
            if seen.get(j.platform_id, 0) < max_jobs_per_platform:
                kept.append(j)
                seen[j.platform_id] = seen.get(j.platform_id, 0) + 1
        js = kept
    if max_jobs is not None:
        js = js[:max_jobs]
    gh = []
    for j in js:
        (outp / f"{j.job_id}.json").write_text(j.model_dump_json(indent=1))
        plat = m.platforms[j.platform_id]
        gh.append({"job_id": j.job_id, "platform": j.platform_id, "shard": j.shard, "labels": plat.runner_labels, "os": plat.os, "cells": len(j.cells), "est_minutes": round(j.est_minutes, 1)})
    (outp / "gh-matrix.json").write_text(json.dumps({"include": gh}))
    (outp / "policy.json").write_text(json.dumps({"job_budget_minutes": m.job_budget_minutes, "kill_minutes": m.kill_minutes}))
    click.echo(json.dumps(gh, indent=1))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"kill_minutes={m.kill_minutes}\nbench_timeout_minutes={int(m.kill_minutes) + 15}\njobs={len(js)}\n")  # workflow expressions cannot add


@main.command("collect")
@click.option("--tier", type=click.Choice([t.value for t in Tier]), required=True)
@click.argument("node_out", nargs=-1, type=click.Path(exists=True))
@click.pass_obj
def collect(obj, tier, node_out):
    """Import node output directories (records.jsonl) into the store."""
    st: Store = obj["store"]
    problems = []
    for d in node_out:
        d = Path(d)
        run_id = (d / "run_id.txt").read_text().strip() if (d / "run_id.txt").exists() else d.name
        try:
            p = st.import_node_output(d, tier=Tier(tier), run_id=run_id)
        except Exception as e:
            problems.append(f"{d}: {e}")
            click.echo(f"{d}: IMPORT FAILED: {e}", err=True)
            continue
        click.echo(f"{d} -> {p if p else 'no records (node crashed before its first record)'}")
    if problems:
        sys.exit(1)


@main.command("report")
@click.option("--tier", type=click.Choice([t.value for t in Tier]), required=True)
@click.option("--out", type=click.Path(), default="reports")
@click.pass_obj
def report(obj, tier, out):
    """Render coverage, regressions, baseline ratios."""
    m: Matrix = obj["matrix"]
    st: Store = obj["store"]
    t = Tier(tier)
    outp = Path(out) / t.value
    outp.mkdir(parents=True, exist_ok=True)
    views = rp.coverage(m, st, t)
    (outp / "coverage.md").write_text(rp.coverage_markdown(views, t))
    regs = rp.regressions(m, st, t)
    (outp / "regressions.md").write_text(rp.regressions_markdown(regs))
    (outp / "regressions.json").write_text(rp.to_json(regs))
    ratios = rp.baseline_ratios(m, st, t)
    (outp / "baselines.md").write_text(rp.baseline_markdown(ratios))
    (outp / "baselines.json").write_text(rp.to_json(ratios))
    n_reg = sum(1 for r in regs if r["kind"] == "regression")
    trails = sum(1 for r in ratios if not r["pie_leads"])
    click.echo(json.dumps({"cells": len(views), "regressions": n_reg, "pie_trails": trails}))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"regressions={n_reg}\npie_trails={trails}\n")



if __name__ == "__main__":
    main()
