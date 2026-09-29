"""``pie-evals-node`` — what runs on the machine under test."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from pie_evals.schema import JobSpec

from .build import DEFAULT_PIE_ROOT


@click.group()
def main():
    pass


@main.command("run")
@click.option("--job", "job_path", type=click.Path(exists=True), required=True)
@click.option("--pie-root", type=click.Path(exists=True), default=os.environ.get("PIE_ROOT", str(DEFAULT_PIE_ROOT)))
@click.option("--out", type=click.Path(), default="out")
@click.option("--hf-cache", type=click.Path(), default=None)
@click.option("--no-build", is_flag=True, help="use the pie binary already in target/release")
@click.option("--only", multiple=True, help="restrict to cells whose key contains this substring")
@click.option("--download", is_flag=True, help="fetch missing checkpoints and Ollama tags here (default: no — `prepare` does that)")
def run(job_path, pie_root, out, hf_cache, no_build, only, download):
    """Execute a JobSpec and write records.jsonl under --out."""
    from .runner import NodeRunner

    job = JobSpec.model_validate_json(Path(job_path).read_text())
    if only:
        job = job.model_copy(update={"cells": [c for c in job.cells if any(s in c.cell_key for s in only)]})
    runner = NodeRunner(job, pie_root=Path(pie_root), out_dir=Path(out), hf_cache=Path(hf_cache) if hf_cache else None, build=not no_build, download=download)
    recs = runner.run()
    by = {}
    for r in recs:
        by[str(r.status)] = by.get(str(r.status), 0) + 1
    click.echo(json.dumps({"run_id": runner.run_id, "records": len(recs), "by_status": by}))


@main.command("build")
@click.option("--pie-commit", required=True)
@click.option("--features", default="metal", show_default=True, help="comma-separated pie cargo features")
@click.option("--pie-root", type=click.Path(), default=os.environ.get("PIE_ROOT", str(DEFAULT_PIE_ROOT)))
@click.option("--mirror", type=click.Path(), default=os.environ.get("PIE_MIRROR"), help="bare mirror; updated, then used as clone source")
@click.option("--target-dir", type=click.Path(), default=os.environ.get("CARGO_TARGET_DIR"))
@click.option("--force", is_flag=True)
def build_cmd(pie_commit, features, pie_root, mirror, target_dir, force):
    """Build pie at a commit and cache the artifacts (binary, bench wasm) under $PIE_EVALS_CACHE/builds."""
    from . import build as pb

    feats = [f for f in features.split(",") if f]
    if mirror:
        pb.update_mirror(Path(mirror))
    pb.ensure_checkout(Path(pie_root), pie_commit, mirror=Path(mirror) if mirror else None)
    if pb.is_cached(pie_commit, feats) and not force:
        click.echo(f"cache hit: {pb.cache_dir(pie_commit, feats)}")
        return
    out = pb.build(Path(pie_root), pie_commit, feats, target_dir=Path(target_dir) if target_dir else None, python=os.environ.get("PIE_PY", "python3"))
    click.echo(str(out))


@main.command("prepare")
@click.option("--tier", type=click.Choice(["targeted"]), default="targeted", show_default=True)
@click.option("--matrix", "matrix_dir", default="matrix")
@click.option("--platform", "platforms", multiple=True, help="restrict to artifacts these platforms run (default: all of the tier)")
@click.option("--engine", "engines_f", multiple=True)
@click.option("--program", "programs_f", multiple=True)
@click.option("--pie-root", type=click.Path(), default=os.environ.get("PIE_ROOT", str(DEFAULT_PIE_ROOT)))
@click.option("--pie-commit", default=None, help="import quantized checkpoints as .zt artifacts for this commit (needs the built pie binary)")
@click.option("--hf-cache", type=click.Path(), default=None)
def prepare_cmd(tier, matrix_dir, platforms, engines_f, programs_f, pie_root, pie_commit, hf_cache):
    """Fetch every checkpoint and Ollama tag a tier needs, so a run finds them on disk.
    Failures are reported per artifact and never abort the rest; exit 1 if any failed."""
    from pie_evals.orchestrate.matrix import Matrix
    from pie_evals.schema import Tier

    from .snapshots import ensure_snapshot, hf_cache_dir

    m = Matrix.load(matrix_dir)
    cells = m.runnable(Tier(tier))
    if platforms:
        cells = [c for c in cells if c.platform.id in platforms]
    if engines_f:
        cells = [c for c in cells if str(c.engine) in engines_f]
    if programs_f:
        cells = [c for c in cells if c.program.id in programs_f]
    arts = {c.artifact.id: c.artifact for c in cells}
    cache = Path(hf_cache) if hf_cache else hf_cache_dir()
    failed = {}
    from .importer import ensure_artifact, needs_import

    pie_bin = Path(pie_root) / "target/release/pie"
    say = lambda m_: click.echo(m_, err=True)  # noqa: E731
    for aid, art in sorted(arts.items()):
        try:
            p = ensure_snapshot(art, cache, log=say)
            if art.ollama_tag:
                from .engines.ollama import ensure_ollama_model

                ensure_ollama_model(art.ollama_tag, pull=True, log=say)
            if pie_commit and needs_import(art) and pie_bin.exists():
                p = ensure_artifact(art, p, pie_bin, pie_commit, log=say)
            click.echo(f"ok    {aid}: {p}")
        except Exception as e:
            failed[aid] = str(e).strip().splitlines()[-1][:200] if str(e).strip() else repr(e)
            click.echo(f"FAIL  {aid}: {failed[aid]}", err=True)
    click.echo(json.dumps({"prepared": len(arts) - len(failed), "failed": failed}))
    sys.exit(1 if failed else 0)


@main.command("quality", help="Score output quality (GSM8K, IFEval) on the same sampled questions for each artifact.")
@click.option("--artifact", "artifacts", multiple=True, required=True, help="artifact id from matrix/models.yaml (a pie checkpoint or an Ollama arm)")
@click.option("--task", "tasks", multiple=True, type=click.Choice(["gsm8k", "ifeval"]), default=("gsm8k", "ifeval"), show_default=True)
@click.option("--n", type=int, default=100, show_default=True)
@click.option("--matrix", "matrix_dir", default="matrix")
@click.option("--store", "store_dir", default="store")
@click.option("--pie-root", type=click.Path(), default=os.environ.get("PIE_ROOT", str(DEFAULT_PIE_ROOT)))
def quality_cmd(artifacts, tasks, n, matrix_dir, store_dir, pie_root):
    from pie_evals.quality.run import run_quality

    for a in artifacts:
        run_quality(a, list(tasks), n, matrix_dir=matrix_dir, store_dir=store_dir, pie_root=Path(pie_root), log=lambda m_: click.echo(m_, err=True))


@main.command("preflight")
def preflight_cmd():
    """Print the hardware fingerprint and machine state checks."""
    from . import preflight as pf
    from . import provenance as prov

    click.echo(json.dumps({"fingerprint": prov.hardware_fingerprint(), "versions": prov.versions(), "state": pf.Preflight().before_job(None)}, indent=1, default=str))



if __name__ == "__main__":
    main()
