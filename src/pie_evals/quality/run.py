from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pie_evals.schema import SourceFormat

from .tasks import TASKS, wilson

SYSTEM = "You are a helpful assistant."


def _tokenizer(snapshot: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)


def _render(snapshot: Path, prompts: list[str]) -> list[str]:
    tok = _tokenizer(snapshot)
    out = []
    for p in prompts:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
        try:
            out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, reasoning_strength="low"))
        except Exception:  # noqa: BLE001
            out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def _ollama_generate(tag: str, snapshot: Path, prompts: list[str], max_tokens: int, log) -> tuple[list[str], str]:
    from pie_evals.node.engines.ollama import server_version, start_server, stop_server

    rendered = _render(snapshot, prompts)
    proc, url = start_server({"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_CONTEXT_LENGTH": "4096", "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_NOHISTORY": "1"}, None, 60)
    try:
        def one(p: str) -> str:
            body = {"model": tag, "prompt": p, "raw": True, "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0, "top_p": 1, "seed": 0}, "keep_alive": "30m"}
            req = urllib.request.Request(f"{url}/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                return json.loads(r.read()).get("response", "")

        with ThreadPoolExecutor(max_workers=4) as pool:
            texts = list(pool.map(one, rendered))
        return texts, server_version()
    finally:
        stop_server(proc)


def _pie_generate(art, snapshot: Path, pie_root: Path, prompts: list[str], max_tokens: int, log) -> tuple[list[str], str]:
    from pie_evals.node.build import bench_python
    from pie_evals.node.importer import ensure_artifact, needs_import

    commit = subprocess.run(["git", "-C", str(pie_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    model = ensure_artifact(art, snapshot, pie_root / "target/release/pie", commit, log=log) if needs_import(art) else snapshot
    script = Path(__file__).resolve().parents[1] / "node" / "engines" / "scripts" / "quality_pie.py"
    env = {**os.environ, "PYTHONPATH": ":".join([str(pie_root / "scripts/bench"), str(pie_root / "python/client/src"), str(pie_root / "python/server/python")])}
    tok = _tokenizer(snapshot)
    token_ids = [tok.encode(r, add_special_tokens=False) for r in _render(snapshot, prompts)]
    with tempfile.TemporaryDirectory() as tmp:
        items, out = Path(tmp) / "items.json", Path(tmp) / "out.json"
        items.write_text(json.dumps([{"prompt": p, "prompt_tokens": ids} for p, ids in zip(prompts, token_ids, strict=True)]))
        argv = [str(bench_python()), str(script), "--items", str(items), "--out", str(out), "--system", SYSTEM, "--max-tokens", str(max_tokens), "--",
                "latency", "--model", str(model), "--engine", "metal", "--inferlet-dir", str(pie_root / "examples/text-completion-bench"),
                "--max-model-len", "4096", "--requests", "1"]
        subprocess.run(argv, cwd=str(pie_root / "scripts/bench"), env=env, check=True, timeout=6 * 3600)
        return json.loads(out.read_text()), commit[:12]


def run_quality(artifact_id: str, tasks: list[str], n: int, *, matrix_dir: str = "matrix", store_dir: str = "store", pie_root: Path, log=print) -> list[Path]:
    from pie_evals.node.snapshots import ensure_snapshot, hf_cache_dir
    from pie_evals.orchestrate.matrix import Matrix

    m = Matrix.load(matrix_dir)
    art = m.artifacts[artifact_id]
    engine = "ollama" if art.source_format == SourceFormat.OLLAMA else "pie"
    snapshot = ensure_snapshot(art, hf_cache_dir(), download=True, log=log)
    written = []
    for task in tasks:
        load, score, max_tokens = TASKS[task]
        items = load(n)
        t0 = time.monotonic()
        if engine == "ollama":
            texts, version = _ollama_generate(art.ollama_tag, snapshot, [i.prompt for i in items], max_tokens, log)
        else:
            texts, version = _pie_generate(art, snapshot, pie_root, [i.prompt for i in items], max_tokens, log)
        scores = [score(i, t) for i, t in zip(items, texts, strict=True)]
        correct = sum(scores)
        lo, hi = wilson(correct, len(scores))
        summary = {
            "artifact": art.id, "model": art.baseline_of or art.id, "engine": engine, "label": art.baseline_label or ("pie" if engine == "pie" else "Ollama"),
            "task": task, "n": len(scores), "correct": correct, "score": correct / len(scores) if scores else None, "ci95": [lo, hi],
            "version": version, "empty": sum(1 for t in texts if not t), "minutes": round((time.monotonic() - t0) / 60, 1),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "samples": [{"id": i.id, "score": s, "text": t} for i, s, t in zip(items, scores, texts, strict=True)],
        }
        path = Path(store_dir) / "quality" / f"{art.id}-{task}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=1))
        log(f"quality {art.id} {task}: {correct:.0f}/{len(scores)} = {summary['score']:.1%} (95% {lo:.1%}..{hi:.1%}) in {summary['minutes']} min")
        written.append(path)
    return written
