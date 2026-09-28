#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from common import (
    RequestResult,
    add_mode_subcommands,
    add_output_dump_args,
    finish,
    hf_chat_prompts_and_counts,
    make_prompts,
    request_max_tokens,
    summarize,
)

FLUSH_PROMPT = "0"


def post_stream(url: str, payload: dict[str, Any], timeout: float, on_chunk) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    last: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                raise RuntimeError(f"ollama: {obj['error']}")
            on_chunk(obj)
            last = obj
    return last


def generate_payload(args: argparse.Namespace, prompt: str, max_tokens: int) -> dict[str, Any]:
    options: dict[str, Any] = {"num_predict": max_tokens, "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed}
    options.update(json.loads(args.options) if args.options else {})
    return {"model": args.ollama_model, "prompt": prompt, "raw": True, "stream": True, "options": options, "keep_alive": args.keep_alive}


def run_one(args: argparse.Namespace, endpoint: str, prompt: str, prompt_count: int, max_tokens: int, server: list[dict[str, Any]]) -> RequestResult:
    start = time.perf_counter()
    stamps: list[float] = []

    def on_chunk(obj: dict[str, Any]) -> None:
        if obj.get("response") or obj.get("thinking"):
            stamps.append(time.perf_counter())

    try:
        done = post_stream(endpoint, generate_payload(args, prompt, max_tokens), args.request_timeout, on_chunk)
    except Exception as e:  # noqa: BLE001
        return RequestResult(False, time.perf_counter() - start, 0, error=f"{type(e).__name__}: {e}")
    end = time.perf_counter()
    n_out = int(done.get("eval_count") or 0)
    server.append({
        "prompt_eval_count": done.get("prompt_eval_count"),
        "prompt_eval_cached_count": done.get("prompt_eval_cached_count"),
        "prompt_eval_s": (done.get("prompt_eval_duration") or 0) / 1e9,
        "eval_count": n_out,
        "eval_s": (done.get("eval_duration") or 0) / 1e9,
        "load_s": (done.get("load_duration") or 0) / 1e9,
        "done_reason": done.get("done_reason"),
        "hf_prompt_tokens": prompt_count,
        "max_tokens": max_tokens,
    })
    ttft = (stamps[0] - start) if stamps else None
    gaps = [int((b - a) * 1e6) for a, b in zip(stamps, stamps[1:])]
    return RequestResult(True, end - start, n_out, int(done.get("prompt_eval_count") or prompt_count), ttft_s=ttft, intertoken_us=gaps)


def flush(args: argparse.Namespace, endpoint: str) -> None:
    def one(i: int) -> None:
        try:
            post_stream(endpoint, generate_payload(args, f"{FLUSH_PROMPT} {i}", 1), args.request_timeout, lambda _o: None)
        except Exception as e:  # noqa: BLE001
            print(f"flush failed: {e}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, args.flush_slots)) as pool:
        list(pool.map(one, range(max(1, args.flush_slots))))


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


async def run(args: argparse.Namespace):
    if args.ollama_model is None:
        sys.exit("--ollama-model is required")
    n = args.requests if args.mode == "latency" else args.num_requests
    batch = 1 if args.mode == "latency" else (args.concurrency or max(1, args.num_requests))
    prompts, counts = hf_chat_prompts_and_counts(args.model, args.system, make_prompts(args, n + args.warmup), getattr(args, "think", None))
    budgets = [request_max_tokens(args, i) for i in range(len(prompts))]
    endpoint = args.url.rstrip("/") + "/api/generate"
    server: list[dict[str, Any]] = []
    latency = args.mode == "latency"

    for i in range(args.warmup):
        if args.flush_cache and latency:
            flush(args, endpoint)
        await asyncio.to_thread(run_one, args, endpoint, prompts[i], counts[i], budgets[i], [])

    run_p, run_c, run_b = prompts[args.warmup:], counts[args.warmup:], budgets[args.warmup:]
    gate = asyncio.Semaphore(batch)

    async def one(p: str, c: int, m: int) -> RequestResult:
        async with gate:
            return await asyncio.to_thread(run_one, args, endpoint, p, c, m, server)

    if latency:
        results = []
        wall = 0.0
        for p, c, m in zip(run_p, run_c, run_b):
            if args.flush_cache:
                flush(args, endpoint)
            t0 = time.perf_counter()
            results.append(await one(p, c, m))
            wall += time.perf_counter() - t0
    else:
        if args.flush_cache:
            flush(args, endpoint)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(p, c, m) for p, c, m in zip(run_p, run_c, run_b)))
        wall = time.perf_counter() - t0

    pre = [s["prompt_eval_count"] / s["prompt_eval_s"] for s in server if s["prompt_eval_s"] and s["prompt_eval_count"]]
    dec = [s["eval_count"] / s["eval_s"] for s in server if s["eval_s"] and s["eval_count"]]
    cached = sum(int(s["prompt_eval_cached_count"] or 0) for s in server)
    evaluated = sum(int(s["prompt_eval_count"] or 0) for s in server)
    summary = summarize(
        mode=args.mode, engine="ollama", model=args.ollama_model, results=results, wall_s=wall,
        config={
            "ollama_model": args.ollama_model,
            "client_in_flight": batch,
            "raw_prompt": True,
            "flush_cache": bool(args.flush_cache),
            "ignore_eos": "unsupported by Ollama; every request stopped at EOS or num_predict",
            "short_responses": sum(1 for s in server if s["eval_count"] < s["max_tokens"]),
            "cached_prompt_tokens": cached,
            "cached_prompt_fraction": (cached / evaluated) if evaluated else 0.0,
            "hf_prompt_tokens": sum(s["hf_prompt_tokens"] for s in server),
            "server_prefill_tok_s": _median(pre),
            "server_decode_tok_s": _median(dec),
            "server_load_s_max": max((s["load_s"] for s in server), default=0.0),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "options": args.options or "",
        },
    )
    return summary, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama latency/throughput benchmark on pie's common client")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        add_output_dump_args(sp)
        sp.add_argument("--url", default="http://127.0.0.1:11434")
        sp.add_argument("--ollama-model", default=None, help="Ollama tag; --model is the HF tokenizer")
        sp.add_argument("--keep-alive", default="30m")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--options", default="", help="extra Ollama options as JSON")
        sp.add_argument("--flush-cache", action=argparse.BooleanOptionalAction, default=True, help="one-token unrelated prompt per slot before each request")
        sp.add_argument("--flush-slots", type=int, default=1, help="server slots to flush, OLLAMA_NUM_PARALLEL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, results = asyncio.run(run(args))
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
