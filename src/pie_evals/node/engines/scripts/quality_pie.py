#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pie_bench import bench_inferlet_wasm, build_parser, pie_client


async def generate(bench_args: list[str], items: list[dict], system: str, max_tokens: int, timeout: float, concurrency: int) -> list[str]:
    from pie_client import Event

    args = build_parser().parse_args(bench_args)
    out = [""] * len(items)
    async with pie_client(args) as (client, _):
        pkg = await client.install_program(bench_inferlet_wasm(args.inferlet_dir), force_overwrite=True)
        gate = asyncio.Semaphore(concurrency)

        async def one(i: int, item: dict) -> None:
            async with gate:
                inp = {"system": system, "prompt": item["prompt"], "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
                       "ignore_eos": False, "return_text": True, "report_timing": False}
                proc = await client.launch_process(pkg, input=inp)
                while True:
                    ev, msg = await asyncio.wait_for(proc.recv(), timeout=timeout)
                    if ev == Event.Return:
                        out[i] = json.loads(msg).get("text", "")
                        return
                    if ev not in (Event.Message,):
                        out[i] = ""
                        return

        await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("bench_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    items = json.loads(Path(a.items).read_text())
    bench_args = a.bench_args[1:] if a.bench_args[:1] == ["--"] else a.bench_args
    texts = asyncio.run(generate(bench_args, items, a.system, a.max_tokens, a.timeout, a.concurrency))
    Path(a.out).write_text(json.dumps(texts))
    print(f"generated {sum(1 for t in texts if t)} of {len(texts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
