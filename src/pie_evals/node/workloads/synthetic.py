"""Workload shapes → ``scripts/bench/common.py`` arguments.

common.py builds prompts from words (``--prompt``, ``--shared-prefix-words``,
``--mixed-phase`` ...), so a shape defined in tokens is realised as a
deterministic word block sized by ``WORDS_PER_TOKEN`` and then *checked*:
after a run, ``check_input_parity`` compares the prompt/output token counts
every engine reported. If they differ the tok/s column is not a comparison
and the record is marked INPUT_MISMATCH.

TODO(pie): add ``--prompt-tokens-file`` to scripts/bench/common.py so shapes can be
pre-tokenized once and handed to every engine as ids; until then prompts are
words and parity is verified rather than guaranteed.
"""

from __future__ import annotations

import hashlib
import random

from pie_evals.schema import WorkloadSpec

WORDS_PER_TOKEN = 1.0  # measured: a 5898-word block of these common words became 5968 tokens on Qwen3.5 (one token per word); verified per run by check_input_parity
SEED = 20260730  # same as common.py's --arrival-seed default

_WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or his from at which but have an "
    "had they you were their one all we can her has there been if more when will would who so no she what "
    "about up out many then them these some him time into only its two could other new than first any my "
    "now such like our over man me even most made after also did much before here through years where "
    "must way well down should because each just those people how too little state good very make world "
    "still own see men work long get life never day another know while last might great old year off "
    "come since against go came right used take three states himself few house use during without again "
    "place american around however home small found mrs thought went say part once general high upon "
    "school every don does got united left number course war until always away something fact though "
    "water less public put think almost hand enough far took head yet government system better set told "
    "nothing night end why called didn eyes find going look asked later knew point next program city"
).split()


def prompt_words_for_tokens(n_tokens: int, seed: int = SEED, salt: str = "") -> str:
    """A deterministic block of prose of about ``n_tokens`` tokens."""
    n_words = max(1, int(round(n_tokens * WORDS_PER_TOKEN)))
    rng = random.Random(f"{seed}:{salt}:{n_tokens}")
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def common_args_for(workload: WorkloadSpec, warmup: int = 2) -> list[str]:
    """common.py flags shared by every engine for this shape."""
    p = workload.params
    kind = str(workload.kind)
    args: list[str] = ["--temperature", str(p.get("temperature", 0)), "--top-p", "1", "--ignore-eos", "--warmup", str(warmup), "--no-think"]
    if kind == "long_context":
        # a 32k-word prompt does not fit one argv element (E2BIG at 128 KB): the bench
        # synthesizes the bulk itself from --shared-prefix-words, deterministically and
        # identically for every engine; only a short tail travels on the command line
        prefill = int(p.get("prefill", 2048))
        tail = 64
        args += ["--prompt", prompt_words_for_tokens(tail, salt=workload.id)]
        args += ["--shared-prefix-words", str(int(round((prefill - tail) * WORDS_PER_TOKEN)))]
        args += ["--max-tokens", str(int(p.get("decode", 64)))]
    elif kind in ("single_stream", "control_aa", "concurrency"):
        prompt = str(p["prompt"]) if p.get("prompt") else prompt_words_for_tokens(int(p.get("prefill", 128)), salt=workload.id)
        args += ["--prompt", prompt]
        args += ["--max-tokens", str(int(p.get("decode", 64)))]
        if kind == "concurrency":
            args += ["--unique-prompts"]
    elif kind == "prefix_shared":
        args += ["--prompt", prompt_words_for_tokens(int(p.get("unique_suffix", 64)), salt=workload.id)]
        args += ["--shared-prefix-words", str(int(round(int(p.get("shared_prefix", 1024)) * WORDS_PER_TOKEN)))]
        args += ["--max-tokens", str(int(p.get("decode", 64))), "--unique-prompts"]
    elif kind == "mixed_length":
        mu_p, _ = p.get("prompt_lognormal", [6.0, 0.8])
        mu_o, _ = p.get("output_lognormal", [5.0, 0.7])
        import math

        long_words = int(min(int(p.get("max_prompt", 4096)), math.exp(mu_p) * 4) * WORDS_PER_TOKEN)
        args += ["--prompt", prompt_words_for_tokens(int(math.exp(mu_p)), salt=workload.id)]
        args += ["--mixed-phase", "--mixed-long-prompt-words", str(long_words)]
        args += ["--mixed-short-output", str(max(8, int(math.exp(mu_o) / 4)))]
        args += ["--max-tokens", str(int(min(int(p.get("max_output", 1024)), math.exp(mu_o) * 2)))]
        args += ["--output-spread", "4", "--unique-prompts"]
    elif kind == "replay":
        # open loop; trace-driven arrivals are approximated by the trace's mean rate until
        # common.py grows a --trace flag (tracked in docs/design.md, "replay").
        args += ["--prompt", prompt_words_for_tokens(512, salt=workload.id), "--max-tokens", "256"]
        args += ["--arrival-rate", str(p.get("arrival_rate", 4.0)), "--arrival-process", "poisson", "--arrival-seed", str(SEED)]
        args += ["--unique-prompts"]
    else:
        raise ValueError(kind)
    return args


def check_input_parity(summaries: dict[str, dict]) -> str | None:
    """Given engine -> summary, return None if prompt/output token totals agree, else a message."""
    pt = {e: s.get("prompt_tokens") for e, s in summaries.items()}
    ot = {e: s.get("output_tokens") for e, s in summaries.items()}
    if len(set(pt.values())) > 1 or len(set(ot.values())) > 1:
        return f"input mismatch: prompt_tokens={pt} output_tokens={ot}"
    return None


def workload_fingerprint(workload: WorkloadSpec) -> str:
    return hashlib.sha256(workload.model_dump_json().encode()).hexdigest()[:12]
