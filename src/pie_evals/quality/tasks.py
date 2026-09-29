from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Item:
    id: str
    prompt: str
    gold: object


GSM8K_SUFFIX = "\n\nSolve the problem step by step. End with a line of the form '#### <number>'."


def _download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def load_gsm8k(n: int, seed: int = 20260929) -> list[Item]:
    import pyarrow.parquet as pq

    rows = pq.read_table(_download("openai/gsm8k", "main/test-00000-of-00001.parquet")).to_pylist()
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return [Item(f"gsm8k-{i}", r["question"] + GSM8K_SUFFIX, r["answer"].split("####")[-1].strip().replace(",", "")) for i, r in enumerate(rows)]


def load_ifeval(n: int, seed: int = 20260929) -> list[Item]:
    rows = [json.loads(line) for line in _download("google/IFEval", "ifeval_input_data.jsonl").read_text().splitlines() if line.strip()]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return [Item(f"ifeval-{r['key']}", r["prompt"], r) for r in rows]


MC_SUFFIX = "\n\nAnswer with the letter of the correct choice. End your response with a line of the form 'Answer: <letter>'."
MATH_SUFFIX = "\n\nSolve the problem. Put the final answer in \\boxed{}."
LETTERS = "ABCDEFGH"


def _choices(question: str, options: list[str]) -> str:
    return question.strip() + "\n\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options)) + MC_SUFFIX


def load_mmlu(n: int, seed: int = 20260929) -> list[Item]:
    import pyarrow.parquet as pq

    rows = pq.read_table(_download("cais/mmlu", "all/test-00000-of-00001.parquet")).to_pylist()
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return [Item(f"mmlu-{i}", _choices(r["question"], list(r["choices"])), LETTERS[int(r["answer"])]) for i, r in enumerate(rows)]


def load_arc(n: int, seed: int = 20260929) -> list[Item]:
    import pyarrow.parquet as pq

    rows = pq.read_table(_download("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet")).to_pylist()
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    out = []
    for r in rows:
        labels = list(r["choices"]["label"])
        out.append(Item(f"arc-{r['id']}", _choices(r["question"], list(r["choices"]["text"])), LETTERS[labels.index(r["answerKey"])]))
    return out


def load_math500(n: int, seed: int = 20260929) -> list[Item]:
    rows = [json.loads(line) for line in _download("HuggingFaceH4/MATH-500", "test.jsonl").read_text().splitlines() if line.strip()]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    return [Item(f"math500-{r['unique_id']}", r["problem"] + MATH_SUFFIX, r["answer"]) for r in rows]


def _number(text: str) -> str | None:
    tail = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    return nums[-1 if "####" not in text else 0].replace(",", "").rstrip(".")


def answer_part(text: str) -> str:
    marks = list(re.finditer(r"(?:to=(\w+))?\s*<\|message\|>", text))
    if marks:
        final = [m for m in marks if m.group(1) != "self"]
        if not final:
            return ""
        last = final[-1]
        nxt = next((m.start() for m in marks if m.start() > last.end()), len(text))
        return re.sub(r"<\|(eot|eom|end|start)\|>.*$", "", text[last.end():nxt], flags=re.S).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
    elif text.lstrip().startswith("<|channel>"):
        return ""
    return text.strip()


def score_gsm8k(item: Item, text: str) -> float:
    got = _number(answer_part(text))
    try:
        return float(got is not None and abs(float(got) - float(item.gold)) < 1e-6)
    except ValueError:
        return 0.0


def score_ifeval(item: Item, text: str) -> float:
    from lm_eval.tasks.ifeval import utils

    return float(utils.process_results(item.gold, [answer_part(text)])["prompt_level_strict_acc"])


def _letter(text: str) -> str | None:
    found = re.findall(r"answer\s*(?:is)?\s*[:\-]?\s*\**\(?([A-H])\b", text, flags=re.I)
    if found:
        return found[-1].upper()
    bare = re.fullmatch(r"\W*([A-H])\W*", text.strip(), flags=re.I)
    return bare.group(1).upper() if bare else None


def score_choice(item: Item, text: str) -> float:
    return float(_letter(answer_part(text)) == item.gold)


def _boxed(text: str) -> str | None:
    at = text.rfind("\\boxed")
    if at < 0:
        return None
    start = text.find("{", at)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1:i]
    return None


def _normalize_math(answer: str) -> str:
    a = answer.strip()
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\(?:left|right|!|,|;|:| )", "", a)
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("$", "")
    a = re.sub(r"\s+", "", a).rstrip(".")
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def score_math(item: Item, text: str) -> float:
    got = _boxed(answer_part(text))
    return float(got is not None and _normalize_math(got) == _normalize_math(str(item.gold)))


TASKS = {
    "gsm8k": (load_gsm8k, score_gsm8k, 512),
    "ifeval": (load_ifeval, score_ifeval, 1024),
    "mmlu": (load_mmlu, score_choice, 512),
    "arc": (load_arc, score_choice, 512),
    "math500": (load_math500, score_math, 1024),
}


def wilson(correct: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
