from __future__ import annotations

import json
from functools import cache
from typing import Any


@cache
def model_config(repo: str) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(repo, "config.json")) as f:
            config = json.load(f)
    except Exception:
        return None
    return config.get("text_config", config)


def _step_cost(c: dict[str, Any], context: float) -> dict[str, float]:
    h = c["hidden_size"]
    n_layers = c["num_hidden_layers"]
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", heads)
    hd = c.get("head_dim") or h // heads
    types = c.get("layer_types") or ["full_attention"] * n_layers

    q_width = heads * hd * (2 if c.get("attn_output_gate") else 1)
    full_proj = h * q_width + 2 * h * kv_heads * hd + heads * hd * h
    full_attn = 2 * heads * hd * context

    linear_proj = linear_state = 0
    if "linear_attention" in types:
        nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = 2 * nk * dk + nv * dv
        linear_proj = h * qkv + h * nv * dv + 2 * h * nv + nv * dv * h + c.get("linear_conv_kernel_dim", 4) * qkv
        linear_state = 4 * nv * dk * dv

    if c.get("num_experts_per_tok"):
        mlp = (c["num_experts_per_tok"] * 3 * h * c["moe_intermediate_size"]
               + 3 * h * c.get("shared_expert_intermediate_size", 0)
               + h * c.get("num_experts", 0))
    else:
        mlp = 3 * h * c["intermediate_size"]

    n_full = sum(t == "full_attention" for t in types)
    n_linear = len(types) - n_full
    return {
        "linear": n_full * full_proj + n_linear * (linear_proj + linear_state) + n_layers * mlp,
        "attention": n_full * full_attn,
        "head": h * c["vocab_size"],
    }


def prefill_flops(c: dict[str, Any], tokens: int) -> float:
    per_position = _step_cost(c, 1)["attention"]
    fixed = _step_cost(c, 0)
    return 2 * (tokens * fixed["linear"] + per_position * tokens * (tokens + 1) / 2 + fixed["head"])


def decode_flops_per_token(c: dict[str, Any], context: float) -> float:
    cost = _step_cost(c, context)
    return 2 * (cost["linear"] + cost["attention"] + cost["head"])


def tflops(row: dict[str, Any], params: dict[str, Any], config: dict[str, Any] | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {"prefill_tflops": None, "decode_tflops": None}
    if not config:
        return out
    try:
        prefill = int(params.get("prefill") or params.get("shared_prefix") or 0)
        decode = int(params.get("decode") or 0)
        single = int(params.get("concurrency") or 1) == 1
        if single and row.get("prefill_tok_s") and prefill:
            out["prefill_tflops"] = row["prefill_tok_s"] * prefill_flops(config, prefill) / prefill / 1e12
        rate = row.get("decode_tok_s") if single else row.get("output_tok_s")
        if rate:
            out["decode_tflops"] = rate * decode_flops_per_token(config, prefill + decode / 2) / 1e12
    except (KeyError, TypeError, ZeroDivisionError):
        return {"prefill_tflops": None, "decode_tflops": None}
    return out
