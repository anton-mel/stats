"""Engine configuration recipes.

``recipes/<engine>.yaml`` carries named knob sets: pie runs ``default``, a
baseline (Ollama) runs ``competitive``::

    default:
      knobs:
        num_parallel:
          value: null
          rationale: why this is needed for a fair comparison
    competitive:
      extends: default
      knobs: {...}
      family:          # per artifact.family overrides (qwen3_6, gemma4 ...)
        qwen3_6: {knobs: {...}}
      arch:            # per platform.arch overrides (apple9, apple10 ...)
        apple10: {knobs: {...}}
      backend:         # per platform.backend overrides (metal)
        metal: {knobs: {...}}
      workload_kind:   # per workload.kind overrides (prefix_shared, mixed_length ...)
        prefix_shared: {knobs: {...}}

A knob may be written as ``{value: X, rationale: "..."}`` or as the bare
value. Values that depend on the workload are written as ``"$concurrency"``,
``"$num_requests"`` or ``"$max_model_len"`` and resolved here, so the yaml can
say "slots = concurrency" without knowing the shape. ``"$serve_concurrency"``
and ``"$serve_context"`` are left as written: they depend on every shape the
process's one server must hold, and the adapter resolves them when it boots.

``load_recipe`` returns the flat ``{knob: value}`` mapping the adapters read,
plus two bookkeeping keys the adapters ignore: ``_recipe`` (the section name)
and ``_rationale`` (``{knob: rationale}``) for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pie_evals.schema import PlatformSpec, WorkloadSpec

from ..shape import max_model_len_for, workload_concurrency, workload_num_requests

RECIPE_DIR = Path(__file__).resolve().parent

# family first, so an arch override (e.g. FA3 on Hopper) still wins over a family default
_OVERRIDE_AXES = ("family", "arch", "backend", "workload_kind")


def recipe_path(engine: str) -> Path:
    return RECIPE_DIR / f"{engine}.yaml"


def _load_yaml(engine: str) -> dict[str, Any]:
    path = recipe_path(engine)
    if not path.is_file():
        raise FileNotFoundError(f"no recipe file for engine {engine!r}: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping of recipe names")
    return data


def _split_knob(raw: Any) -> tuple[Any, str]:
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"], str(raw.get("rationale", ""))
    return raw, ""


def _merge_knobs(into: dict[str, Any], rationale: dict[str, str], knobs: dict[str, Any] | None) -> None:
    for k, raw in (knobs or {}).items():
        v, why = _split_knob(raw)
        into[k] = v
        if why:
            rationale[k] = why


def _section_chain(data: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """The section and everything it ``extends``, base first."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur: str | None = name
    while cur:
        if cur in seen:
            raise ValueError(f"recipe {name!r}: cyclic extends through {cur!r}")
        seen.add(cur)
        section = data.get(cur)
        if section is None:
            raise KeyError(f"recipe section {cur!r} not found (have {sorted(data)})")
        chain.append(section or {})
        cur = (section or {}).get("extends")
    chain.reverse()
    return chain


def _resolve_value(v: Any, subs: dict[str, Any]) -> Any:
    if isinstance(v, str) and v.startswith("$"):
        key = v[1:]
        if key not in subs:
            raise KeyError(f"unknown recipe substitution {v!r} (have {sorted('$' + s for s in subs)})")
        return subs[key]
    if isinstance(v, list):
        return [_resolve_value(x, subs) for x in v]
    if isinstance(v, dict):
        return {k: _resolve_value(x, subs) for k, x in v.items()}
    return v


def load_recipe(engine: str, name: str, platform: PlatformSpec, workload: WorkloadSpec, family: str | None = None) -> dict[str, Any]:
    """Flat ``{knob: value}`` for ``recipes/<engine>.yaml`` section ``name``,
    with per-family / per-arch / per-backend / per-workload-kind overrides
    applied and ``$concurrency``-style values resolved against ``workload``."""
    data = _load_yaml(engine)
    knobs: dict[str, Any] = {}
    rationale: dict[str, str] = {}
    selectors = {
        "family": family or "",
        "arch": platform.arch,
        "backend": str(platform.backend),
        "workload_kind": str(workload.kind),
    }
    for section in _section_chain(data, name):
        _merge_knobs(knobs, rationale, section.get("knobs"))
        for axis in _OVERRIDE_AXES:
            override = (section.get(axis) or {}).get(selectors[axis])
            if override:
                _merge_knobs(knobs, rationale, override.get("knobs", override))
                # one level of nesting: an arch override may except a family (FA3 on Hopper, but not for MLA)
                for inner in _OVERRIDE_AXES:
                    exc = (override.get(inner) or {}).get(selectors[inner]) if inner != axis else None
                    if exc:
                        _merge_knobs(knobs, rationale, exc.get("knobs", exc))
    subs = {
        "concurrency": workload_concurrency(workload),
        "num_requests": workload_num_requests(workload),
        "max_model_len": max_model_len_for(workload),
        "serve_concurrency": "$serve_concurrency",
        "serve_context": "$serve_context",
    }
    out = {k: _resolve_value(v, subs) for k, v in knobs.items()}
    out["_recipe"] = name
    out["_rationale"] = rationale
    return out


def recipe_names(engine: str) -> list[str]:
    return sorted(_load_yaml(engine))


__all__ = ["RECIPE_DIR", "load_recipe", "recipe_names", "recipe_path"]
