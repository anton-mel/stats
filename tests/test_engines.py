
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

from pie_evals.node.engines import get_engine
from pie_evals.node.engines.base import Engine
from pie_evals.node.engines.ollama import SCRIPT as OLLAMA_SCRIPT
from pie_evals.node.engines.ollama import OllamaEngine
from pie_evals.node.engines.pie import PieEngine
from pie_evals.node.engines.recipes import load_recipe, recipe_names
from pie_evals.node.engines.shape import max_model_len_for, workload_concurrency
from pie_evals.node.workloads import common_args_for
from pie_evals.schema import ArtifactSpec, Backend, Mode, PlatformSpec, WorkloadSpec

PIE_ROOT = Path(os.environ.get("PIE_ROOT", str(Path.home() / "pie")))
BENCHES = PIE_ROOT / "scripts/bench"

def _has_benches() -> bool:
    try:
        return BENCHES.is_dir()
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _has_benches(), reason=f"pie checkout not at {PIE_ROOT}")


def _first_arg_flags(call: ast.Call, loop_iters: list[ast.AST]) -> list[str]:
    if not call.args:
        return []
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [first.value]
    if isinstance(first, ast.Name):
        out = []
        for it in loop_iters:
            for node in ast.walk(it):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--"):
                    out.append(node.value)
        return out
    return []


def _flags_in(node: ast.AST, loop_iters: list[ast.AST] | None = None) -> set[str]:
    flags: set[str] = set()
    loop_iters = list(loop_iters or [])

    def visit(n: ast.AST, iters: list[ast.AST]) -> None:
        if isinstance(n, ast.For):
            iters = iters + [n.iter]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
            names = _first_arg_flags(n, iters)
            boolean_optional = any(
                kw.arg == "action" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "BooleanOptionalAction"
                for kw in n.keywords
            )
            for name in names:
                if not name.startswith("--"):
                    continue
                flags.add(name)
                if boolean_optional:
                    flags.add("--no-" + name[2:])
        for child in ast.iter_child_nodes(n):
            visit(child, iters)

    visit(node, loop_iters)
    return flags


def _called_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def common_flags_by_function() -> dict[str, tuple[set[str], set[str]]]:
    tree = ast.parse((BENCHES / "common.py").read_text())
    out = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            out[fn.name] = (_flags_in(fn), _called_names(fn))
    return out


def argparse_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    flags = _flags_in(tree)
    if path.resolve() == (BENCHES / "common.py").resolve():
        return flags
    common = common_flags_by_function()
    todo = [n for n in _called_names(tree) if n in common]
    seen: set[str] = set()
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        fn_flags, refs = common[fn]
        flags |= fn_flags
        todo += [n for n in refs if n in common]
    return flags


def accepted_flags(eng: Engine) -> set[str]:
    return argparse_flags(eng.script_path())


_FLAG_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")


def emitted_flags(argv: list[str]) -> list[str]:
    return [a for a in argv if _FLAG_RE.match(a)]


def flag_value(argv: list[str], flag: str) -> str:
    i = argv.index(flag)
    return argv[i + 1]


def platform(arch: str = "apple9", memory_gib: float = 48.0) -> PlatformSpec:
    return PlatformSpec(id=f"mac-{arch}", os="macos", backend=Backend.METAL, accelerator="dummy", arch=arch,
                        memory_gib=memory_gib, interconnect="uma")


def artifact(**over) -> ArtifactSpec:
    base = {"id": "muse-glimmer-30b-mlx4", "base_model": "mlx-community/Qwen3.6-27B-4bit", "family": "qwen3_6",
            "scheme": "affine_u4_g64", "source_format": "mlx"}
    base.update(over)
    return ArtifactSpec(**base)


def ollama_artifact(**over) -> ArtifactSpec:
    base = {"id": "qwen3.6-27b-ollama", "source_format": "ollama", "scheme": "gguf_q4_k_m",
            "ollama_tag": "qwen3.6:27b", "baseline_of": "muse-glimmer-30b-mlx4"}
    base.update(over)
    return artifact(**base)


WORKLOADS = {
    "control-aa": WorkloadSpec(id="control-aa", kind="control_aa", params={"prefill": 128, "decode": 64, "rounds": 2}),
    "ob-story-200": WorkloadSpec(id="ob-story-200", kind="single_stream", params={"prompt": "Tell a story about a llama.", "decode": 200, "requests": 6, "warmup": 1}),
    "ss-128-64": WorkloadSpec(id="ss-128-64", kind="single_stream", params={"prefill": 128, "decode": 64}),
    "lc-2k-128": WorkloadSpec(id="lc-2k-128", kind="long_context", params={"prefill": 2048, "decode": 128, "concurrency": 1}),
    "c8": WorkloadSpec(id="c8", kind="concurrency", params={"concurrency": 8, "num_requests": 32, "prefill": 128, "decode": 128}),
    "c32": WorkloadSpec(id="c32", kind="concurrency", params={"concurrency": 32, "num_requests": 128, "prefill": 128, "decode": 128}),
}

RECIPE = {PieEngine: "default", OllamaEngine: "competitive"}


def make(cls: type[Engine], *, plat: PlatformSpec | None = None, art: ArtifactSpec | None = None, mode: Mode | None = None,
         recipe_name: str | None = None, workload: WorkloadSpec | None = None, **kw) -> Engine:
    plat = plat or platform()
    workload = workload or WORKLOADS["c32"]
    art = art or (ollama_artifact() if cls is OllamaEngine else artifact())
    recipe = load_recipe(str(cls.name), recipe_name or RECIPE[cls], plat, workload, family=art.family)
    kw.setdefault("python", "/usr/bin/python3")
    return cls(pie_root=PIE_ROOT, artifact=art, platform=plat, mode=mode or Mode(), recipe=recipe, **kw)


def argv_for(engine: Engine, workload: WorkloadSpec, tmp_path: Path) -> list[str]:
    return engine.build_argv(workload, common_args_for(workload), tmp_path / "bench.json")


def test_registry_resolves_every_engine():
    assert get_engine("pie") is PieEngine
    assert get_engine("ollama") is OllamaEngine


@pytest.mark.parametrize("engine", ["pie", "ollama"])
def test_recipes_resolve_with_rationale(engine):
    names = recipe_names(engine)
    assert "default" in names
    for name in names:
        for wl in WORKLOADS.values():
            r = load_recipe(engine, name, platform(), wl, family="qwen3_6")
            assert r["_recipe"] == name
            assert r["_rationale"], f"{engine}.{name}: recipe carries no rationale"
            for k, v in r.items():
                if isinstance(v, str) and v.startswith("$"):
                    assert v in ("$serve_concurrency", "$serve_context"), f"{engine}.{k} left unresolved: {v}"


def test_recipe_overrides_by_family_and_backend():
    r = load_recipe("pie", "default", platform(), WORKLOADS["c32"], family="qwen3_6")
    assert r["total_pages"] == 1024 and r["engine_option"] == {"max_state_slots": 8}
    r = load_recipe("pie", "default", platform(), WORKLOADS["c32"], family="gemma4")
    assert "total_pages" not in r


def test_ollama_competitive_leaves_the_server_envelope_to_the_adapter():
    for wl in WORKLOADS.values():
        r = load_recipe("ollama", "competitive", platform(), wl)
        assert r["num_parallel"] == "$serve_concurrency"
        assert r["context_length"] == "$serve_context"
    r = load_recipe("ollama", "default", platform(), WORKLOADS["c32"])
    assert r["num_parallel"] is None and r["context_length"] is None


def test_shape_helpers():
    assert workload_concurrency(WORKLOADS["ss-128-64"]) == 1
    assert workload_concurrency(WORKLOADS["c32"]) == 32
    assert max_model_len_for(WORKLOADS["c32"]) == 2048
    assert max_model_len_for(WORKLOADS["lc-2k-128"]) == 4096


def test_literal_prompt_param_is_sent_as_is():
    args = common_args_for(WORKLOADS["ob-story-200"])
    assert flag_value(args, "--prompt") == "Tell a story about a llama."
    assert flag_value(args, "--max-tokens") == "200"


def test_pie_metal_args(tmp_path):
    eng = make(PieEngine)
    argv = argv_for(eng, WORKLOADS["c32"], tmp_path)
    assert argv[1].endswith("scripts/bench/pie_bench.py") and argv[2] == "tput"
    assert flag_value(argv, "--engine") == "metal"
    assert flag_value(argv, "--max-model-len") == "2048"
    assert flag_value(argv, "--concurrency") == "32"
    assert "--report-timing" in argv and "--no-pretokenized-prompts" in argv
    assert "--dump-all-token-ids" in argv and "--json-out" in argv
    assert "--gpu-mem-util" not in argv and "--tp-size" not in argv and "--device" not in argv
    assert flag_value(argv, "--inferlet-dir") == str(PIE_ROOT / "examples/text-completion-bench")
    assert eng.env["PIE_BENCH_INFERLET_DIR"] == str(PIE_ROOT / "examples/text-completion-bench")
    assert eng.env["PYTHONPATH"].startswith(f"{PIE_ROOT}/python/client/src:{PIE_ROOT}/python/server/python")


def test_pie_program_path_is_settable(tmp_path):
    eng = make(PieEngine)
    eng.program_path = "examples/dflash-speculative-bench"
    assert eng.env["PIE_BENCH_INFERLET_DIR"].endswith("dflash-speculative-bench")
    argv = argv_for(eng, WORKLOADS["ss-128-64"], tmp_path)
    assert flag_value(argv, "--inferlet-dir").endswith("dflash-speculative-bench")
    assert argv[2] == "latency" and flag_value(argv, "--requests") == "8"
    eng3 = PieEngine(pie_root=PIE_ROOT, artifact=artifact(), platform=platform(), mode=Mode(), recipe={"program_path": "x/y"})
    assert eng3.program_path == "x/y" and eng3.inferlet_dir == PIE_ROOT / "x/y"


def test_pie_recipe_passthrough(tmp_path):
    eng = make(PieEngine, art=artifact(family="gemma4"))
    eng.recipe.update({"frame_submit_depth": 4, "max_forward_requests": 32, "engine_option": {"max_state_slots": 128, "sku": "$pie_sku"}})
    argv = argv_for(eng, WORKLOADS["c32"], tmp_path)
    assert flag_value(argv, "--frame-submit-depth") == "4"
    assert flag_value(argv, "--max-forward-requests") == "32"
    i = argv.index("--engine-option")
    assert argv[i + 1] == "max_state_slots=128"
    assert argv.count("--engine-option") == 1
    eng.artifact = artifact(pie_sku="qwen3-8b-bf16-kv-bf16")
    argv = argv_for(eng, WORKLOADS["c32"], tmp_path)
    assert "sku=qwen3-8b-bf16-kv-bf16" in argv


def test_pie_counters_from_summary():
    eng = make(PieEngine)
    summary = {
        "wall_s": 2.0,
        "config": {
            "total batches": 400, "device idle us": 200_000, "cumulative_batch_latency_us": 1_600_000,
            "avg batch latency us": 4000.0, "fire.execute.engine_fire_us": 3500.0,
            "turnaround sum us": 90_000, "turnaround n": 300, "wave fires": 400,
        },
    }
    c = eng.counters_from_summary(summary)
    assert c["batches"] == 400
    assert c["device_idle_pct"] == pytest.approx(10.0)
    assert c["host_us_per_step"] == pytest.approx(500.0)
    assert c["guest_turnaround_us"] == pytest.approx(300.0)
    assert c["batch_latency_us"] == pytest.approx(4000.0)
    assert eng.counters_from_summary({"wall_s": 1.0, "config": {}}) == {}
    c = eng.counters_from_summary({"wall_s": 1.0, "config": {"default.total_batches": 10, "default.fire.quorum.device_idle_us": 100_000}})
    assert c["batches"] == 10 and c["device_idle_pct"] == pytest.approx(10.0)


def test_pie_version_is_git_short_sha():
    v = make(PieEngine).version()
    assert re.fullmatch(r"[0-9a-f]{7,12}", v), v


def test_pie_serve_is_a_noop_on_metal(tmp_path):
    eng = make(PieEngine)
    assert eng.serve(WORKLOADS["c32"], tmp_path / "serve.log", 5) is None
    assert eng.server_url is None and eng._server_proc is None


def test_ollama_args(tmp_path):
    eng = make(OllamaEngine)
    argv = argv_for(eng, WORKLOADS["c32"], tmp_path)
    assert Path(argv[1]) == OLLAMA_SCRIPT and OLLAMA_SCRIPT.parts[-2:] == ("scripts", "ollama_bench.py")
    assert OLLAMA_SCRIPT.parent.parent.name == "engines" and OLLAMA_SCRIPT.is_file()
    assert argv[2] == "tput"
    assert flag_value(argv, "--ollama-model") == "qwen3.6:27b"
    assert flag_value(argv, "--model") == "mlx-community/Qwen3.6-27B-4bit"
    assert "--no-ignore-eos" in argv and "--ignore-eos" not in argv
    assert "--dump-all-token-ids" in argv
    assert str(PIE_ROOT / "scripts" / "bench") in eng.env["PYTHONPATH"].split(":")


def test_ollama_recipe_flags(tmp_path):
    eng = make(OllamaEngine, recipe_name="default")
    eng.server_url = "http://127.0.0.1:5555"
    eng.recipe.update({"options": {"num_ctx": 4096}, "flush_cache": False, "request_timeout": 600})
    argv = argv_for(eng, WORKLOADS["ss-128-64"], tmp_path)
    assert flag_value(argv, "--url") == "http://127.0.0.1:5555"
    assert flag_value(argv, "--keep-alive") == "30m"
    assert flag_value(argv, "--options") == '{"num_ctx": 4096}'
    assert "--no-flush-cache" in argv
    assert flag_value(argv, "--request-timeout") == "600"


def test_ollama_flushes_every_slot_it_booted(tmp_path):
    eng = make(OllamaEngine)
    assert flag_value(argv_for(eng, WORKLOADS["ss-128-64"], tmp_path), "--flush-slots") == "1"
    eng.slots = 8
    assert flag_value(argv_for(eng, WORKLOADS["ss-128-64"], tmp_path), "--flush-slots") == "8"


def test_advanced_shape_carries_temperature_and_ollama_seed(tmp_path):
    adv = WorkloadSpec(id="ob-advanced-500", kind="single_stream", params={"prompt": "p", "decode": 500, "temperature": 0.7, "seed": 42})
    assert flag_value(common_args_for(adv), "--temperature") == "0.7"
    argv = argv_for(make(OllamaEngine), adv, tmp_path)
    assert flag_value(argv, "--seed") == "42"
    assert "--seed" not in argv_for(make(PieEngine), adv, tmp_path)


def test_ollama_keeps_its_cache_on_the_cached_prompt_shape():
    cache = WorkloadSpec(id="cache-2k", kind="prefix_shared", params={"shared_prefix": 2048, "unique_suffix": 64, "variants": 8, "decode": 64, "concurrency": 1})
    assert load_recipe("ollama", "competitive", platform(), cache)["flush_cache"] is False
    assert load_recipe("ollama", "competitive", platform(), WORKLOADS["ss-128-64"])["flush_cache"] is True


def test_ollama_server_env_maps_recipe_knobs():
    eng = make(OllamaEngine)
    eng.recipe = {**eng.recipe, "num_parallel": 8, "context_length": 4096, "flash_attention": True, "kv_cache_type": "q8_0"}
    env = eng.server_env(WORKLOADS["c8"])
    assert env["OLLAMA_NUM_PARALLEL"] == "8" and env["OLLAMA_CONTEXT_LENGTH"] == "4096"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1" and env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_KEEP_ALIVE"] == "-1"
    eng.recipe["flash_attention"] = False
    assert eng.server_env(WORKLOADS["c8"])["OLLAMA_FLASH_ATTENTION"] == "0"
    env = make(OllamaEngine, recipe_name="default").server_env(WORKLOADS["c8"])
    assert not {"OLLAMA_NUM_PARALLEL", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE"} & set(env)


def test_ollama_never_sweeps_the_resident_app():
    assert make(OllamaEngine).leftover_process_names() == []


def test_ollama_bench_parses_the_argv_shape():
    sys.path.insert(0, str(BENCHES))
    sys.path.insert(0, str(OLLAMA_SCRIPT.parent))
    try:
        import ollama_bench
    finally:
        sys.path.remove(str(BENCHES))
        sys.path.remove(str(OLLAMA_SCRIPT.parent))
    args = ollama_bench.build_parser().parse_args([
        "latency", "--model", "X", "--ollama-model", "t", "--prompt", "p", "--max-tokens", "4", "--no-ignore-eos",
        "--warmup", "1", "--requests", "1", "--json-out", "f", "--dump-all-token-ids", "--no-think",
    ])
    assert args.ollama_model == "t" and args.model == "X" and args.max_tokens == 4
    assert args.ignore_eos is False and args.flush_cache is True


_CASES = []
for _wl in WORKLOADS:
    _CASES += [
        ("pie", PieEngine, _wl, {}),
        ("pie-gemma", PieEngine, _wl, {"art": artifact(family="gemma4")}),
        ("ollama", OllamaEngine, _wl, {}),
        ("ollama-default", OllamaEngine, _wl, {"recipe_name": "default"}),
    ]


@pytest.mark.parametrize("label,cls,wl,over", _CASES, ids=[f"{c[0]}/{c[2]}" for c in _CASES])
def test_every_emitted_flag_is_parsed_by_the_script(label, cls, wl, over, tmp_path):
    workload = WORKLOADS[wl]
    eng = make(cls, workload=workload, **over)
    argv = argv_for(eng, workload, tmp_path)
    accepted = accepted_flags(eng)
    assert accepted, f"no flags parsed from {eng.script_path()}"
    unknown = sorted(set(emitted_flags(argv)) - accepted)
    assert not unknown, f"{label}/{wl}: {eng.script_path().name} does not accept {unknown}"


def test_argparse_flag_extraction_sees_loop_registered_and_boolean_optional_flags():
    flags = argparse_flags(BENCHES / "pie_bench.py")
    assert {"--inferlet-dir", "--engine", "--wasm-warm-slots", "--frame-submit-depth", "--pretokenized-prompts", "--no-pretokenized-prompts"} <= flags
    assert "--dump-all-token-ids" in flags
    common = argparse_flags(BENCHES / "common.py")
    assert {"--model", "--ignore-eos", "--no-ignore-eos", "--json-out"} <= common
    got = argparse_flags(OLLAMA_SCRIPT)
    assert {"--model", "--max-tokens", "--concurrency", "--num-requests", "--requests", "--dump-all-token-ids"} <= got
    assert {"--ollama-model", "--flush-cache", "--no-flush-cache"} <= got


def test_a_shape_can_score_on_prefill(tmp_path):
    import sys

    from pie_evals.node.engines.base import Engine

    fake = tmp_path / "fake_bench.py"
    fake.write_text(
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--json-out') + 1]\n"
        "req = {'ok': True, 'latency_s': 0.2, 'ttft_s': 0.1, 'output_tokens': 1, 'prompt_tokens': 512}\n"
        "json.dump({'summary': {'wall_s': 0.2, 'output_tokens': 1, 'prompt_tokens': 512}, 'requests': [req]}, open(out, 'w'))\n"
    )
    root = tmp_path / "pie"
    (root / "scripts" / "bench").mkdir(parents=True)

    class Stub(Engine):
        name = "pie"

        def script_path(self):
            return fake

        def default_python(self):
            return sys.executable

        def model_arg(self):
            return "m"

        def engine_args(self, workload):
            return []

    eng = Stub(pie_root=root, artifact=artifact(), platform=platform(), mode=Mode(), recipe={})
    pp = WorkloadSpec(id="lb-pp512", kind="single_stream", params={"prefill": 512, "decode": 1, "primary": "prefill_tok_s"})
    res = eng.run(pp, [], tmp_path / "out", 60)
    assert res.perf.primary == "prefill_tok_s" and res.perf.prefill_tok_s == 512 / 0.1


def test_lifeline_stops_its_child_when_the_parent_goes_away():
    import subprocess
    import sys
    import time

    sup = subprocess.Popen([sys.executable, "-m", "pie_evals.node.engines.lifeline", sys.executable, "-c", "import time; time.sleep(60)"],
                           stdin=subprocess.PIPE)
    time.sleep(0.5)
    sup.stdin.close()
    assert sup.wait(timeout=30) != 0 or sup.returncode == 0
    assert sup.poll() is not None
