
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from pie_evals.schema import EngineName, ErrorClass, WorkloadSpec

from .base import Engine, EngineLaunchError, register
from .shape import max_model_len_for, workload_concurrency

SCRIPT = Path(__file__).resolve().parent / "scripts" / "ollama_bench.py"


def ollama_bin() -> str:
    return os.environ.get("OLLAMA_BIN") or shutil.which("ollama") or "/usr/local/bin/ollama"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}


def server_version(binary: str | None = None) -> str:
    try:
        out = subprocess.run([binary or ollama_bin(), "--version"], capture_output=True, text=True, timeout=30, check=False)
    except OSError:
        return "unknown"
    found = re.findall(r"(\d+\.\d+\.\d+(?:-\S+)?)", (out.stdout or "") + (out.stderr or ""))
    return found[-1] if found else "unknown"


def start_server(env_extra: dict[str, str], log_path: Path | None, timeout_s: float) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, **env_extra, "OLLAMA_HOST": f"127.0.0.1:{port}"}
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen([ollama_bin(), "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise EngineLaunchError(ErrorClass.LOAD_FAIL, f"ollama serve exited during startup (see {log_path})")
        try:
            _get(f"{url}/api/version", timeout=2)
            return proc, url
        except OSError:
            time.sleep(0.3)
    proc.terminate()
    raise EngineLaunchError(ErrorClass.HANG, f"ollama serve did not answer within {timeout_s:.0f}s")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def has_model(url: str, tag: str) -> bool:
    try:
        _post(f"{url}/api/show", {"model": tag}, timeout=30)
        return True
    except OSError:
        return False


def ensure_ollama_model(tag: str, *, pull: bool, url: str | None = None, log=print) -> None:
    proc = None
    if url is None:
        proc, url = start_server({}, None, 60)
    try:
        if has_model(url, tag):
            return
        if not pull:
            raise EngineLaunchError(ErrorClass.LOAD_FAIL, f"ollama model {tag} not found in the store (run `pie-evals-node prepare`, or `run --download`)")
        log(f"ollama: pulling {tag}")
        done = _post(f"{url}/api/pull", {"model": tag, "stream": True}, timeout=7200)
        if done.get("status") != "success" and not has_model(url, tag):
            raise EngineLaunchError(ErrorClass.LOAD_FAIL, f"ollama pull {tag} failed: {done}")
    finally:
        stop_server(proc)


@register
class OllamaEngine(Engine):
    name = EngineName.OLLAMA
    script = "ollama_bench.py"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.slots = 1
        existing = self.env.get("PYTHONPATH") or os.environ.get("PYTHONPATH")
        self.env["PYTHONPATH"] = ":".join([str(self.pie_root / "scripts" / "bench")] + ([existing] if existing else []))

    def script_path(self) -> Path:
        return SCRIPT

    def default_python(self) -> str:
        if os.environ.get("OLLAMA_BENCH_PY"):
            return os.environ["OLLAMA_BENCH_PY"]
        from pie_evals.node.build import bench_python

        return str(bench_python())

    @property
    def tag(self) -> str:
        if not self.artifact.ollama_tag:
            raise EngineLaunchError(ErrorClass.LOAD_FAIL, f"artifact {self.artifact.id} has no ollama_tag")
        return self.artifact.ollama_tag

    def model_arg(self) -> str:
        return str(self.recipe.get("snapshot_dir") or self.artifact.base_model)

    def server_env(self, workload: WorkloadSpec) -> dict[str, str]:
        r = self.recipe
        env = {"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NOHISTORY": "1"}
        for knob, var in (("num_parallel", "OLLAMA_NUM_PARALLEL"), ("context_length", "OLLAMA_CONTEXT_LENGTH"),
                          ("flash_attention", "OLLAMA_FLASH_ATTENTION"), ("kv_cache_type", "OLLAMA_KV_CACHE_TYPE")):
            v = r.get(knob)
            if v is not None:
                env[var] = ("1" if v else "0") if isinstance(v, bool) else str(v)
        return env

    def engine_args(self, workload: WorkloadSpec) -> list[str]:
        r = self.recipe
        args = ["--ollama-model", self.tag, "--url", self.server_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")]
        if r.get("keep_alive"):
            args += ["--keep-alive", str(r["keep_alive"])]
        if r.get("options"):
            args += ["--options", json.dumps(r["options"], sort_keys=True)]
        if r.get("flush_cache") is False:
            args.append("--no-flush-cache")
        else:
            args += ["--flush-slots", str(self.slots)]
        if r.get("request_timeout"):
            args += ["--request-timeout", str(r["request_timeout"])]
        if workload.params.get("seed") is not None:
            args += ["--seed", str(workload.params["seed"])]
        return args

    def serve(self, workload: WorkloadSpec, log_path: Path, timeout_s: int) -> None:
        if self._server_proc is not None:
            return
        self.recipe = {**self.recipe}
        if self.recipe.get("num_parallel") == "$serve_concurrency":
            self.recipe["num_parallel"] = workload_concurrency(workload)
        if self.recipe.get("context_length") == "$serve_context":
            self.recipe["context_length"] = max_model_len_for(workload)
        self.slots = int(self.recipe.get("num_parallel") or 1)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc, url = start_server(self.server_env(workload), log_path, 60)
        self._server_proc, self.server_url = proc, url
        try:
            if not has_model(url, self.tag):
                raise EngineLaunchError(ErrorClass.LOAD_FAIL, f"ollama model {self.tag} not found in the store (run `pie-evals-node prepare`)")
            t0 = time.monotonic()
            _post(f"{url}/api/generate", {"model": self.tag, "prompt": "", "keep_alive": -1}, timeout=timeout_s)
            self.load_s = time.monotonic() - t0
            ps = _get(f"{url}/api/ps").get("models", [])
            m = next((m for m in ps if m.get("name") == self.tag or m.get("model") == self.tag), {})
            self.loaded = {"size": m.get("size"), "size_vram": m.get("size_vram"), "context_length": m.get("context_length")}
        except EngineLaunchError:
            self.stop()
            raise
        except OSError as e:
            self.stop()
            tail = log_path.read_text(errors="replace").strip().splitlines()[-5:] if log_path.exists() else []
            raise EngineLaunchError(ErrorClass.LOAD_FAIL, f"ollama could not load {self.tag}: {e}; " + " | ".join(tail)[:400]) from e

    def stop(self) -> None:
        stop_server(self._server_proc)
        self._server_proc = None
        self.server_url = None

    def resident_gib(self) -> float | None:
        size = (getattr(self, "loaded", None) or {}).get("size_vram")
        return round(size / 2**30, 2) if size else None

    def counters_from_summary(self, summary: dict[str, Any]) -> dict[str, float]:
        out = {}
        if getattr(self, "load_s", None) is not None:
            out["server_load_s"] = float(self.load_s)
        ctx = (getattr(self, "loaded", None) or {}).get("context_length")
        if ctx:
            out["server_context_length"] = float(ctx)
        return out

    def build_argv(self, workload: WorkloadSpec, common_args: list[str], json_out: Path) -> list[str]:
        return [a for a in super().build_argv(workload, common_args, json_out) if a not in ("--ignore-eos",)] + ["--no-ignore-eos"]

    def version(self) -> str:
        return server_version()

    def leftover_process_names(self) -> list[str]:
        return []
