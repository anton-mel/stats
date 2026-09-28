"""Build pie once per (commit, features) and cache the artifacts under
``$PIE_EVALS_CACHE/builds``, so a Mac that measures a commit again (another
model, a re-run) never rebuilds it.

Artifacts per build: the ``pie`` CLI binary (Metal) and the guest programs
(wasm). ``restore`` puts them where the bench scripts look.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

PIE_UPSTREAM = "https://github.com/pie-project/pie.git"  # public; the runner needs no SSH key
BENCH_DEPS = ["websockets", "msgpack", "blake3", "cryptography", "numpy", "transformers", "jinja2"]  # pie_client + scripts/bench/common.py (pie_client rides on PYTHONPATH); transformers renders the Ollama arm's chat template


DEFAULT_CACHE = Path.home() / ".cache/pie-stats"
DEFAULT_PIE_ROOT = Path.home() / "pie-stats"


def _cache_root(root: Path | None) -> Path:
    return root or Path(os.environ.get("PIE_EVALS_CACHE", DEFAULT_CACHE))


def bench_python(cache_root: Path | None = None, log=print) -> Path:
    """A venv on the cache that runs pie's bench scripts (and the Ollama
    client, which shares their prompt construction): the client deps plus
    transformers for the chat template. Created with uv when present, else
    python3 -m venv."""
    venv = _cache_root(cache_root) / "venv-pie"
    py = venv / "bin" / "python"
    if not py.exists():
        log(f"venv: creating {venv}")
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", "--quiet", "--python", "3.12", str(venv)], check=True)
        else:
            subprocess.run(["python3", "-m", "venv", str(venv)], check=True)
        _pip_install(py, BENCH_DEPS)
    return py


def _pip_install(py: Path, args: list[str]) -> None:
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(py), *args], check=True)
    else:
        subprocess.run([str(py), "-m", "pip", "install", "-q", *args], check=True)


def cache_key(commit: str, features: list[str]) -> str:
    return f"{commit[:12]}-{'+'.join(features) or 'none'}"


def cache_dir(commit: str, features: list[str], root: Path | None = None) -> Path:
    return _cache_root(root) / "builds" / cache_key(commit, features)


def is_cached(commit: str, features: list[str], root: Path | None = None) -> bool:
    d = cache_dir(commit, features, root)
    return (d / "pie").exists() and (d / "wasm").is_dir() and any((d / "wasm").glob("*.wasm"))


def _git(args: list[str], cwd: Path | None = None, retries: int = 3, timeout_s: int = 1800) -> None:
    """git with HTTP/1.1 (some links cancel HTTP/2 streams on long transfers) and retries."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(["git", "-c", "http.version=HTTP/1.1", "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=120", *args], cwd=cwd, check=True, timeout=timeout_s)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            last = e
            if attempt < retries:
                time.sleep(15 * attempt)
    assert last is not None
    raise last


def ensure_checkout(pie_root: Path, commit: str, mirror: Path | None = None, upstream: str = PIE_UPSTREAM) -> None:
    """A checkout at ``pie_root`` pinned to ``commit``.

    With a bare mirror it is the clone source (fast, no network). Without one
    only the pinned commit is fetched (shallow)."""
    if not (pie_root / ".git").exists():
        if mirror and mirror.exists():
            _git(["clone", "--quiet", str(mirror), str(pie_root)])
            subprocess.run(["git", "-C", str(pie_root), "remote", "set-url", "origin", upstream], check=False)
        else:
            pie_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "--quiet", str(pie_root)], check=True)
            subprocess.run(["git", "-C", str(pie_root), "remote", "add", "origin", upstream], check=True)
            _git(["-C", str(pie_root), "fetch", "--quiet", "--depth", "1", "origin", commit])
            subprocess.run(["git", "-C", str(pie_root), "checkout", "--quiet", "FETCH_HEAD"], check=True)
            return
    head = subprocess.run(["git", "-C", str(pie_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not head.startswith(commit) and not commit.startswith(head):
        r = subprocess.run(["git", "-C", str(pie_root), "checkout", "--quiet", "--force", commit], capture_output=True, text=True)
        if r.returncode != 0:
            _git(["-C", str(pie_root), "fetch", "--quiet", "--depth", "1", "origin", commit])
            subprocess.run(["git", "-C", str(pie_root), "checkout", "--quiet", "--force", "FETCH_HEAD"], check=True)


def update_mirror(mirror: Path, upstream: str = PIE_UPSTREAM) -> None:
    """Maintain a bare mirror of pie (clone it the first time)."""
    if not mirror.exists():
        _git(["clone", "--quiet", "--mirror", upstream, str(mirror)], timeout_s=3600)
    else:
        subprocess.run(["git", "-C", str(mirror), "fetch", "--quiet", "--prune"], check=False, timeout=1800)


def build(pie_root: Path, commit: str, features: list[str], *, cache_root: Path | None = None, target_dir: Path | None = None, python: str = "python3", timeout_s: int = 3600, log=print) -> Path:
    """Build the binary and the guest programs and store them under the cache dir."""
    out = cache_dir(commit, features, cache_root)
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    if target_dir:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    feats = ",".join(features)
    log(f"build: pie {commit[:12]} features={feats} target={env.get('CARGO_TARGET_DIR', 'in-tree')}")
    subprocess.run(["cargo", "build", "--release", "-p", "pie", "--bin", "pie", "--features", feats], cwd=pie_root, env=env, check=True, timeout=timeout_s)
    tdir = Path(env.get("CARGO_TARGET_DIR", pie_root / "target"))
    shutil.copy2(tdir / "release" / "pie", out / "pie")
    # guest programs: the whole examples/ workspace (every inferlet the matrix can name), built
    # in-tree because bench_inferlet_paths walks up to examples/target; all .wasm are cached
    log("build: every inferlet in examples/ (wasm32-wasip2)")
    genv = {k: v for k, v in env.items() if k != "CARGO_TARGET_DIR"}
    subprocess.run(["cargo", "build", "--release", "--target", "wasm32-wasip2", "--workspace"], cwd=pie_root / "examples", env=genv, check=True, timeout=timeout_s)
    (out / "wasm").mkdir(exist_ok=True)
    n = 0
    for w in (pie_root / "examples/target/wasm32-wasip2/release").glob("*.wasm"):
        shutil.copy2(w, out / "wasm" / w.name)
        n += 1
    log(f"build: {n} guest programs cached")
    (out / "COMMIT").write_text(commit + "\n")
    log(f"build: cached at {out}")
    return out


def restore(pie_root: Path, commit: str, features: list[str], *, cache_root: Path | None = None, python: str = "python3", set_env: bool = True, log=print) -> bool:
    """Put cached artifacts where the bench scripts look. Returns False on a cache miss."""
    d = cache_dir(commit, features, cache_root)
    if not is_cached(commit, features, cache_root):
        return False
    binary = pie_root / "target/release/pie"
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d / "pie", binary)
    wdir = pie_root / "examples/target/wasm32-wasip2/release"
    wdir.mkdir(parents=True, exist_ok=True)
    for w in (d / "wasm").glob("*.wasm"):
        shutil.copy(w, wdir / w.name)
        os.utime(wdir / w.name, None)  # newer than the freshly cloned sources (bench_inferlet_paths' staleness check)
    if set_env:
        os.environ.setdefault("PIE_PY", str(bench_python(cache_root, log=log)))  # the pie adapter's interpreter
    log(f"restore: {d} -> {pie_root}")
    return True


def ensure(pie_root: Path, commit: str, features: list[str], *, cache_root: Path | None = None, target_dir: Path | None = None, python: str = "python3", log=print) -> None:
    if restore(pie_root, commit, features, cache_root=cache_root, python=python, log=log):
        return
    build(pie_root, commit, features, cache_root=cache_root, target_dir=target_dir, python=python, log=log)
    restore(pie_root, commit, features, cache_root=cache_root, python=python, log=log)
