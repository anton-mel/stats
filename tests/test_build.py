
from pie_evals.node import build as pb


def test_cache_key_and_miss(tmp_path):
    assert pb.cache_key("fa26669562dc7829c0f89c212529bc7f606a5e5c", ["metal"]) == "fa26669562dc-metal"
    assert not pb.is_cached("fa26669562dc", ["metal"], tmp_path)


def test_restore_places_artifacts(tmp_path, monkeypatch):
    d = pb.cache_dir("abc123def456", ["metal"], tmp_path)
    d.mkdir(parents=True)
    (d / "pie").write_bytes(b"bin")
    (d / "wasm").mkdir()
    (d / "wasm" / "text_completion_bench.wasm").write_bytes(b"wasm")
    pie_root = tmp_path / "pie"
    monkeypatch.setenv("PIE_PY", "")
    monkeypatch.delenv("PIE_PY")
    monkeypatch.setattr(pb, "bench_python", lambda cache_root=None, log=print: tmp_path / "venv/bin/python")
    assert pb.restore(pie_root, "abc123def456", ["metal"], cache_root=tmp_path, log=lambda *_: None)
    assert (pie_root / "target/release/pie").read_bytes() == b"bin"
    assert (pie_root / "examples/target/wasm32-wasip2/release/text_completion_bench.wasm").exists()
    assert pb.os.environ["PIE_PY"] == str(tmp_path / "venv/bin/python")


def test_checkout_moves_past_a_build_dirtied_tree(tmp_path):
    import subprocess

    src = tmp_path / "src"
    subprocess.run(["git", "init", "-q", str(src)], check=True)

    def run(*a):
        return subprocess.run(["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True, text=True).stdout.strip()

    (src / "Cargo.lock").write_text("a\n")
    run("add", ".")
    run("commit", "-qm", "a")
    first = run("rev-parse", "HEAD")
    (src / "Cargo.lock").write_text("b\n")
    run("commit", "-qam", "b")
    second = run("rev-parse", "HEAD")
    run("checkout", "-q", first)
    (src / "Cargo.lock").write_text("rewritten by a build\n")
    pb.ensure_checkout(src, second)
    assert run("rev-parse", "HEAD") == second
    assert (src / "Cargo.lock").read_text() == "b\n"
