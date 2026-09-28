from pathlib import Path

from pie_evals.node import importer as im
from pie_evals.schema import ArtifactSpec, QuantScheme, SourceFormat


def _art(fmt=SourceFormat.MLX, sku=None):
    return ArtifactSpec(id="qwen36-mini", base_model="mlx-community/Qwen3.6-35B-A3B-4bit", family="qwen3_6_moe", scheme=QuantScheme.AFFINE_U4_G64, source_format=fmt, pie_sku=sku)


def test_needs_import_only_for_mlx():
    assert im.needs_import(_art())
    assert not im.needs_import(_art(fmt=SourceFormat.HF_SAFETENSORS))


def test_ensure_artifact_runs_import_once(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        (out / "qwen36.zt").write_bytes(b"zt")
        return type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(im.subprocess, "run", fake_run)
    zt = im.ensure_artifact(_art(sku="qwen36-35b-a3b-mini-u4g64-kv-bf16"), tmp_path / "snap", Path("/pie/target/release/pie"), "e06ff359377b", root=tmp_path / "cache", log=lambda *_: None)
    assert zt.name == "qwen36.zt" and zt.parent.name == "qwen36-mini-e06ff359"
    assert calls[0][:3] == ["/pie/target/release/pie", "model", "import"] and "--sku" in calls[0]
    zt2 = im.ensure_artifact(_art(), tmp_path / "snap", Path("/pie/target/release/pie"), "e06ff359377b", root=tmp_path / "cache", log=lambda *_: None)
    assert zt2 == zt and len(calls) == 1
