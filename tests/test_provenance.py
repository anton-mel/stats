import json
import subprocess

from pie_evals.node.provenance import (
    build_provenance,
    checkpoint_revision,
    git_commit,
    quant_block,
    quant_block_hash,
    relevant_env,
)
from pie_evals.schema import Provenance


def _write_cfg(d, cfg):
    (d / "config.json").write_text(json.dumps(cfg))


MLX_CFG = {
    "model_type": "deepseek_v4",
    "quantization": {
        "bits": 2,
        "group_size": 64,
        "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 64},
        "model.layers.3.mlp.experts.gate_proj": {"bits": 4, "group_size": 32},
        "lm_head": {"bits": 8, "group_size": 64},
    },
}


def test_quant_block_hash_covers_per_tensor_overrides(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_cfg(a, MLX_CFG)
    cfg_b = json.loads(json.dumps(MLX_CFG))
    cfg_b["quantization"]["model.layers.3.mlp.experts.gate_proj"]["bits"] = 2
    _write_cfg(b, cfg_b)
    ha, hb = quant_block_hash(a), quant_block_hash(b)
    assert ha and hb and len(ha) == 64
    assert ha != hb
    assert quant_block(a)["bits"] == 2 and quant_block(b)["bits"] == 2


def test_quant_block_hash_is_order_independent_and_stable(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_cfg(a, MLX_CFG)
    reordered = {"quantization": dict(reversed(list(MLX_CFG["quantization"].items()))), "model_type": "x"}
    _write_cfg(b, reordered)
    assert quant_block_hash(a) == quant_block_hash(b)


def test_quant_block_hash_hf_and_missing(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    _write_cfg(hf, {"quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                            "ignored_layers": ["lm_head"]}})
    h = quant_block_hash(hf)
    assert h
    plain = tmp_path / "plain"
    plain.mkdir()
    _write_cfg(plain, {"model_type": "qwen3", "torch_dtype": "bfloat16"})
    assert quant_block_hash(plain) is None
    assert quant_block_hash(tmp_path / "nope") is None
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_cfg(nested, {"text_config": {"quantization_config": {"quant_method": "mxfp4"}}})
    assert quant_block_hash(nested)


def test_checkpoint_revision(tmp_path):
    snap = tmp_path / "models--Qwen--Qwen3-8B" / "snapshots" / "abc123def"
    snap.mkdir(parents=True)
    assert checkpoint_revision(snap) == "abc123def"
    mini = tmp_path / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots" / "mini-l0-7-e32"
    mini.mkdir(parents=True)
    assert checkpoint_revision(mini) == "mini-l0-7-e32"


def test_relevant_env():
    env = {
        "PIE_HOME": "/x", "OLLAMA_NUM_PARALLEL": "8", "HF_HUB_CACHE": "/hf", "MTL_DEBUG_LAYER": "0",
        "CUDA_VISIBLE_DEVICES": "0,1", "HOME": "/Users/x", "PATH": "/bin", "SECRET_TOKEN": "nope",
    }
    out = relevant_env(env)
    assert set(out) == {"PIE_HOME", "OLLAMA_NUM_PARALLEL", "HF_HUB_CACHE", "MTL_DEBUG_LAYER"}
    assert list(out) == sorted(out)


def test_git_commit(tmp_path):
    assert git_commit(tmp_path / "missing") is None
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "f").write_text("1")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "x"], check=True, env=env)
    sha = git_commit(repo)
    assert sha and len(sha) == 40
    (repo / "f").write_text("2")
    assert git_commit(repo) == sha + "-dirty"


def test_build_provenance_fills_record_fields(tmp_path):
    snap = tmp_path / "snapshots" / "rev0"
    snap.mkdir(parents=True)
    _write_cfg(snap, MLX_CFG)
    pv = build_provenance(
        pie_root=tmp_path / "no-pie", snapshot_dir=snap, engine_version="pie@deadbeef",
        engine_config={"tp": 1}, recipe_name="default", runner="m4-pro-48g",
        machine_state={"thermal_warning": "no"}, harness_path=tmp_path / "no-harness",
    )
    assert isinstance(pv, Provenance)
    assert pv.checkpoint_revision == "rev0"
    assert pv.quant_block_hash == quant_block_hash(snap)
    assert pv.engine_version == "pie@deadbeef"
    assert pv.engine_config == {"tp": 1}
    assert pv.engine_config_recipe == "default"
    assert pv.runner == "m4-pro-48g"
    assert pv.hardware_fingerprint.get("os") == "macos"
    assert pv.hardware_fingerprint.get("hostname")
    assert pv.os_version
    assert pv.machine_state == {"thermal_warning": "no"}
    assert pv.pie_commit is None
