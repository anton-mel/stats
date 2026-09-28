from pathlib import Path

from pie_evals.node.snapshots import snapshot_dir_if_present
from pie_evals.schema.cell import ArtifactSpec


def _art(**over) -> ArtifactSpec:
    base = dict(id="qwen3.6-35b-a3b-mlx4", base_model="mlx-community/Qwen3.6-35B-A3B-4bit", family="qwen3_6", scheme="affine_u4_g64", source_format="mlx")
    base.update(over)
    return ArtifactSpec(**base)


def test_a_pinned_revision_wins_over_the_newest_snapshot(tmp_path: Path):
    snaps = tmp_path / "models--mlx-community--Qwen3.6-35B-A3B-4bit" / "snapshots"
    old, new = "38740b847e4cb78f352aba30aa41c76e08e6eb46", "f000000000000000000000000000000000000000"
    for d in (old, new):
        (snaps / d).mkdir(parents=True)
        (snaps / d / "config.json").write_text("{}")
        (snaps / d / "model.safetensors").write_bytes(b"x")
    assert snapshot_dir_if_present(_art(), tmp_path) == snaps / new
    assert snapshot_dir_if_present(_art(revision=old), tmp_path) == snaps / old


def test_an_ollama_arm_needs_only_the_tokenizer(tmp_path: Path):
    snaps = tmp_path / "models--mlx-community--Qwen3.6-35B-A3B-4bit" / "snapshots"
    (snaps / "abc").mkdir(parents=True)
    (snaps / "abc" / "config.json").write_text("{}")
    (snaps / "abc" / "tokenizer.json").write_text("{}")
    ollama = _art(id="qwen3.6-35b-a3b-ollama", source_format="ollama", scheme="gguf_q4_k_m", ollama_tag="qwen3.6:35b")
    assert snapshot_dir_if_present(ollama, tmp_path) == snaps / "abc"
    assert snapshot_dir_if_present(_art(), tmp_path) is None


def test_a_snapshot_whose_weights_are_gone_is_not_present(tmp_path: Path):
    snaps = tmp_path / "models--mlx-community--Qwen3.6-35B-A3B-4bit" / "snapshots"
    rev = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
    (snaps / rev).mkdir(parents=True)
    (snaps / rev / "config.json").write_text("{}")
    (snaps / rev / "model.safetensors").symlink_to(tmp_path / "blobs" / "gone")
    assert snapshot_dir_if_present(_art(), tmp_path) is None
    assert snapshot_dir_if_present(_art(revision=rev), tmp_path) is None
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "gone").write_bytes(b"x")
    assert snapshot_dir_if_present(_art(), tmp_path) == snaps / rev
