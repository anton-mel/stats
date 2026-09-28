from pathlib import Path

import pytest
from pydantic import ValidationError

from pie_evals.orchestrate.matrix import Matrix, selector_matches, summarize
from pie_evals.schema import ArtifactSpec, CellStatus, Tier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def matrix():
    return Matrix.load(ROOT / "matrix")


@pytest.fixture(scope="module")
def cells(matrix):
    return matrix.expand()


def test_expansion_is_nonempty_and_stable(cells):
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids)), "cell ids must be unique"
    s = summarize(cells)
    assert s["total"] > 100 and s["tier:targeted"] == s["total"]


def test_every_cell_is_a_mac(cells):
    assert {c.platform.backend.value for c in cells} == {"metal"}
    assert {c.platform.os for c in cells} == {"macos"}
    assert {str(c.engine) for c in cells} == {"pie", "ollama"}
    assert all(c.mode.tp == 1 for c in cells)


def test_engines_only_get_their_own_weights(cells):
    for c in cells:
        if str(c.engine) == "pie":
            assert c.artifact.source_format.value != "ollama", c.cell_key
        else:
            assert c.artifact.source_format.value == "ollama" and c.artifact.ollama_tag, c.cell_key


def test_every_ollama_artifact_pairs_with_a_pie_artifact(matrix):
    for a in matrix.artifacts.values():
        if a.source_format.value == "ollama":
            assert a.baseline_of in matrix.artifacts, a.id
            assert matrix.artifacts[a.baseline_of].source_format.value != "ollama", a.id


def test_control_aa_is_pie_only(cells):
    assert all(str(c.engine) == "pie" for c in cells if c.workload.kind.value == "control_aa")


def test_ollama_artifact_needs_a_tag():
    base = {"id": "x-ollama", "base_model": "org/x", "family": "x", "scheme": "gguf_q4_k_m", "source_format": "ollama"}
    with pytest.raises(ValidationError, match="ollama_tag"):
        ArtifactSpec(**base)
    a = ArtifactSpec(**base, ollama_tag="x:7b")
    assert a.artifact_key.endswith("#x:7b")
    assert ArtifactSpec(**{**base, "ollama_tag": "x:7b-mlx"}).artifact_key != a.artifact_key


def test_fit_rule_marks_big_models_on_small_macs(matrix, cells):
    c = next(c for c in cells if c.platform.memory_gib <= 32)
    big = c.model_copy(update={"artifact": c.artifact.model_copy(update={"expected_gib": 30.0})})
    assert "does not fit" in matrix._physical_reason(big)
    assert matrix._physical_reason(c) is None


def test_context_rule_marks_shapes_past_max_context(matrix, cells):
    c = next(c for c in cells if c.workload.id == "lc-2k-128")
    short = c.model_copy(update={"artifact": c.artifact.model_copy(update={"max_context": 1024})})
    assert "max_context" in matrix._physical_reason(short)


def test_pie_only_programs_have_no_baseline_cells(cells):
    for c in cells:
        if c.program.pie_only:
            assert str(c.engine) == "pie"


def test_selector_negation_and_comparison(cells):
    c = next(c for c in cells if str(c.engine) == "ollama")
    assert selector_matches({"tp": ">=1"}, c)
    assert not selector_matches({"tp": ">1"}, c)
    assert selector_matches({"not_engine": ["pie"]}, c)
    assert not selector_matches({"engine": "ollama", "not_source_format": "ollama"}, c)


def test_targeted_budget_holds(matrix, cells):
    assert matrix.check_budget(Tier.TARGETED, cells) == []


def test_cell_status_enum_complete():
    assert {s.value for s in CellStatus} == {"pass", "fail", "declared_unsupported", "not_run", "noisy"}
