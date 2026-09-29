import json

from pie_evals.orchestrate import dashboard
from pie_evals.orchestrate.store import Store
from pie_evals.quality.tasks import Item, score_gsm8k, wilson


def test_gsm8k_reads_the_marked_answer_first():
    item = Item("q", "p", "42")
    assert score_gsm8k(item, "6 * 7 = 42\n#### 42") == 1.0
    assert score_gsm8k(item, "#### 1,042") == 0.0
    assert score_gsm8k(item, "so the answer is 42.") == 1.0
    assert score_gsm8k(item, "no idea") == 0.0


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson(70, 100)
    assert lo < 0.7 < hi and round(lo, 2) == 0.60 and round(hi, 2) == 0.78
    assert wilson(0, 0) == (0.0, 0.0)


def test_site_reads_quality_summaries(tmp_path):
    q = tmp_path / "quality"
    q.mkdir()
    (q / "gemma-4-26b-a4b-ollama-gsm8k.json").write_text(json.dumps({"artifact": "gemma-4-26b-a4b-ollama", "task": "gsm8k", "score": 0.9, "ci95": [0.8, 0.95], "n": 100, "samples": [1]}))
    got = dashboard.quality(Store(tmp_path))
    assert got == {"gemma-4-26b-a4b-ollama": {"gsm8k": {"score": 0.9, "ci95": [0.8, 0.95], "n": 100, "version": None, "date": None, "engine": None, "empty": None}}}
