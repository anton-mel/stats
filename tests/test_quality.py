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


def test_scores_grade_the_answer_not_the_thinking():
    from pie_evals.quality.tasks import answer_part

    assert answer_part("<|channel>thought\nlet me think 41<channel|>#### 42") == "#### 42"
    assert answer_part("<think>41</think>\n#### 42") == "#### 42"
    assert answer_part("<|channel>thought\nnever closed") == ""
    assert score_gsm8k(Item("q", "p", "42"), "<|channel>thought\n#### 41<channel|>#### 42") == 1.0


def test_glimmer_answers_are_the_user_message():
    from pie_evals.quality.tasks import answer_part

    out = " to=self<|message|>We need to capitalize.<|eom|><|start|>assistant to=user<|message|>HELLO THERE<|eot|>"
    assert answer_part(out) == "HELLO THERE"
    assert answer_part(" to=self<|message|>still thinking when the budget ran out") == ""


def test_choice_tasks_read_the_marked_letter():
    from pie_evals.quality.tasks import score_choice

    item = Item("q", "p", "C")
    assert score_choice(item, "It is the third.\nAnswer: C") == 1.0
    assert score_choice(item, "Answer: **C**") == 1.0
    assert score_choice(item, "the answer is (C)") == 1.0
    assert score_choice(item, "C") == 1.0
    assert score_choice(item, "Answer: B") == 0.0
    assert score_choice(item, "A looks right, then B, so\nAnswer: C") == 1.0
    assert score_choice(item, "no idea") == 0.0
    assert score_choice(item, "<think>Answer: B</think>Answer: C") == 1.0


def test_math_tasks_compare_the_boxed_answer():
    from pie_evals.quality.tasks import score_math

    item = Item("q", "p", "\\frac{1}{2}")
    assert score_math(item, "so \\boxed{\\dfrac{1}{2}}") == 1.0
    assert score_math(item, "so \\boxed{ \\frac{1}{2} }.") == 1.0
    assert score_math(item, "\\boxed{\\frac{1}{3}}") == 0.0
    assert score_math(item, "the answer is 1/2") == 0.0
    assert score_math(Item("q", "p", "\\left( 3, \\frac{\\pi}{2} \\right)"), "\\boxed{(3, \\frac{\\pi}{2})}") == 1.0
    assert score_math(Item("q", "p", "10"), "\\boxed{\\text{10}}") == 1.0
    assert score_math(Item("q", "p", "5"), "first \\boxed{4} then \\boxed{5}") == 1.0
    assert score_math(Item("q", "p", "{1}"), "\\boxed{{1}}") == 1.0


def test_choice_loaders_shuffle_deterministically_and_mark_the_gold_letter(monkeypatch, tmp_path):
    import csv

    from pie_evals.quality import tasks

    path = tmp_path / "gpqa.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"])
        w.writeheader()
        for i in range(5):
            w.writerow({"Question": f"q{i}", "Correct Answer": f"right{i}", "Incorrect Answer 1": "a", "Incorrect Answer 2": "b", "Incorrect Answer 3": "c"})
    monkeypatch.setattr(tasks, "_download", lambda repo, filename: path)
    first, second = tasks.load_gpqa(5), tasks.load_gpqa(5)
    assert [i.prompt for i in first] == [i.prompt for i in second]
    for item in first:
        lines = {row[0]: row[3:] for row in item.prompt.splitlines() if len(row) > 3 and row[1:3] == ". "}
        assert lines[item.gold].startswith("right")
        assert tasks.score_choice(item, f"Answer: {item.gold}") == 1.0


def test_every_task_has_a_loader_scorer_and_budget():
    from pie_evals.quality.tasks import TASKS

    assert set(TASKS) == {"gsm8k", "ifeval", "mmlu", "arc", "math500", "gpqa"}
    for load, score, budget in TASKS.values():
        assert callable(load) and callable(score) and budget > 0


def test_a_task_that_cannot_load_is_skipped_not_fatal(monkeypatch):
    from pie_evals.quality import run, tasks

    class GatedRepoError(Exception):
        pass

    def gated(n):
        raise GatedRepoError("401 Client Error. Cannot access gated repo")

    monkeypatch.setitem(tasks.TASKS, "gpqa", (gated, tasks.score_choice, 1024))
    lines = []
    assert run.load_items("gpqa", 100, lines.append) is None
    assert "skipped" in lines[0] and "HF_TOKEN" in lines[0]
    monkeypatch.setitem(tasks.TASKS, "gpqa", (lambda n: ["item"], tasks.score_choice, 1024))
    assert run.load_items("gpqa", 100, lines.append) == ["item"]
