from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pie_evals.orchestrate.jobs import recorded_cell_keys
from pie_evals.orchestrate.store import Store
from pie_evals.schema.cell import Tier
from pie_evals.schema.record import ARROW_SCHEMA


def test_recorded_cell_keys_are_per_commit_and_pin(tmp_path: Path):
    st = Store(tmp_path)
    out = st.records_dir / "targeted" / "2026-09" / "run1.parquet"
    out.parent.mkdir(parents=True)
    rows = [
        {"cell_key": "pie|m4-pro-48g|a|ss-128-64|text-completion-bench|tp1", "pie_commit": "aaaa", "status": "pass", "engine": "pie"},
        {"cell_key": "pie|m4-pro-48g|a|c8|text-completion-bench|tp1", "pie_commit": "aaaa", "status": "crash", "engine": "pie"},
        {"cell_key": "pie|m4-pro-48g|a|c32|text-completion-bench|tp1", "pie_commit": "bbbb", "status": "pass", "engine": "pie"},
        {"cell_key": "ollama|m4-pro-48g|a#t:1b|c8|text-completion-bench|tp1", "pie_commit": None, "status": "pass", "engine": "ollama", "engine_version": "0.34.4"},
        {"cell_key": "ollama|m4-pro-48g|a#t:1b|c32|text-completion-bench|tp1", "pie_commit": None, "status": "pass", "engine": "ollama", "engine_version": "0.33.0"},
        {"cell_key": "pie|m4-pro-48g|a|lc-1k-128|text-completion-bench|tp1", "pie_commit": "aaaa", "status": "not_run", "engine": "pie"},
        {"cell_key": "pie|m4-pro-48g|a|lc-2k-128|text-completion-bench|tp1", "pie_commit": "aaaa", "status": "fail", "engine": "pie", "error_class": "harness_invalid"},
        {"cell_key": "pie|m4-pro-48g|a|ob-512-200|text-completion-bench|tp1", "pie_commit": "aaaa", "status": "noisy", "engine": "pie"},
        {"cell_key": "ollama|m4-pro-48g|a#t:1b|ss-128-64|text-completion-bench|tp1", "pie_commit": None, "status": "fail", "engine": "ollama", "engine_version": "0.34.4", "error_class": "crash"},
    ]
    for row in rows:
        row.setdefault("started_at", datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc))
    cols = {name: [row.get(name) for row in rows] for name in ARROW_SCHEMA.names}
    pq.write_table(pa.table(cols, schema=ARROW_SCHEMA), out)
    done = recorded_cell_keys(st, Tier.TARGETED, "aaaa", {"ollama": "0.34.4"})
    assert done == {rows[0]["cell_key"], rows[1]["cell_key"], rows[3]["cell_key"]}
    assert recorded_cell_keys(st, Tier.TARGETED, "aaaa") == {rows[0]["cell_key"], rows[1]["cell_key"]}
    assert recorded_cell_keys(Store(tmp_path / "empty"), Tier.TARGETED, "aaaa") == set()
