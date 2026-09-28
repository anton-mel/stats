"""The in-repo result store: ``store/records/<tier>/<YYYY-MM>/<run_id>.parquet``.

Append-only. A run writes one file; ``collect`` moves node output into the
store; readers concatenate. History for a cell is the ordered list of
primary-metric values of its PASS records, newest last.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import pyarrow as pa
import pyarrow.parquet as pq

from pie_evals.schema import CellStatus, Record, Tier
from pie_evals.schema.record import ARROW_SCHEMA, records_to_table


class Store:
    def __init__(self, root: Path | str = "store"):
        self.root = Path(root)
        self.records_dir = self.root / "records"

    # ---- write -----------------------------------------------------------------------
    def write_run(self, records: list[Record], *, tier: Tier, run_id: str, when: datetime | None = None) -> Path:
        when = when or datetime.now(timezone.utc)
        problems = {r.cell_id: r.validate_for_store() for r in records}
        bad = {k: v for k, v in problems.items() if v}
        if bad:
            raise ValueError(f"records missing mandatory provenance: {json.dumps(bad)[:2000]}")
        out = self.records_dir / str(tier) / when.strftime("%Y-%m") / f"{run_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(records_to_table(records), out, compression="zstd")
        return out

    def import_node_output(self, node_out: Path, *, tier: Tier, run_id: str) -> Path | None:
        """Node writes ``records.jsonl``; the collector lifts it into parquet.
        A node that crashed before its first record leaves no file: that is
        reported by the caller, not an exception that loses the other nodes."""
        f = node_out / "records.jsonl"
        if not f.exists():
            return None
        recs = [Record.model_validate_json(line) for line in f.read_text().splitlines() if line.strip()]
        if not recs:
            return None
        return self.write_run(recs, tier=tier, run_id=run_id)

    # ---- read ------------------------------------------------------------------------
    def files(self, tier: Tier | None = None) -> list[Path]:
        base = self.records_dir / str(tier) if tier else self.records_dir
        return sorted(base.rglob("*.parquet")) if base.exists() else []

    def table(self, tier: Tier | None = None) -> pa.Table:
        fs = self.files(tier)
        if not fs:
            return ARROW_SCHEMA.empty_table()
        return pa.concat_tables([pq.read_table(f, schema=ARROW_SCHEMA) for f in fs], promote_options="default")

    def history(self, tier: Tier | None = None, *, limit: int = 20, engine: str | None = None) -> dict[str, list[float]]:
        """cell_id -> primary metric values of PASS records, oldest→newest, last ``limit``."""
        t = self.table(tier)
        if t.num_rows == 0:
            return {}
        rows = t.select(["cell_id", "status", "primary_value", "started_at", "engine"]).to_pylist()
        rows = [r for r in rows if r["status"] == str(CellStatus.PASS) and r["primary_value"] is not None]
        if engine:
            rows = [r for r in rows if r["engine"] == engine]
        rows.sort(key=lambda r: r["started_at"])
        hist: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            hist[r["cell_id"]].append(float(r["primary_value"]))
        return {k: v[-limit:] for k, v in hist.items()}

    def measured_minutes(self, *, limit: int = 5) -> dict[str, float]:
        """cell_id -> median wall minutes of its last ``limit`` records that ran."""
        t = self.table(None)
        if t.num_rows == 0:
            return {}
        rows = sorted((r for r in t.select(["cell_id", "status", "duration_s", "started_at"]).to_pylist() if r["status"] != "not_run" and r["duration_s"]), key=lambda r: r["started_at"])
        by: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by[r["cell_id"]].append(r["duration_s"] / 60)
        return {k: median(v[-limit:]) for k, v in by.items()}

    def latest_by_cell(self, tier: Tier | None = None) -> dict[str, dict]:
        t = self.table(tier)
        if t.num_rows == 0:
            return {}
        rows = sorted(t.to_pylist(), key=lambda r: r["started_at"])
        latest: dict[str, dict] = {}
        for r in rows:
            # a not_run row says the job never reached the cell; it carries no
            # measurement and must not hide an earlier one (nightly 35926457671
            # dropped every vLLM gpt-oss row from baselines.md that way)
            if r["status"] == "not_run" and r["cell_id"] in latest:
                continue
            latest[r["cell_id"]] = r
        return latest
