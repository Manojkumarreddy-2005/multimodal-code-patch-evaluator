"""SQLite schema and persistence engine for `patch_pairs.db`.

Stores normalized Java snippets, dataset provenance (IBM Project CodeNet Java250
and SWE-Refactor), Sonar-aligned technical debt scores, and paired patches
(Patch A = Clean/Maintainable, Patch B = Complex/Duplicated) with pre-computed
technical debt deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SCHEMA_DDL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS snippets (
    snippet_id TEXT PRIMARY KEY,
    dataset_source TEXT NOT NULL,
    problem_or_repo_id TEXT NOT NULL,
    raw_code TEXT NOT NULL,
    normalized_code TEXT NOT NULL,
    sloc INTEGER NOT NULL,
    passes_tests INTEGER NOT NULL DEFAULT 1,
    sonar_tdi REAL NOT NULL,
    sonar_breakdown_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patch_pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_source TEXT NOT NULL,
    split TEXT NOT NULL,
    group_id TEXT NOT NULL,
    patch_a_id TEXT NOT NULL,
    patch_b_id TEXT NOT NULL,
    refactoring_type TEXT DEFAULT '',
    delta_tdi REAL NOT NULL,
    delta_metrics_json TEXT NOT NULL,
    is_length_matched INTEGER NOT NULL DEFAULT 0,
    triggered_dimensions_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patch_a_id) REFERENCES snippets(snippet_id),
    FOREIGN KEY (patch_b_id) REFERENCES snippets(snippet_id),
    UNIQUE (patch_a_id, patch_b_id)
);

CREATE INDEX IF NOT EXISTS idx_snippets_source_group
    ON snippets(dataset_source, problem_or_repo_id);

CREATE INDEX IF NOT EXISTS idx_pairs_source_split
    ON patch_pairs(dataset_source, split);

CREATE INDEX IF NOT EXISTS idx_pairs_group
    ON patch_pairs(group_id);

CREATE INDEX IF NOT EXISTS idx_pairs_length_matched
    ON patch_pairs(is_length_matched);
"""


@dataclass
class SnippetRecord:
    """Represents a single verified Java implementation stored in `snippets`."""

    snippet_id: str
    dataset_source: str
    problem_or_repo_id: str
    raw_code: str
    normalized_code: str
    sloc: int
    sonar_tdi: float
    sonar_breakdown: Dict[str, Any] = field(default_factory=dict)
    passes_tests: int = 1
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchPairRecord:
    """Represents a paired (Patch A = Clean, Patch B = Complex/Duplicated) record."""

    dataset_source: str
    split: str
    group_id: str
    patch_a_id: str
    patch_b_id: str
    delta_tdi: float
    delta_metrics: Dict[str, float]
    is_length_matched: bool
    triggered_dimensions: Dict[str, bool]
    refactoring_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    pair_id: Optional[int] = None


class PatchDatabase:
    """SQLite database manager for `patch_pairs.db`."""

    def __init__(self, db_path: str | Path = "data/patch_pairs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PatchDatabase":
        self.connect()
        self.initialize_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def initialize_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self.connect()
        conn.executescript(SCHEMA_DDL)
        conn.commit()

    def reset_database(self) -> None:
        """Drop and recreate all tables for a clean ingestion run."""
        conn = self.connect()
        conn.executescript(
            """
            DROP TABLE IF EXISTS patch_pairs;
            DROP TABLE IF EXISTS snippets;
            """
        )
        conn.executescript(SCHEMA_DDL)
        conn.commit()

    def upsert_snippets(self, snippets: Iterable[SnippetRecord]) -> int:
        """Batch insert or update snippet records."""
        conn = self.connect()
        rows = [
            (
                s.snippet_id,
                s.dataset_source,
                s.problem_or_repo_id,
                s.raw_code,
                s.normalized_code,
                int(s.sloc),
                int(s.passes_tests),
                float(s.sonar_tdi),
                json.dumps(s.sonar_breakdown, sort_keys=True),
                json.dumps(s.metrics, sort_keys=True),
                json.dumps(s.metadata, sort_keys=True),
            )
            for s in snippets
        ]
        if not rows:
            return 0
        with conn:
            conn.executemany(
                """
                INSERT INTO snippets (
                    snippet_id, dataset_source, problem_or_repo_id,
                    raw_code, normalized_code, sloc, passes_tests,
                    sonar_tdi, sonar_breakdown_json, metrics_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snippet_id) DO UPDATE SET
                    sonar_tdi = excluded.sonar_tdi,
                    sonar_breakdown_json = excluded.sonar_breakdown_json,
                    metrics_json = CASE
                        WHEN excluded.metrics_json != '{}' THEN excluded.metrics_json
                        ELSE snippets.metrics_json
                    END,
                    metadata_json = excluded.metadata_json
                """,
                rows,
            )
        return len(rows)

    def insert_pairs(self, pairs: Iterable[PatchPairRecord]) -> int:
        """Batch insert or update patch pairs."""
        conn = self.connect()
        rows = [
            (
                p.dataset_source,
                p.split,
                p.group_id,
                p.patch_a_id,
                p.patch_b_id,
                p.refactoring_type,
                float(p.delta_tdi),
                json.dumps(p.delta_metrics, sort_keys=True),
                1 if p.is_length_matched else 0,
                json.dumps(p.triggered_dimensions, sort_keys=True),
                json.dumps(p.metadata, sort_keys=True),
            )
            for p in pairs
        ]
        if not rows:
            return 0
        with conn:
            conn.executemany(
                """
                INSERT INTO patch_pairs (
                    dataset_source, split, group_id, patch_a_id, patch_b_id,
                    refactoring_type, delta_tdi, delta_metrics_json,
                    is_length_matched, triggered_dimensions_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patch_a_id, patch_b_id) DO UPDATE SET
                    split = excluded.split,
                    delta_tdi = excluded.delta_tdi,
                    delta_metrics_json = excluded.delta_metrics_json,
                    is_length_matched = excluded.is_length_matched,
                    triggered_dimensions_json = excluded.triggered_dimensions_json,
                    metadata_json = excluded.metadata_json
                """,
                rows,
            )
        return len(rows)

    def fetch_pairs(
        self,
        split: Optional[str] = None,
        dataset_source: Optional[str] = None,
        length_matched_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query joined patch pairs with full Patch A and Patch B snippet fields."""
        conn = self.connect()
        clauses: List[str] = []
        params: List[Any] = []

        if split is not None:
            clauses.append("p.split = ?")
            params.append(split)
        if dataset_source is not None:
            clauses.append("p.dataset_source = ?")
            params.append(dataset_source)
        if length_matched_only:
            clauses.append("p.is_length_matched = 1")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT
                p.pair_id,
                p.dataset_source,
                p.split,
                p.group_id,
                p.patch_a_id,
                p.patch_b_id,
                p.refactoring_type,
                p.delta_tdi,
                p.delta_metrics_json,
                p.is_length_matched,
                p.triggered_dimensions_json,
                p.metadata_json AS pair_metadata_json,
                sa.raw_code AS code_a_raw,
                sa.normalized_code AS code_a_norm,
                sa.sloc AS sloc_a,
                sa.sonar_tdi AS tdi_a,
                sa.metrics_json AS metrics_a_json,
                sb.raw_code AS code_b_raw,
                sb.normalized_code AS code_b_norm,
                sb.sloc AS sloc_b,
                sb.sonar_tdi AS tdi_b,
                sb.metrics_json AS metrics_b_json
            FROM patch_pairs p
            JOIN snippets sa ON p.patch_a_id = sa.snippet_id
            JOIN snippets sb ON p.patch_b_id = sb.snippet_id
            {where_sql}
            ORDER BY p.pair_id ASC
        """
        cursor = conn.execute(query, params)
        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            item = dict(row)
            item["delta_metrics"] = json.loads(item.pop("delta_metrics_json"))
            item["triggered_dimensions"] = json.loads(item.pop("triggered_dimensions_json"))
            item["pair_metadata"] = json.loads(item.pop("pair_metadata_json"))
            item["metrics_a"] = json.loads(item.pop("metrics_a_json"))
            item["metrics_b"] = json.loads(item.pop("metrics_b_json"))
            item["is_length_matched"] = bool(item["is_length_matched"])
            results.append(item)
        return results

    def verify_no_group_leakage(self) -> Tuple[bool, Dict[str, Set[str]]]:
        """Verify that no `group_id` (problem_id or repo) overlaps across train/val/test."""
        conn = self.connect()
        cursor = conn.execute(
            "SELECT DISTINCT split, group_id FROM patch_pairs WHERE split IN ('train', 'val', 'test')"
        )
        split_groups: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        for row in cursor.fetchall():
            split_groups[row["split"]].add(row["group_id"])

        overlaps = (
            (split_groups["train"] & split_groups["val"])
            | (split_groups["train"] & split_groups["test"])
            | (split_groups["val"] & split_groups["test"])
        )
        return (len(overlaps) == 0), split_groups

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return comprehensive dataset and split statistics from `patch_pairs.db`."""
        conn = self.connect()
        total_snippets = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        total_pairs = conn.execute("SELECT COUNT(*) FROM patch_pairs").fetchone()[0]

        by_source = {}
        for row in conn.execute(
            """
            SELECT dataset_source, COUNT(*) AS cnt,
                   ROUND(AVG(delta_tdi), 2) AS avg_delta_tdi,
                   SUM(is_length_matched) AS length_matched_cnt
            FROM patch_pairs
            GROUP BY dataset_source
            """
        ).fetchall():
            by_source[row["dataset_source"]] = dict(row)

        by_split = {}
        for row in conn.execute(
            """
            SELECT split, COUNT(*) AS cnt, COUNT(DISTINCT group_id) AS unique_groups
            FROM patch_pairs
            GROUP BY split
            """
        ).fetchall():
            by_split[row["split"]] = dict(row)

        leakage_free, _ = self.verify_no_group_leakage()

        return {
            "database_path": str(self.db_path),
            "total_snippets": total_snippets,
            "total_pairs": total_pairs,
            "by_source": by_source,
            "by_split": by_split,
            "zero_group_leakage_verified": leakage_free,
        }
