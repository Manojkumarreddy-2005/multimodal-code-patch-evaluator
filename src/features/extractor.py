"""Unified 12-Metric Tree-Sitter Feature Extractor & SQLite Batch Updater.

Combines Complexity (4), Duplication & Redundancy (3), and Maintainability & Coupling (5)
into a canonical 12-dimensional feature vector with <50ms per-patch execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.data.schema import PatchDatabase
from src.features.ast_parser import FastJavaASTParser
from src.features.complexity import extract_complexity_metrics
from src.features.duplication import extract_duplication_metrics
from src.features.maintainability import extract_maintainability_metrics


FEATURE_NAMES: Tuple[str, ...] = (
    # Complexity (4)
    "cyclomatic_complexity",
    "cognitive_complexity",
    "max_ast_depth",
    "max_nesting_depth",
    # Duplication & Redundancy (3)
    "ast_clone_ratio",
    "token_4gram_overlap",
    "halstead_volume",
    # Maintainability & Coupling (5)
    "dfg_edge_density",
    "mean_live_var_span",
    "fan_out_calls",
    "local_var_param_count",
    "sloc",
)


@dataclass(frozen=True)
class StaticFeatureVector:
    """Canonical 12-dimensional static feature representation of a Java patch."""

    cyclomatic_complexity: float
    cognitive_complexity: float
    max_ast_depth: float
    max_nesting_depth: float
    ast_clone_ratio: float
    token_4gram_overlap: float
    halstead_volume: float
    dfg_edge_density: float
    mean_live_var_span: float
    fan_out_calls: float
    local_var_param_count: float
    sloc: float
    extraction_time_ms: float = 0.0

    def to_dict(self, include_latency: bool = False) -> Dict[str, float]:
        d = {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "max_ast_depth": self.max_ast_depth,
            "max_nesting_depth": self.max_nesting_depth,
            "ast_clone_ratio": self.ast_clone_ratio,
            "token_4gram_overlap": self.token_4gram_overlap,
            "halstead_volume": self.halstead_volume,
            "dfg_edge_density": self.dfg_edge_density,
            "mean_live_var_span": self.mean_live_var_span,
            "fan_out_calls": self.fan_out_calls,
            "local_var_param_count": self.local_var_param_count,
            "sloc": self.sloc,
        }
        if include_latency:
            d["extraction_time_ms"] = self.extraction_time_ms
        return d

    def to_numpy(self) -> np.ndarray:
        """Return 1D float32 numpy array of shape (12,) in canonical FEATURE_NAMES order."""
        return np.array(
            [
                self.cyclomatic_complexity,
                self.cognitive_complexity,
                self.max_ast_depth,
                self.max_nesting_depth,
                self.ast_clone_ratio,
                self.token_4gram_overlap,
                self.halstead_volume,
                self.dfg_edge_density,
                self.mean_live_var_span,
                self.fan_out_calls,
                self.local_var_param_count,
                self.sloc,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaticFeatureVector":
        return cls(
            cyclomatic_complexity=float(data["cyclomatic_complexity"]),
            cognitive_complexity=float(data["cognitive_complexity"]),
            max_ast_depth=float(data["max_ast_depth"]),
            max_nesting_depth=float(data["max_nesting_depth"]),
            ast_clone_ratio=float(data["ast_clone_ratio"]),
            token_4gram_overlap=float(data["token_4gram_overlap"]),
            halstead_volume=float(data["halstead_volume"]),
            dfg_edge_density=float(data["dfg_edge_density"]),
            mean_live_var_span=float(data["mean_live_var_span"]),
            fan_out_calls=float(data["fan_out_calls"]),
            local_var_param_count=float(data["local_var_param_count"]),
            sloc=float(data["sloc"]),
            extraction_time_ms=float(data.get("extraction_time_ms", 0.0)),
        )


class TreeSitterFeatureExtractor:
    """Extracts all 12 static metrics in-memory using `tree-sitter-java` (<50ms SLA)."""

    def __init__(self) -> None:
        self._ast_parser = FastJavaASTParser()

    def extract(self, code: str) -> StaticFeatureVector:
        """Parse Java code once and extract all 12 metrics in <50ms."""
        t0 = time.perf_counter()
        parsed = self._ast_parser.parse(code)

        comp = extract_complexity_metrics(parsed)
        dup = extract_duplication_metrics(parsed)
        maint = extract_maintainability_metrics(parsed)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return StaticFeatureVector(
            cyclomatic_complexity=comp.cyclomatic_complexity,
            cognitive_complexity=comp.cognitive_complexity,
            max_ast_depth=comp.max_ast_depth,
            max_nesting_depth=comp.max_nesting_depth,
            ast_clone_ratio=dup.ast_clone_ratio,
            token_4gram_overlap=dup.token_4gram_overlap,
            halstead_volume=dup.halstead_volume,
            dfg_edge_density=maint.dfg_edge_density,
            mean_live_var_span=maint.mean_live_var_span,
            fan_out_calls=maint.fan_out_calls,
            local_var_param_count=maint.local_var_param_count,
            sloc=maint.sloc,
            extraction_time_ms=round(elapsed_ms, 4),
        )

    @staticmethod
    def compute_delta_dict(
        vec_a: StaticFeatureVector,
        vec_b: StaticFeatureVector,
    ) -> Dict[str, float]:
        """Compute Δm = m(Patch B) - m(Patch A) across all 12 features."""
        dict_a = vec_a.to_dict()
        dict_b = vec_b.to_dict()
        return {
            f"delta_{name}": round(dict_b[name] - dict_a[name], 4)
            for name in FEATURE_NAMES
        }

    @staticmethod
    def compute_triggered_dimensions(deltas: Dict[str, float]) -> Dict[str, bool]:
        """Ground Track B's 4 reasoning rubric dimensions in the 12 Tree-Sitter deltas."""
        return {
            "maintainability_reasoning": bool(
                deltas.get("delta_sloc", 0.0) >= 5.0
                or deltas.get("delta_halstead_volume", 0.0) >= 80.0
                or deltas.get("delta_local_var_param_count", 0.0) >= 2.0
            ),
            "complexity_awareness": bool(
                deltas.get("delta_cognitive_complexity", 0.0) >= 2.0
                or deltas.get("delta_cyclomatic_complexity", 0.0) >= 2.0
                or deltas.get("delta_max_nesting_depth", 0.0) >= 1.0
            ),
            "duplication_detection": bool(
                deltas.get("delta_ast_clone_ratio", 0.0) >= 0.04
                or deltas.get("delta_token_4gram_overlap", 0.0) >= 0.05
                or deltas.get("delta_halstead_volume", 0.0) >= 150.0
            ),
            "future_change_impact": bool(
                deltas.get("delta_dfg_edge_density", 0.0) >= 0.25
                or deltas.get("delta_mean_live_var_span", 0.0) >= 2.0
                or deltas.get("delta_fan_out_calls", 0.0) >= 2.0
            ),
        }

    def populate_database_metrics(self, db: PatchDatabase) -> Dict[str, Any]:
        """Extract 12 metrics for all snippets in `patch_pairs.db` and update all pair deltas."""
        conn = db.connect()
        rows = conn.execute(
            "SELECT snippet_id, normalized_code FROM snippets ORDER BY snippet_id"
        ).fetchall()

        snippet_vectors: Dict[str, StaticFeatureVector] = {}
        snippet_updates: List[Tuple[str, str]] = []
        latencies_ms: List[float] = []

        t_start = time.perf_counter()
        for row in rows:
            sid = row["snippet_id"]
            norm_code = row["normalized_code"]
            vec = self.extract(norm_code)
            snippet_vectors[sid] = vec
            latencies_ms.append(vec.extraction_time_ms)
            snippet_updates.append(
                (json.dumps(vec.to_dict(include_latency=True), sort_keys=True), sid)
            )

        with conn:
            conn.executemany(
                "UPDATE snippets SET metrics_json = ? WHERE snippet_id = ?",
                snippet_updates,
            )

        # Update all patch pairs with exact 12-metric deltas and grounded rubric triggers
        pair_rows = conn.execute(
            "SELECT pair_id, patch_a_id, patch_b_id, delta_tdi FROM patch_pairs ORDER BY pair_id"
        ).fetchall()

        pair_updates: List[Tuple[str, str, int]] = []
        delta_matrix: List[np.ndarray] = []

        for prow in pair_rows:
            pid = prow["pair_id"]
            vec_a = snippet_vectors[prow["patch_a_id"]]
            vec_b = snippet_vectors[prow["patch_b_id"]]
            deltas = self.compute_delta_dict(vec_a, vec_b)
            deltas["delta_sonar_tdi"] = round(float(prow["delta_tdi"]), 3)
            triggers = self.compute_triggered_dimensions(deltas)

            delta_matrix.append(vec_b.to_numpy() - vec_a.to_numpy())
            pair_updates.append(
                (
                    json.dumps(deltas, sort_keys=True),
                    json.dumps(triggers, sort_keys=True),
                    pid,
                )
            )

        with conn:
            conn.executemany(
                """
                UPDATE patch_pairs
                SET delta_metrics_json = ?, triggered_dimensions_json = ?
                WHERE pair_id = ?
                """,
                pair_updates,
            )

        total_elapsed_s = time.perf_counter() - t_start
        lat_arr = np.array(latencies_ms, dtype=np.float64) if latencies_ms else np.zeros(1)
        delta_arr = np.vstack(delta_matrix) if delta_matrix else np.zeros((1, 12))

        mean_deltas = {
            name: round(float(np.mean(delta_arr[:, idx])), 3)
            for idx, name in enumerate(FEATURE_NAMES)
        }

        return {
            "snippets_processed": len(snippet_updates),
            "pairs_updated": len(pair_updates),
            "total_elapsed_seconds": round(total_elapsed_s, 3),
            "latency_ms": {
                "mean_ms": round(float(np.mean(lat_arr)), 4),
                "median_ms": round(float(np.median(lat_arr)), 4),
                "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
                "p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
                "max_ms": round(float(np.max(lat_arr)), 4),
                "sla_target_ms": 50.0,
                "sla_passed_99pct": bool(float(np.percentile(lat_arr, 99)) < 50.0),
            },
            "mean_pair_deltas_B_minus_A": mean_deltas,
        }
