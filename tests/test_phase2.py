"""Unit and integration tests for Phase 2: Tree-Sitter 12-Metric Feature Engine."""

from __future__ import annotations

import numpy as np

from src.data.schema import PatchDatabase
from src.features.extractor import FEATURE_NAMES, TreeSitterFeatureExtractor


def test_12_feature_vector_ordering_and_latency_sla() -> None:
    extractor = TreeSitterFeatureExtractor()
    sample_java = """
    public class Calculator {
        public int compute(int[] items, int threshold) {
            int sum = 0;
            for (int x : items) {
                if (x > threshold && x < 1000) {
                    sum += Math.abs(x);
                }
            }
            return sum;
        }
    }
    """
    vec = extractor.extract(sample_java)
    arr = vec.to_numpy()
    assert len(FEATURE_NAMES) == 12
    assert arr.shape == (12,)
    assert arr.dtype == np.float32
    assert vec.extraction_time_ms < 50.0  # Strict <50ms SLA
    assert vec.sloc == 11.0
    assert vec.local_var_param_count >= 4.0  # items, threshold, sum, x
    assert vec.fan_out_calls == 1.0  # Math.abs(x)


def test_complexity_metrics_distinguish_deep_nesting_from_guard_clauses() -> None:
    extractor = TreeSitterFeatureExtractor()

    guard_clause_code = """
    public class GuardProcessor {
        public int evaluate(int a, int b, int c) {
            if (a <= 0) return 0;
            if (b <= 0) return 0;
            if (c <= 0) return 0;
            return a + b + c;
        }
    }
    """

    deeply_nested_code = """
    public class NestedProcessor {
        public int evaluate(int a, int b, int c) {
            int res = 0;
            if (a > 0) {
                for (int i = 0; i < a; i++) {
                    if (b > 0) {
                        while (c > 0) {
                            if (a > b && b > c || a == 42) {
                                res += a + b + c;
                            }
                            c--;
                        }
                    }
                }
            }
            return res;
        }
    }
    """

    v_guard = extractor.extract(guard_clause_code)
    v_nested = extractor.extract(deeply_nested_code)

    # Guard clauses have max control nesting depth = 1 and cognitive complexity = 3 (1+1+1)
    assert v_guard.max_nesting_depth == 1.0
    assert v_guard.cognitive_complexity == 3.0

    # Deeply nested code has nesting depth = 5 and much higher cognitive complexity
    assert v_nested.max_nesting_depth == 5.0
    assert v_nested.cognitive_complexity >= 17.0
    assert v_nested.cyclomatic_complexity > v_guard.cyclomatic_complexity
    assert v_nested.max_ast_depth > v_guard.max_ast_depth


def test_duplication_metrics_detect_type2_clones_and_bloat() -> None:
    extractor = TreeSitterFeatureExtractor()

    dry_code = """
    public class DryService {
        public int processAll(int[] alpha, int[] beta) {
            return sumScaled(alpha, 3) + sumScaled(beta, 5);
        }
        private int sumScaled(int[] arr, int factor) {
            int total = 0;
            for (int v : arr) {
                if (v > 0) {
                    total += v * factor + 17;
                }
            }
            return total;
        }
    }
    """

    cloned_code = """
    public class ClonedService {
        public int processAll(int[] alpha, int[] beta) {
            int totalA = 0;
            for (int i = 0; i < alpha.length; i++) {
                if (alpha[i] > 0) {
                    totalA = totalA + alpha[i] * 3 + 17;
                }
            }
            int totalB = 0;
            for (int j = 0; j < beta.length; j++) {
                if (beta[j] > 0) {
                    totalB = totalB + beta[j] * 5 + 17;
                }
            }
            int totalC = 0;
            for (int k = 0; k < alpha.length; k++) {
                if (alpha[k] > 0) {
                    totalC = totalC + alpha[k] * 7 + 17;
                }
            }
            return totalA + totalB + totalC;
        }
    }
    """

    v_dry = extractor.extract(dry_code)
    v_cloned = extractor.extract(cloned_code)

    assert v_cloned.ast_clone_ratio > v_dry.ast_clone_ratio + 0.25
    assert v_cloned.token_4gram_overlap > v_dry.token_4gram_overlap + 0.15
    assert v_cloned.halstead_volume > v_dry.halstead_volume


def test_maintainability_dfg_and_live_variable_span() -> None:
    extractor = TreeSitterFeatureExtractor()

    short_span_code = """
    public int computeClean(int x) {
        int a = x + 1;
        return a * 2;
    }
    """

    long_span_coupled_code = """
    public int computeTangled(int x, int y, int z) {
        int earlyVar = x + 1;
        int mid1 = Math.max(y, 0);
        int mid2 = Math.min(z, 100);
        int mid3 = Integer.bitCount(mid1 + mid2);
        for (int i = 0; i < mid3; i++) {
            mid1 += Math.abs(i - mid2);
            earlyVar = earlyVar + mid1 - mid2;
        }
        return earlyVar + mid1 + mid2 + mid3;
    }
    """

    v_short = extractor.extract(short_span_code)
    v_long = extractor.extract(long_span_coupled_code)

    assert v_long.dfg_edge_density > v_short.dfg_edge_density
    assert v_long.mean_live_var_span > v_short.mean_live_var_span
    assert v_long.fan_out_calls >= 4.0
    assert v_long.local_var_param_count > v_short.local_var_param_count


def test_database_metrics_populated_in_sqlite() -> None:
    with PatchDatabase("data/patch_pairs.db") as db:
        sample_pairs = db.fetch_pairs(split="train", dataset_source="codenet_java250")
        assert len(sample_pairs) > 0
        first = sample_pairs[0]
        assert len(first["metrics_a"]) >= 12
        assert len(first["metrics_b"]) >= 12
        for fname in FEATURE_NAMES:
            assert fname in first["metrics_a"]
            assert fname in first["metrics_b"]
            assert f"delta_{fname}" in first["delta_metrics"]
