"""Unit and integration tests for Phase 1: Dataset Collection & SQLite Pipeline."""

from __future__ import annotations

from pathlib import Path

from scripts.run_phase1 import run_phase1_pipeline
from src.data.normalizer import JavaCodeNormalizer
from src.data.schema import PatchDatabase
from src.data.sonar_proxy import SonarDebtCalculator


def test_java_normalizer_strips_comments_preserves_string_literals() -> None:
    normalizer = JavaCodeNormalizer()
    raw_java = """
    // Top-level line comment that must be stripped
    /* Multi-line block comment
       spanning two lines */
    public class Sample {
        /** Javadoc comment */
        public static String getUrl() {
            String url = "http://example.com/api/*keep_this*/"; // trailing comment
            return url;
        }
    }
    """
    result = normalizer.normalize(raw_java)
    assert not result.has_syntax_error
    assert result.comment_count == 4
    assert "Top-level line comment" not in result.normalized_code
    assert "Multi-line block comment" not in result.normalized_code
    assert "Javadoc comment" not in result.normalized_code
    assert "trailing comment" not in result.normalized_code
    assert '"http://example.com/api/*keep_this*/"' in result.normalized_code
    assert result.sloc == 6


def test_sonar_debt_calculator_ranks_clean_over_duplicated() -> None:
    normalizer = JavaCodeNormalizer()
    calc = SonarDebtCalculator()

    clean_java = """
    public class CleanCalculator {
        public int sumPositive(int[] items) {
            int total = 0;
            for (int item : items) {
                if (item > 0) {
                    total += item;
                }
            }
            return total;
        }
    }
    """

    bloated_java = """
    public class BloatedCalculator {
        public int sumPositive(int[] items, int p1, int p2, int p3, int p4, int p5) {
            int total = 0;
            int shadow = 0;
            for (int i = 0; i < items.length; i++) {
                if (items[i] > 0 && p1 >= 0 && p2 >= 0) {
                    if (p3 > 0 || p4 > 0) {
                        if (items[i] != 42 && items[i] != 99) {
                            total = total + items[i] * 37 - 19;
                            shadow = shadow + items[i] * 37 - 19;
                        } else {
                            total = total + items[i] * 37 - 19;
                            shadow = shadow + items[i] * 37 - 19;
                        }
                    }
                }
            }
            return total + (shadow - shadow);
        }
    }
    """

    norm_a = normalizer.normalize(clean_java)
    norm_b = normalizer.normalize(bloated_java)

    prof_a = calc.analyze(norm_a.normalized_code)
    prof_b = calc.analyze(norm_b.normalized_code)

    assert prof_b.sonar_tdi_minutes > prof_a.sonar_tdi_minutes + 10.0
    assert prof_b.max_control_nesting > prof_a.max_control_nesting
    assert prof_b.duplicate_block_count >= 1

    deltas, triggers = calc.compute_pair_deltas(prof_a, prof_b)
    assert deltas["delta_sonar_tdi"] > 10.0
    assert triggers["complexity_awareness"] is True
    assert triggers["duplication_detection"] is True


def test_phase1_pipeline_populates_sqlite_and_prevents_leakage(tmp_path: Path) -> None:
    db_file = tmp_path / "test_patch_pairs.db"
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(
        f"""
project:
  seed: 42
database:
  sqlite_path: "{db_file.as_posix()}"
phase1_ingestion:
  codenet_raw_dir: "{(tmp_path / 'codenet').as_posix()}"
  swe_refactor_raw_path: "{(tmp_path / 'swe_refactor.json').as_posix()}"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  length_matched_max_rel_diff: 0.15
  min_tdi_delta_minutes: 5.0
  max_pairs_per_problem: 6
""",
        encoding="utf-8",
    )

    stats = run_phase1_pipeline(config_path=cfg_file, reset_db=True)
    assert db_file.exists()
    assert stats["total_snippets"] >= 100
    assert stats["total_pairs"] >= 70
    assert stats["zero_group_leakage_verified"] is True
    assert "codenet_java250" in stats["by_source"]
    assert "swe_refactor" in stats["by_source"]

    with PatchDatabase(db_file) as db:
        train_pairs = db.fetch_pairs(split="train", dataset_source="codenet_java250")
        test_pairs = db.fetch_pairs(split="test", dataset_source="codenet_java250")
        ood_pairs = db.fetch_pairs(split="ood_test", dataset_source="swe_refactor")
        len_matched = db.fetch_pairs(length_matched_only=True)

        assert len(train_pairs) > 0
        assert len(test_pairs) > 0
        assert len(ood_pairs) == 15
        assert len(len_matched) > 0

        # Verify Patch B always has strictly greater technical debt than Patch A
        for pair in train_pairs + test_pairs + ood_pairs:
            assert pair["tdi_b"] > pair["tdi_a"]
            assert pair["delta_tdi"] > 0
            assert "//" not in pair["code_a_norm"].splitlines()[0]
