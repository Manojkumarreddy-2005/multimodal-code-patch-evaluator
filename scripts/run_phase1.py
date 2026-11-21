"""CLI entrypoint for Phase 1: Public Dataset Collection & SQLite Pipeline.

Usage:
    python -m scripts.run_phase1 --config configs/default_config.yaml --reset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import yaml

# Ensure workspace root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ingest_codenet import CodeNetJava250Ingestor
from src.data.ingest_swerefactor import SWERefactorIngestor
from src.data.normalizer import JavaCodeNormalizer
from src.data.schema import PatchDatabase
from src.data.sonar_proxy import SonarDebtCalculator


def run_phase1_pipeline(
    config_path: str | Path = "configs/default_config.yaml",
    reset_db: bool = True,
) -> dict:
    """Execute Phase 1 ingestion for CodeNet Java250 and SWE-Refactor into `patch_pairs.db`."""
    start_time = time.perf_counter()
    cfg_file = Path(config_path)
    config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}

    db_path = config.get("database", {}).get("sqlite_path", "data/patch_pairs.db")
    p1_cfg = config.get("phase1_ingestion", {})
    p1_cfg["seed"] = config.get("project", {}).get("seed", 42)
    sonar_cfg = config.get("sonar_proxy", {})

    normalizer = JavaCodeNormalizer()
    debt_calc = SonarDebtCalculator(sonar_cfg)

    codenet_ingestor = CodeNetJava250Ingestor(
        raw_dir=p1_cfg.get("codenet_raw_dir", "data/raw/Project_CodeNet_Java250"),
        normalizer=normalizer,
        debt_calc=debt_calc,
        config=p1_cfg,
    )
    swe_ingestor = SWERefactorIngestor(
        raw_path=p1_cfg.get("swe_refactor_raw_path", "data/raw/swe_refactor.json"),
        normalizer=normalizer,
        debt_calc=debt_calc,
        config=p1_cfg,
    )

    with PatchDatabase(db_path) as db:
        if reset_db:
            db.reset_database()

        cn_snippets, cn_pairs = codenet_ingestor.ingest_to_db(db, bootstrap_if_missing=True)
        swe_snippets, swe_pairs = swe_ingestor.ingest_to_db(db, bootstrap_if_missing=True)
        stats = db.get_summary_stats()

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    stats["ingestion_Summary"] = {
        "codenet_java250_snippets": cn_snippets,
        "codenet_java250_pairs": cn_pairs,
        "swe_refactor_snippets": swe_snippets,
        "swe_refactor_pairs": swe_pairs,
        "elapsed_ms": round(elapsed_ms, 2),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1: Dataset Collection & SQLite Pipeline (`patch_pairs.db`)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=True,
        help="Reset existing tables in patch_pairs.db before ingesting.",
    )
    args = parser.parse_args()

    stats = run_phase1_pipeline(config_path=args.config, reset_db=args.reset)
    print("=" * 72)
    print("PHASE 1 COMPLETE: SQLite Dataset Pipeline (`patch_pairs.db`)")
    print("=" * 72)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
