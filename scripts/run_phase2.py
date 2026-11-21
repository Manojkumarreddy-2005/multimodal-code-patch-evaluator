"""CLI entrypoint for Phase 2: Tree-Sitter Feature Engineering (12 Static Metrics).

Extracts the 12 deterministic metrics for all snippets in `data/patch_pairs.db`,
verifies the <50ms per-patch latency SLA, and updates all pairwise feature deltas.

Usage:
    python -m scripts.run_phase2 --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.schema import PatchDatabase
from src.features.extractor import TreeSitterFeatureExtractor


def run_phase2_pipeline(config_path: str | Path = "configs/default_config.yaml") -> dict:
    """Run Phase 2 12-metric extraction and update `patch_pairs.db`."""
    cfg_file = Path(config_path)
    config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    db_path = config.get("database", {}).get("sqlite_path", "data/patch_pairs.db")

    extractor = TreeSitterFeatureExtractor()
    with PatchDatabase(db_path) as db:
        report = extractor.populate_database_metrics(db)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 2: Tree-Sitter 12-Metric Feature Engineering on `patch_pairs.db`."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    report = run_phase2_pipeline(config_path=args.config)
    print("=" * 72)
    print("PHASE 2 COMPLETE: Tree-Sitter 12 Static Metrics (`patch_pairs.db`)")
    print("=" * 72)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
