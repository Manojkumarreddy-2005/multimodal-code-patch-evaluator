"""End-to-End Orchestrator for Phases 1 -> 2 -> 3 -> 4 (`scripts/run_pipeline.py`).

Usage:
    python -m scripts.run_pipeline --config configs/default_config.yaml
    python -m scripts.run_pipeline --skip-phase1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_phase1 import run_phase1_pipeline
from scripts.run_phase2 import run_phase2_pipeline
from scripts.run_phase3 import run_phase3_pipeline
from scripts.run_phase4 import run_phase4_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Multimodal Code Patch Evaluator Pipeline (Phases 1 -> 4)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 dataset ingestion if `data/patch_pairs.db` is already populated.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    if not args.skip_phase1 or not Path("data/patch_pairs.db").exists():
        print("[1/4] Running Phase 1: Dataset Collection & SQLite Pipeline...")
        run_phase1_pipeline(config_path=args.config, reset_db=True)

    print("[2/4] Running Phase 2: Tree-Sitter 12-Metric Feature Engineering...")
    run_phase2_pipeline(config_path=args.config)

    print("[3/4] Running Phase 3: Dual-Track System Architecture (Track A + Track B)...")
    run_phase3_pipeline(config_path=args.config)

    print("[4/4] Running Phase 4: Evaluation Metrics, Ablation Studies & Publication Tables...")
    run_phase4_pipeline(config_path=args.config)

    total_s = time.perf_counter() - t0
    print(f"\nAll Phases Complete in {total_s:.2f}s. Artifacts saved in `data/`.")


if __name__ == "__main__":
    main()
