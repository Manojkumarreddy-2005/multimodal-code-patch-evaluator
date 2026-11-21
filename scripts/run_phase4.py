"""CLI entrypoint for Phase 4: Evaluation Metrics, Ablation Studies & Publication Tables.

Runs:
  - Ablation Study 1 (Modality Impact): Static Metrics Only vs GraphCodeBERT Only vs Full Multimodal Fusion
  - Ablation Study 2 (Objective Comparison): Tabular XGBoost vs Pointwise Regression vs Bradley-Terry vs Multi-Objective DQN
  - Track B Generative Evaluation: Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B, Meta-Llama-3-8B
  - Exports JSON, Markdown (`data/phase4_publication_tables.md`), and LaTeX (`data/phase4_publication_tables.tex`).

Usage:
    python -m scripts.run_phase4 --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.export_tables import save_publication_artifacts
from scripts.run_phase3 import prepare_split_tensors
from src.data.schema import PatchDatabase
from src.evaluation.ablations import AblationSuiteRunner
from src.track_a.encoder import CodeDFGEmbedder


def run_phase4_pipeline(config_path: str | Path = "configs/default_config.yaml") -> Dict[str, Any]:
    """Execute Phase 4 ablation studies and export publication tables."""
    t0 = time.perf_counter()
    cfg_file = Path(config_path)
    config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    db_path = config.get("database", {}).get("sqlite_path", "data/patch_pairs.db")
    seed = int(config.get("project", {}).get("seed", 42))

    embedder = CodeDFGEmbedder(cache_path="data/code_embeddings_768.npz")

    with PatchDatabase(db_path) as db:
        conn = db.connect()
        snip_rows = conn.execute(
            "SELECT snippet_id, normalized_code FROM snippets"
        ).fetchall()
        embedder.embed_batch([(r["snippet_id"], r["normalized_code"]) for r in snip_rows])

        train_pairs = db.fetch_pairs(split="train")
        val_pairs = db.fetch_pairs(split="val")
        test_pairs = db.fetch_pairs(split="test")
        ood_pairs = db.fetch_pairs(split="ood_test")

    ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr, lm_tr = prepare_split_tensors(train_pairs, embedder)
    ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te = prepare_split_tensors(test_pairs, embedder)
    ca_oo, sa_oo, ta_oo, cb_oo, sb_oo, tb_oo, lm_oo = prepare_split_tensors(ood_pairs, embedder)

    runner = AblationSuiteRunner(seed=seed)

    modality_ablation = runner.run_modality_ablation(
        ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr,
        ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te,
        ca_oo, sa_oo, ta_oo, cb_oo, sb_oo, tb_oo, lm_oo,
    )

    objective_ablation = runner.run_objective_ablation(
        ca_tr, sa_tr, ta_tr, cb_tr, sb_tr, tb_tr,
        ca_te, sa_te, ta_te, cb_te, sb_te, tb_te, lm_te,
        ca_oo, sa_oo, ta_oo, cb_oo, sb_oo, tb_oo, lm_oo,
    )

    track_b_eval = runner.run_track_b_evaluation(
        test_pairs=test_pairs,
        ood_pairs=ood_pairs[:250],
    )

    elapsed_s = time.perf_counter() - t0
    report = {
        "elapsed_seconds": round(elapsed_s, 2),
        "dataset_splits": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "ood_test_pairs": len(ood_pairs),
        },
        "ablation_1_modality_impact": modality_ablation,
        "ablation_2_objective_comparison": objective_ablation,
        "track_b_generative_evaluation": track_b_eval,
    }

    json_path = Path("data/phase4_ablation_report.json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_publication_artifacts(
        report,
        md_path="data/phase4_publication_tables.md",
        tex_path="data/phase4_publication_tables.tex",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4: Quantitative & Qualitative Evaluation + Ablation Studies."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    report = run_phase4_pipeline(config_path=args.config)
    print("=" * 72)
    print("PHASE 4 COMPLETE: Evaluation Metrics, Ablations & Publication Tables")
    print("=" * 72)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
