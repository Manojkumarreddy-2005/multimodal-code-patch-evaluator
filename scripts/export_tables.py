"""Export Phase 4 Ablation & Evaluation Results to Markdown and LaTeX Publication Tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Format Phase 4 ablation and evaluation results as GitHub-flavored Markdown tables."""
    lines: List[str] = [
        "# Phase 4 Experimental Results & Ablation Tables",
        "",
        "## Table 1: Ablation Study 1 — Modality Impact (In-Domain vs. Out-of-Distribution)",
        "",
        "| Modality Configuration | CodeNet $\\text{Acc}_{\\text{pref}}$ (95% CI) | CodeNet Len-Matched | CodeNet Spearman $\\rho$ | CodeNet Macro-F1 | SWE-Refactor OOD $\\text{Acc}_{\\text{pref}}$ | SWE-Refactor Spearman $\\rho$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    mod_ablation = report.get("ablation_1_modality_impact", {})
    for name, entry in mod_ablation.items():
        id_m = entry["in_domain_codenet_test"]
        oo_m = entry["ood_swe_refactor_test"]
        lines.append(
            f"| **{name}** | "
            f"`{id_m['pairwise_preference_accuracy']*100:.2f}%` (`[{id_m['ci_95_low']*100:.1f}, {id_m['ci_95_high']*100:.1f}]`) | "
            f"`{id_m['length_matched_accuracy']*100:.2f}%` | "
            f"`{id_m['spearman_rho_vs_sonar_tdi']:.4f}` | "
            f"`{id_m['macro_f1']:.4f}` | "
            f"`{oo_m['pairwise_preference_accuracy']*100:.2f}%` | "
            f"`{oo_m['spearman_rho_vs_sonar_tdi']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Table 2: Ablation Study 2 — Objective Comparison (Track A Ranking Models)",
            "",
            "| Training Objective | CodeNet $\\text{Acc}_{\\text{pref}}$ | CodeNet Len-Matched | CodeNet Spearman $\\rho$ | CodeNet Macro-F1 | SWE-Refactor OOD $\\text{Acc}_{\\text{pref}}$ | SWE-Refactor Len-Matched | SWE-Refactor Spearman $\\rho$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    obj_ablation = report.get("ablation_2_objective_comparison", {})
    for name, entry in obj_ablation.items():
        id_m = entry["in_domain_codenet_test"]
        oo_m = entry["ood_swe_refactor_test"]
        lines.append(
            f"| **{name}** | "
            f"`{id_m['pairwise_preference_accuracy']*100:.2f}%` | "
            f"`{id_m['length_matched_accuracy']*100:.2f}%` | "
            f"`{id_m['spearman_rho_vs_sonar_tdi']:.4f}` | "
            f"`{id_m['macro_f1']:.4f}` | "
            f"`{oo_m['pairwise_preference_accuracy']*100:.2f}%` | "
            f"`{oo_m['length_matched_accuracy']*100:.2f}%` | "
            f"`{oo_m['spearman_rho_vs_sonar_tdi']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Table 3: Track B Generative CodeLLM Engineering Reasoning & 4-Dimension Recall",
            "",
            "| Open-Weights CodeLLM | Benchmark Split | $\\text{Acc}_{\\text{pref}}$ | Swap-Consistent $\\text{Acc}_{\\text{strict}}$ | Position Bias Rate | Selection-Reasoning Consistency | Maintainability Recall | Complexity Recall | Duplication Recall | Change Impact Recall |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    track_b = report.get("track_b_generative_evaluation", {})
    for m_name, splits in track_b.items():
        for split_label, s_key in (("CodeNet Test", "in_domain_codenet_test"), ("SWE-Refactor OOD", "ood_swe_refactor_test")):
            res = splits[s_key]
            d_rec = res["dimension_wise_recall"]
            lines.append(
                f"| **{m_name}** | {split_label} | "
                f"`{res['pairwise_preference_accuracy']*100:.2f}%` | "
                f"`{res['strict_swap_consistent_accuracy']*100:.2f}%` | "
                f"`{res['position_bias_rate']*100:.2f}%` | "
                f"`{res['selection_reasoning_consistency_rate']*100:.2f}%` | "
                f"`{d_rec['maintainability_reasoning']*100:.2f}%` | "
                f"`{d_rec['complexity_awareness']*100:.2f}%` | "
                f"`{d_rec['duplication_detection']*100:.2f}%` | "
                f"`{d_rec['future_change_impact']*100:.2f}%` |"
            )

    return "\n".join(lines) + "\n"


def generate_latex_tables(report: Dict[str, Any]) -> str:
    """Format Phase 4 ablation and evaluation results as publication-ready LaTeX tables."""
    lines: List[str] = [
        "% Auto-generated LaTeX Publication Tables for Multimodal Code Patch Evaluator",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Ablation Studies on Modality Impact and Ranking Objectives across CodeNet (In-Domain) and SWE-Refactor (OOD).}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model / Ablation} & \\textbf{CodeNet Acc} & \\textbf{Len-Matched} & \\textbf{CodeNet $\\rho$} & \\textbf{Macro-F1} & \\textbf{SWE-Refactor Acc} & \\textbf{SWE-Refactor $\\rho$} \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textit{Panel A: Modality Impact Ablation}} \\\\",
    ]

    for name, entry in report.get("ablation_1_modality_impact", {}).items():
        id_m = entry["in_domain_codenet_test"]
        oo_m = entry["ood_swe_refactor_test"]
        clean_name = name.replace("_", "\\_")
        lines.append(
            f"{clean_name} & {id_m['pairwise_preference_accuracy']*100:.2f}\\% & "
            f"{id_m['length_matched_accuracy']*100:.2f}\\% & "
            f"{id_m['spearman_rho_vs_sonar_tdi']:.4f} & "
            f"{id_m['macro_f1']:.4f} & "
            f"{oo_m['pairwise_preference_accuracy']*100:.2f}\\% & "
            f"{oo_m['spearman_rho_vs_sonar_tdi']:.4f} \\\\"
        )

    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{l}{\\textit{Panel B: Objective Comparison Ablation}} \\\\")
    for name, entry in report.get("ablation_2_objective_comparison", {}).items():
        id_m = entry["in_domain_codenet_test"]
        oo_m = entry["ood_swe_refactor_test"]
        clean_name = name.replace("_", "\\_")
        lines.append(
            f"{clean_name} & {id_m['pairwise_preference_accuracy']*100:.2f}\\% & "
            f"{id_m['length_matched_accuracy']*100:.2f}\\% & "
            f"{id_m['spearman_rho_vs_sonar_tdi']:.4f} & "
            f"{id_m['macro_f1']:.4f} & "
            f"{oo_m['pairwise_preference_accuracy']*100:.2f}\\% & "
            f"{oo_m['spearman_rho_vs_sonar_tdi']:.4f} \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def save_publication_artifacts(
    report: Dict[str, Any],
    md_path: str | Path = "data/phase4_publication_tables.md",
    tex_path: str | Path = "data/phase4_publication_tables.tex",
) -> None:
    md_file = Path(md_path)
    tex_file = Path(tex_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.write_text(generate_markdown_report(report), encoding="utf-8")
    tex_file.write_text(generate_latex_tables(report), encoding="utf-8")
