# Multimodal Code Patch Quality Evaluator

**Authors:** Yeddula Manoj Kumar Reddy (`manojymkr05@gmail.com`), Pabbineedi Vignesh Surya, Duvvada Abhishek — *Reinforcement Learning Course Project, National Institute of Technology Warangal (NIT Warangal)*

**Research Question:** *Can AI distinguish functional correctness from engineering quality when two distinct solutions satisfy 100% of unit tests?*

**Research Paper & Publication Artifacts:**
- Compiled IEEE Two-Column PDF: [`paper/research_paper.pdf`](paper/research_paper.pdf)
- Overleaf-Ready IEEEtran LaTeX Source: [`paper/main.tex`](paper/main.tex) + [`paper/references.bib`](paper/references.bib)
- Reproducible Vector Diagrams (SVG & High-DPI PNG): [`paper/figures/`](paper/figures/) (regenerable via `python scripts/generate_paper_figures.py` and `python scripts/build_paper_html_pdf.py`)

This repository implements an end-to-end multimodal reinforcement learning, discriminative ranking, and generative CodeLLM evaluation pipeline that quantitatively ranks maintainable Java patches over complex/duplicated ones while testing whether open-weights CodeLLMs can accurately reason about technical debt.

---

## Architecture Overview

1. **Phase 1: Public Dataset Collection & SQLite Pipeline (`src/data/`)**
   - **IBM Project CodeNet (`Java250` Subset)**: 250 problem folders (`75,000` accepted Java solutions passing 100% of test cases), stratified-sampled and paired (`Patch A = cleanest` vs. `Patch B = highest technical debt`) with strict group-level splits (`train` / `val` / `test`) by `problem_id`.
   - **SWE-Refactor (`DOI: 10.5281/zenodo.17196850`)**: Real-world Java refactorings (`sourceCodeAfterRefactoring` vs. `sourceCodeBeforeRefactoring`) across 18 open-source Java repositories verified by RefactoringMiner 3.0 and JUnit tests, evaluated as an Out-of-Distribution (`ood_test`) generalization benchmark.
   - **AST Normalization & Storage**: Strips all comments via `tree-sitter-java` byte spans without corrupting string literals and stores `7,650` snippets and `2,341` pairs (`1,178` length-matched pairs with $|\Delta \text{SLOC}| \le 15\%$) in `data/patch_pairs.db`.

2. **Phase 2: Fast Tree-Sitter Feature Engineering (`src/features/`)**
   - Extracts **12 deterministic static metrics** in memory (`1.58 ms` median, `7.54 ms` mean per patch):
     - **Complexity (4)**: `cyclomatic_complexity`, `cognitive_complexity`, `max_ast_depth`, `max_nesting_depth`
     - **Duplication & Redundancy (3)**: `ast_clone_ratio` (Merkle/WL subtree hashing), `token_4gram_overlap`, `halstead_volume`
     - **Maintainability & Coupling (5)**: `dfg_edge_density` (intra-procedural Def-Use chains), `mean_live_var_span`, `fan_out_calls`, `local_var_param_count`, `sloc`

3. **Phase 3: Dual-Track System Architecture (`src/track_a/`, `src/track_b/`)**
   - **Track A (Discriminative Ranking & Multi-Objective RL)**:
     - Fuses $768\text{-dim}$ GraphCodeBERT source/DFG embeddings with a $64\text{-dim}$ static metric projection into a $256\text{-dim}$ multimodal representation.
     - Compares Tabular XGBoost ($\Delta m$), Pointwise Regression (MSE), Bradley-Terry Pairwise Preference (`-log σ(s_A - s_B)`), and a Gymnasium (`PatchRankingEnv`) Dueling Double-DQN agent optimizing a 4-part multi-objective reward.
   - **Track B (Generative CodeLLM Engineering Reasoning)**:
     - Evaluates `Qwen2.5-Coder-7B-Instruct`, `DeepSeek-Coder-6.7B-Instruct`, and `Meta-Llama-3-8B-Instruct` via bidirectional blind prompting (`(A, B)` and `(B, A)`) on a structured 4-dimension JSON rubric (`maintainability_reasoning`, `complexity_awareness`, `duplication_detection`, `future_change_impact`).

4. **Phase 4: Evaluation Metrics & Ablation Suite (`src/evaluation/`)**
   - Computes Pairwise Preference Accuracy ($\text{Acc}_{\text{pref}}$) with 95% bootstrap CIs, Length-Matched Accuracy, Spearman's $\rho$ vs. SonarQube TDI, Macro-F1, Strict Swap-Consistent Accuracy ($\text{Acc}_{\text{strict}}$), Position Bias Rate ($\text{PBR}$), Selection-Reasoning Consistency Rate ($\text{SRCR}$), and 4-Axis Dimension Recall.
   - Exports publication-ready Markdown (`data/phase4_publication_tables.md`) and LaTeX (`data/phase4_publication_tables.tex`) tables.

---

## Quickstart & CLI Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual phases
python -m scripts.run_phase1 --config configs/default_config.yaml
python -m scripts.run_phase2 --config configs/default_config.yaml
python -m scripts.run_phase3 --config configs/default_config.yaml
python -m scripts.run_phase4 --config configs/default_config.yaml

# Or run the complete pipeline (Phases 1 -> 4)
python -m scripts.run_pipeline --skip-phase1

# Run the complete unit and integration test suite
python -m pytest -v
```
