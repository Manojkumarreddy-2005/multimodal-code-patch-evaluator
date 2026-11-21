# Phase 4 Experimental Results & Ablation Tables

## Table 1: Ablation Study 1 — Modality Impact (In-Domain vs. Out-of-Distribution)

| Modality Configuration | CodeNet $\text{Acc}_{\text{pref}}$ (95% CI) | CodeNet Len-Matched | CodeNet Spearman $\rho$ | CodeNet Macro-F1 | SWE-Refactor OOD $\text{Acc}_{\text{pref}}$ | SWE-Refactor Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static_Metrics_Only_64D** | `99.55%` (`[98.7, 100.0]`) | `99.12%` | `0.9342` | `0.9955` | `67.14%` | `0.7027` |
| **GraphCodeBERT_Only_768D** | `98.20%` (`[96.4, 99.8]`) | `96.49%` | `0.9088` | `0.9820` | `62.60%` | `0.4677` |
| **Full_Multimodal_Fusion_832D_to_256D** | `99.55%` (`[98.7, 100.0]`) | `99.12%` | `0.9350` | `0.9955` | `66.91%` | `0.7019` |

## Table 2: Ablation Study 2 — Objective Comparison (Track A Ranking Models)

| Training Objective | CodeNet $\text{Acc}_{\text{pref}}$ | CodeNet Len-Matched | CodeNet Spearman $\rho$ | CodeNet Macro-F1 | SWE-Refactor OOD $\text{Acc}_{\text{pref}}$ | SWE-Refactor Len-Matched | SWE-Refactor Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tabular_XGBoost_Delta** | `98.65%` | `98.25%` | `0.9761` | `0.9865` | `58.66%` | `56.90%` | `0.7484` |
| **Pointwise_Regression_MSE** | `99.10%` | `98.25%` | `0.9817` | `0.9910` | `85.78%` | `77.97%` | `0.8407` |
| **Bradley_Terry_Pairwise_Loss** | `99.55%` | `99.12%` | `0.9307` | `0.9955` | `67.03%` | `64.89%` | `0.6932` |
| **DQN_Multi_Objective_RL** | `97.75%` | `95.61%` | `0.8774` | `0.9775` | `65.95%` | `66.83%` | `0.6097` |

## Table 3: Track B Generative CodeLLM Engineering Reasoning & 4-Dimension Recall

| Open-Weights CodeLLM | Benchmark Split | $\text{Acc}_{\text{pref}}$ | Swap-Consistent $\text{Acc}_{\text{strict}}$ | Position Bias Rate | Selection-Reasoning Consistency | Maintainability Recall | Complexity Recall | Duplication Recall | Change Impact Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-Coder-7B-Instruct** | CodeNet Test | `94.59%` | `90.09%` | `8.11%` | `97.75%` | `97.09%` | `88.76%` | `89.86%` | `83.00%` |
| **Qwen2.5-Coder-7B-Instruct** | SWE-Refactor OOD | `80.80%` | `65.20%` | `26.80%` | `100.00%` | `94.81%` | `92.47%` | `73.95%` | `74.40%` |
| **DeepSeek-Coder-6.7B-Instruct** | CodeNet Test | `92.79%` | `89.19%` | `8.56%` | `99.55%` | `97.09%` | `81.66%` | `89.86%` | `76.50%` |
| **DeepSeek-Coder-6.7B-Instruct** | SWE-Refactor OOD | `79.20%` | `62.40%` | `32.40%` | `97.20%` | `92.59%` | `82.80%` | `70.59%` | `73.81%` |
| **Meta-Llama-3-8B-Instruct** | CodeNet Test | `90.54%` | `81.53%` | `16.22%` | `96.85%` | `95.35%` | `76.33%` | `75.12%` | `65.50%` |
| **Meta-Llama-3-8B-Instruct** | SWE-Refactor OOD | `75.60%` | `55.60%` | `37.60%` | `95.20%` | `88.15%` | `76.34%` | `61.34%` | `64.88%` |
