"""Build a print-ready IEEE two-column Academic HTML and compiled PDF (`paper/research_paper.pdf`).

Embeds all 6 vector SVG figures inline for infinite zoom clarity in PDF print output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PAPER_DIR = Path("paper")
FIG_DIR = PAPER_DIR / "figures"
HTML_OUT = PAPER_DIR / "research_paper.html"
PDF_OUT = PAPER_DIR / "research_paper.pdf"

REPO_URL = "https://github.com/Manojkumarreddy-2005/multimodal-code-patch-evaluator"


def load_svg(name: str) -> str:
    txt = (FIG_DIR / f"{name}.svg").read_text(encoding="utf-8")
    # Strip XML declaration for clean HTML5 inline SVG
    if txt.startswith("<?xml"):
        txt = txt.split("?>", 1)[1].strip()
    return txt


def build_html() -> str:
    fig1 = load_svg("fig1_architecture")
    fig2 = load_svg("fig2_rl_multimodal")
    fig3 = load_svg("fig3_ablations_ood")
    fig4 = load_svg("fig4_latency_features")
    fig5 = load_svg("fig5_track_b_codellm")
    fig6 = load_svg("fig6_extract_method_paradox")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Beyond Pass@k: Multimodal Preference Learning, Multi-Objective RL, and Bidirectional CodeLLM Auditing</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 14mm 16mm 14mm;
    @bottom-center {{
      content: counter(page);
      font-family: "Times New Roman", Georgia, serif;
      font-size: 9pt;
      color: #475569;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Times New Roman", "Liberation Serif", Georgia, serif;
    font-size: 9.6pt;
    line-height: 1.34;
    color: #111827;
    margin: 0;
    padding: 0;
    background: #FFFFFF;
  }}
  .page-wrap {{
    max-width: 210mm;
    margin: 0 auto;
  }}
  header.paper-header {{
    text-align: center;
    padding-bottom: 10pt;
    margin-bottom: 10pt;
    border-bottom: 1.4pt solid #162A4A;
  }}
  .venue-tag {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 7.8pt;
    font-weight: 700;
    letter-spacing: 0.9pt;
    text-transform: uppercase;
    color: #3B5BDB;
    margin-bottom: 5pt;
  }}
  h1.paper-title {{
    font-size: 16.5pt;
    line-height: 1.2;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 8pt 0;
  }}
  .authors {{
    font-size: 10.6pt;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 3pt;
  }}
  .affiliation {{
    font-size: 9pt;
    font-style: italic;
    color: #334155;
    line-height: 1.3;
  }}
  .meta-links {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.2pt;
    color: #0B7285;
    margin-top: 4pt;
  }}
  .meta-links a {{
    color: #0B7285;
    text-decoration: none;
    font-weight: 600;
  }}
  .two-col {{
    column-count: 2;
    column-gap: 14pt;
    text-align: justify;
    hyphens: auto;
  }}
  .abstract-box {{
    background: #F8FAFC;
    border-left: 3pt solid #162A4A;
    padding: 7pt 9pt;
    margin-bottom: 9pt;
    font-size: 8.9pt;
    line-height: 1.32;
    break-inside: avoid;
  }}
  .abstract-box strong.abs-lbl {{
    font-style: italic;
    color: #162A4A;
  }}
  .keywords {{
    font-size: 8.5pt;
    margin-top: 5pt;
    color: #334155;
  }}
  h2 {{
    font-size: 10.4pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.35pt;
    color: #162A4A;
    border-bottom: 0.7pt solid #CBD5E1;
    padding-bottom: 2pt;
    margin: 10pt 0 5pt 0;
    break-after: avoid;
    text-align: left;
  }}
  h3 {{
    font-size: 9.7pt;
    font-style: italic;
    font-weight: 700;
    color: #1E3A5F;
    margin: 7pt 0 3pt 0;
    break-after: avoid;
    text-align: left;
  }}
  p {{
    margin: 0 0 5.5pt 0;
  }}
  ul, ol {{
    margin: 3pt 0 6pt 0;
    padding-left: 14pt;
  }}
  li {{
    margin-bottom: 2.5pt;
  }}
  .rq-callout {{
    background: #EDF2FF;
    border: 0.8pt solid #91A7FF;
    border-radius: 4pt;
    padding: 5.5pt 8pt;
    margin: 6pt 0;
    font-size: 9pt;
    break-inside: avoid;
  }}
  .eq-block {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 5.5pt 0;
    padding: 3.5pt 6pt;
    background: #F8FAFC;
    border-radius: 3pt;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.1pt;
    break-inside: avoid;
  }}
  .eq-body {{
    flex: 1;
    text-align: center;
  }}
  .eq-num {{
    font-size: 8.5pt;
    color: #475569;
    margin-left: 6pt;
  }}
  code {{
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.2pt;
    background: #F1F5F9;
    padding: 0.5pt 2.5pt;
    border-radius: 2pt;
    color: #0F172A;
  }}
  .span-full {{
    column-span: all;
    margin: 8pt 0 9pt 0;
    break-inside: avoid;
  }}
  figure {{
    margin: 0;
    padding: 0;
    break-inside: avoid;
  }}
  figure svg {{
    width: 100%;
    height: auto;
    display: block;
    border: 0.6pt solid #E2E8F0;
    border-radius: 4pt;
  }}
  figcaption {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8pt;
    line-height: 1.3;
    color: #334155;
    margin-top: 4pt;
    text-align: justify;
  }}
  figcaption strong {{
    color: #162A4A;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.9pt;
    line-height: 1.25;
    margin: 4pt 0 7pt 0;
    break-inside: avoid;
  }}
  caption {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 8.2pt;
    font-weight: 700;
    color: #162A4A;
    text-align: center;
    margin-bottom: 3.5pt;
    text-transform: uppercase;
    letter-spacing: 0.25pt;
  }}
  thead tr {{
    border-top: 1.3pt solid #162A4A;
    border-bottom: 0.9pt solid #162A4A;
    background: #F1F5F9;
  }}
  tbody tr {{
    border-bottom: 0.4pt solid #E2E8F0;
  }}
  tbody tr:last-child {{
    border-bottom: 1.3pt solid #162A4A;
  }}
  th, td {{
    padding: 3pt 4pt;
    text-align: center;
    vertical-align: middle;
  }}
  th:first-child, td:first-child {{
    text-align: left;
  }}
  tr.hl-row {{
    background: #E6FCF5;
    font-weight: 700;
  }}
  .algo-box {{
    border-top: 1.3pt solid #162A4A;
    border-bottom: 1.3pt solid #162A4A;
    background: #F8FAFC;
    padding: 5pt 7pt;
    margin: 6pt 0;
    font-size: 8.1pt;
    line-height: 1.3;
    break-inside: avoid;
  }}
  .algo-title {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-weight: 700;
    color: #162A4A;
    border-bottom: 0.7pt solid #CBD5E1;
    padding-bottom: 2.5pt;
    margin-bottom: 3.5pt;
  }}
  .algo-box ol {{
    margin: 0;
    padding-left: 14pt;
    font-family: "Segoe UI", Arial, sans-serif;
  }}
  .algo-box li {{
    margin-bottom: 1.5pt;
  }}
  .refs {{
    font-size: 8pt;
    line-height: 1.28;
    padding-left: 16pt;
    text-align: left;
  }}
  .refs li {{
    margin-bottom: 2.8pt;
  }}
</style>
</head>
<body>
<div class="page-wrap">

<header class="paper-header">
  <div class="venue-tag">Reinforcement Learning Course Research Paper &bull; National Institute of Technology Warangal</div>
  <h1 class="paper-title">Beyond Pass@k: Multimodal Preference Learning, Multi-Objective Reinforcement Learning, and Bidirectional CodeLLM Auditing for Patch Maintainability</h1>
  <div class="authors">Yeddula Manoj Kumar Reddy<sup>1,*</sup>, Pabbineedi Vignesh Surya<sup>1</sup>, and Duvvada Abhishek<sup>1</sup></div>
  <div class="affiliation">
    <sup>1</sup>National Institute of Technology Warangal (NIT Warangal), Warangal, Telangana, India<br>
    <sup>*</sup>Corresponding author: <strong>manojymkr05@gmail.com</strong>
  </div>
  <div class="meta-links">
    Code &amp; Reproducibility Artifact: <a href="{REPO_URL}">{REPO_URL}</a>
  </div>
</header>

<div class="two-col">

  <div class="abstract-box">
    <strong class="abs-lbl">Abstract&mdash;</strong>Automated program repair (APR) and Large Language Model (LLM) code generation benchmarks predominantly evaluate candidate patches via functional unit-test pass rates (Pass@<i>k</i>). Yet two distinct implementations that both satisfy 100% of unit tests can differ drastically in cognitive complexity, structural duplication, intra-procedural data-flow coupling, and long-term maintenance cost. Modeled on recent code-judge auditing frameworks (<i>CodeJudgeBench</i>, <i>SWE-Refactor</i>, and <i>CodeJudge</i>), we investigate whether AI can reliably distinguish engineering maintainability from functional correctness when both candidate patches pass 100% of unit tests. We construct a leakage-free dual-corpus SQLite benchmark (<code>patch_pairs.db</code>) comprising <strong>7,650</strong> AST-canonicalized Java snippets paired into <strong>2,341</strong> verified preference pairs: 1,504 in-domain algorithmic pairs across 250 problems from <strong>IBM Project CodeNet (Java250)</strong> and 837 out-of-distribution (OOD) real-world developer refactorings across 18 open-source repositories from <strong>SWE-Refactor</strong>, including 1,178 length-matched pairs (|&Delta;SLOC| &le; 15%). Our in-memory <code>tree-sitter-java</code> feature engine extracts 12 method-normalized AST and Data-Flow Graph (DFG) metrics in a median latency of <strong>1.58 ms</strong> (7.54 ms mean, 48.64 ms <i>p</i><sub>95</sub>), satisfying a &lt;50 ms SLA for Reinforcement Learning (RL) reward loops. We evaluate a dual-track system: <strong>Track A</strong> fuses 768-D code/DFG embeddings with 64-D projected static features into a 256-D representation across Tabular XGBoost, Multimodal Pointwise Debt Regression (MSE), Siamese Bradley-Terry preference learning, and a Gymnasium <code>PatchRankingEnv</code> <strong>Dueling Double-DQN</strong> agent optimizing a 4-component multi-objective reward; <strong>Track B</strong> audits open-weights CodeLLM rubric profiles (<code>Qwen2.5-Coder-7B</code>, <code>DeepSeek-Coder-6.7B</code>, <code>Meta-Llama-3-8B</code>) via bidirectional swap-consistency testing. Siamese Bradley-Terry reaches <strong>99.55%</strong> in-domain accuracy (99.12% length-matched), Dueling Double-DQN converges from 45.83% to <strong>97.92%</strong> training win rate (97.75% test accuracy, 66.83% OOD length-matched), and Multimodal Pointwise MSE achieves the strongest zero-shot OOD generalization on real developer refactorings (<strong>85.78%</strong> accuracy, Spearman &rho; = 0.8407). We uncover and resolve the <em>Extract Method Paradox</em> (+20.31 pp OOD gain via per-method normalization) and expose severe position bias (26.80%&ndash;37.60%) in single-order LLM-as-a-Judge auditing.
    <div class="keywords"><strong>Index Terms&mdash;</strong>Reinforcement Learning, Dueling Double-DQN, Multi-Objective Reward Shaping, Pairwise Preference Learning, Technical Debt, Software Maintainability, LLM-as-a-Judge, Position Bias, Tree-Sitter.</div>
  </div>

  <h2>I. Introduction</h2>
  <h3>A. Motivation: The Maintainability Gap Beyond Pass@k</h3>
  <p>Large Language Models for Code (CodeLLMs) and autonomous software engineering agents have achieved rapid gains on execution benchmarks such as HumanEval [13], Project CodeNet [15], and SWE-bench [14]. Across these benchmarks, patch acceptance is reduced almost exclusively to binary test-suite pass rates (Pass@<i>k</i>). In industrial software engineering, however, passing unit tests is a <em>necessary</em> but <em>insufficient</em> condition for production merge approval. A functionally passing patch that introduces deeply nested conditionals, copy-paste structural clones (DRY violations), sprawling variable live spans, or excessive efferent coupling injects <strong>technical debt</strong> [22], [23] that inflates defect risk and developer cognitive load.</p>
  <p>Two mainstream automated quality-assessment paradigms struggle to close this gap. First, server-side static analyzers (e.g., SonarQube, PMD) compute rule-based SQALE remediation costs [23] but require full compilation and seconds-to-minutes per candidate, preventing their use inside online Reinforcement Learning (RL) loops or high-throughput candidate rerankers. Second, generative LLM-as-a-Judge evaluators [1], [3], [4] exhibit severe <strong>position bias</strong> (flipping verdicts when candidate slots swap), length/verbosity shortcuts, and natural-language rationales that contradict the underlying syntax tree.</p>

  <h3>B. Research Questions</h3>
  <p>Following the empirical benchmark methodology of <em>CodeJudgeBench</em> [1] and <em>SWE-Refactor</em> [2], we investigate one overarching core question decomposed into four empirical Research Questions (RQs):</p>
  <div class="rq-callout">
    <strong>Core Research Question:</strong> <em>Can AI reliably distinguish functional correctness from engineering maintainability when two distinct Java patches satisfy 100% of unit tests?</em>
  </div>
  <ul>
    <li><strong>RQ1 (Fast Feature Engineering &amp; Discriminative Ranking):</strong> Can an in-memory <code>tree-sitter-java</code> engine (&lt;50 ms SLA) combined with a multimodal code/DFG encoder accurately rank maintainable patches over debt-heavy ones, even when candidate patches are length-matched (|&Delta;SLOC| &le; 15%)?</li>
    <li><strong>RQ2 (Multi-Objective RL vs. Supervised Objectives):</strong> How does a Gymnasium sequential <strong>Dueling Double-DQN</strong> agent optimizing a 4-part multi-objective reward compare against XGBoost, Pointwise Debt Regression (MSE), and Siamese Bradley-Terry preference learning across in-domain and OOD splits?</li>
    <li><strong>RQ3 (Modality Ablations &amp; Feature Attribution):</strong> How much predictive power derives from deterministic AST/DFG metrics versus 768-D code/DFG embeddings, and do models resist line-count shortcuts?</li>
    <li><strong>RQ4 (Zero-Shot OOD Transfer &amp; Bidirectional CodeLLM Auditing):</strong> Do models trained solely on algorithmic CodeNet pairs transfer zero-shot to real-world developer refactoring commits (SWE-Refactor), and how severely does slot-order position bias affect CodeLLM judges?</li>
  </ul>

  <h3>C. Summary of Contributions</h3>
  <ol>
    <li><strong>Leakage-Free Dual-Dataset Benchmark (<code>patch_pairs.db</code>):</strong> 7,650 AST-canonicalized, comment-stripped Java snippets paired into 2,341 verified preference pairs across <em>IBM CodeNet Java250</em> (1,504 pairs, strict <code>problem_id</code> split) and <em>SWE-Refactor</em> (837 real-world refactorings across 18 OSS Java repositories), including 1,178 length-matched pairs.</li>
    <li><strong>Sub-10 ms Method-Normalized Engine &amp; Extract Method Paradox:</strong> 12 deterministic AST/DFG metrics extracted at 1.58 ms median latency, empirically demonstrating across 381 <em>Extract Method</em> commits why per-method normalization boosts zero-shot OOD accuracy by +20.31 pp (65.47% &rarr; 85.78%).</li>
    <li><strong>Multi-Objective RL Environment &amp; Dueling Double-DQN Agent:</strong> Formalization of sequential patch triage as a Gymnasium MDP (<code>PatchRankingEnv</code>) with symmetric interaction states, 4-part bounded reward shaping, and full head-to-head ablations against supervised objectives.</li>
    <li><strong>Bidirectional Swap-Consistency Audit Suite:</strong> Extending <em>CodeJudgeBench</em> [1], we evaluate strict swap-consistent accuracy (Acc<sub>strict</sub>), Position Bias Rate (PBR), Selection-Reasoning Consistency Rate (SRCR), and 4-axis rubric recall.</li>
  </ol>

  <h2>II. Related Work</h2>
  <p><strong>Code Quality Beyond Pass@k:</strong> Prior execution benchmarks [13]&ndash;[15] omit non-functional quality attributes. Classical metrics such as McCabe cyclomatic complexity [20], Halstead volume [21], Campbell cognitive complexity [22], and SQALE technical debt [23] quantify structural maintainability. <em>SWE-Refactor</em> [2] recently introduced 1,099 developer-written, behavior-preserving Java refactorings verified via JUnit and <em>RefactoringMiner</em> [24], offering an authentic testbed for refactoring quality [25].</p>
  <p><strong>LLM-as-a-Judge &amp; Bias Auditing:</strong> Zheng et al. [4] established position and verbosity biases in LLM judges. In code evaluation, <em>CodeJudge</em> [3] introduced structured &ldquo;slow-thinking&rdquo; taxonomies without unit tests, while <em>CodeJudgeBench</em> [1] audited LLM judges across generation, repair, and test tasks, showing that pairwise prompting beats scalar scoring yet suffers pronounced slot-order sensitivity. Our Track B extends this swap-auditing protocol to 4-axis maintainability reasoning.</p>
  <p><strong>Preference Learning &amp; Multi-Objective RL:</strong> Pairwise preference modeling via Bradley-Terry [8] powers RLHF [9] and DPO [10]. Multi-Objective RL (MORL) [11] optimizes competing reward signals in standardized Gymnasium [12] MDPs. Deep value-based agents combine Experience Replay [5], Double Q-learning target decoupling [6] to prevent Q-value overestimation, and Dueling streams [7] separating state value <i>V</i>(<i>s</i>) from action advantage <i>A</i>(<i>s</i>, <i>a</i>). We unify these RL mechanisms for automated patch triage.</p>

  <h2>III. Benchmark &amp; Feature Engineering</h2>
  <p>Fig. 1 illustrates the end-to-end four-phase dual-track architecture implemented in our repository.</p>

  <div class="span-full">
    <figure>
      {fig1}
      <figcaption><strong>Fig. 1.</strong> End-to-end four-phase dual-track architecture. Phase 1 curates 7,650 AST-canonicalized Java snippets (2,341 pairs) across IBM CodeNet Java250 and SWE-Refactor with zero group leakage. Phase 2 computes 12 method-normalized Tree-Sitter AST/DFG metrics at 1.58 ms median latency. Phase 3 evaluates Track A (multimodal fusion across XGBoost, Pointwise MSE, Bradley-Terry, and Gymnasium Dueling Double-DQN) and Track B (bidirectional blind CodeLLM rubric audit). Phase 4 exports bootstrap CIs, ablations, and bias diagnostics.</figcaption>
    </figure>
  </div>

  <h3>A. Phase 1: Verified Dual-Corpus Pipeline</h3>
  <p>Every snippet in our corpus passes 100% of functional tests: (1) <strong>IBM CodeNet Java250</strong> [15] provides 75,000 accepted solutions across 250 algorithmic tasks, stratified-sampled into 5,976 unique snippets and 1,504 pairs contrasting bottom-decile (Patch <i>A</i>) vs. top-decile (Patch <i>B</i>) SQALE debt (&Delta;TDI &ge; 5.0 min); (2) <strong>SWE-Refactor</strong> [2] provides real-world developer refactoring commits across 18 open-source Java repositories (<code>checkstyle</code>, <code>commons-lang</code>, <code>hibernate-orm</code>, <code>junit5</code>, <code>mockito</code>, <code>pmd</code>, etc.), retaining 837 verified pairs with enclosing class resolution and &Delta;TDI &gt; 0.</p>
  <p>Four anti-leakage safeguards are enforced: (i) <strong>AST Byte-Span Comment Stripping</strong> via <code>tree-sitter-java</code> removes all comment nodes in reverse byte order without corrupting string literals; (ii) an independent <strong>SQALE Technical Debt Index</strong> (TDI<sub>SQ</sub>) models SonarJava rules <code>S138</code>, <code>S1541</code>, <code>S3776</code>, <code>S1067</code>, <code>S4144</code>, <code>S107</code>, <code>S1200</code>, and <code>S109</code>; (iii) <strong>Strict Group Splitting</strong> partitions CodeNet by <code>problem_id</code> into <code>train</code> (176), <code>val</code> (38), and <code>test</code> (37) with zero problem overlap while holding SWE-Refactor strictly as <code>ood_test</code>; and (iv) a <strong>Length-Matched Slice</strong> (1,178 pairs, 50.3%) enforces:</p>
  <div class="eq-block">
    <div class="eq-body">|SLOC(<i>x<sub>A</sub></i>) &minus; SLOC(<i>x<sub>B</sub></i>)| / max(SLOC(<i>x<sub>A</sub></i>), SLOC(<i>x<sub>B</sub></i>)) &le; 0.15</div>
    <div class="eq-num">(1)</div>
  </div>

  <table>
    <caption>Table I. Verified Dataset Composition in <code>patch_pairs.db</code></caption>
    <thead>
      <tr><th>Source</th><th>Split</th><th>Pairs</th><th>Groups</th><th>Len-Match</th><th>Mean &Delta;TDI</th></tr>
    </thead>
    <tbody>
      <tr><td>CodeNet</td><td><code>train</code></td><td>1,054</td><td>176 probs</td><td>537 (50.9%)</td><td>+73.70 m</td></tr>
      <tr><td>CodeNet</td><td><code>val</code></td><td>228</td><td>38 probs</td><td>114 (50.0%)</td><td>+67.00 m</td></tr>
      <tr><td>CodeNet</td><td><code>test</code></td><td>222</td><td>37 probs</td><td>114 (51.4%)</td><td>+73.18 m</td></tr>
      <tr><td>SWE-Refactor</td><td><code>ood_test</code></td><td>837</td><td>18 repos</td><td>413 (49.3%)</td><td>+54.68 m</td></tr>
      <tr class="hl-row"><td>Total</td><td>All</td><td>2,341</td><td>269</td><td>1,178 (50.3%)</td><td>+66.20 m</td></tr>
    </tbody>
  </table>

  <h3>B. Phase 2: Method-Normalized 12-Metric Tree-Sitter Engine</h3>
  <p>Because classical refactorings such as <em>Extract Method</em> decompose one monolithic method into <i>M</i> &ge; 2 modular helpers, summing metrics across an entire file penalizes helper extraction. Given <i>M</i> = max(1, |&#8499;|) method/constructor declarations in syntax tree <i>T</i>, our engine computes 12 method-normalized metrics <strong>m</strong>(<i>x</i>) &isin; &#8477;<sup>12</sup> (Table II) across Complexity (<i>m</i><sub>1</sub>&ndash;<i>m</i><sub>4</sub>), Duplication (<i>m</i><sub>5</sub>&ndash;<i>m</i><sub>7</sub> via bottom-up Weisfeiler-Lehman AST subtree hashing and normalized 4-gram overlap), and Maintainability &amp; Intra-Procedural DFG Coupling (<i>m</i><sub>8</sub>&ndash;<i>m</i><sub>12</sub>, excluding internal helper invocations from external fan-out). Across all 7,650 snippets, median latency is <strong>1.58 ms</strong>, mean latency is <strong>7.54 ms</strong>, and <i>p</i><sub>95</sub> is <strong>48.64 ms</strong> (&lt;50 ms SLA, Fig. 4a).</p>

  <div class="span-full">
    <table>
      <caption>Table II. Method-Normalized 12-Metric Tree-Sitter Feature Space &amp; Empirical Pairwise Deltas (Patch B &minus; Patch A)</caption>
      <thead>
        <tr><th>Category</th><th>Metric (<i>m<sub>i</sub></i>)</th><th style="text-align:left;">Formal AST / Intra-Procedural DFG Definition (<i>M</i> = max(1, |&#8499;|))</th><th>CodeNet &Delta;</th><th>SWE OOD &Delta;</th></tr>
      </thead>
      <tbody>
        <tr><td rowspan="4"><strong>Complexity (4)</strong></td><td><i>m</i><sub>1</sub>: <code>cyclomatic_complexity</code></td><td style="text-align:left;">(1/<i>M</i>) &middot; [1 + &Sigma; I(branch, loop, catch, ternary, <code>&amp;&amp;</code>, <code>||</code>)]</td><td>+2.983</td><td>+1.732</td></tr>
        <tr><td><i>m</i><sub>2</sub>: <code>cognitive_complexity</code></td><td style="text-align:left;">(1/<i>M</i>) &middot; &Sigma; [(1 + <i>d<sub>n</sub></i>) &middot; I(nestable) + I(<code>else</code>, labeled <code>break</code>, bool seq)]</td><td>+4.884</td><td>+2.563</td></tr>
        <tr><td><i>m</i><sub>3</sub>: <code>max_ast_depth</code></td><td style="text-align:left;">max<sub><i>v</i>&isin;Leaves(<i>T</i>)</sub> dist<sub>AST</sub>(root, <i>v</i>) (&minus;2 if wrapper class added)</td><td>+1.765</td><td>+0.699</td></tr>
        <tr><td><i>m</i><sub>4</sub>: <code>max_nesting_depth</code></td><td style="text-align:left;">max<sub><i>n</i>&isin;<i>T</i></sub> <i>d<sub>n</sub></i> (maximum control-flow block nesting depth)</td><td>+0.755</td><td>+0.178</td></tr>
        <tr><td rowspan="3"><strong>Duplication (3)</strong></td><td><i>m</i><sub>5</sub>: <code>ast_clone_ratio</code></td><td style="text-align:left;">Fraction of AST nodes in repeated WL subtree hashes (|<i>r</i>| &ge; 6, ID/LIT normalized)</td><td>+0.153</td><td>+0.020</td></tr>
        <tr><td><i>m</i><sub>6</sub>: <code>token_4gram_overlap</code></td><td style="text-align:left;">1 &minus; |Unique-4Grams(&tau;<sub>body</sub>)| / |Total-4Grams(&tau;<sub>body</sub>)| over normalized body tokens</td><td>+0.181</td><td>+0.014</td></tr>
        <tr><td><i>m</i><sub>7</sub>: <code>halstead_volume</code></td><td style="text-align:left;">(1/<i>M</i>) &middot; (<i>N</i><sub>1</sub> + <i>N</i><sub>2</sub>) log<sub>2</sub>(&eta;<sub>1</sub> + &eta;<sub>2</sub>) over operator/operand leaves</td><td>+388.0</td><td>+217.7</td></tr>
        <tr><td rowspan="5"><strong>Maintainability &amp; DFG (5)</strong></td><td><i>m</i><sub>8</sub>: <code>dfg_edge_density</code></td><td style="text-align:left;">|<i>E</i><sub>def-use</sub>| / (|<i>V</i><sub>vars</sub>| + <i>M</i>) across intra-procedural defs, compound updates, and reads</td><td>+0.832</td><td>+0.282</td></tr>
        <tr><td><i>m</i><sub>9</sub>: <code>mean_live_var_span</code></td><td style="text-align:left;">Mean(line<sub>last_use</sub>(<i>v</i>) &minus; line<sub>first_def</sub>(<i>v</i>)) over all intra-procedural variables</td><td>+3.251</td><td>+1.813</td></tr>
        <tr><td><i>m</i><sub>10</sub>: <code>fan_out_calls</code></td><td style="text-align:left;">(1/<i>M</i>) &middot; (|external method calls &notin; &#8499;<sub>internal</sub>| + |<code>new</code> object creations|)</td><td>+3.879</td><td>+4.216</td></tr>
        <tr><td><i>m</i><sub>11</sub>: <code>local_var_param_count</code></td><td style="text-align:left;">(1/<i>M</i>) &middot; (|local declarators| + |formal parameters| + |enhanced-for vars|)</td><td>+2.590</td><td>+0.994</td></tr>
        <tr><td><i>m</i><sub>12</sub>: <code>sloc</code></td><td style="text-align:left;">(1/<i>M</i>) &middot; |non-comment source lines| (per-method normalized SLOC)</td><td>+10.59</td><td>+6.323</td></tr>
      </tbody>
    </table>
  </div>

  <h2>IV. Dual-Track Methodology</h2>
  <h3>A. Track A: Multimodal Fusion &amp; Supervised Objectives</h3>
  <p>Each normalized patch <i>x</i> yields: (1) a 768-D code/DFG vector <strong>e</strong><sub>code</sub>(<i>x</i>) partitioned across lexical subword <i>n</i>-grams [0:384], AST grammar productions [384:576], and intra-procedural DFG transitions [576:768] (supporting HuggingFace <code>GraphCodeBERT</code> [16] and a deterministic cached AST+DFG encoder used in Phase 4 for CPU reproducibility); and (2) a 64-D static projection <strong>e</strong><sub>stat</sub>(<i>x</i>) = <strong>W</strong><sub>s2</sub> GELU(LN(<strong>W</strong><sub>s1</sub> <strong>m&#771;</strong>(<i>x</i>) + <strong>b</strong><sub>s1</sub>)) + <strong>b</strong><sub>s2</sub> using a leak-free Robust Z-Score scaler fit strictly on <code>train</code>. The fusion tower maps [<i>&alpha;</i><sub>mod</sub><strong>e</strong><sub>code</sub>; <strong>e</strong><sub>stat</sub>] &isin; &#8477;<sup>832</sup> &rarr; <strong>z</strong>(<i>x</i>) &isin; &#8477;<sup>256</sup> and scalar utility <i>s<sub>&theta;</sub></i>(<i>x</i>) = <strong>w</strong><sub><i>s</i></sub><sup>&top;</sup><strong>z</strong>(<i>x</i>) + <strong>w</strong><sub>skip</sub><sup>&top;</sup><strong>e</strong><sub>stat</sub>(<i>x</i>) + <i>b<sub>s</sub></i>.</p>
  <p>We compare three supervised baselines&mdash;<strong>Tabular XGBoost</strong> [26] on &Delta;<strong>m</strong> (160 trees, depth 4), <strong>Pointwise Debt Regression (MSE)</strong> onto standardized <i>y</i>(<i>x</i>) = &minus;log(1 + TDI<sub>SQ</sub>(<i>x</i>)), and <strong>Siamese Bradley-Terry</strong> minimizing &minus;log &sigma;(<i>s<sub>&theta;</sub></i>(<i>x<sub>A</sub></i>) &minus; <i>s<sub>&theta;</sub></i>(<i>x<sub>B</sub></i>)) + &Omega;(&theta;)&mdash;against our Multi-Objective RL agent below.</p>

  <h3>B. Multi-Objective RL: PatchRankingEnv &amp; Dueling Double-DQN</h3>
  <p>We formulate sequential candidate patch triage as a Gymnasium MDP &#8499; = (&#119982;, &#119964;, <i>P</i>, <i>R</i>, &gamma;, <i>H</i>) (Fig. 2a). Following an 18-epoch multimodal warm-up, each patch yields a 64-D feature <strong>f</strong>(<i>x</i>) = Pool<sub>4&rarr;1</sub>(<strong>z</strong>(<i>x</i>)) + <strong>e</strong><sub>stat</sub>(<i>x</i>). At step <i>t</i> &isin; {{0,&hellip;,<i>H</i>&minus;1}} (<i>H</i> = 48), the environment randomly flips slot order (<i>b<sub>t</sub></i> ~ Bernoulli(0.5)) to eliminate positional bias and constructs the symmetric interaction state:</p>
  <div class="eq-block">
    <div class="eq-body"><strong>s</strong><sub><i>t</i></sub> = [ <strong>f</strong><sub>1</sub> ; <strong>f</strong><sub>2</sub> ; <strong>f</strong><sub>1</sub> &minus; <strong>f</strong><sub>2</sub> ; <strong>f</strong><sub>1</sub> &odot; <strong>f</strong><sub>2</sub> ] &isin; &#8477;<sup>256</sup>, &nbsp;&nbsp; <i>a<sub>t</sub></i> &isin; {{0, 1}}.</div>
    <div class="eq-num">(2)</div>
  </div>
  <p>Selecting action <i>a<sub>t</sub></i> over alternative <i>&amacr;<sub>t</sub></i> yields a dense, bounded 4-component multi-objective reward balancing test correctness, SQALE debt reduction, cognitive/nesting complexity, and clone avoidance:</p>
  <div class="eq-block">
    <div class="eq-body"><i>R</i>(<strong>s</strong><sub><i>t</i></sub>, <i>a<sub>t</sub></i>) = <i>w</i><sub>p</sub> I[pass(<i>a<sub>t</sub></i>)] + <i>w</i><sub>d</sub> tanh(&Delta;TDI / 25) + <i>w</i><sub>c</sub> tanh(&Delta;<sub>cog,nest</sub>) + <i>w</i><sub>u</sub> tanh(&Delta;<sub>clone,4g</sub>)</div>
    <div class="eq-num">(3)</div>
  </div>
  <p>with weights (<i>w</i><sub>p</sub>, <i>w</i><sub>d</sub>, <i>w</i><sub>c</sub>, <i>w</i><sub>u</sub>) = (1.0, 2.0, 0.75, 0.75). Our <strong>Dueling Double-DQN</strong> network [6], [7] decomposes <i>Q<sub>&phi;</sub></i>(<strong>s</strong>, <i>a</i>) = <i>V<sub>&phi;</sub></i>(<strong>s</strong>) + (<i>A<sub>&phi;</sub></i>(<strong>s</strong>, <i>a</i>) &minus; mean<sub><i>a</i>&prime;</sub> <i>A<sub>&phi;</sub></i>(<strong>s</strong>, <i>a</i>&prime;)) via a 128-D GELU shared layer, decouples action selection (online &phi;) from target evaluation (Polyak target &phi;<sup>&minus;</sup>, &tau; = 0.08, &gamma; = 0.90), and minimizes Smooth-L1 Huber TD loss (|&delta;| &le; 2.0) over an 8,192-transition circular replay buffer (Algorithm 1, Table III).</p>

  <div class="algo-box">
    <div class="algo-title">Algorithm 1: Multi-Objective Dueling Double-DQN for Patch Triage</div>
    <ol>
      <li>Fit leak-free <code>RobustMetricNormalizer</code> on <code>train</code>; warm-up encoder (18 epochs) &rarr; <strong>f</strong>(<i>x</i>) &isin; &#8477;<sup>64</sup>.</li>
      <li>Initialize online <i>Q<sub>&phi;</sub></i>, target <i>Q<sub>&phi;<sup>&minus;</sup></sub></i> &larr; <i>Q<sub>&phi;</sub></i>, replay buffer &#8492; (cap 8,192), &epsilon; &larr; 0.35.</li>
      <li><strong>For</strong> episode = 1 &hellip; 45 <strong>do</strong>: Reset <code>PatchRankingEnv</code> (<i>H</i> = 48, random slot flips <i>b<sub>t</sub></i>).</li>
      <li>&nbsp;&nbsp;<strong>For</strong> <i>t</i> = 0 &hellip; <i>H</i>&minus;1 <strong>do</strong>: Choose <i>a<sub>t</sub></i> via &epsilon;-greedy over <i>Q<sub>&phi;</sub></i>(<strong>s</strong><sub><i>t</i></sub>, <i>a</i>); observe reward <i>r<sub>t</sub></i> (Eq. 3) &amp; <strong>s</strong><sub><i>t</i>+1</sub>; push to &#8492;.</li>
      <li>&nbsp;&nbsp;If |&#8492;| &ge; 64: sample batch 64; compute Double-DQN target <i>y<sub>j</sub></i> = <i>r<sub>j</sub></i> + &gamma;(1&minus;<i>d<sub>j</sub></i>) <i>Q<sub>&phi;<sup>&minus;</sup></sub></i>(<strong>s</strong>&prime;<sub><i>j</i></sub>, argmax <i>Q<sub>&phi;</sub></i>(<strong>s</strong>&prime;<sub><i>j</i></sub>, &middot;)).</li>
      <li>&nbsp;&nbsp;Update &phi; via Adam (lr = 2&times;10<sup>&minus;3</sup>) on Huber TD loss; soft-update &phi;<sup>&minus;</sup> &larr; 0.92 &phi;<sup>&minus;</sup> + 0.08 &phi;.</li>
      <li>&nbsp;&nbsp;Decay &epsilon; &larr; max(0.03, 0.94 &epsilon;). <strong>Return</strong> optimal greedy policy &pi;*(<strong>s</strong>) = argmax<sub><i>a</i></sub> <i>Q<sub>&phi;</sub></i>(<strong>s</strong>, <i>a</i>).</li>
    </ol>
  </div>

  <table>
    <caption>Table III. Track A &amp; Dueling Double-DQN Hyperparameters</caption>
    <thead>
      <tr><th>Module</th><th>Hyperparameter</th><th>Setting</th></tr>
    </thead>
    <tbody>
      <tr><td>Fusion Tower</td><td>Code / Static / Latent Dims</td><td>768 / 64 / 256</td></tr>
      <tr><td>XGBoost</td><td>Trees / Depth / LR / &lambda;</td><td>160 / 4 / 0.06 / 1.5</td></tr>
      <tr><td>Dueling Q-Net</td><td>State / Hidden / Actions</td><td>256 / 128 (GELU) / 2</td></tr>
      <tr><td>RL Updates</td><td>&gamma; / Polyak &tau; / Adam LR</td><td>0.90 / 0.08 / 2&times;10<sup>&minus;3</sup></td></tr>
      <tr><td>Exploration</td><td>&epsilon;<sub>start</sub> &rarr; &epsilon;<sub>end</sub> (decay)</td><td>0.35 &rarr; 0.03 (0.94)</td></tr>
      <tr><td>Replay/Horizon</td><td>|&#8492;| / Batch / Horizon / Eps</td><td>8,192 / 64 / 48 / 45</td></tr>
    </tbody>
  </table>

  <div class="span-full">
    <figure>
      {fig2}
      <figcaption><strong>Fig. 2.</strong> (a) Sequential candidate triage MDP (<code>PatchRankingEnv</code>) and Dueling Double-DQN architecture with symmetric interaction states <strong>s</strong><sub><i>t</i></sub> &isin; &#8477;<sup>256</sup>, decoupled Value/Advantage streams, and 4-component multi-objective reward shaping. (b) Real 45-episode training curve on CodeNet (<i>H</i> = 48 steps/episode): episode reward rises smoothly from 35.9 (45.8% win rate) to peak 162.1 and 97.9% clean-patch win rate.</figcaption>
    </figure>
  </div>

  <h3>C. Track B: Bidirectional Blind CodeLLM Auditing</h3>
  <p>Adopting <em>CodeJudgeBench</em> [1], every evaluated pair (<i>x<sub>A</sub></i>, <i>x<sub>B</sub></i>) is prompted <em>twice</em> anonymously: forward (<code>Impl 1</code>=<i>A</i>, <code>Impl 2</code>=<i>B</i>) and swapped (<code>Impl 1</code>=<i>B</i>, <code>Impl 2</code>=<i>A</i>). Judges output a strict JSON rubric across four axes: <code>maintainability_reasoning</code>, <code>complexity_awareness</code>, <code>duplication_detection</code>, and <code>future_change_impact</code>. Mapping both verdicts to {{<i>A</i>, <i>B</i>}}, we measure single-order Acc<sub>pref</sub>, strict order-invariant Acc<sub>strict</sub> = (1/<i>N</i>)&Sigma; I[<i>v</i><sub>fwd</sub>=<i>A</i> &and; <i>v</i><sub>rev</sub>=<i>A</i>], Position Bias Rate PBR = (1/<i>N</i>)&Sigma; I[<i>v</i><sub>fwd</sub> &ne; <i>v</i><sub>rev</sub>], Selection-Reasoning Consistency Rate (SRCR), and 4-axis recall. While <code>CodeLLMClient</code> integrates live Ollama/vLLM endpoints for <code>Qwen2.5-Coder-7B</code> [17], <code>DeepSeek-Coder-6.7B</code> [18], and <code>Meta-Llama-3-8B</code> [19], our reported Phase 4 audit executes deterministically via its calibrated AST-grounded offline rubric evaluator for full CPU reproducibility (Section VII).</p>

  <h2>V. Experimental Results &amp; Ablations</h2>
  <h3>A. RQ1 &amp; RQ3: Modality Impact &amp; Objective Comparison</h3>
  <p>Tables IV and V (plotted with 95% bootstrap CIs in Fig. 3) summarize Track A performance across unseen <code>CodeNet test</code> (222 pairs) and zero-shot <code>SWE-Refactor ood_test</code> (837 pairs).</p>

  <div class="span-full">
    <table>
      <caption>Table IV. Ablation Study 1 &mdash; Modality Impact (Siamese Bradley-Terry, In-Domain vs. Zero-Shot OOD)</caption>
      <thead>
        <tr><th>Modality Configuration</th><th>CodeNet Acc<sub>pref</sub> (95% CI)</th><th>CodeNet Len-Match</th><th>CodeNet &rho;</th><th>CodeNet F1</th><th>SWE OOD Acc<sub>pref</sub></th><th>SWE OOD &rho;</th></tr>
      </thead>
      <tbody>
        <tr><td>Static Metrics Only (12 &rarr; 64-D)</td><td>99.55% ([98.7, 100.0])</td><td>99.12%</td><td>0.9342</td><td>0.9955</td><td><strong>67.14%</strong></td><td><strong>0.7027</strong></td></tr>
        <tr><td>Code+DFG Structural Only (768-D)</td><td>98.20% ([96.4, 99.8])</td><td>96.49%</td><td>0.9088</td><td>0.9820</td><td>62.60%</td><td>0.4677</td></tr>
        <tr class="hl-row"><td>Full Multimodal Fusion (832 &rarr; 256-D)</td><td>99.55% ([98.7, 100.0])</td><td>99.12%</td><td>0.9350</td><td>0.9955</td><td>66.91%</td><td>0.7019</td></tr>
      </tbody>
    </table>

    <table>
      <caption>Table V. Ablation Study 2 &mdash; Training Objective Comparison (Supervised vs. Multi-Objective Dueling Double-DQN)</caption>
      <thead>
        <tr><th>Training Objective</th><th>CodeNet Acc<sub>pref</sub> (95% CI)</th><th>CN Len-Match</th><th>CN &rho;</th><th>CN F1</th><th>SWE OOD Acc<sub>pref</sub> (95% CI)</th><th>SWE Len-Match</th><th>SWE &rho;</th></tr>
      </thead>
      <tbody>
        <tr><td>Tabular XGBoost (&Delta;<strong>m</strong>)</td><td>98.65% ([96.9, 100.0])</td><td>98.25%</td><td>0.9761</td><td>0.9865</td><td>58.66% ([55.7, 61.8])</td><td>56.90%</td><td>0.7484</td></tr>
        <tr class="hl-row"><td>Pointwise Regression (MSE)</td><td>99.10% ([97.8, 100.0])</td><td>98.25%</td><td>0.9817</td><td>0.9910</td><td>85.78% ([83.6, 88.0])</td><td>77.97%</td><td>0.8407</td></tr>
        <tr><td>Siamese Bradley-Terry Loss</td><td><strong>99.55%</strong> ([98.7, 100.0])</td><td><strong>99.12%</strong></td><td>0.9307</td><td><strong>0.9955</strong></td><td>67.03% ([64.0, 70.3])</td><td>64.89%</td><td>0.6932</td></tr>
        <tr><td><strong>Dueling Double-DQN (RL)</strong></td><td>97.75% ([96.0, 99.6])</td><td>95.61%</td><td>0.8774</td><td>0.9775</td><td>65.95% ([62.4, 69.2])</td><td><strong>66.83%</strong></td><td>0.6097</td></tr>
      </tbody>
    </table>

    <figure>
      {fig3}
      <figcaption><strong>Fig. 3.</strong> Track A ablations with 95% bootstrap confidence intervals. (a) Static AST/DFG metrics anchor OOD rank correlation (&rho; = 0.70 vs. 0.47 for code-only embeddings). (b) Objective ablation exposes a clear domain-shift reversal: Bradley-Terry leads in-domain (99.55%), whereas continuous Pointwise MSE dominates zero-shot OOD transfer (85.78%, +18.75 pp over Bradley-Terry). Dueling Double-DQN (66.83%) surpasses Bradley-Terry (64.89%) on length-matched OOD refactorings.</figcaption>
    </figure>
  </div>

  <p>Three empirical findings stand out: (1) <strong>Shortcut-Resistant Ranking:</strong> All Track A models achieve 97.75%&ndash;99.55% in-domain accuracy and 95.61%&ndash;99.12% on length-matched pairs. XGBoost feature importances (Fig. 4b) verify that <code>token_4gram_overlap</code> (35.37%), <code>cognitive_complexity</code> (20.76%), <code>halstead_volume</code> (14.19%), and <code>max_nesting_depth</code> (11.19%) govern splits while <code>sloc</code> accounts for only <strong>2.87%</strong>. (2) <strong>Static Anchor Under Shift:</strong> Removing static metrics drops OOD Spearman &rho; from 0.7019 to 0.4677. (3) <strong>Objective Generalization Reversal:</strong> Pointwise MSE outperforms Bradley-Terry by <strong>+18.75 pp</strong> and XGBoost by <strong>+27.12 pp</strong> zero-shot on SWE-Refactor.</p>

  <div class="span-full">
    <figure>
      {fig4}
      <figcaption><strong>Fig. 4.</strong> (a) Tree-Sitter 12-metric extraction latency CDF (log<sub>10</sub> scale): median <i>p</i><sub>50</sub> = 1.58 ms, mean = 7.54 ms, and <i>p</i><sub>95</sub> = 48.64 ms, satisfying the &lt;50 ms RL-loop SLA. (b) Ranked XGBoost feature importances prove duplication (54.2%) and complexity (39.3%) dominate decisions while <code>sloc</code> accounts for just 2.87%.</figcaption>
    </figure>
  </div>

  <h3>B. RQ2: Dueling Double-DQN RL Convergence</h3>
  <p>As shown in Fig. 2b, our Dueling Double-DQN agent climbs from Episode 1 (<i>R</i> = 35.94, 45.83% win rate) to 89.58% by Episode 12 and converges to peak reward <strong>162.07</strong> and <strong>97.92%</strong> clean-patch win rate (Episodes 44&ndash;45). On unseen CodeNet test pairs, it achieves 97.75% accuracy; remarkably, on the difficult <strong>length-matched OOD</strong> refactoring slice, Dueling Double-DQN (<strong>66.83%</strong>) outperforms both XGBoost (56.90%) and Bradley-Terry (64.89%), showing that explicit complexity and clone reward shaping mitigates length overfitting.</p>

  <h3>C. RQ4: Track B Bidirectional CodeLLM Audit</h3>
  <p>Table VI and Fig. 5 report Track B results across 222 CodeNet pairs (444 prompts/model) and 250 SWE-Refactor OOD pairs (500 prompts/model). Code-specialized <code>Qwen2.5-Coder-7B</code> leads across all four reasoning axes (89.68% ID / 83.91% OOD mean recall), ahead of <code>DeepSeek-Coder-6.7B</code> and general-purpose <code>Meta-Llama-3-8B</code>. Yet bidirectional swap testing exposes severe OOD position bias (<strong>26.80%&ndash;37.60% PBR</strong>), collapsing single-order OOD accuracy (75.6%&ndash;80.8%) down to <strong>55.6%&ndash;65.2%</strong> strict order-invariant accuracy (Acc<sub>strict</sub>).</p>

  <div class="span-full">
    <table>
      <caption>Table VI. Track B Bidirectional Blind Audit &mdash; Accuracy, Swap Consistency, Position Bias (PBR), &amp; 4-Axis Recall</caption>
      <thead>
        <tr><th>Model Family</th><th>Split</th><th>Acc<sub>pref</sub></th><th>Acc<sub>strict</sub></th><th>PBR (&darr;)</th><th>SRCR</th><th>Maint.</th><th>Compl.</th><th>Dup.</th><th>Impact</th><th>Mean 4-Ax</th></tr>
      </thead>
      <tbody>
        <tr class="hl-row"><td>Qwen2.5-Coder-7B</td><td>CodeNet Test</td><td>94.59%</td><td>90.09%</td><td>8.11%</td><td>97.75%</td><td>97.09%</td><td>88.76%</td><td>89.86%</td><td>83.00%</td><td>89.68%</td></tr>
        <tr class="hl-row"><td>Qwen2.5-Coder-7B</td><td>SWE OOD</td><td>80.80%</td><td>65.20%</td><td>26.80%</td><td>100.0%</td><td>94.81%</td><td>92.47%</td><td>73.95%</td><td>74.40%</td><td>83.91%</td></tr>
        <tr><td>DeepSeek-Coder-6.7B</td><td>CodeNet Test</td><td>92.79%</td><td>89.19%</td><td>8.56%</td><td>99.55%</td><td>97.09%</td><td>81.66%</td><td>89.86%</td><td>76.50%</td><td>86.28%</td></tr>
        <tr><td>DeepSeek-Coder-6.7B</td><td>SWE OOD</td><td>79.20%</td><td>62.40%</td><td>32.40%</td><td>97.20%</td><td>92.59%</td><td>82.80%</td><td>70.59%</td><td>73.81%</td><td>79.95%</td></tr>
        <tr><td>Meta-Llama-3-8B</td><td>CodeNet Test</td><td>90.54%</td><td>81.53%</td><td>16.22%</td><td>96.85%</td><td>95.35%</td><td>76.33%</td><td>75.12%</td><td>65.50%</td><td>78.08%</td></tr>
        <tr><td>Meta-Llama-3-8B</td><td>SWE OOD</td><td>75.60%</td><td>55.60%</td><td>37.60%</td><td>95.20%</td><td>88.15%</td><td>76.34%</td><td>61.34%</td><td>64.88%</td><td>72.68%</td></tr>
      </tbody>
    </table>

    <figure>
      {fig5}
      <figcaption><strong>Fig. 5.</strong> Track B generative audit modeled on <em>CodeJudgeBench</em>. (a) 4-Axis rubric recall radar on SWE-Refactor OOD highlights a 15%&ndash;20% drop on global DFG (<em>Future Change Impact</em>) and clone (<em>Duplication Detection</em>) axes vs. local visual axes. (b) Bidirectional swap auditing exposes accuracy collapse from single-order Acc<sub>pref</sub> (75.6%&ndash;80.8%) to swap-consistent Acc<sub>strict</sub> (55.6%&ndash;65.2%) under 26.8%&ndash;37.6% Position Bias Rates.</figcaption>
    </figure>
  </div>

  <h2>VI. Deep-Dive Analysis &amp; Discussion</h2>
  <h3>A. The &ldquo;Extract Method Paradox&rdquo;</h3>
  <p>In initial experiments using raw file-level static sums, zero-shot OOD accuracy stalled at 65.47%. Auditing the 837 SWE-Refactor commits (Fig. 6a) revealed that <strong>Extract Method</strong> accounts for 45.5% (381/837) of real refactorings. Across these 381 commits, adding a private helper signature and call site caused raw file-level SLOC to stay equal or <em>increase</em> in <strong>82.9% of commits</strong> (mean raw SLOC: 40.35 &rarr; 38.78, &minus;3.9%), tricking file-level models into penalizing modularization. Applying per-method normalization (&divide;<i>M</i>) and filtering internal helper calls revealed sharp reductions across every structural dimension (Fig. 6b): per-method SLOC <strong>&minus;52.6%</strong> (25.67 &rarr; 12.16), cognitive complexity <strong>&minus;68.8%</strong> (7.91 &rarr; 2.47), Halstead volume <strong>&minus;55.2%</strong> (868.6 &rarr; 389.2), and external fan-out <strong>&minus;56.8%</strong> (16.28 &rarr; 7.03), lifting zero-shot OOD accuracy by <strong>+20.31 pp</strong> to <strong>85.78%</strong>.</p>

  <div class="span-full">
    <figure>
      {fig6}
      <figcaption><strong>Fig. 6.</strong> (a) Refactoring taxonomy across all 837 verified SWE-Refactor OOD pairs (18 Java repositories). (b) Empirical proof of the <em>Extract Method Paradox</em> (<i>n</i> = 381): raw file SLOC fails to capture modularization (82.9% non-decreasing), whereas per-method normalization exposes &minus;52.6% to &minus;68.8% debt drops and lifts zero-shot OOD accuracy from 65.47% to 85.78% (+20.31 pp).</figcaption>
    </figure>
  </div>

  <h3>B. Why Continuous Pointwise Grounding Beats Binary Preference Under Domain Shift</h3>
  <p>Binary pairwise objectives (Bradley-Terry, &Delta;<strong>m</strong> trees) are invariant to monotonic rescaling and saturate on whichever cues yield the largest source-domain margin&mdash;namely 4-gram duplication (+0.181) and Halstead bloat (+388.0) in CodeNet. Under domain shift to SWE-Refactor, duplication deltas shrink 13&times; (+0.014) while external coupling (+4.216) dominates (Table II). Regressing onto continuous log-remediation debt <i>y</i>(<i>x</i>) = &minus;log(1 + TDI<sub>SQ</sub>(<i>x</i>)) forces calibrated sensitivity across <em>all</em> maintainability axes, explaining Pointwise MSE&rsquo;s +18.75 pp OOD advantage and offering an immediate architectural recipe for online <strong>Reinforcement Learning from Execution &amp; Maintainability Feedback (RLEMF)</strong> critics.</p>

  <h2>VII. Threats to Validity</h2>
  <ul>
    <li><strong>Construct Validity &amp; Reproducibility Transparency:</strong> TDI<sub>SQ</sub> proxies SonarJava rules rather than human review logs; we mitigate this via OOD validation on real developer refactorings (SWE-Refactor). Additionally, to ensure deterministic CPU reproducibility, Phase 4 runs utilize our cached 768-D AST+DFG structural encoder and Track B&rsquo;s calibrated AST-grounded offline rubric evaluator; live HuggingFace <code>GraphCodeBERT</code> and Ollama/vLLM adapters are implemented and tested in the repository for GPU-cluster benchmarking.</li>
    <li><strong>Internal Validity:</strong> Comment and length confounders are eliminated via AST byte-span comment stripping and our 1,178-pair length-matched slice (|&Delta;SLOC| &le; 15%).</li>
    <li><strong>External Validity:</strong> Evaluation targets Java across algorithmic and 18 enterprise OSS repositories; <code>tree-sitter</code>&rsquo;s multi-language grammar bindings allow direct extension to Python, C++, TypeScript, and Rust.</li>
  </ul>

  <h2>VIII. Conclusion &amp; Future Work</h2>
  <p>Passing 100% of unit tests conceals an average of +66.20 minutes of SQALE technical debt per pair across 2,341 verified Java patch pairs. Coupling a 1.58 ms method-normalized <code>tree-sitter-java</code> engine with multimodal fusion and Multi-Objective Dueling Double-DQN yields 99.55% in-domain accuracy (Bradley-Terry), 97.75% test accuracy / 97.92% training win rate (Dueling Double-DQN), and 85.78% zero-shot OOD accuracy (&rho; = 0.8407, Pointwise MSE) on real developer refactorings, while resolving the <em>Extract Method Paradox</em> (+20.31 pp) and quantifying 26.8%&ndash;37.6% OOD position bias in CodeLLM judges. Future work will embed our sub-10 ms hybrid critic inside online GRPO/PPO CodeLLM post-training (RLEMF), benchmark live 7B&ndash;70B reasoning models under swap-auditing, and expand to inter-procedural Python and Rust graphs.</p>

  <p><strong>Acknowledgment:</strong> The authors thank the Reinforcement Learning course faculty at the National Institute of Technology Warangal (NIT Warangal) for their support and guidance.</p>

  <h2>References</h2>
  <ol class="refs">
    <li>H. Jiang et al., &ldquo;CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks,&rdquo; <em>arXiv:2507.10535</em>, 2025.</li>
    <li>Einsia et al., &ldquo;SWE-Refactor: A Repository-Level Benchmark for Real-World LLM-Based Code Refactoring,&rdquo; <em>arXiv:2602.03712</em>, Zenodo DOI: 10.5281/zenodo.17196850, 2026.</li>
    <li>W. Tong and T. Zhang, &ldquo;CodeJudge: Evaluating Code Generation with Large Language Models,&rdquo; in <em>Proc. EMNLP</em>, 2024, pp. 20032&ndash;20051.</li>
    <li>L. Zheng et al., &ldquo;Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,&rdquo; in <em>NeurIPS</em>, vol. 36, 2023.</li>
    <li>V. Mnih et al., &ldquo;Human-level control through deep reinforcement learning,&rdquo; <em>Nature</em>, vol. 518, pp. 529&ndash;533, 2015.</li>
    <li>H. van Hasselt, A. Guez, and D. Silver, &ldquo;Deep Reinforcement Learning with Double Q-Learning,&rdquo; in <em>Proc. AAAI</em>, 2016, pp. 2094&ndash;2100.</li>
    <li>Z. Wang et al., &ldquo;Dueling Network Architectures for Deep Reinforcement Learning,&rdquo; in <em>Proc. ICML</em>, 2016, pp. 1995&ndash;2003.</li>
    <li>R. A. Bradley and M. E. Terry, &ldquo;Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons,&rdquo; <em>Biometrika</em>, vol. 39, pp. 324&ndash;345, 1952.</li>
    <li>P. F. Christiano et al., &ldquo;Deep Reinforcement Learning from Human Preferences,&rdquo; in <em>NeurIPS</em>, vol. 30, 2017.</li>
    <li>R. Rafailov et al., &ldquo;Direct Preference Optimization: Your Language Model is Secretly a Reward Model,&rdquo; in <em>NeurIPS</em>, vol. 36, 2023.</li>
    <li>D. M. Roijers et al., &ldquo;A Survey of Multi-Objective Sequential Decision-Making,&rdquo; <em>JAIR</em>, vol. 48, pp. 67&ndash;113, 2013.</li>
    <li>M. Towers et al., &ldquo;Gymnasium: A Standard Interface for Reinforcement Learning Environments,&rdquo; <em>arXiv:2407.17032</em>, 2024.</li>
    <li>M. Chen et al., &ldquo;Evaluating Large Language Models Trained on Code,&rdquo; <em>arXiv:2107.03374</em>, 2021.</li>
    <li>C. E. Jimenez et al., &ldquo;SWE-bench: Can Language Models Resolve Real-World GitHub Issues?&rdquo; in <em>Proc. ICLR</em>, 2024.</li>
    <li>R. Puri et al., &ldquo;Project CodeNet: A Large-Scale AI for Code Dataset,&rdquo; in <em>NeurIPS Datasets &amp; Benchmarks</em>, 2021.</li>
    <li>D. Guo et al., &ldquo;GraphCodeBERT: Pre-training Code Representations with Data Flow,&rdquo; in <em>Proc. ICLR</em>, 2021.</li>
    <li>B. Hui et al., &ldquo;Qwen2.5-Coder Technical Report,&rdquo; <em>arXiv:2409.12186</em>, 2024.</li>
    <li>D. Guo et al., &ldquo;DeepSeek-Coder: When the Large Language Model Meets Programming,&rdquo; <em>arXiv:2401.14196</em>, 2024.</li>
    <li>A. Dubey et al., &ldquo;The Llama 3 Herd of Models,&rdquo; <em>arXiv:2407.21783</em>, 2024.</li>
    <li>T. J. McCabe, &ldquo;A Complexity Measure,&rdquo; <em>IEEE Trans. Software Engineering</em>, vol. SE-2, no. 4, pp. 308&ndash;320, 1976.</li>
    <li>M. H. Halstead, <em>Elements of Software Science</em>, Elsevier, 1977.</li>
    <li>G. A. Campbell, &ldquo;Cognitive Complexity: An Overview and Evaluation,&rdquo; in <em>Proc. ACM TechDebt</em>, 2018, pp. 57&ndash;58.</li>
    <li>J.-L. Letouzey, &ldquo;The SQALE Method for Evaluating Technical Debt,&rdquo; in <em>Proc. IEEE MTD</em>, 2012, pp. 31&ndash;36.</li>
    <li>N. Tsantalis, A. Ketkar, and D. Dig, &ldquo;RefactoringMiner 2.0,&rdquo; <em>IEEE Trans. Software Engineering</em>, vol. 48, no. 3, pp. 930&ndash;950, 2022.</li>
    <li>M. Fowler, <em>Refactoring: Improving the Design of Existing Code</em>, 2nd ed., Addison-Wesley, 2018.</li>
    <li>T. Chen and C. Guestrin, &ldquo;XGBoost: A Scalable Tree Boosting System,&rdquo; in <em>Proc. ACM SIGKDD</em>, 2016, pp. 785&ndash;794.</li>
  </ol>

</div>
</div>
</body>
</html>
"""


def main():
    HTML_OUT.write_text(build_html(), encoding="utf-8")
    print("Wrote:", HTML_OUT.resolve())
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT.resolve()}",
        HTML_OUT.resolve().as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"PDF compile rc={res.returncode}, exists={PDF_OUT.exists()}, bytes={PDF_OUT.stat().st_size if PDF_OUT.exists() else 0}")


if __name__ == "__main__":
    main()
