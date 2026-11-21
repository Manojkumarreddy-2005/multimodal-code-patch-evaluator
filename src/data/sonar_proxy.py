"""SonarQube / SQALE-aligned Technical Debt Index (TDI) & Delta Calculator.

Computes an independent technical debt remediation score (in minutes) modeled on
SonarJava rules (S138, S107, S3776, S1067, S4144, S109, S1200) so that:
1. CodeNet Java250 solutions can be paired (Patch A = cleanest, Patch B = highest debt)
   without circular feature leakage into the 12 raw Tree-Sitter features.
2. Phase 1 stores pre-computed technical debt deltas in `patch_pairs.db`.
3. Phase 4 can evaluate Spearman's rank correlation (rho) against Sonar debt indexes
   and ground-truth 4-axis reasoning triggers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser


JAVA_LANGUAGE = Language(tsjava.language())


@dataclass(frozen=True)
class SonarDebtProfile:
    """SQALE / SonarJava-aligned technical debt profile for a single Java patch."""

    sonar_tdi_minutes: float
    sloc: int
    branch_count: int
    max_control_nesting: int
    boolean_operator_chains: int
    duplicate_block_count: int
    normalized_4gram_overlap: float
    parameter_count: int
    local_variable_count: int
    external_call_count: int
    magic_literal_count: int
    smell_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SonarDebtCalculator:
    """Computes SonarQube-aligned remediation cost (TDI) and pairwise debt deltas."""

    CONTROL_NESTING_NODES = frozenset(
        {
            "if_statement",
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
            "switch_expression",
            "switch_statement",
            "catch_clause",
            "ternary_expression",
        }
    )

    BRANCH_NODES = frozenset(
        {
            "if_statement",
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
            "switch_block_statement_group",
            "switch_rule",
            "catch_clause",
            "ternary_expression",
        }
    )

    STATEMENT_BLOCK_CANDIDATES = frozenset(
        {
            "if_statement",
            "for_statement",
            "while_statement",
            "expression_statement",
            "local_variable_declaration",
            "return_statement",
        }
    )

    BENIGN_NUMBERS = frozenset({"0", "1", "-1", "2", "0L", "1L", "0.0", "1.0", "10", "100"})

    def __init__(self, config: Optional[Dict[str, float]] = None) -> None:
        self._parser = Parser(JAVA_LANGUAGE)
        cfg = config or {}
        self.base_sloc_threshold = int(cfg.get("base_method_sloc_threshold", 20))
        self.sloc_overage_rate = float(cfg.get("sloc_overage_minutes_per_line", 0.5))
        self.max_params_threshold = int(cfg.get("max_params_threshold", 4))
        self.extra_param_minutes = float(cfg.get("extra_param_minutes", 4.0))
        self.deep_nesting_threshold = int(cfg.get("deep_nesting_threshold", 2))
        self.extra_nesting_minutes = float(cfg.get("extra_nesting_minutes", 6.0))
        self.boolean_ops_threshold = int(cfg.get("complex_boolean_ops_threshold", 2))
        self.boolean_op_minutes = float(cfg.get("boolean_op_minutes", 3.0))
        self.duplicate_block_minutes = float(cfg.get("duplicate_statement_block_minutes", 8.0))
        self.magic_number_minutes = float(cfg.get("magic_number_minutes", 1.5))
        self.high_coupling_threshold = int(cfg.get("high_coupling_calls_threshold", 6))
        self.extra_coupling_minutes = float(cfg.get("extra_coupling_call_minutes", 1.5))

    def analyze(self, normalized_code: str) -> SonarDebtProfile:
        """Compute SonarQube-style technical debt breakdown on normalized Java code."""
        code_bytes = normalized_code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        if tree.root_node.has_error and "class " not in normalized_code and "interface " not in normalized_code:
            wrapped_bytes = b"class _Wrapper {\n" + code_bytes + b"\n}"
            wrapped_tree = self._parser.parse(wrapped_bytes)
            if not wrapped_tree.root_node.has_error:
                tree = wrapped_tree

        active_lines = set()
        branch_count = 0
        max_nesting = 0
        nesting_excess_sum = 0
        boolean_chains = 0
        param_count = 0
        local_var_count = 0
        external_calls = 0
        magic_literals = 0

        method_nodes: List[Node] = []
        statement_signatures: List[str] = []
        normalized_tokens: List[str] = []

        # DFS with control-flow nesting depth tracking
        stack: List[tuple[Node, int]] = [(tree.root_node, 0)]
        while stack:
            node, depth = stack.pop()
            ntype = node.type

            if ntype in ("method_declaration", "constructor_declaration"):
                method_nodes.append(node)

            if ntype in self.BRANCH_NODES:
                branch_count += 1

            next_depth = depth
            if ntype in self.CONTROL_NESTING_NODES:
                # Special handling: 'else if' does not increment nesting depth in Sonar S3776
                is_else_if = (
                    ntype == "if_statement"
                    and node.parent is not None
                    and node.parent.type == "if_statement"
                    and node.parent.child_by_field_name("alternative") == node
                )
                if not is_else_if:
                    next_depth = depth + 1
                    if next_depth > max_nesting:
                        max_nesting = next_depth
                    if next_depth > self.deep_nesting_threshold:
                        nesting_excess_sum += next_depth - self.deep_nesting_threshold

            if ntype == "binary_expression":
                op_node = node.child_by_field_name("operator")
                if op_node is not None and op_node.type in ("&&", "||"):
                    boolean_chains += 1

            elif ntype == "formal_parameter":
                param_count += 1

            elif ntype == "local_variable_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        local_var_count += 1

            elif ntype in ("method_invocation", "object_creation_expression"):
                external_calls += 1

            elif ntype in ("decimal_integer_literal", "decimal_floating_point_literal"):
                lit_text = node.text.decode("utf-8", errors="ignore") if node.text else ""
                if lit_text not in self.BENIGN_NUMBERS:
                    magic_literals += 1

            # Capture structural signatures of non-trivial statements for S4144 clone detection
            if ntype in self.STATEMENT_BLOCK_CANDIDATES and node.end_byte - node.start_byte >= 24:
                sig = self._canonical_subtree_signature(node)
                if len(sig) >= 5:
                    statement_signatures.append(sig)

            # Leaf token collection for SLOC and Type-2 4-gram duplication (only inside executable statements, not method headers)
            if node.child_count == 0 and ntype not in ("line_comment", "block_comment", "comment"):
                for row in range(node.start_point[0], node.end_point[0] + 1):
                    active_lines.add(row)
                if depth > 0 or (node.parent is not None and node.parent.type not in ("modifiers", "formal_parameter", "formal_parameters", "method_declaration")):
                    if ntype in ("identifier", "type_identifier"):
                        normalized_tokens.append("ID")
                    elif "literal" in ntype:
                        normalized_tokens.append("LIT")
                    else:
                        normalized_tokens.append(ntype)

            for child in reversed(node.children):
                stack.append((child, next_depth))

        sloc = sum(1 for ln in normalized_code.splitlines() if ln.strip())
        num_methods = max(1, len(method_nodes))

        # Per-method metrics (SonarJava rules S138, S107, S1541 apply per method, not summed across helpers)
        max_method_sloc = sloc
        max_method_params = param_count
        max_method_branches = branch_count
        max_method_vars = local_var_count
        if method_nodes:
            m_slocs: List[int] = []
            m_params: List[int] = []
            m_branches: List[int] = []
            m_vars: List[int] = []
            for mnode in method_nodes:
                m_slocs.append(max(1, mnode.end_point[0] - mnode.start_point[0] + 1))
                mp, mb, mv = 0, 0, 0
                mstack = [mnode]
                while mstack:
                    cur = mstack.pop()
                    if cur.type == "formal_parameter":
                        mp += 1
                    elif cur.type in self.BRANCH_NODES:
                        mb += 1
                    elif cur.type == "variable_declarator":
                        mv += 1
                    for ch in cur.children:
                        mstack.append(ch)
                m_params.append(mp)
                m_branches.append(mb)
                m_vars.append(mv)
            max_method_sloc = max(m_slocs)
            max_method_params = max(m_params)
            max_method_branches = max(m_branches)
            max_method_vars = max(m_vars)

        mean_method_sloc = sloc / num_methods
        mean_method_branches = branch_count / num_methods

        # Count duplicate statement blocks (S4144 / CPD proxy)
        sig_counts = Counter(statement_signatures)
        duplicate_blocks = sum(count - 1 for count in sig_counts.values() if count > 1)

        # Normalized token 4-gram overlap (only meaningful when snippet has sufficient body tokens)
        if len(normalized_tokens) >= 40:
            ngrams = [
                tuple(normalized_tokens[i : i + 4])
                for i in range(len(normalized_tokens) - 3)
            ]
            ngram_overlap = 1.0 - (len(set(ngrams)) / len(ngrams))
        else:
            ngram_overlap = 0.0

        # Compute SQALE remediation cost in minutes across per-method SonarJava rule categories
        smell_s138_length = (
            max(0.0, max_method_sloc - self.base_sloc_threshold) * self.sloc_overage_rate
            + max(0.0, mean_method_sloc - 10.0) * 0.45
        )
        smell_s1541_branches = (
            max(0, max_method_branches - 3) * 2.5
            + mean_method_branches * 1.5
        )
        smell_s3776_nesting = (nesting_excess_sum / num_methods) * self.extra_nesting_minutes
        smell_s1067_bool = (
            max(0, boolean_chains - self.boolean_ops_threshold) * self.boolean_op_minutes
            + (boolean_chains / num_methods) * 1.5
        )
        smell_s4144_clones = (
            duplicate_blocks * self.duplicate_block_minutes
            + max(0.0, ngram_overlap - 0.25) * 35.0
        )
        smell_s107_params = max(0, max_method_params - self.max_params_threshold) * self.extra_param_minutes
        smell_s1200_coupling = (
            max(0.0, (external_calls / num_methods) - self.high_coupling_threshold) * self.extra_coupling_minutes
            + max(0, max_method_vars - 4) * 1.25
        )
        smell_s109_magic = max(0.0, (magic_literals / num_methods) - 1.0) * self.magic_number_minutes

        breakdown = {
            "S138_method_length": round(smell_s138_length, 2),
            "S1541_cyclomatic_brain_overload": round(smell_s1541_branches, 2),
            "S3776_cognitive_nesting": round(smell_s3776_nesting, 2),
            "S1067_boolean_complexity": round(smell_s1067_bool, 2),
            "S4144_code_duplication": round(smell_s4144_clones, 2),
            "S107_parameter_overload": round(smell_s107_params, 2),
            "S1200_coupling_and_state": round(smell_s1200_coupling, 2),
            "S109_magic_literals": round(smell_s109_magic, 2),
        }

        # Per-method structural cost rewards modular helper extraction (lower mean_method_sloc & nesting)
        base_structural_cost = (
            0.30 * mean_method_sloc
            + 0.08 * max_method_sloc
            + 1.00 * mean_method_branches
            + 1.60 * (max_nesting / num_methods)
            + 4.50 * ngram_overlap
            + 0.40 * (local_var_count / num_methods)
        )
        total_tdi = round(sum(breakdown.values()) + base_structural_cost, 3)

        return SonarDebtProfile(
            sonar_tdi_minutes=total_tdi,
            sloc=sloc,
            branch_count=branch_count,
            max_control_nesting=max_nesting,
            boolean_operator_chains=boolean_chains,
            duplicate_block_count=duplicate_blocks,
            normalized_4gram_overlap=round(ngram_overlap, 4),
            parameter_count=max_method_params,
            local_variable_count=max_method_vars,
            external_call_count=external_calls,
            magic_literal_count=magic_literals,
            smell_breakdown=breakdown,
        )

    def _canonical_subtree_signature(self, node: Node) -> str:
        """Build an identifier-agnostic structural sequence for clone detection."""
        tokens: List[str] = []
        stack: List[Node] = [node]
        while stack:
            curr = stack.pop()
            if curr.child_count == 0:
                if curr.type in ("identifier", "type_identifier"):
                    tokens.append("ID")
                elif "literal" in curr.type:
                    tokens.append("LIT")
                else:
                    tokens.append(curr.type)
            else:
                for child in reversed(curr.children):
                    stack.append(child)
        return "|".join(tokens)

    @staticmethod
    def compute_pair_deltas(
        profile_a: SonarDebtProfile,
        profile_b: SonarDebtProfile,
        trigger_cfg: Optional[Dict[str, float]] = None,
    ) -> tuple[Dict[str, float], Dict[str, bool]]:
        """Compute (Patch B - Patch A) technical debt deltas and 4-axis rubric triggers.

        Patch A is the clean/refactored reference; Patch B is the complex/duplicated variant.
        Positive deltas indicate higher debt/complexity in Patch B.
        """
        tcfg = trigger_cfg or {}
        deltas = {
            "delta_sonar_tdi": round(profile_b.sonar_tdi_minutes - profile_a.sonar_tdi_minutes, 3),
            "delta_sloc": float(profile_b.sloc - profile_a.sloc),
            "delta_branch_count": float(profile_b.branch_count - profile_a.branch_count),
            "delta_max_nesting": float(
                profile_b.max_control_nesting - profile_a.max_control_nesting
            ),
            "delta_boolean_chains": float(
                profile_b.boolean_operator_chains - profile_a.boolean_operator_chains
            ),
            "delta_duplicate_blocks": float(
                profile_b.duplicate_block_count - profile_a.duplicate_block_count
            ),
            "delta_4gram_overlap": round(
                profile_b.normalized_4gram_overlap - profile_a.normalized_4gram_overlap, 4
            ),
            "delta_parameter_count": float(
                profile_b.parameter_count - profile_a.parameter_count
            ),
            "delta_local_var_count": float(
                profile_b.local_variable_count - profile_a.local_variable_count
            ),
            "delta_external_calls": float(
                profile_b.external_call_count - profile_a.external_call_count
            ),
            "delta_magic_literals": float(
                profile_b.magic_literal_count - profile_a.magic_literal_count
            ),
        }

        # Determine which of the 4 Track B engineering dimensions are structurally active
        triggered_dimensions = {
            "maintainability_reasoning": bool(
                deltas["delta_sloc"] >= tcfg.get("maintainability_sloc_delta", 6)
                or deltas["delta_local_var_count"] >= tcfg.get("maintainability_var_delta", 3)
                or deltas["delta_magic_literals"] >= 2
                or deltas["delta_sonar_tdi"] >= 8.0
            ),
            "complexity_awareness": bool(
                deltas["delta_max_nesting"] >= tcfg.get("complexity_nesting_delta", 1)
                or deltas["delta_branch_count"] >= tcfg.get("complexity_branch_delta", 3)
                or deltas["delta_boolean_chains"] >= 2
            ),
            "duplication_detection": bool(
                deltas["delta_duplicate_blocks"] >= tcfg.get("duplication_block_delta", 1)
                or deltas["delta_4gram_overlap"] >= tcfg.get("duplication_ngram_delta", 0.10)
            ),
            "future_change_impact": bool(
                deltas["delta_external_calls"] >= tcfg.get("change_impact_coupling_delta", 2)
                or deltas["delta_parameter_count"] >= tcfg.get("change_impact_param_delta", 1)
                or deltas["delta_local_var_count"] >= 4
            ),
        }

        return deltas, triggered_dimensions
