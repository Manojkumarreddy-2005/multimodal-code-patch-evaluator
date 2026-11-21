"""Phase 2 Complexity Metrics (4 Static Features) via `tree-sitter-java`.

Extracts:
1. `cyclomatic_complexity`: McCabe Cyclomatic Complexity (1 + branch & boolean decision points)
2. `cognitive_complexity`: SonarSource nesting-weighted Cognitive Complexity
3. `max_ast_depth`: Maximum root-to-leaf path length in the AST
4. `max_nesting_depth`: Maximum nested control-flow block depth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from tree_sitter import Node

from src.features.ast_parser import ParsedJavaAST


@dataclass(frozen=True)
class ComplexityMetrics:
    """Four deterministic complexity features."""

    cyclomatic_complexity: float
    cognitive_complexity: float
    max_ast_depth: float
    max_nesting_depth: float


# Control structures that increment both Cyclomatic and Cognitive complexity (with nesting penalty)
NESTING_DECISION_NODES = frozenset(
    {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "ternary_expression",
    }
)

SWITCH_NODES = frozenset({"switch_expression", "switch_statement"})
SWITCH_CASE_NODES = frozenset({"switch_block_statement_group", "switch_rule"})
NESTING_ONLY_NODES = frozenset({"lambda_expression"})


def extract_complexity_metrics(parsed: ParsedJavaAST) -> ComplexityMetrics:
    """Compute the 4 complexity metrics in a single iterative AST pass."""
    cyclomatic = 1
    cognitive = 0
    max_ast_depth = 0
    max_nesting_depth = 0

    # Stack elements: (node, ast_depth, control_nesting_depth)
    stack: List[Tuple[Node, int, int]] = [(parsed.root_node, 0, 0)]

    while stack:
        node, ast_depth, nesting_depth = stack.pop()
        ntype = node.type

        if ast_depth > max_ast_depth:
            max_ast_depth = ast_depth

        next_nesting = nesting_depth

        if ntype in NESTING_DECISION_NODES:
            cyclomatic += 1
            # Check if this `if_statement` is an `else if` branch
            is_else_if = (
                ntype == "if_statement"
                and node.parent is not None
                and node.parent.type == "if_statement"
                and node.parent.child_by_field_name("alternative") == node
            )
            if is_else_if:
                # SonarSource Cognitive Complexity: `else if` adds +1 hybrid increment,
                # does NOT add nesting penalty, and preserves current nesting depth.
                cognitive += 1
            else:
                cognitive += 1 + nesting_depth
                next_nesting = nesting_depth + 1
                if next_nesting > max_nesting_depth:
                    max_nesting_depth = next_nesting

            # Check for plain `else` block (when alternative is a block rather than another if_statement)
            if ntype == "if_statement":
                alt = node.child_by_field_name("alternative")
                if alt is not None and alt.type != "if_statement":
                    cognitive += 1

        elif ntype in SWITCH_NODES:
            # In SonarSource Cognitive Complexity, an entire `switch` counts as +(1 + nesting_depth)
            cognitive += 1 + nesting_depth
            next_nesting = nesting_depth + 1
            if next_nesting > max_nesting_depth:
                max_nesting_depth = next_nesting

        elif ntype in SWITCH_CASE_NODES:
            # In McCabe Cyclomatic Complexity, each `case` group adds +1 decision point
            has_default_only = False
            for child in node.children:
                if child.type == "switch_label":
                    label_text = child.text.decode("utf-8", errors="ignore") if child.text else ""
                    if label_text.strip().startswith("default"):
                        has_default_only = True
                    else:
                        has_default_only = False
                        break
            if not has_default_only:
                cyclomatic += 1

        elif ntype in NESTING_ONLY_NODES:
            next_nesting = nesting_depth + 1
            if next_nesting > max_nesting_depth:
                max_nesting_depth = next_nesting

        elif ntype == "binary_expression":
            op_node = node.child_by_field_name("operator")
            if op_node is not None and op_node.type in ("&&", "||"):
                op_type = op_node.type
                cyclomatic += 1
                # Cognitive complexity increments +1 per contiguous sequence of identical logical operators
                parent_same_op = False
                if node.parent is not None and node.parent.type == "binary_expression":
                    p_op = node.parent.child_by_field_name("operator")
                    if p_op is not None and p_op.type == op_type:
                        parent_same_op = True
                if not parent_same_op:
                    cognitive += 1

        elif ntype in ("break_statement", "continue_statement"):
            # Labeled break/continue adds +1 in Cognitive Complexity
            for child in node.children:
                if child.type == "identifier":
                    cognitive += 1
                    break

        for child in reversed(node.children):
            stack.append((child, ast_depth + 1, next_nesting))

    # Adjust max_ast_depth if the parser added a synthetic 2-level class wrapper
    if parsed.is_wrapped_method and max_ast_depth > 2:
        max_ast_depth -= 2

    # Count method declarations so modular decomposition (Extract Method) measures per-method complexity
    method_count = 0
    mstack: List[Node] = [parsed.root_node]
    while mstack:
        cur = mstack.pop()
        if cur.type in ("method_declaration", "constructor_declaration"):
            method_count += 1
        for ch in cur.children:
            mstack.append(ch)
    num_methods = max(1, method_count)

    # Per-method cyclomatic and cognitive complexity (SonarJava S1541 and S3776 are method-level rules)
    per_method_cyclomatic = round(float(cyclomatic) / num_methods, 4)
    per_method_cognitive = round(float(cognitive) / num_methods, 4)

    return ComplexityMetrics(
        cyclomatic_complexity=per_method_cyclomatic,
        cognitive_complexity=per_method_cognitive,
        max_ast_depth=float(max_ast_depth),
        max_nesting_depth=float(max_nesting_depth),
    )
