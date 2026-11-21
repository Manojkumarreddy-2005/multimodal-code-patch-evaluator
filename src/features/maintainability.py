"""Phase 2 Maintainability & Coupling Metrics (5 Static Features) via `tree-sitter-java`.

Extracts:
8.  `dfg_edge_density`: Intra-procedural Data-Flow Graph (Def-Use) edge density
9.  `mean_live_var_span`: Mean live line span (last_use_line - first_def_line) per variable
10. `fan_out_calls`: External method invocations and object creation expressions
11. `local_var_param_count`: Total local variables and formal parameters
12. `sloc`: Physical non-blank, non-comment Source Lines of Code
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from tree_sitter import Node

from src.features.ast_parser import ParsedJavaAST


@dataclass(frozen=True)
class MaintainabilityMetrics:
    """Five deterministic maintainability and coupling features."""

    dfg_edge_density: float
    mean_live_var_span: float
    fan_out_calls: float
    local_var_param_count: float
    sloc: float


COMMENT_TYPES = frozenset({"line_comment", "block_comment", "comment"})


def extract_maintainability_metrics(parsed: ParsedJavaAST) -> MaintainabilityMetrics:
    """Compute DFG edge density, mean live variable span, external fan-out calls, var count, and SLOC."""
    active_lines: Set[int] = set()
    method_nodes: List[Node] = []
    declared_method_names: Set[str] = set()
    invocation_nodes: List[Node] = []
    object_creations = 0
    stack: List[Node] = [parsed.root_node]

    raw_lines = parsed.code_bytes.splitlines()
    max_line_idx = len(raw_lines) - 1

    while stack:
        node = stack.pop()
        ntype = node.type

        if ntype in COMMENT_TYPES:
            continue

        if ntype in ("method_declaration", "constructor_declaration"):
            method_nodes.append(node)
            name_node = node.child_by_field_name("name")
            if name_node is not None and name_node.text:
                declared_method_names.add(name_node.text.decode("utf-8", errors="ignore"))

        elif ntype == "method_invocation":
            invocation_nodes.append(node)

        elif ntype == "object_creation_expression":
            object_creations += 1

        if node.child_count == 0:
            tok_text = node.text.decode("utf-8", errors="ignore") if node.text else ""
            if tok_text.strip():
                for row in range(node.start_point[0], node.end_point[0] + 1):
                    if parsed.is_wrapped_method and (row == 0 or row == max_line_idx):
                        continue
                    active_lines.add(row)

        for child in reversed(node.children):
            stack.append(child)

    # Count only EXTERNAL method calls (calls to extracted internal helper methods are modular decomposition, not external fan-out)
    external_invocations = 0
    for inv in invocation_nodes:
        name_node = inv.child_by_field_name("name")
        obj_node = inv.child_by_field_name("object")
        callee_name = name_node.text.decode("utf-8", errors="ignore") if (name_node and name_node.text) else ""
        is_internal_helper = (
            callee_name in declared_method_names
            and (obj_node is None or (obj_node.text and obj_node.text.decode("utf-8", errors="ignore") == "this"))
        )
        if not is_internal_helper:
            external_invocations += 1

    num_methods = max(1, len(method_nodes))
    fan_out_calls = round(float(external_invocations + object_creations) / num_methods, 4)

    scopes = method_nodes if method_nodes else [parsed.root_node]

    total_dfg_edges = 0
    total_tracked_vars = 0
    live_spans: List[int] = []

    for scope_node in scopes:
        edges, n_vars, spans = _analyze_scope_dfg(scope_node)
        total_dfg_edges += edges
        total_tracked_vars += n_vars
        live_spans.extend(spans)

    local_var_param_count = round(float(total_tracked_vars) / num_methods, 4)
    dfg_edge_density = total_dfg_edges / (total_tracked_vars + float(num_methods))
    mean_live_var_span = (
        sum(live_spans) / len(live_spans) if live_spans else 0.0
    )
    raw_sloc = max(1, len(active_lines)) if parsed.code_bytes.strip() else 0
    sloc = round(float(raw_sloc) / num_methods, 4) if num_methods > 1 else float(raw_sloc)

    return MaintainabilityMetrics(
        dfg_edge_density=round(float(dfg_edge_density), 4),
        mean_live_var_span=round(float(mean_live_var_span), 4),
        fan_out_calls=float(fan_out_calls),
        local_var_param_count=float(local_var_param_count),
        sloc=float(sloc),
    )


def _analyze_scope_dfg(scope_root: Node) -> Tuple[int, int, List[int]]:
    """Construct intra-procedural Def-Use chains in pre-order execution sequence."""
    # Map variable_name -> (first_def_line, latest_def_line, last_use_line, def_count)
    var_table: Dict[str, List[int]] = {}
    def_id_Set: Set[int] = set()
    dfg_edges = 0

    # Pre-order traversal over AST within the method scope
    stack: List[Node] = [scope_root]
    while stack:
        node = stack.pop()
        ntype = node.type

        if ntype in COMMENT_TYPES:
            continue

        # 1. Formal parameters & catch parameters
        if ntype in ("formal_parameter", "spread_parameter", "catch_formal_parameter"):
            name_node = node.child_by_field_name("name")
            if name_node is None:
                for ch in node.children:
                    if ch.type == "identifier":
                        name_node = ch
            if name_node is not None and name_node.text:
                vname = name_node.text.decode("utf-8", errors="ignore")
                line = name_node.start_point[0]
                def_id_Set.add(name_node.id)
                if vname not in var_table:
                    var_table[vname] = [line, line, line, 1]

        # 2. Enhanced for-loop variable declaration: `for (int x : arr)`
        elif ntype == "enhanced_for_statement":
            name_node = node.child_by_field_name("name")
            if name_node is not None and name_node.text:
                vname = name_node.text.decode("utf-8", errors="ignore")
                line = name_node.start_point[0]
                def_id_Set.add(name_node.id)
                if vname not in var_table:
                    var_table[vname] = [line, line, line, 1]

        # 3. Local variable declarations: `int a = expr;`
        elif ntype == "variable_declarator":
            name_node = node.child_by_field_name("name")
            if name_node is not None and name_node.text:
                vname = name_node.text.decode("utf-8", errors="ignore")
                line = name_node.start_point[0]
                def_id_Set.add(name_node.id)
                if vname not in var_table:
                    var_table[vname] = [line, line, line, 1]
                else:
                    var_table[vname][1] = line
                    var_table[vname][3] += 1

        # 4. Assignment expressions: `x = expr` or compound `x += expr`
        elif ntype == "assignment_expression":
            left_node = node.child_by_field_name("left")
            op_node = node.child_by_field_name("operator")
            op_text = op_node.text.decode("utf-8", errors="ignore") if (op_node and op_node.text) else "="
            if left_node is not None:
                target_id_node = left_node if left_node.type == "identifier" else None
                if target_id_node is not None and target_id_node.text:
                    vname = target_id_node.text.decode("utf-8", errors="ignore")
                    line = target_id_node.start_point[0]
                    def_id_Set.add(target_id_node.id)
                    if vname in var_table:
                        # Compound assignment (`+=`, `-=`, `*=`) both reads and redefines `vname`
                        if op_text != "=":
                            dfg_edges += 1
                            var_table[vname][2] = max(var_table[vname][2], line)
                        # Def-Def dependency edge from prior definition
                        dfg_edges += 1
                        var_table[vname][1] = line
                        var_table[vname][3] += 1

        # 5. Unary update expressions: `i++`, `++i`, `i--`
        elif ntype == "update_expression":
            for ch in node.children:
                if ch.type == "identifier" and ch.text:
                    vname = ch.text.decode("utf-8", errors="ignore")
                    line = ch.start_point[0]
                    def_id_Set.add(ch.id)
                    if vname in var_table:
                        dfg_edges += 1  # Read-modify-write def-use edge
                        var_table[vname][1] = line
                        var_table[vname][2] = max(var_table[vname][2], line)

        # 6. Variable Use (identifier read in expressions, conditions, arguments, returns)
        elif ntype == "identifier" and node.id not in def_id_Set:
            # Exclude method names in `method_invocation`
            is_method_callee = (
                node.parent is not None
                and node.parent.type == "method_invocation"
                and node.parent.child_by_field_name("name") == node
            )
            if not is_method_callee and node.text:
                vname = node.text.decode("utf-8", errors="ignore")
                if vname in var_table:
                    line = node.start_point[0]
                    dfg_edges += 1
                    if line > var_table[vname][2]:
                        var_table[vname][2] = line

        for child in reversed(node.children):
            stack.append(child)

    spans = [max(0, info[2] - info[0]) for info in var_table.values()]
    return dfg_edges, len(var_table), spans
