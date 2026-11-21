"""Phase 2 Duplication & Redundancy Metrics (3 Static Features) via `tree-sitter-java`.

Extracts:
5. `ast_clone_ratio`: Bottom-up Weisfeiler-Lehman / Merkle structural subtree clone ratio
6. `token_4gram_overlap`: Normalized token 4-gram repetition ratio (DRY violation score)
7. `halstead_volume`: Halstead implementation volume V = (N1 + N2) * log2(eta1 + eta2)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Dict, List, Set, Tuple
from tree_sitter import Node

from src.features.ast_parser import ParsedJavaAST


@dataclass(frozen=True)
class DuplicationMetrics:
    """Three deterministic duplication and redundancy features."""

    ast_clone_ratio: float
    token_4gram_overlap: float
    halstead_volume: float


COMMENT_TYPES = frozenset({"line_comment", "block_comment", "comment"})
BOILERPLATE_SUBTREES = frozenset(
    {
        "program",
        "package_declaration",
        "import_declaration",
        "modifiers",
        "formal_parameters",
        "formal_parameter",
    }
)
OPERAND_LEAF_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "decimal_integer_literal",
        "hex_integer_literal",
        "octal_integer_literal",
        "binary_integer_literal",
        "decimal_floating_point_literal",
        "hex_floating_point_literal",
        "true",
        "false",
        "character_literal",
        "string_literal",
        "null_literal",
    }
)
PUNCTUATION_TO_IGNORE_HALSTEAD = frozenset({";", ",", ")", "]", "}"})


def extract_duplication_metrics(parsed: ParsedJavaAST, min_clone_nodes: int = 6) -> DuplicationMetrics:
    """Compute AST subtree clone ratio, token 4-gram overlap, and Halstead volume."""
    # Post-order traversal for bottom-up Merkle subtree hashing + leaf token pass
    postorder_nodes: List[Node] = []
    stack: List[Tuple[Node, bool]] = [(parsed.root_node, False)]

    while stack:
        node, visited = stack.pop()
        if node.type in COMMENT_TYPES:
            continue
        if visited:
            postorder_nodes.append(node)
        else:
            stack.append((node, True))
            for child in reversed(node.children):
                if child.type not in COMMENT_TYPES:
                    stack.append((child, False))

    node_hash: Dict[int, int] = {}
    node_size: Dict[int, int] = {}
    hash_to_nodes: Dict[int, List[Node]] = defaultdict(list)

    normalized_tokens: List[str] = []
    operators: List[str] = []
    operands: List[str] = []

    for node in postorder_nodes:
        nid = node.id
        ntype = node.type

        # Ignore synthetic wrapper root if method was wrapped
        if parsed.is_wrapped_method and ntype == "class_declaration":
            raw_text = node.text.decode("utf-8", errors="ignore") if node.text else ""
            if raw_text.startswith("class _PatchWrapper"):
                continue

        if node.child_count == 0:
            raw_tok = node.text.decode("utf-8", errors="ignore") if node.text else ntype
            if not raw_tok.strip():
                node_hash[nid] = 0
                node_size[nid] = 0
                continue

            # Ignore synthetic wrapper tokens
            if parsed.is_wrapped_method and raw_tok in ("_PatchWrapper",):
                node_hash[nid] = 0
                node_size[nid] = 0
                continue

            # Normalize identifiers and literals for Type-2 clone invariance
            if ntype in ("identifier", "type_identifier"):
                norm_tok = "ID"
            elif "literal" in ntype or ntype in ("true", "false"):
                norm_tok = "LIT"
            else:
                norm_tok = ntype

            # Only include executable body tokens (exclude method signature modifiers/params) in 4-gram overlap
            in_signature = (
                node.parent is not None
                and node.parent.type in ("modifiers", "formal_parameter", "formal_parameters", "method_declaration")
            )
            if not in_signature:
                normalized_tokens.append(norm_tok)

            # Halstead operator vs operand classification
            if ntype in OPERAND_LEAF_TYPES:
                operands.append(raw_tok)
            elif raw_tok not in PUNCTUATION_TO_IGNORE_HALSTEAD:
                operators.append(raw_tok)

            node_hash[nid] = hash(norm_tok)
            node_size[nid] = 1
        else:
            child_hashes: List[int] = []
            sz = 1
            for ch in node.children:
                cid = ch.id
                ch_h = node_hash.get(cid, 0)
                if ch_h != 0:
                    child_hashes.append(ch_h)
                    sz += node_size.get(cid, 0)

            digest = hash((ntype, tuple(child_hashes)))
            node_hash[nid] = digest
            node_size[nid] = sz

            if sz >= min_clone_nodes and ntype not in BOILERPLATE_SUBTREES:
                hash_to_nodes[digest].append(node)

    method_count = sum(
        1 for n in postorder_nodes if n.type in ("method_declaration", "constructor_declaration")
    )
    num_methods = max(1, method_count)

    # Compute AST subtree clone ratio: fraction of AST nodes belonging to duplicate subtrees
    total_ast_nodes = max(1, len(node_size))
    cloned_node_ids: Set[int] = set()
    for digest, dup_roots in hash_to_nodes.items():
        if len(dup_roots) >= 2:
            for root_node in dup_roots[1:]:
                sub_stack = [root_node]
                while sub_stack:
                    cur = sub_stack.pop()
                    if cur.id in node_size:
                        cloned_node_ids.add(cur.id)
                    for ch in cur.children:
                        sub_stack.append(ch)

    ast_clone_ratio = min(1.0, len(cloned_node_ids) / total_ast_nodes)

    # Compute normalized token 4-gram overlap
    if len(normalized_tokens) >= 12:
        total_ngrams = len(normalized_tokens) - 3
        unique_ngrams = len(
            {
                (
                    normalized_tokens[i],
                    normalized_tokens[i + 1],
                    normalized_tokens[i + 2],
                    normalized_tokens[i + 3],
                )
                for i in range(total_ngrams)
            }
        )
        token_4gram_overlap = max(0.0, 1.0 - (unique_ngrams / total_ngrams))
    else:
        token_4gram_overlap = 0.0

    # Compute Per-Method Halstead Volume: V = (N1 + N2) * log2(eta1 + eta2) / num_methods
    n1 = len(operators)
    n2 = len(operands)
    eta1 = len(set(operators))
    eta2 = len(set(operands))
    vocabulary = eta1 + eta2
    length = n1 + n2
    if vocabulary > 1 and length > 0:
        halstead_volume = float((length * math.log2(vocabulary)) / num_methods)
    else:
        halstead_volume = 0.0

    return DuplicationMetrics(
        ast_clone_ratio=round(float(ast_clone_ratio), 4),
        token_4gram_overlap=round(float(token_4gram_overlap), 4),
        halstead_volume=round(float(halstead_volume), 2),
    )
