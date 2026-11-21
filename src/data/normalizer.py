"""AST-aware Java code normalizer using tree-sitter-java.

Strips comments (line_comment, block_comment, Javadoc) using exact AST byte spans
so string literals containing '//' or '/*' are never corrupted. Standardizes
whitespace and calculates physical SLOC.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import List, Tuple

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser


JAVA_LANGUAGE = Language(tsjava.language())


@dataclass(frozen=True)
class NormalizedSnippet:
    """Result of AST-canonical normalization on a Java code snippet."""

    raw_code: str
    normalized_code: str
    sha256_id: str
    sloc: int
    comment_count: int
    has_syntax_error: bool


class JavaCodeNormalizer:
    """Deterministic AST-based normalizer for Java source files and method patches."""

    COMMENT_NODE_TYPES = frozenset({"line_comment", "block_comment", "comment"})

    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    def parse_tree(self, code_bytes: bytes):
        """Parse UTF-8 encoded Java source bytes into a Tree-sitter Tree."""
        return self._parser.parse(code_bytes)

    def normalize(self, raw_code: str) -> NormalizedSnippet:
        """Strip comments via AST spans, canonicalize whitespace, and compute SLOC."""
        if not raw_code or not raw_code.strip():
            empty_hash = hashlib.sha256(b"").hexdigest()[:16]
            return NormalizedSnippet(
                raw_code=raw_code,
                normalized_code="",
                sha256_id=empty_hash,
                sloc=0,
                comment_count=0,
                has_syntax_error=False,
            )

        canonical_raw = raw_code.replace("\r\n", "\n").replace("\r", "\n")
        code_bytes = canonical_raw.encode("utf-8")
        tree = self._parser.parse(code_bytes)

        comment_spans: List[Tuple[int, int]] = []
        has_error = False

        # Iterative DFS traversal to collect comments and detect syntax errors
        stack: List[Node] = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in self.COMMENT_NODE_TYPES:
                comment_spans.append((node.start_byte, node.end_byte))
            elif node.type == "ERROR" or node.is_missing:
                has_error = True
            for child in reversed(node.children):
                stack.append(child)

        # Method-level patches (e.g., SWE-Refactor) may lack an outer class declaration.
        # Verify if wrapping in a class body resolves top-level compilation unit errors.
        if has_error and "class " not in canonical_raw and "interface " not in canonical_raw:
            wrapped_bytes = b"class _Wrapper {\n" + code_bytes + b"\n}"
            wrapped_tree = self._parser.parse(wrapped_bytes)
            wrap_error = False
            wstack: List[Node] = [wrapped_tree.root_node]
            while wstack:
                wnode = wstack.pop()
                if wnode.type == "ERROR" or wnode.is_missing:
                    wrap_error = True
                    break
                for wchild in wnode.children:
                    wstack.append(wchild)
            if not wrap_error:
                has_error = False

        # Remove comment byte spans in reverse order to preserve byte offsets
        if comment_spans:
            comment_spans.sort(key=lambda span: span[0], reverse=True)
            mutable_bytes = bytearray(code_bytes)
            for start_b, end_b in comment_spans:
                del mutable_bytes[start_b:end_b]
            stripped_text = mutable_bytes.decode("utf-8", errors="replace")
        else:
            stripped_text = canonical_raw

        # Canonicalize line formatting: convert tabs to 4 spaces, strip trailing spaces,
        # and remove blank lines so formatting differences cannot leak signals.
        non_empty_lines: List[str] = []
        for line in stripped_text.split("\n"):
            expanded = line.expandtabs(4).rstrip()
            if expanded.strip():
                non_empty_lines.append(expanded)

        normalized_code = "\n".join(non_empty_lines)

        # Re-parse normalized code to verify token-bearing physical SLOC
        norm_bytes = normalized_code.encode("utf-8")
        norm_tree = self._parser.parse(norm_bytes)
        active_lines = set()
        leaf_stack: List[Node] = [norm_tree.root_node]
        while leaf_stack:
            n = leaf_stack.pop()
            if n.child_count == 0 and n.type not in self.COMMENT_NODE_TYPES:
                if n.text and n.text.strip():
                    for row in range(n.start_point[0], n.end_point[0] + 1):
                        active_lines.add(row)
            else:
                for c in n.children:
                    leaf_stack.append(c)

        sloc = len(active_lines) if active_lines else len(non_empty_lines)
        sha256_id = hashlib.sha256(norm_bytes).hexdigest()[:24]

        return NormalizedSnippet(
            raw_code=canonical_raw,
            normalized_code=normalized_code,
            sha256_id=sha256_id,
            sloc=sloc,
            comment_count=len(comment_spans),
            has_syntax_error=has_error,
        )
