"""Fast in-memory tree-sitter-java AST parser wrapper (<50ms SLA).

Supports both full Java compilation units (e.g., IBM Project CodeNet `Main.java`)
and standalone method-level patches (e.g., SWE-Refactor `sourceCodeBeforeRefactoring`).
"""

from __future__ import annotations

from dataclasses import dataclass
import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser, Tree


JAVA_LANGUAGE = Language(tsjava.language())


@dataclass(frozen=True)
class ParsedJavaAST:
    """Container for a parsed Tree-sitter Java syntax tree and source buffer."""

    tree: Tree
    root_node: Node
    code_bytes: bytes
    is_wrapped_method: bool
    line_offset: int


class FastJavaASTParser:
    """Reusable in-memory `tree-sitter-java` parser."""

    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    def parse(self, code: str) -> ParsedJavaAST:
        """Parse Java source code into a Tree-sitter AST.

        If `code` is a method-level snippet without an enclosing `class` or `interface`
        declaration and produces top-level syntax errors, transparently wraps it in
        `class _PatchWrapper { ... }` so method declarations, formal parameters, and
        statements parse into their exact grammar nodes.
        """
        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        root = tree.root_node

        if (
            root.has_error
            and "class " not in code
            and "interface " not in code
            and "enum " not in code
            and "record " not in code
        ):
            wrapped_bytes = b"class _PatchWrapper {\n" + code_bytes + b"\n}"
            wrapped_tree = self._parser.parse(wrapped_bytes)
            if not wrapped_tree.root_node.has_error:
                return ParsedJavaAST(
                    tree=wrapped_tree,
                    root_node=wrapped_tree.root_node,
                    code_bytes=wrapped_bytes,
                    is_wrapped_method=True,
                    line_offset=1,
                )

        return ParsedJavaAST(
            tree=tree,
            root_node=root,
            code_bytes=code_bytes,
            is_wrapped_method=False,
            line_offset=0,
        )
