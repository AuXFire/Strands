"""Optional tree-sitter AST encoder.

Provides higher-fidelity code encoding when ``tree_sitter`` and a language
grammar are importable. Falls back gracefully when they're not — callers
should check ``is_available()`` first.

Mapping strategy: each tree-sitter node type maps to a structural codon.
Identifier and string-literal leaves are routed through the regex encoder's
identifier-splitter pipeline so the same semantic vocabulary is used.
"""

from __future__ import annotations

from dataclasses import dataclass

from strands.codebook import Codebook, default_codebook
from strands.encoder import encode as encode_text
from strands.identifier import split_identifier
from strands.lemmatizer import lemmatize
from strands.shade import compute_shade
from strands.strand import CodonEntry, Strand


_LANGUAGE_CACHE: dict[str, object] = {}
_PARSER_CACHE: dict[str, object] = {}


def is_available(language: str = "python") -> bool:
    try:
        _get_parser(language)
        return True
    except Exception:
        return False


def _get_parser(language: str):
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    from tree_sitter import Language, Parser

    if language == "python":
        import tree_sitter_python as ts_lang
    elif language == "javascript":
        import tree_sitter_javascript as ts_lang
    else:
        raise ImportError(f"No tree-sitter grammar bundled for {language!r}")

    lang = Language(ts_lang.language())
    parser = Parser(lang)
    _LANGUAGE_CACHE[language] = lang
    _PARSER_CACHE[language] = parser
    return parser


# Node type -> codebook code-lookup token. Hand-mapped per spec §5.2.
_NODE_TYPE_KEYWORDS: dict[str, str] = {
    # Python
    "function_definition": "function",
    "class_definition": "class",
    "if_statement": "if",
    "for_statement": "for",
    "while_statement": "while",
    "try_statement": "try",
    "except_clause": "catch",
    "with_statement": "with",
    "import_statement": "import",
    "import_from_statement": "import",
    "return_statement": "return",
    "break_statement": "break",
    "continue_statement": "continue",
    "raise_statement": "throw",
    "yield_statement": "yield",
    "lambda": "lambda",
    "assignment": "assign",
    "augmented_assignment": "assign",
    "list": "array",
    "dictionary": "map",
    "set": "set",
    "tuple": "tuple",
    "binary_operator": "operator",
    "comparison_operator": "compare",
    "boolean_operator": "and",
    # JavaScript
    "function_declaration": "function",
    "function_expression": "function",
    "arrow_function": "lambda",
    "class_declaration": "class",
    "method_definition": "method",
    "variable_declaration": "var",
    "lexical_declaration": "let",
    "switch_statement": "switch",
    "throw_statement": "throw",
    "await_expression": "await",
    "yield_expression": "yield",
    "object": "object",
    "array": "array",
}


@dataclass(slots=True)
class AstEncodeResult:
    strand: Strand
    text: str
    language: str
    structural_count: int
    semantic_count: int
    unknowns: list[str]

    @property
    def byte_size(self) -> int:
        return self.strand.byte_size

    @property
    def strand_text(self) -> str:
        return self.strand.to_text()


def encode_ast(
    source: str,
    *,
    language: str = "python",
    codebook: Codebook | None = None,
) -> AstEncodeResult:
    """AST-aware code encoding. Raises ImportError if tree-sitter is unavailable."""
    cb = codebook or default_codebook()
    parser = _get_parser(language)
    tree = parser.parse(source.encode("utf-8"))

    entries: list[CodonEntry] = []
    unknowns: list[str] = []
    structural = 0
    semantic = 0

    def emit_keyword(keyword: str) -> bool:
        nonlocal structural
        cb_entry = cb.lookup(keyword, mode="code")
        if cb_entry is None:
            return False
        entries.append(
            CodonEntry(
                codon=cb_entry.codon,
                shade=compute_shade(keyword, cb_entry.shade_hint),
                word=keyword,
            )
        )
        structural += 1
        return True

    def emit_identifier(text: str) -> None:
        nonlocal semantic
        for part in split_identifier(text):
            cb_entry = cb.lookup(part, mode="auto")
            if cb_entry is None:
                lemma = lemmatize(part)
                cb_entry = cb.lookup(lemma, mode="auto")
                word = lemma
            else:
                word = part
            if cb_entry is None:
                unknowns.append(part)
                continue
            entries.append(
                CodonEntry(
                    codon=cb_entry.codon,
                    shade=compute_shade(word, cb_entry.shade_hint),
                    word=word,
                )
            )
            semantic += 1

    def visit(node) -> None:
        nonlocal semantic
        kw = _NODE_TYPE_KEYWORDS.get(node.type)
        if kw:
            emit_keyword(kw)

        if node.child_count == 0:
            text = node.text.decode("utf-8") if node.text else ""
            if node.type == "identifier":
                emit_identifier(text)
            elif node.type == "string" and len(text) >= 2:
                inner = text.strip("\"'`")
                for ce in encode_text(inner, codebook=cb).strand.codons:
                    entries.append(ce)
                    semantic += 1
            return

        for child in node.children:
            visit(child)

    visit(tree.root_node)

    return AstEncodeResult(
        strand=Strand(codons=entries),
        text=source,
        language=language,
        structural_count=structural,
        semantic_count=semantic,
        unknowns=unknowns,
    )
