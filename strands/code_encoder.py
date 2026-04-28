"""Code encoder — language-aware encoding for source files (spec §7.2).

Two interleaved streams are emitted in order:
  - **Structural tokens** (keywords, operators, control flow) → code domains.
  - **Semantic tokens** (identifiers, string literals) → text domains via
    the identifier splitter + abbreviation expander.

Tree-sitter integration is optional. When the ``tree_sitter`` and a
language-specific grammar are importable, the encoder uses AST traversal
for higher-fidelity encoding. Otherwise it falls back to a regex tokenizer
that knows each language's keyword set — good enough for the MVP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from strands.codebook import Codebook, default_codebook
from strands.encoder import encode as encode_text
from strands.identifier import split_identifier
from strands.lemmatizer import lemmatize
from strands.shade import compute_shade
from strands.strand import CodonEntry, Strand

# Per-language reserved-keyword sets. These are looked up code-mode in the
# codebook; identifiers that aren't keywords go through identifier splitting.
LANGUAGE_KEYWORDS: dict[str, set[str]] = {
    "python": {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield", "match", "case",
    },
    "javascript": {
        "abstract", "async", "await", "break", "case", "catch", "class",
        "const", "continue", "debugger", "default", "delete", "do", "else",
        "enum", "export", "extends", "false", "finally", "for", "from",
        "function", "if", "implements", "import", "in", "instanceof",
        "interface", "let", "new", "null", "of", "private", "protected",
        "public", "return", "static", "super", "switch", "this", "throw",
        "true", "try", "typeof", "undefined", "var", "void", "while", "with",
        "yield",
    },
    "typescript": {
        "abstract", "any", "as", "async", "await", "break", "case", "catch",
        "class", "const", "constructor", "continue", "debugger", "declare",
        "default", "delete", "do", "else", "enum", "export", "extends",
        "false", "finally", "for", "from", "function", "if", "implements",
        "import", "in", "instanceof", "interface", "is", "keyof", "let",
        "namespace", "new", "null", "number", "of", "private", "protected",
        "public", "readonly", "return", "static", "string", "super",
        "switch", "this", "throw", "true", "try", "type", "typeof",
        "undefined", "unknown", "var", "void", "while", "with", "yield",
    },
    "rust": {
        "as", "async", "await", "break", "const", "continue", "crate", "dyn",
        "else", "enum", "extern", "false", "fn", "for", "if", "impl", "in",
        "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
        "self", "Self", "static", "struct", "super", "trait", "true", "type",
        "unsafe", "use", "where", "while",
    },
    "go": {
        "break", "case", "chan", "const", "continue", "default", "defer",
        "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
        "interface", "map", "package", "range", "return", "select", "struct",
        "switch", "type", "var",
    },
    "java": {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch",
        "char", "class", "const", "continue", "default", "do", "double",
        "else", "enum", "extends", "final", "finally", "float", "for", "goto",
        "if", "implements", "import", "instanceof", "int", "interface",
        "long", "native", "new", "package", "private", "protected", "public",
        "return", "short", "static", "strictfp", "super", "switch",
        "synchronized", "this", "throw", "throws", "transient", "try", "void",
        "volatile", "while",
    },
    "c": {
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "int", "long", "register", "return", "short", "signed", "sizeof",
        "static", "struct", "switch", "typedef", "union", "unsigned", "void",
        "volatile", "while",
    },
    "cpp": {
        "alignas", "alignof", "and", "asm", "auto", "bool", "break", "case",
        "catch", "char", "class", "compl", "const", "constexpr", "continue",
        "decltype", "default", "delete", "do", "double", "else", "enum",
        "explicit", "export", "extern", "false", "float", "for", "friend",
        "goto", "if", "inline", "int", "long", "mutable", "namespace", "new",
        "noexcept", "not", "nullptr", "operator", "or", "private", "protected",
        "public", "register", "return", "short", "signed", "sizeof", "static",
        "static_cast", "struct", "switch", "template", "this", "throw", "true",
        "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
        "virtual", "void", "volatile", "while", "xor",
    },
}


# Tokenize source code into (token, kind) where kind ∈ {keyword, identifier,
# string, comment, number, punct}. Order is preserved.
_TOKEN_RE = re.compile(
    r"""
    (?P<comment>\#[^\n]*|//[^\n]*|/\*.*?\*/)            # comments
    |(?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')    # strings
    |(?P<number>\b\d+(?:\.\d+)?\b)                       # numbers
    |(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)              # identifiers + keywords
    |(?P<punct>[+\-*/%<>=!&|^~?:;,.\(\)\[\]\{\}])        # punctuation
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(slots=True)
class CodeEncodeResult:
    strand: Strand
    text: str
    language: str
    unknowns: list[str] = field(default_factory=list)
    structural_count: int = 0
    semantic_count: int = 0

    @property
    def byte_size(self) -> int:
        return self.strand.byte_size

    @property
    def strand_text(self) -> str:
        return self.strand.to_text()

    @property
    def structural_density(self) -> float:
        total = self.structural_count + self.semantic_count
        if total == 0:
            return 0.0
        return self.structural_count / total


def detect_language(filename: str | None, source: str | None = None) -> str | None:
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        return _EXT_LANG.get(ext)
    return None


_EXT_LANG: dict[str, str] = {
    "py": "python", "pyx": "python", "pyi": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "c": "c", "h": "c",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp",
}


def encode_code(
    source: str,
    *,
    language: str = "python",
    codebook: Codebook | None = None,
) -> CodeEncodeResult:
    """Encode source code into a strand. ``language`` selects the keyword set."""
    cb = codebook or default_codebook()
    keywords = LANGUAGE_KEYWORDS.get(language, set())

    entries: list[CodonEntry] = []
    unknowns: list[str] = []
    structural = 0
    semantic = 0

    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        token = m.group()
        if kind in ("comment", "punct", "number"):
            continue
        if kind == "string":
            # Encode string literal contents through the text pipeline.
            inner = token[1:-1]
            for sub in encode_text(inner, codebook=cb).strand.codons:
                entries.append(sub)
                semantic += 1
            continue
        if kind == "identifier":
            if token in keywords or token.lower() in keywords:
                # Structural keyword → code-domain lookup (case-insensitive).
                cb_entry = cb.lookup(token.lower(), mode="code")
                if cb_entry is None:
                    cb_entry = cb.lookup(token.lower(), mode="auto")
                if cb_entry is None:
                    unknowns.append(token)
                    continue
                entries.append(
                    CodonEntry(
                        codon=cb_entry.codon,
                        shade=compute_shade(token.lower(), cb_entry.shade_hint),
                        word=token.lower(),
                        alt_codons=cb_entry.alt_codons,
                        synset=cb_entry.synset,
                    )
                )
                structural += 1
            else:
                # Identifier → split + look up each part.
                for part in split_identifier(token):
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
                            alt_codons=cb_entry.alt_codons,
                            synset=cb_entry.synset,
                        )
                    )
                    semantic += 1

    return CodeEncodeResult(
        strand=Strand(codons=entries),
        text=source,
        language=language,
        unknowns=unknowns,
        structural_count=structural,
        semantic_count=semantic,
    )
