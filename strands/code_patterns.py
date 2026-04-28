"""Code pattern detection (spec §7.2 step 4).

Lightweight regex matchers that classify a source snippet into a small
set of reusable patterns. The detected patterns are intersected at compare
time and used as a +0.1 alignment bonus per spec §8.3 rule 2.
"""

from __future__ import annotations

import re

# Each pattern is (name, regex). All regexes operate on lowercased source
# with whitespace normalized.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("async-await", re.compile(r"\b(async|await)\b")),
    ("error-handling", re.compile(r"\b(try|catch|except|raise|throw|panic|recover|finally|rescue)\b")),
    ("loop-iteration", re.compile(r"\b(for|while|foreach|range|enumerate|map|filter|reduce|fold)\b")),
    ("conditional-branch", re.compile(r"\b(if|else|elif|switch|case|match|when|unless)\b")),
    ("recursion", re.compile(r"")),  # populated dynamically
    ("network-request", re.compile(r"\b(fetch|http|request|get\b|post|put|delete|axios|reqwest|urllib)\b")),
    ("file-io", re.compile(r"\b(open|read|write|fopen|fread|fwrite|readfile|writefile|fs\.)\b")),
    ("database-crud", re.compile(r"\b(select|insert|update|delete|where|from|join|query|sql)\b")),
    ("json-parse", re.compile(r"\b(json|parse|stringify|unmarshal|marshal|loads|dumps|deserialize|serialize)\b")),
    ("class-oop", re.compile(r"\b(class|extends|implements|interface|trait|new|self|super|this)\b")),
    ("function-def", re.compile(r"\b(def\s+\w+|function\s+\w+|fn\s+\w+|func\s+\w+|public\s+\w+|private\s+\w+)\s*\(")),
    ("logging", re.compile(r"\b(log|logger|console\.(log|warn|error|info|debug)|print|println|fmt\.print)")),
    ("collection-ops", re.compile(r"\b(map|filter|reduce|sort|find|group|flatten|zip|push|pop|append|insert)\b\s*\(")),
    ("string-format", re.compile(r"(\.format\(|f['\"]|sprintf|printf|fmt\.sprintf|template)")),
    ("auth-token", re.compile(r"\b(auth|token|jwt|oauth|login|password|session|cookie|bearer)\b")),
    ("test-assert", re.compile(r"\b(assert|expect|should|describe|it\s*\(|test\s*\(|context\s*\(|setup|teardown)\b")),
]


def detect_patterns(source: str) -> set[str]:
    """Return the set of pattern names matched in the source code."""
    text = source.lower()
    out: set[str] = set()
    for name, pattern in _PATTERNS:
        if name == "recursion":
            # Special: function name appears in its own body. Quick heuristic.
            for m in re.finditer(r"\b(?:def|function|fn|func)\s+(\w+)\s*\(", source, re.IGNORECASE):
                fn_name = m.group(1)
                # Look for fn_name( elsewhere in source that isn't its own definition
                refs = re.findall(rf"\b{re.escape(fn_name)}\s*\(", source)
                if len(refs) >= 2:
                    out.add("recursion")
                    break
            continue
        if pattern.search(text):
            out.add(name)
    return out


def pattern_bonus(patterns_a: set[str], patterns_b: set[str]) -> float:
    """Return alignment bonus per spec §8.3 rule 2: +0.1 when both share at
    least one pattern, scaled slightly by overlap size."""
    shared = patterns_a & patterns_b
    if not shared:
        return 0.0
    return min(0.20, 0.10 + 0.02 * (len(shared) - 1))
