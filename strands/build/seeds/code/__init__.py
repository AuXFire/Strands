"""Code-domain seed concepts (Phase 2 — spec §5.2)."""

from __future__ import annotations

from strands.build.seeds.code import (
    api as _api,
)
from strands.build.seeds.code import (
    control_flow as _control_flow,
)
from strands.build.seeds.code import (
    data_structure as _data_structure,
)
from strands.build.seeds.code import (
    error as _error,
)
from strands.build.seeds.code import (
    infrastructure as _infrastructure,
)
from strands.build.seeds.code import (
    io as _io,
)
from strands.build.seeds.code import (
    module as _module,
)
from strands.build.seeds.code import (
    operation as _operation,
)
from strands.build.seeds.code import (
    pattern as _pattern,
)
from strands.build.seeds.code import (
    testing as _testing,
)
from strands.build.seeds.code import (
    type_system as _type_system,
)

CODE_DOMAIN_MODULES: dict[str, object] = {
    "CF": _control_flow,
    "DS": _data_structure,
    "TS": _type_system,
    "OP": _operation,
    "IO": _io,
    "ER": _error,
    "PT": _pattern,
    "MD": _module,
    "TE": _testing,
    "AP": _api,
    "IN": _infrastructure,
}


def collect_code_seeds() -> dict[tuple[str, int, int], list[str]]:
    out: dict[tuple[str, int, int], list[str]] = {}
    for code, module in CODE_DOMAIN_MODULES.items():
        for category, concept, words in module.SEEDS:  # type: ignore[attr-defined]
            key = (code, category, concept)
            if key in out:
                raise ValueError(f"Duplicate code seed key: {key}")
            out[key] = list(words)
    return out
