"""Aggregates per-domain seed concepts into a single mapping.

Each per-domain module exports `SEEDS: list[tuple[category, concept, words]]`.
This module collects them into `ALL_SEEDS: dict[(domain_code, cat, concept)] = words`.
"""

from __future__ import annotations

from strands.build.seeds import (
    abstract as _abstract,
)
from strands.build.seeds import (
    action as _action,
)
from strands.build.seeds import (
    body as _body,
)
from strands.build.seeds import (
    communication as _communication,
)
from strands.build.seeds import (
    economy as _economy,
)
from strands.build.seeds import (
    emotion as _emotion,
)
from strands.build.seeds import (
    food as _food,
)
from strands.build.seeds import (
    movement as _movement,
)
from strands.build.seeds import (
    nature as _nature,
)
from strands.build.seeds import (
    object as _object,
)
from strands.build.seeds import (
    person as _person,
)
from strands.build.seeds import (
    quality as _quality,
)
from strands.build.seeds import (
    quantity as _quantity,
)
from strands.build.seeds import (
    relation as _relation,
)
from strands.build.seeds import (
    sensory as _sensory,
)
from strands.build.seeds import (
    social as _social,
)
from strands.build.seeds import (
    space as _space,
)
from strands.build.seeds import (
    tech as _tech,
)
from strands.build.seeds import (
    time as _time,
)

_DOMAIN_MODULES: dict[str, object] = {
    "EM": _emotion,
    "AC": _action,
    "OB": _object,
    "QU": _quality,
    "AB": _abstract,
    "NA": _nature,
    "PE": _person,
    "SP": _space,
    "TM": _time,
    "QT": _quantity,
    "BD": _body,
    "SO": _social,
    "TC": _tech,
    "FD": _food,
    "CM": _communication,
    "SN": _sensory,
    "MV": _movement,
    "RL": _relation,
    "EC": _economy,
}


def _collect() -> dict[tuple[str, int, int], list[str]]:
    out: dict[tuple[str, int, int], list[str]] = {}
    for domain_code, module in _DOMAIN_MODULES.items():
        for category, concept, words in module.SEEDS:  # type: ignore[attr-defined]
            key = (domain_code, category, concept)
            if key in out:
                raise ValueError(f"Duplicate seed key: {key}")
            out[key] = list(words)
    return out


ALL_SEEDS: dict[tuple[str, int, int], list[str]] = _collect()


def all_code_seeds() -> dict[tuple[str, int, int], list[str]]:
    """Code-domain seeds (Phase 2). Lazy import to keep text-only callers cheap."""
    from strands.build.seeds.code import collect_code_seeds

    return collect_code_seeds()


ALL_CODE_SEEDS: dict[tuple[str, int, int], list[str]] = all_code_seeds()
