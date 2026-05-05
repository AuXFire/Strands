"""TemplateStore — lookup phrasal templates by response shape, by
relation type, or by both. Returns ranked candidates so the realizer
can fall back to a lower-weight template when the top one's slots
can't be filled.

The store is an in-memory dict for now. Per spec §2.4 templates would
eventually live alongside the backbone surface form tables, but for
B1 they're a Python module loaded at startup.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Iterable

from strands.realization.template import Register, Template


class TemplateStore:
    """Indexed collection of templates. Two indices:
      response_shape → list[Template] sorted by weight desc
      relation_type  → list[Template] sorted by weight desc

    A template registered with both is reachable from both indices."""

    def __init__(self) -> None:
        self._by_shape: dict[str, list[Template]] = defaultdict(list)
        self._by_relation: dict[int, list[Template]] = defaultdict(list)
        self._by_id: dict[str, Template] = {}

    def add(self, template: Template) -> None:
        if template.template_id in self._by_id:
            raise ValueError(
                f"duplicate template_id: {template.template_id}"
            )
        self._by_id[template.template_id] = template
        if template.response_shape:
            self._by_shape[template.response_shape].append(template)
            self._by_shape[template.response_shape].sort(
                key=lambda t: -t.weight,
            )
        if template.relation_type:
            self._by_relation[template.relation_type].append(template)
            self._by_relation[template.relation_type].sort(
                key=lambda t: -t.weight,
            )

    def add_all(self, templates: Iterable[Template]) -> None:
        for t in templates:
            self.add(t)

    def get(self, template_id: str) -> Template | None:
        return self._by_id.get(template_id)

    def by_shape(
        self, shape: str, *, register: Register | None = None,
    ) -> list[Template]:
        candidates = self._by_shape.get(shape, [])
        if register is not None:
            candidates = [t for t in candidates if t.register == register]
        return list(candidates)

    def by_relation(
        self, relation_type: int, *, register: Register | None = None,
    ) -> list[Template]:
        candidates = self._by_relation.get(relation_type, [])
        if register is not None:
            candidates = [t for t in candidates if t.register == register]
        return list(candidates)

    def best(
        self, *, shape: str = "", relation_type: int = 0,
        register: Register | None = None,
    ) -> Template | None:
        """Highest-weight template matching the criteria, or None.

        When BOTH ``shape`` and ``relation_type`` are supplied, the
        result must match both — useful for shape-bucketed relation
        templates like ('elaborate', HYPERNYM) vs ('elaborate', MERONYM).
        When only one is supplied, that index is used.
        """
        pool: list[Template] = []
        if shape and relation_type:
            shape_pool = self.by_shape(shape, register=register)
            pool = [t for t in shape_pool if t.relation_type == relation_type]
        elif shape:
            pool = self.by_shape(shape, register=register)
        elif relation_type:
            pool = self.by_relation(relation_type, register=register)
        return pool[0] if pool else None

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Template]:
        return iter(self._by_id.values())
