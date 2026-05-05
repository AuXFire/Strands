"""Seed templates for the BDRM realization layer (B1.2).

Hand-curated initial set covering every response shape the
deterministic system currently produces. Each shape gets at least
one template; high-frequency shapes get multiple registers / phrasings
so the planner has variety even before learned ranking is in place.

This is the bootstrap. Later phases extract templates from corpora
or distill them from a teacher model.

Note: ``Rel`` is imported lazily inside ``_seed_templates()`` to avoid
a circular import — ``strands.backbone`` reaches the realization layer
through ``response.py``, so the realization layer cannot eagerly pull
on ``strands.backbone.__init__`` at module load.
"""

from __future__ import annotations

from strands.realization.template import (
    Register,
    SlotKind,
    SlotSpec,
    Template,
)


# Common slot specs reused across templates.
_SUBJECT = SlotSpec(name="subject", kind=SlotKind.LEMMA, capitalize=True)
_SUBJECT_AN = SlotSpec(
    name="subject", kind=SlotKind.NOUN_PHRASE, article=True, capitalize=False,
)
_TARGET = SlotSpec(name="target", kind=SlotKind.LEMMA)
_TARGET_AN = SlotSpec(
    name="target", kind=SlotKind.NOUN_PHRASE, article=True,
)
_GLOSS = SlotSpec(name="gloss", kind=SlotKind.GLOSS)
_HYPERNYM = SlotSpec(
    name="hypernym", kind=SlotKind.NOUN_PHRASE, article=True,
)


def _seed_templates() -> tuple[Template, ...]:
    """Build the seed list. Lazy so the Rel import doesn't trigger
    a backbone-package init at realization load time."""
    from strands.backbone.schema import Rel
    return (
        # ---- Definition (gloss-backed) ----
        Template(
            template_id="def.basic",
            pattern="{subject} is {gloss}.",
            slots=(_SUBJECT, _GLOSS),
            response_shape="definition",
            weight=1.0,
        ),
        Template(
            template_id="def.with_hypernym",
            pattern="{subject} is {gloss}, {hypernym}.",
            slots=(_SUBJECT, _GLOSS, _HYPERNYM),
            response_shape="definition_with_hypernym",
            weight=1.1,
        ),
        Template(
            template_id="def.fallback_hypernym",
            pattern="{subject} is {hypernym}.",
            slots=(_SUBJECT, _HYPERNYM),
            response_shape="definition_fallback",
            weight=0.7,
        ),
        Template(
            template_id="def.no_information",
            pattern="I don't have enough information about {subject}.",
            slots=(_SUBJECT,),
            response_shape="definition_unknown",
            weight=0.5,
        ),

        # ---- Yes/no questions ----
        Template(
            template_id="yesno.yes_hypernym",
            pattern="Yes, {subject} is {target}.",
            slots=(_SUBJECT_AN, _TARGET_AN),
            response_shape="yesno_yes_hypernym",
            weight=1.0,
        ),
        Template(
            template_id="yesno.no_with_actual",
            pattern="No, {subject} is not {target}; {subject} is {actual}.",
            slots=(
                _SUBJECT_AN, _TARGET_AN,
                SlotSpec(
                    name="actual", kind=SlotKind.NOUN_PHRASE, article=True,
                ),
            ),
            response_shape="yesno_no_with_actual",
            weight=1.0,
        ),
        Template(
            template_id="yesno.no_plain",
            pattern="No, {subject} is not {target}.",
            slots=(_SUBJECT_AN, _TARGET_AN),
            response_shape="yesno_no_plain",
            weight=0.7,
        ),
        Template(
            template_id="yesno.yes_capable",
            pattern="Yes, {subject} can {target}.",
            slots=(_SUBJECT_AN, _TARGET),
            response_shape="yesno_yes_capable",
            weight=1.0,
        ),
        Template(
            template_id="yesno.yes_property",
            pattern="Yes, {subject} is {target}.",
            slots=(_SUBJECT_AN, _TARGET),
            response_shape="yesno_yes_property",
            weight=1.0,
        ),
        Template(
            template_id="yesno.yes_has",
            pattern="Yes, {subject} has {target}.",
            slots=(_SUBJECT_AN, _TARGET),
            response_shape="yesno_yes_has",
            weight=1.0,
        ),
        Template(
            template_id="yesno.unsure",
            pattern=(
                "I don't have direct evidence either way about whether "
                "{positive_phrase}."
            ),
            slots=(SlotSpec(name="positive_phrase", kind=SlotKind.LITERAL),),
            response_shape="yesno_unsure",
            weight=1.0,
        ),

        # ---- Typed wh-questions (relation-walks) ----
        Template(
            template_id="rel.location",
            pattern="{subject} can be found at {target}.",
            slots=(_SUBJECT, SlotSpec(
                name="target", kind=SlotKind.LEMMA_LIST,
            )),
            response_shape="answer_location",
            relation_type=int(Rel.AT_LOCATION),
            weight=1.0,
        ),
        Template(
            template_id="rel.composition",
            pattern="{subject} is made of {target}.",
            slots=(_SUBJECT, SlotSpec(
                name="target", kind=SlotKind.LEMMA_LIST,
            )),
            response_shape="answer_composition",
            relation_type=int(Rel.MADE_OF),
            weight=1.0,
        ),
        Template(
            template_id="rel.purpose",
            pattern="{subject} is used for {target}.",
            slots=(_SUBJECT, SlotSpec(
                name="target", kind=SlotKind.LEMMA_LIST,
            )),
            response_shape="answer_purpose",
            relation_type=int(Rel.USED_FOR),
            weight=1.0,
        ),
        Template(
            template_id="rel.ability",
            pattern="{subject} is capable of {target}.",
            slots=(_SUBJECT, SlotSpec(
                name="target", kind=SlotKind.LEMMA_LIST,
            )),
            response_shape="answer_ability",
            relation_type=int(Rel.CAPABLE_OF),
            weight=1.0,
        ),
        Template(
            template_id="rel.cause",
            pattern="{subject} is caused by {target}.",
            slots=(_SUBJECT, SlotSpec(
                name="target", kind=SlotKind.LEMMA_LIST,
            )),
            response_shape="answer_cause",
            relation_type=int(Rel.CAUSED_BY),
            weight=1.0,
        ),
        Template(
            template_id="rel.effect",
            pattern="{subject} causes {target}.",
            slots=(_SUBJECT, SlotSpec(
                name="target", kind=SlotKind.LEMMA_LIST,
            )),
            response_shape="answer_effect",
            relation_type=int(Rel.CAUSES),
            weight=1.0,
        ),

        # ---- Elaboration (relation-walk single-target) ----
        Template(
            template_id="elab.hypernym",
            pattern="{subject} is a kind of {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.HYPERNYM),
            weight=1.0,
        ),
        Template(
            template_id="elab.has_property",
            pattern="{subject} is {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.HAS_PROPERTY),
            weight=0.95,
        ),
        Template(
            template_id="elab.meronym",
            pattern="{subject} has {target} as a part.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.MERONYM),
            weight=0.9,
        ),
        Template(
            template_id="elab.holonym",
            pattern="{subject} is part of {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.HOLONYM),
            weight=0.85,
        ),
        Template(
            template_id="elab.at_location",
            pattern="{subject} can be found at {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.AT_LOCATION),
            weight=0.9,
        ),
        Template(
            template_id="elab.used_for",
            pattern="{subject} is used for {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.USED_FOR),
            weight=0.9,
        ),
        Template(
            template_id="elab.capable_of",
            pattern="{subject} is capable of {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.CAPABLE_OF),
            weight=0.9,
        ),
        Template(
            template_id="elab.made_of",
            pattern="{subject} is made of {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.MADE_OF),
            weight=0.9,
        ),
        Template(
            template_id="elab.hyponym",
            pattern="examples of {subject} include {target}.",
            slots=(_SUBJECT, _TARGET),
            response_shape="elaborate",
            relation_type=int(Rel.HYPONYM),
            weight=0.85,
        ),
        Template(
            template_id="elab.exhausted",
            pattern="That's all I have about {subject}.",
            slots=(_SUBJECT,),
            response_shape="elaborate_exhausted",
            weight=0.5,
        ),

        # ---- Inform-turn acknowledgement ----
        Template(
            template_id="inform.belief_ack",
            pattern="Got it — noted that {raw}.",
            slots=(SlotSpec(name="raw", kind=SlotKind.LITERAL),),
            response_shape="inform_belief_ack",
            weight=1.0,
        ),
        Template(
            template_id="inform.generic_ack",
            pattern="Got it — noted about {subject}.",
            slots=(_SUBJECT,),
            response_shape="inform_generic_ack",
            weight=0.9,
        ),
        Template(
            template_id="inform.bare",
            pattern="Got it.",
            slots=(),
            response_shape="inform_bare",
            weight=0.7,
        ),

        # ---- Instruction acknowledgement ----
        Template(
            template_id="instruct.with_subject",
            pattern="OK — I'll work on {subject}.",
            slots=(_SUBJECT,),
            response_shape="instruct_with_subject",
            weight=1.0,
        ),
        Template(
            template_id="instruct.bare",
            pattern="OK.",
            slots=(),
            response_shape="instruct_bare",
            weight=0.7,
        ),

        # ---- Social ----
        Template(
            template_id="social.greeting",
            pattern="Hello.",
            slots=(),
            response_shape="social_greeting",
            weight=1.0,
        ),
        Template(
            template_id="social.thanks",
            pattern="You're welcome.",
            slots=(),
            response_shape="social_thanks",
            weight=1.0,
        ),
        Template(
            template_id="social.apology",
            pattern="No worries.",
            slots=(),
            response_shape="social_apology",
            weight=1.0,
        ),
        Template(
            template_id="social.farewell",
            pattern="Goodbye.",
            slots=(),
            response_shape="social_farewell",
            weight=1.0,
        ),
        Template(
            template_id="social.fallback",
            pattern="Hello.",
            slots=(),
            response_shape="social_fallback",
            weight=0.5,
        ),

        # ---- Fallback ----
        Template(
            template_id="fallback.no_anchor",
            pattern="I don't know what you're asking about.",
            slots=(),
            response_shape="fallback_no_anchor",
            weight=1.0,
        ),
        Template(
            template_id="fallback.rephrase",
            pattern="Could you rephrase that?",
            slots=(),
            response_shape="fallback_rephrase",
            weight=1.0,
        ),
    )


# Module-level cache so consumers can iterate without rebuilding.
_CACHED_SEEDS: tuple[Template, ...] | None = None


def seed_templates() -> tuple[Template, ...]:
    """Public accessor — builds (and caches) the seed list."""
    global _CACHED_SEEDS
    if _CACHED_SEEDS is None:
        _CACHED_SEEDS = _seed_templates()
    return _CACHED_SEEDS


def __getattr__(name):
    """Lazy module-level ``SEED_TEMPLATES`` for backwards compat."""
    if name == "SEED_TEMPLATES":
        return seed_templates()
    raise AttributeError(name)


def build_default_store():
    """Create a TemplateStore loaded with the seeds."""
    from strands.realization.store import TemplateStore
    store = TemplateStore()
    store.add_all(seed_templates())
    return store
