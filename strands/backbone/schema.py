"""Backbone schema — fixed-width binary node and edge layouts.

Per BDRM spec §2.3:
  Nodes are 128 bytes.
  Edges are 32 bytes.

Both are designed to be memory-mappable as numpy structured arrays,
giving O(1) random access without parsing overhead. Field byte offsets
match the spec; ``reserved`` regions are zeroed.

Node concept_type is a bitfield (low 7 bits used; high bit reserved):
  0x01  ENTITY      (concrete things, places, named entities)
  0x02  EVENT       (actions, processes, occurrences)
  0x04  PROPERTY    (attributes, qualities)
  0x08  RELATION    (predicates, relational concepts)
  0x10  FRAME       (FrameNet frames, multi-role events)
  0x20  ABSTRACT    (concepts without physical instantiation)
  0x40  QUANTIFIER  (counts, measures, modal qualifiers)

Edge relation_type uses a 16-bit ID drawn from RELATION_TAXONOMY below.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

# 16-bit relation taxonomy (spec §2.3.3, with stable numeric IDs).
class Rel(IntEnum):
    HYPERNYM             = 0x0001
    HYPONYM              = 0x0002
    MERONYM              = 0x0003
    HOLONYM              = 0x0004
    SYNONYM              = 0x0005
    ANTONYM              = 0x0006
    CAUSES               = 0x0010
    CAUSED_BY            = 0x0011
    HAS_PROPERTY         = 0x0020
    PROPERTY_OF          = 0x0021
    CAPABLE_OF           = 0x0022
    AT_LOCATION          = 0x0030
    LOCATION_OF          = 0x0031
    PART_OF              = 0x0040
    USED_FOR             = 0x0050
    HAS_PREREQUISITE     = 0x0060
    ENTAILS              = 0x0070
    TEMPORAL_BEFORE      = 0x0080
    TEMPORAL_AFTER       = 0x0081
    COREFERENTIAL        = 0x0090
    DERIVED_FROM         = 0x00A0
    RELATED_TO           = 0x00B0   # ConceptNet generic /r/RelatedTo
    SIMILAR_TO           = 0x00B1
    INSTANCE_OF          = 0x00C0
    HAS_A                = 0x00D0
    MADE_OF              = 0x00E0
    HAS_CONTEXT          = 0x00F0
    DESIRES              = 0x0100
    NOT_DESIRES          = 0x0101
    MOTIVATED_BY_GOAL    = 0x0110
    HAS_SUBEVENT         = 0x0120
    HAS_FIRST_SUBEVENT   = 0x0121
    HAS_LAST_SUBEVENT    = 0x0122
    DEFINED_AS           = 0x0130
    FORM_OF              = 0x0140
    MANNER_OF            = 0x0150


class ConceptType:
    ENTITY     = 0x01
    EVENT      = 0x02
    PROPERTY   = 0x04
    RELATION   = 0x08
    FRAME      = 0x10
    ABSTRACT   = 0x20
    QUANTIFIER = 0x40


# Source attribution byte for edges (spec §2.3.2)
class Source:
    WORDNET     = 0x01
    CONCEPTNET  = 0x02
    WIKIDATA    = 0x04
    FRAMENET    = 0x08
    PROPBANK    = 0x10
    HUMAN       = 0x20
    INFERRED    = 0x40


# 128-byte node layout — matches spec §2.3.1 exactly.
NODE_DTYPE = np.dtype([
    ("node_id",                  np.uint32),    # offset  0,  4 bytes
    ("concept_type",             np.uint8),     # offset  4,  1 byte
    ("activation_default",       np.uint16),    # offset  5,  2 bytes
    ("volatility_flag",          np.uint8),     # offset  7,  1 byte
    ("embedding_compressed",     "S16"),        # offset  8, 16 bytes (LSH digest)
    ("lemma_count",              np.uint8),     # offset 24,  1 byte
    ("lemma_offset",             np.uint32),    # offset 25,  4 bytes
    ("relationship_count",       np.uint8),     # offset 29,  1 byte
    ("relationship_offset",      np.uint32),    # offset 30,  4 bytes
    ("frame_id",                 np.uint16),    # offset 34,  2 bytes
    ("language_independent_id",  np.uint32),    # offset 36,  4 bytes
    # Strand-link bytes (BDRM extension): the codon this node renders as
    # in strand encoding. Multiple nodes can share a codon (synonyms).
    ("codon_domain",             np.uint8),     # offset 40,  1 byte
    ("codon_category",           np.uint8),     # offset 41,  1 byte
    ("codon_concept",            np.uint8),     # offset 42,  1 byte
    # Gloss table pointer — offset of this node's null-terminated UTF-8
    # definition string in the gloss_buffer file. 0 = no gloss.
    ("gloss_offset",             np.uint32),    # offset 43,  4 bytes
    ("reserved",                 "S81"),        # offset 47, 81 bytes
], align=False)

# Sanity check: dtype matches the 128-byte spec.
assert NODE_DTYPE.itemsize == 128, (
    f"NODE_DTYPE is {NODE_DTYPE.itemsize} bytes; spec requires 128"
)


# 32-byte edge layout — matches spec §2.3.2.
EDGE_DTYPE = np.dtype([
    ("target_id",          np.uint32),  # offset  0,  4 bytes
    ("relation_type",      np.uint16),  # offset  4,  2 bytes
    ("weight",             np.uint16),  # offset  6,  2 bytes
    ("confidence",         np.uint16),  # offset  8,  2 bytes
    ("context_volatility", np.uint16),  # offset 10,  2 bytes
    ("source_attribution", np.uint8),   # offset 12,  1 byte
    ("bidirectional_flag", np.uint8),   # offset 13,  1 byte
    ("compute_on_conflict",np.uint8),   # offset 14,  1 byte
    ("reserved",           "S17"),      # offset 15, 17 bytes
], align=False)

assert EDGE_DTYPE.itemsize == 32, (
    f"EDGE_DTYPE is {EDGE_DTYPE.itemsize} bytes; spec requires 32"
)
