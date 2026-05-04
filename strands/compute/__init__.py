"""Neural Compute Module — small transformer that plugs into the
deterministic BDRM pipeline as a fluent NLG fallback for low-
confidence cases.

See ``strands/backbone/compute_module.py`` for the protocol; this
package is one implementation."""

from strands.compute.format import format_context, format_training_example
from strands.compute.model import ModelConfig, TinyTransformer
from strands.compute.module import NeuralComputeModule
from strands.compute.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    VOCAB_SIZE,
    decode,
    encode,
    encode_context,
    encode_pair,
)

__all__ = [
    "BOS_ID",
    "EOS_ID",
    "ModelConfig",
    "NeuralComputeModule",
    "PAD_ID",
    "SEP_ID",
    "TinyTransformer",
    "VOCAB_SIZE",
    "decode",
    "encode",
    "encode_context",
    "encode_pair",
    "format_context",
    "format_training_example",
]
