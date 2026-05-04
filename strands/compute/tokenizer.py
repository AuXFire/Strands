"""Byte-level tokenizer for the neural Compute Module.

We use raw UTF-8 bytes (256 vocab) plus a small set of special tokens.
Byte-level is the simplest universal tokenizer — no OOV, no
preprocessing, no language assumptions. Sequences are longer than
with subword tokenization, but for a small CPU-trained model the
trade-off is fine.

Total vocab: 256 (bytes) + 4 special = 260 tokens.

Special token IDs:
  256  PAD  — padding
  257  BOS  — beginning of sequence
  258  EOS  — end of sequence
  259  SEP  — separates context from answer

The byte → int mapping is straightforward: byte b maps to id b.
"""

from __future__ import annotations

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
SEP_ID = 259
VOCAB_SIZE = 260


def encode(text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
    """Encode a string to a list of token ids."""
    ids = list(text.encode("utf-8"))
    if bos:
        ids = [BOS_ID] + ids
    if eos:
        ids = ids + [EOS_ID]
    return ids


def decode(ids: list[int]) -> str:
    """Decode a list of token ids back to a string. Special tokens are
    dropped (they have no surface form)."""
    bytes_only = bytes(i for i in ids if 0 <= i < 256)
    # Use 'replace' on the off chance a partial UTF-8 sequence got
    # generated mid-word during sampling.
    return bytes_only.decode("utf-8", errors="replace")


def encode_pair(
    context: str, target: str, *, with_eos: bool = True,
) -> tuple[list[int], int]:
    """Encode a (context, target) pair as ``[BOS] context_bytes [SEP]
    target_bytes [EOS]``. Returns (ids, target_start_index) where
    target_start_index is the position in the id list where the
    target tokens begin (used for masking the loss to only the
    target portion).
    """
    ctx_ids = encode(context)
    tgt_ids = encode(target)
    ids = [BOS_ID] + ctx_ids + [SEP_ID] + tgt_ids
    target_start = 1 + len(ctx_ids) + 1  # [BOS] + ctx + [SEP] = target_start
    if with_eos:
        ids = ids + [EOS_ID]
    return ids, target_start


def encode_context(context: str) -> list[int]:
    """Encode a context for inference: ``[BOS] context_bytes [SEP]``.
    The model continues from here to produce the target."""
    return [BOS_ID] + encode(context) + [SEP_ID]
