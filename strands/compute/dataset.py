"""Dataset for training the neural Compute Module.

Reads JSONL records produced by RecordingComputeModule, formats each
into a (context, target) string pair, tokenizes, and batches.

Each example becomes a sequence:
  [BOS] <context_bytes> [SEP] <target_bytes> [EOS]

The training loss is masked to only the target portion — the model
isn't penalized for not predicting the context tokens.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from strands.backbone import dict_to_conditioning
from strands.compute.format import format_context
from strands.compute.tokenizer import (
    EOS_ID,
    PAD_ID,
    SEP_ID,
    encode_pair,
)


@dataclass(slots=True)
class TrainingRecord:
    ids: list[int]              # full sequence including BOS/SEP/EOS
    target_start: int           # index where target begins (after [SEP])
    raw_context: str
    raw_target: str


class ConditioningDataset(Dataset):
    """Loads a JSONL of captured Conditioning + deterministic_answer
    pairs and exposes them as tokenized training records."""

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        max_seq_len: int = 1024,
        skip_when_truncated: bool = True,
    ) -> None:
        self.path = Path(jsonl_path)
        self.records: list[TrainingRecord] = []
        skipped = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cond = dict_to_conditioning(d["conditioning"])
            ctx = format_context(cond)
            tgt = d["deterministic_answer"].strip()
            if not tgt:
                continue
            ids, target_start = encode_pair(ctx, tgt, with_eos=True)
            if len(ids) > max_seq_len:
                if skip_when_truncated:
                    skipped += 1
                    continue
                ids = ids[:max_seq_len]
                if ids[-1] != EOS_ID:
                    ids[-1] = EOS_ID
            self.records.append(TrainingRecord(
                ids=ids, target_start=target_start,
                raw_context=ctx, raw_target=tgt,
            ))
        if skipped:
            print(
                f"  skipped {skipped} records over max_seq_len={max_seq_len}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TrainingRecord:
        return self.records[idx]


def _pad(ids: list[int], length: int) -> list[int]:
    return ids + [PAD_ID] * (length - len(ids))


def collate_fn(batch: list[TrainingRecord]) -> dict[str, torch.Tensor]:
    """Pad a batch to the longest sequence. Returns:
      input_ids   (B, T)   token ids, PAD-padded
      labels      (B, T)   shifted input_ids; -100 where loss is masked
                           (PAD positions and the entire context portion)
      attention_mask (B, T) 1 for real tokens, 0 for pad
    """
    max_len = max(len(r.ids) for r in batch)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attn: list[list[int]] = []
    for r in batch:
        padded = _pad(r.ids, max_len)
        # Mask: -100 for context tokens (we don't train on them) and
        # for padding positions. The model's loss is computed against
        # the next-token prediction at each position, so labels[i] is
        # the target of input_ids[i]. We shift in the loss
        # computation, not here.
        lbl = [-100] * len(padded)
        for i in range(r.target_start, len(r.ids)):
            lbl[i] = padded[i]
        attn_row = [1] * len(r.ids) + [0] * (max_len - len(r.ids))
        input_ids.append(padded)
        labels.append(lbl)
        attn.append(attn_row)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def split(
    dataset: ConditioningDataset, *, val_fraction: float = 0.1, seed: int = 0,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Deterministic train/val split. Same seed → same split."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_indices = set(indices[:n_val])
    train: list[TrainingRecord] = []
    val: list[TrainingRecord] = []
    for i, r in enumerate(dataset.records):
        (val if i in val_indices else train).append(r)
    return train, val


def iterate_batches(
    records: list[TrainingRecord], *, batch_size: int, shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Simple in-memory batcher. Sorts within bucket-shuffle to keep
    pad waste low."""
    import random
    indices = list(range(len(records)))
    if shuffle:
        rng = random.Random(seed)
        # Bucket-shuffle: shuffle, then sort each batch-sized bucket
        # by length to reduce padding within a batch.
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        # Sort within batch by length so pad waste is bounded.
        chunk.sort(key=lambda i: len(records[i].ids))
        batch = [records[i] for i in chunk]
        yield collate_fn(batch)
