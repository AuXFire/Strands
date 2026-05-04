"""Neural ComputeModule implementation — wraps a trained TinyTransformer
checkpoint and exposes the ComputeModule interface so the deterministic
pipeline can use it.

Drop-in replacement for StubComputeModule:

    from strands.compute.module import NeuralComputeModule
    cm = NeuralComputeModule.from_checkpoint("path/to/best.pt")
    response = respond(backbone, prompt, state=state, compute=cm)

The module:
  1. Receives a Conditioning from the deterministic system
  2. Formats it into a context string via format_context()
  3. Tokenizes ([BOS] context [SEP])
  4. Generates greedy/sampled continuation up to max_new_tokens or [EOS]
  5. Returns the decoded string, OR None to defer to the deterministic
     answer if the generation looks degenerate (length 0, all spaces,
     or the deterministic answer itself which means the model didn't
     improve on it)
"""

from __future__ import annotations

from pathlib import Path

import torch

from strands.backbone.compute_module import Conditioning
from strands.compute.format import format_context
from strands.compute.model import ModelConfig, TinyTransformer
from strands.compute.tokenizer import EOS_ID, decode, encode_context


class NeuralComputeModule:
    """Inference wrapper for a trained TinyTransformer.

    The model loads once at construction; ``complete()`` is the
    per-call hot path. Threadsafe for read-only use.
    """

    def __init__(
        self,
        model: TinyTransformer,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        defer_when_empty: bool = True,
    ) -> None:
        self.model = model
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.defer_when_empty = defer_when_empty

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> "NeuralComputeModule":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ModelConfig(**ckpt["config"])
        model = TinyTransformer(cfg)
        model.load_state_dict(ckpt["model"])
        return cls(
            model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    @torch.no_grad()
    def complete(self, conditioning: Conditioning) -> str | None:
        ctx = format_context(conditioning)
        ids = torch.tensor([encode_context(ctx)], dtype=torch.long)
        if ids.shape[1] >= self.model.cfg.max_seq_len:
            # Context already at limit; nothing to generate.
            return None
        out = self.model.generate(
            ids,
            max_new_tokens=min(
                self.max_new_tokens,
                self.model.cfg.max_seq_len - ids.shape[1],
            ),
            eos_id=EOS_ID,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        generated = decode(out[0, ids.shape[1]:].tolist()).strip()
        if self.defer_when_empty and not generated:
            return None
        # Defer if the model produced exactly the deterministic answer
        # (no value-add) or whitespace-only output.
        if generated == conditioning.deterministic_answer.strip():
            return None
        return generated
