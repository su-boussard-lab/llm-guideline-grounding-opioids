"""
Parse, permute, and reassemble retrieved-evidence chunks in Tier 1 open prompts.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Literal

PermutationMode = Literal["reverse", "random"]



@dataclass(frozen=True)
class RetrievedPromptParts:
    """Split of a retrieved open prompt."""

    prefix: str
    chunks: tuple[tuple[str, str], ...]  # (tag, body) — last body excludes </context>
    tail: str
    context_close_suffix: str  # e.g. "</context>\\n\\n" between chunks and vignette

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(tag for tag, _ in self.chunks)

    def chunk_order_str(self) -> str:
        return ";".join(self.chunk_ids)


def parse_retrieved_open_prompt(open_prompt: str) -> RetrievedPromptParts:
    """
  Split open_prompt into:
    - prefix: through opening <context> header (before first [pdf_...] tag)
    - chunks: list of (tag, body) in current order
    - tail: from first 'Vignette:' onward
    """
    if not open_prompt or not isinstance(open_prompt, str):
        raise ValueError("open_prompt must be a non-empty string")

    vpos = open_prompt.find("Vignette:")
    if vpos < 0:
        raise ValueError("Could not find 'Vignette:' in open_prompt")

    ctx = open_prompt[:vpos]
    tail = open_prompt[vpos:]

    parts = re.split(r"(\[pdf_[^\]]+\])", ctx)
    if len(parts) < 3:
        raise ValueError(f"Expected at least one [pdf_...] chunk tag, got {len(parts) - 1} tags")

    prefix = parts[0]
    chunks: list[tuple[str, str]] = []

    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        chunks.append((parts[i], parts[i + 1]))

    if not chunks:
        raise ValueError("No retrieved chunks parsed from open_prompt")

    context_close_suffix = ""
    last_tag, last_body = chunks[-1]
    close_match = re.search(r"(</context>\s*)$", last_body, flags=re.IGNORECASE | re.DOTALL)
    if close_match:
        context_close_suffix = close_match.group(1)
        last_body = last_body[: close_match.start()]
        chunks[-1] = (last_tag, last_body)

    return RetrievedPromptParts(
        prefix=prefix,
        chunks=tuple(chunks),
        tail=tail,
        context_close_suffix=context_close_suffix,
    )


def permute_chunks(
    chunks: tuple[tuple[str, str], ...],
    mode: PermutationMode,
    *,
    vignette_idx: int,
    random_seed: int = 42,
) -> tuple[tuple[str, str], ...]:
    """Return chunks in permuted order (same multiset of tag/body pairs)."""
    items = list(chunks)
    if mode == "reverse":
        return tuple(reversed(items))
    if mode == "random":
        rng = random.Random(random_seed + int(vignette_idx))
        order = list(range(len(items)))
        rng.shuffle(order)
        return tuple(items[i] for i in order)
    raise ValueError(f"Unknown permutation mode: {mode}")


def reassemble_open_prompt(parts: RetrievedPromptParts, chunks: tuple[tuple[str, str], ...]) -> str:
    """Rebuild open_prompt from prefix, reordered chunks, and vignette tail."""
    if not chunks:
        raise ValueError("chunks must be non-empty")

    out = parts.prefix
    for tag, body in chunks:
        out += tag + body
    out += parts.context_close_suffix
    return out + parts.tail.lstrip()


def apply_chunk_permutation(
    open_prompt: str,
    mode: PermutationMode,
    *,
    vignette_idx: int,
    random_seed: int = 42,
) -> tuple[str, str, str]:
    """
    Permute chunk order in open_prompt.

    Returns:
        new_open_prompt, original_chunk_order (semicolon-separated),
        permuted_chunk_order (semicolon-separated)
    """
    parts = parse_retrieved_open_prompt(open_prompt)
    original_order = parts.chunk_order_str()
    new_chunks = permute_chunks(
        parts.chunks,
        mode,
        vignette_idx=vignette_idx,
        random_seed=random_seed,
    )
    new_open = reassemble_open_prompt(parts, new_chunks)
    new_order = ";".join(tag for tag, _ in new_chunks)
    return new_open, original_order, new_order
