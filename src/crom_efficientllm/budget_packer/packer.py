"""
Budget Packer
-------------
Greedy packing of highest-scoring chunks under a token budget.
- Stable ordering (score desc, tokens asc, original index asc)
- Input validation and optional token estimation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple, Union, Optional

@dataclass(frozen=True)
class Chunk:
    text: str
    score: float
    tokens: int

def _estimate_tokens(text: str) -> int:
    """Lightweight heuristic when `tokens` absent. Avoids heavy tokenizers.
    Why: keeps demo dependency-light and deterministic.
    """
    # approx: 4 chars ≈ 1 token; floor at 1
    return max(1, len(text) // 4)

def _coerce_chunk(obj: Union[Chunk, dict], idx: int) -> Chunk:
    if isinstance(obj, Chunk):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Chunk #{idx} must be Chunk or dict, got {type(obj)}")
    text = str(obj.get("text", ""))
    if not text:
        raise ValueError(f"Chunk #{idx} has empty text")
    score = float(obj.get("score", 0.0))
    tokens = int(obj["tokens"]) if "tokens" in obj else _estimate_tokens(text)
    if tokens <= 0:
        raise ValueError(f"Chunk #{idx} has non-positive tokens: {tokens}")
    return Chunk(text=text, score=score, tokens=tokens)

def budget_pack(
    text_chunks: Sequence[Union[Chunk, dict]],
    budget: int = 1000,
) -> List[Chunk]:
    """
    Args:
        text_chunks: iterable of Chunk or dict with keys {text, score, tokens}
        budget: max token budget (int > 0)
    Returns:
        list of selected chunks (order of selection)
    """
    if budget <= 0:
        raise ValueError("budget must be > 0")

    coerced: List[Chunk] = [_coerce_chunk(c, i) for i, c in enumerate(text_chunks)]

    # stable sort by (-score, tokens, original_index)
    indexed: List[Tuple[int, Chunk]] = list(enumerate(coerced))
    indexed.sort(key=lambda it: (-it[1].score, it[1].tokens, it[0]))

    selected: List[Chunk] = []
    total = 0
    for _, ch in indexed:
        if total + ch.tokens <= budget:
            selected.append(ch)
            total += ch.tokens
    return selected

def pack_summary(selected: Sequence[Chunk]) -> dict:
    tokens = sum(c.tokens for c in selected)
    return {
        "num_chunks": len(selected),
        "tokens": tokens,
        "avg_score": (sum(c.score for c in selected) / len(selected)) if selected else 0.0,
    }
