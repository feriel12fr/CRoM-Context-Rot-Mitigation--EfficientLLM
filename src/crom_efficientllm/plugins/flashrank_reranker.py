from __future__ import annotations
from typing import List, Dict

try:
    from flashrank import Reranker
except Exception as e:  # pragma: no cover
    raise RuntimeError("flashrank not installed. Install extras: pip install .[plugins]") from e

def flashrank_rerank(query: str, docs: List[Dict[str, str]], model_name: str = "ms-marco-TinyBERT-L-2-v2") -> List[Dict]:
    rr = Reranker(model_name)
    pairs = [(query, d["text"]) for d in docs]
    scores = rr.rerank(pairs)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    return [docs[i] | {"score_flashrank": float(scores[i])} for i in order]
