"""
Hybrid Rerank Engine
--------------------
Combines sparse (TF-IDF cosine) and dense (embedding cosine) scores with
min-max normalization for robust fusion.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _to_numpy(x):
    arr = np.asarray(x)
    return arr.astype(np.float32)

def _batch_encode(embed_model, texts: Sequence[str]) -> np.ndarray:
    # Try common API of sentence-transformers: encode(list, convert_to_numpy=True)
    if hasattr(embed_model, "encode"):
        try:
            return _to_numpy(embed_model.encode(list(texts), convert_to_numpy=True))
        except TypeError:
            # Fallback: per-text encode
            return _to_numpy([embed_model.encode(t) for t in texts])
    raise TypeError("embed_model must provide .encode()")

def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn <= 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def hybrid_rerank(
    query: str,
    docs: List[Dict[str, str]],
    embed_model,
    alpha: float = 0.5,
) -> List[Dict[str, object]]:
    """
    Args:
        query: query string
        docs: list of {"text": str}
        embed_model: model with .encode() -> vector(s)
        alpha: weight between sparse/dense in [0,1]
    Returns:
        ranked list of enriched docs with scores {score_sparse, score_dense, score_final}
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not docs:
        return []

    texts = [d.get("text", "") for d in docs]

    # Sparse: TF-IDF cosine
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(texts)
    Q = tfidf.transform([query])
    D = tfidf.transform(texts)
    sparse_scores = cosine_similarity(Q, D).ravel()

    # Dense: cosine(sim) between L2-normalized embeddings
    q_emb = _to_numpy(embed_model.encode(query))
    d_embs = _batch_encode(embed_model, texts)
    # L2 normalize
    def _l2norm(a):
        n = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12
        return a / n

    qn = _l2norm(q_emb.reshape(1, -1))
    dn = _l2norm(d_embs)
    dense_scores = cosine_similarity(qn, dn).ravel()

    # Min-max to [0,1] before fusion to avoid scale issues
    s_sparse = _minmax(sparse_scores)
    s_dense = _minmax(dense_scores)

    final_scores = alpha * s_sparse + (1 - alpha) * s_dense
    order = np.argsort(-final_scores)

    ranked = []
    for i in order:
        item = dict(docs[i])
        item.update(
            score_sparse=float(s_sparse[i]),
            score_dense=float(s_dense[i]),
            score_final=float(final_scores[i]),
        )
        ranked.append(item)
    return ranked
