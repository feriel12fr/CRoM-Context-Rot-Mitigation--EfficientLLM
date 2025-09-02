from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crom_efficientllm.budget_packer.packer import budget_pack, Chunk
from crom_efficientllm.rerank_engine.rerank import hybrid_rerank

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# Optional plugins are imported lazily when flags are set

@dataclass
class Doc:
    id: str
    text: str

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_corpus(path: str) -> List[Doc]:
    rows = load_jsonl(path)
    return [Doc(id=str(r.get("id", i)), text=str(r["text"])) for i, r in enumerate(rows)]

def sparse_retrieval(query: str, corpus: Sequence[Doc], k: int = 100) -> List[Dict]:
    texts = [d.text for d in corpus]
    vect = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
    D = vect.transform(texts)
    Q = vect.transform([query])
    sims = cosine_similarity(Q, D).ravel()
    order = np.argsort(-sims)[:k]
    return [{"id": corpus[i].id, "text": corpus[i].text, "score_sparse": float(sims[i])} for i in order]

def dense_embed_model(name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install with `pip install -e .`.")
    return SentenceTransformer(name)

def _apply_flashrank(query: str, docs: List[Dict], model_name: str) -> List[Dict]:
    try:
        from crom_efficientllm.plugins.flashrank_reranker import flashrank_rerank
    except Exception as e:  # pragma: no cover
        raise RuntimeError("FlashRank plugin not available. Install extras: pip install .[plugins]") from e
    ranked = flashrank_rerank(query, docs, model_name=model_name)
    # Normalize plugin score to 0..1 and put into score_final
    scores = np.array([d.get("score_flashrank", 0.0) for d in ranked], dtype=np.float32)
    if scores.size and float(scores.max() - scores.min()) > 1e-12:
        s = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        s = np.zeros_like(scores)
    for i, d in enumerate(ranked):
        d["score_final"] = float(s[i])
    return ranked

def _apply_llmlingua(text: str, ratio: float) -> str:
    try:
        from crom_efficientllm.plugins.llmlingua_compressor import compress_prompt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("LLMLingua plugin not available. Install extras: pip install .[plugins]") from e
    return compress_prompt(text, target_ratio=ratio)

def _save_evidently_report(all_embs: List[List[float]], out_html: str) -> None:
    try:
        from crom_efficientllm.plugins.evidently_drift import drift_report
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Evidently plugin not available. Install extras: pip install .[plugins]") from e
    n = len(all_embs)
    if n < 4:
        return
    ref = all_embs[: n // 2]
    cur = all_embs[n // 2 :]
    rep = drift_report(ref, cur)
    rep.save_html(out_html)

def mock_llm_generate(prompt: str) -> str:
    time.sleep(0.005)  # simulate small latency
    return "[MOCK] " + prompt[:160]

def e2e(args: argparse.Namespace) -> None:
    corpus = build_corpus(args.corpus)
    queries = [r["query"] for r in load_jsonl(args.queries)]
    embed = dense_embed_model(args.model)
    all_embs: List[List[float]] = []

    t0 = time.perf_counter()
    all_rows = []
    for q in queries:
        t_s = time.perf_counter()
        cands = sparse_retrieval(q, corpus, k=args.k)
        t_sparse = (time.perf_counter() - t_s) * 1000

        t_r = time.perf_counter()
        if args.use_flashrank:
            reranked = _apply_flashrank(q, cands, args.flashrank_model)
        else:
            reranked = hybrid_rerank(q, cands, embed, alpha=args.alpha)
        t_rerank = (time.perf_counter() - t_r) * 1000

        # token heuristic + budget pack
        chunks = [
            Chunk(text=d["text"], score=d.get("score_final", d.get("score_sparse", 0.0)), tokens=max(1, len(d["text"]) // 4))
            for d in reranked
        ]
        budget_tokens = int(sum(c.tokens for c in chunks) * args.budget)
        t_p = time.perf_counter()
        packed = budget_pack(chunks, budget=budget_tokens)
        t_pack = (time.perf_counter() - t_p) * 1000

        prompt = "\n\n".join(c.text for c in packed) + f"\n\nQ: {q}\nA:"
        if args.use_llmlingua:
            prompt = _apply_llmlingua(prompt, ratio=args.compress_ratio)

        # collect embeddings for drift snapshot (mean-pooled)
        with np.errstate(all="ignore"):
            if len(packed) > 0:
                doc_embs = embed.encode([c.text for c in packed], convert_to_numpy=True)
                vec = np.mean(doc_embs, axis=0).tolist()
                all_embs.append(vec)

        t_l = time.perf_counter()
        _ = mock_llm_generate(prompt)
        t_llm = (time.perf_counter() - t_l) * 1000

        total = (time.perf_counter() - t_s) * 1000
        all_rows.append({
            "query": q,
            "sparse_ms": t_sparse,
            "rerank_ms": t_rerank,
            "pack_ms": t_pack,
            "llm_ms": t_llm,
            "total_ms": total,
            "packed_tokens": sum(c.tokens for c in packed),
            "orig_tokens": sum(c.tokens for c in chunks),
            "save_ratio": 1 - (sum(c.tokens for c in packed) / max(1, sum(c.tokens for c in chunks))),
            "used_flashrank": bool(args.use_flashrank),
            "used_llmlingua": bool(args.use_llmlingua),
        })

    elapsed = (time.perf_counter() - t0) * 1000
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "e2e_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved results -> {out_path} ({len(all_rows)} queries) ; elapsed={elapsed:.2f}ms")

    if args.use_evidently and all_embs:
        html_path = os.path.join(args.out_dir, "evidently_report.html")
        _save_evidently_report(all_embs, html_path)
        print(f"evidently report -> {html_path}")

def budget_sweep(args: argparse.Namespace) -> None:
    import itertools
    corpus = build_corpus(args.corpus)
    queries = [r["query"] for r in load_jsonl(args.queries)][: args.max_q]
    embed = dense_embed_model(args.model)

    budgets = [b / 100.0 for b in range(args.b_min, args.b_max + 1, args.b_step)]
    rows = []
    for q, b in itertools.product(queries, budgets):
        cands = sparse_retrieval(q, corpus, k=args.k)
        reranked = hybrid_rerank(q, cands, embed, alpha=args.alpha)
        chunks = [Chunk(text=d["text"], score=d["score_final"], tokens=max(1, len(d["text"]) // 4)) for d in reranked]
        budget_tokens = int(sum(c.tokens for c in chunks) * b)
        packed = budget_pack(chunks, budget=budget_tokens)
        rows.append({
            "query": q,
            "budget": b,
            "packed_tokens": sum(c.tokens for c in packed),
            "orig_tokens": sum(c.tokens for c in chunks),
            "save_ratio": 1 - (sum(c.tokens for c in packed) / max(1, sum(c.tokens for c in chunks))),
            "avg_score": float(np.mean([c.score for c in packed])) if packed else 0.0,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "budget_sweep.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved results -> {out_path} ; points={len(rows)}")

    if args.save_plots:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            import matplotlib.pyplot as _plt
        except Exception:
            print("[warn] matplotlib not installed; install dev extras: pip install -e .[dev]")
        else:
            # Aggregate by budget
            import collections
            agg = collections.defaultdict(list)
            for r in rows:
                agg[r["budget"]].append(r)
            budgets_sorted = sorted(agg.keys())
            avg_save = [float(np.mean([x["save_ratio"] for x in agg[b]])) for b in budgets_sorted]
            avg_score = [float(np.mean([x["avg_score"] for x in agg[b]])) for b in budgets_sorted]

            _plt.figure()
            _plt.plot([b * 100 for b in budgets_sorted], [s * 100 for s in avg_save], marker="o")
            _plt.xlabel("Budget (%)")
            _plt.ylabel("Avg Save Ratio (%)")
            _plt.title("Budget Sweep: Save Ratio vs Budget")
            _plt.grid(True)
            _plt.tight_layout()
            _plt.savefig(os.path.join(args.out_dir, "budget_sweep.png"))

            _plt.figure()
            _plt.plot([s * 100 for s in avg_save], avg_score, marker="o")
            _plt.xlabel("Save Ratio (%)")
            _plt.ylabel("Avg Score (packed)")
            _plt.title("Pareto: Quality vs Savings")
            _plt.grid(True)
            _plt.tight_layout()
            _plt.savefig(os.path.join(args.out_dir, "budget_pareto.png"))
            print("plots ->", os.path.join(args.out_dir, "budget_sweep.png"), ",", os.path.join(args.out_dir, "budget_pareto.png"))

def scaling(args: argparse.Namespace) -> None:
    def make_synth(n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        tokens = np.clip(rng.lognormal(4.0, 0.6, n).astype(int), 5, 2000)
        score = rng.normal(0, 1, n)
        return [Chunk(text="x" * int(t * 4), score=float(s), tokens=int(t)) for s, t in zip(score, tokens)]

    for n in [1000, 5000, 10000, 20000, 50000, 100000]:
        if n > args.n_max:
            break
        chunks = make_synth(n)
        budget = int(sum(c.tokens for c in chunks) * args.budget)
        t0 = time.perf_counter()
        _ = budget_pack(chunks, budget)
        ms = (time.perf_counter() - t0) * 1000
        print(f"n={n:6d}  budget={args.budget:.0%}  time={ms:8.2f} ms")

def dp_curve(args: argparse.Namespace) -> None:
    def make_synth(n: int, seed: int = 123, corr: float = 0.6):
        rng = np.random.default_rng(seed)
        true_rel = rng.normal(0, 1, n)
        noise = rng.normal(0, 1, n) * np.sqrt(1 - corr**2)
        score = corr * true_rel + noise
        tokens = np.clip(rng.lognormal(4.0, 0.6, n).astype(int), 5, 2000)
        chunks = [Chunk(text="x" * int(t * 4), score=float(s), tokens=int(t)) for s, t in zip(score, tokens)]
        return chunks, true_rel

    def optimal(chunks: Sequence[Chunk], values: np.ndarray, budget: int) -> float:
        B = budget
        dp = np.zeros(B + 1, dtype=np.float32)
        for i, ch in enumerate(chunks):
            wt = ch.tokens
            val = max(0.0, float(values[i]))
            for b in range(B, wt - 1, -1):
                dp[b] = max(dp[b], dp[b - wt] + val)
        return float(dp[B])

    chunks, true_rel = make_synth(args.n)
    total = sum(c.tokens for c in chunks)
    budgets = [int(total * b / 100.0) for b in range(args.b_min, args.b_max + 1, args.b_step)]
    out_rows = []

    for B in budgets:
        sel = budget_pack(chunks, B)
        idx_map = {id(c): i for i, c in enumerate(chunks)}
        rel_bp = float(np.sum([max(0.0, true_rel[idx_map[id(c)]]) for c in sel]))
        rel_opt = optimal(chunks[: args.n_opt], true_rel[: args.n_opt], min(B, sum(c.tokens for c in chunks[: args.n_opt])))
        pct = rel_bp / max(rel_opt, 1e-9)
        out_rows.append({"budget": B, "pct": pct, "rel_bp": rel_bp, "rel_opt": rel_opt})
        print(f"budget={B:8d}  rel_bp={rel_bp:8.3f}  rel_opt≈{rel_opt:8.3f}  pct≈{pct*100:5.1f}% (subset n={args.n_opt})")

    if args.save_plots:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            import matplotlib.pyplot as _plt
        except Exception:
            print("[warn] matplotlib not installed; install dev extras: pip install -e .[dev]")
        else:
            _plt.figure()
            xs = [r["budget"] * 100.0 / total for r in out_rows]
            ys = [r["pct"] * 100 for r in out_rows]
            _plt.plot(xs, ys, marker="o")
            _plt.xlabel("Budget (%)")
            _plt.ylabel("% of optimal (subset)")
            _plt.title("DP Curve: Greedy vs Optimal")
            _plt.grid(True)
            _plt.tight_layout()
            os.makedirs(args.out_dir, exist_ok=True)
            _plt.savefig(os.path.join(args.out_dir, "dp_curve.png"))
            print("plot ->", os.path.join(args.out_dir, "dp_curve.png"))

def compare_haystack(args: argparse.Namespace) -> None:
    try:
        from haystack.nodes import BM25Retriever, SentenceTransformersRetriever
        from haystack.document_stores import InMemoryDocumentStore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Install extras: pip install .[haystack]") from e

    corpus = build_corpus(args.corpus)
    docs = [{"content": d.text, "meta": {"id": d.id}} for d in corpus]
    store = InMemoryDocumentStore(use_bm25=True)
    store.write_documents(docs)

    bm25 = BM25Retriever(document_store=store)
    dretr = SentenceTransformersRetriever(document_store=store, model_name_or_path=args.model)

    queries = [r["query"] for r in load_jsonl(args.queries)][: args.max_q]
    for q in queries:
        t0 = time.perf_counter()
        bm = bm25.retrieve(q, top_k=args.k)
        dn = dretr.retrieve(q, top_k=args.k)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{q[:40]:40s}  bm25={len(bm):3d}  dense={len(dn):3d}  time={ms:7.2f} ms")

def main() -> None:
    ap = argparse.ArgumentParser(prog="crom-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("e2e", help="end-to-end: retrieval → rerank → pack → mock LLM")
    p.add_argument("--corpus", default="examples/corpus/sample_docs.jsonl")
    p.add_argument("--queries", default="examples/corpus/sample_queries.jsonl")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--budget", type=float, default=0.3)
    # plugins
    p.add_argument("--use-flashrank", action="store_true")
    p.add_argument("--flashrank-model", default="ms-marco-TinyBERT-L-2-v2")
    p.add_argument("--use-llmlingua", action="store_true")
    p.add_argument("--compress-ratio", type=float, default=0.6)
    p.add_argument("--use-evidently", action="store_true")

    p.add_argument("--out-dir", default="benchmarks/out")
    p.set_defaults(func=e2e)

    p2 = sub.add_parser("sweep", help="budget sweep + Pareto csv")
    p2.add_argument("--corpus", default="examples/corpus/sample_docs.jsonl")
    p2.add_argument("--queries", default="examples/corpus/sample_queries.jsonl")
    p2.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p2.add_argument("--k", type=int, default=200)
    p2.add_argument("--alpha", type=float, default=0.5)
    p2.add_argument("--b-min", type=int, default=10)
    p2.add_argument("--b-max", type=int, default=90)
    p2.add_argument("--b-step", type=int, default=10)
    p2.add_argument("--max-q", type=int, default=20)
    p2.add_argument("--out-dir", default="benchmarks/out")
    p2.add_argument("--save-plots", action="store_true")
    p2.set_defaults(func=budget_sweep)

    p3 = sub.add_parser("scale", help="scaling runtime with synthetic data")
    p3.add_argument("--n-max", type=int, default=100000)
    p3.add_argument("--budget", type=float, default=0.3)
    p3.set_defaults(func=scaling)

    p4 = sub.add_parser("dp-curve", help="% of optimal vs budget (synthetic)")
    p4.add_argument("--n", type=int, default=2000)
    p4.add_argument("--n-opt", type=int, default=200)
    p4.add_argument("--b-min", type=int, default=10)
    p4.add_argument("--b-max", type=int, default=90)
    p4.add_argument("--b-step", type=int, default=10)
    p4.add_argument("--out-dir", default="benchmarks/out")
    p4.add_argument("--save-plots", action="store_true")
    p4.set_defaults(func=dp_curve)

    p5 = sub.add_parser("haystack-compare", help="compare BM25 vs dense retrievers (Haystack)")
    p5.add_argument("--corpus", default="examples/corpus/sample_docs.jsonl")
    p5.add_argument("--queries", default="examples/corpus/sample_queries.jsonl")
    p5.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p5.add_argument("--k", type=int, default=50)
    p5.add_argument("--max-q", type=int, default=10)
    p5.set_defaults(func=compare_haystack)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
