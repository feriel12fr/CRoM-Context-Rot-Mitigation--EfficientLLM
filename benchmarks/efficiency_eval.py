"""
Efficiency Evaluation for CRoM-EfficientLLM
- Synthetic workload to measure token savings, selection quality, and runtime.
- No third-party deps beyond numpy/matplotlib (pandas optional for CSVs).

Usage:
  python benchmarks/efficiency_eval.py --budget 0.3 --n 5000 --seed 123 --plot --save
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt  # optional
except Exception:  # pragma: no cover
    plt = None

# --- Local packers (self-contained to avoid imports during quick eval) ---
@dataclass(frozen=True)
class Chunk:
    text: str
    score: float
    tokens: int

def _estimate_tokens(text: str) -> int:
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

def budget_pack(text_chunks: Sequence[Union[Chunk, dict]], budget: int = 1000) -> List[Chunk]:
    if budget <= 0:
        raise ValueError("budget must be > 0")
    coerced: List[Chunk] = [_coerce_chunk(c, i) for i, c in enumerate(text_chunks)]
    indexed = list(enumerate(coerced))
    indexed.sort(key=lambda it: (-it[1].score, it[1].tokens, it[0]))
    selected: List[Chunk] = []
    total = 0
    for _, ch in indexed:
        if total + ch.tokens <= budget:
            selected.append(ch)
            total += ch.tokens
    return selected

def pack_fcfs(text_chunks: Sequence[Union[Chunk, dict]], budget: int) -> List[Chunk]:
    sel, total = [], 0
    for i, obj in enumerate(text_chunks):
        ch = _coerce_chunk(obj, i)
        if total + ch.tokens <= budget:
            sel.append(ch)
            total += ch.tokens
    return sel

def pack_random(text_chunks: Sequence[Union[Chunk, dict]], budget: int, seed: int = 0) -> List[Chunk]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(text_chunks))
    rng.shuffle(indices)
    sel, total = [], 0
    for i in indices:
        ch = _coerce_chunk(text_chunks[i], i)
        if total + ch.tokens <= budget:
            sel.append(ch)
            total += ch.tokens
    return sel

# --- Data generation and metrics ---

def make_synthetic_chunks(n=2000, seed=42, corr=0.6):
    rng = np.random.default_rng(seed)
    true_rel = rng.normal(0, 1, size=n)
    noise = rng.normal(0, 1, size=n) * math.sqrt(1 - corr**2)
    score = corr * true_rel + noise
    tokens = np.clip(rng.lognormal(mean=4.0, sigma=0.6, size=n).astype(int), 5, 2000)
    chunks = [Chunk(text=("x"*int(t*4)), score=float(s), tokens=int(t)) for s, t in zip(score, tokens)]
    return chunks, true_rel

def eval_once(n=5000, budget_ratio=0.3, seed=123, corr=0.6):
    chunks, true_rel = make_synthetic_chunks(n=n, seed=seed, corr=corr)
    total_tokens = sum(c.tokens for c in chunks)
    budget = int(total_tokens * budget_ratio)

    def run(name, fn):
        t0 = time.perf_counter()
        sel = fn(chunks, budget)
        dt = time.perf_counter() - t0
        idx_map = {id(c): i for i, c in enumerate(chunks)}
        picked_idx = [idx_map[id(c)] for c in sel]
        rel_sum = float(np.sum(true_rel[picked_idx])) if picked_idx else 0.0
        sel_tokens = sum(c.tokens for c in sel)
        return {
            "name": name,
            "time_ms": dt*1000,
            "selected_chunks": len(sel),
            "selected_tokens": sel_tokens,
            "tokens_budget": budget,
            "tokens_total_unpacked": total_tokens,
            "tokens_saved": total_tokens - sel_tokens,
            "save_ratio": (total_tokens - sel_tokens)/total_tokens,
            "relevance_sum": rel_sum,
        }

    rows = [
        run("budget_pack", budget_pack),
        run("fcfs", pack_fcfs),
        run("random", lambda ch, b: pack_random(ch, b, seed=seed)),
    ]
    return rows

def quality_vs_optimal(n=200, budget_ratio=0.3, seed=123, corr=0.6):
    chunks, true_rel = make_synthetic_chunks(n=n, seed=seed, corr=corr)
    budget = int(sum(c.tokens for c in chunks) * budget_ratio)
    values = np.maximum(true_rel, 0.0)

    def optimal(chunks_sub, values, budget):
        items = chunks_sub
        vals = list(values)
        B = budget
        dp = [0.0]*(B+1)
        keep = [[False]*(B+1) for _ in range(len(items))]
        for i, it in enumerate(items):
            wt = it.tokens
            val = vals[i]
            for b in range(B, wt-1, -1):
                alt = dp[b - wt] + val
                if alt > dp[b]:
                    dp[b] = alt
                    keep[i][b] = True
        b = B
        picked_idx = []
        for i in range(len(items)-1, -1, -1):
            if keep[i][b]:
                picked_idx.append(i)
                b -= items[i].tokens
        picked_idx.reverse()
        rel_sum = float(np.sum([values[i] for i in picked_idx])) if picked_idx else 0.0
        total_tokens = sum(items[i].tokens for i in picked_idx)
        return picked_idx, rel_sum, total_tokens

    opt_idx, opt_rel, opt_tokens = optimal(chunks, values, budget)

    # selections
    idx_map = {id(c): i for i, c in enumerate(chunks)}
    def rel_of(selection):
        pid = [idx_map[id(c)] for c in selection]
        return float(np.sum(values[pid])) if pid else 0.0

    sel_bp = budget_pack(chunks, budget)
    sel_fc = pack_fcfs(chunks, budget)
    sel_rd = pack_random(chunks, budget, seed=seed)

    rows = [
        {"name":"optimal_true_rel", "relevance_sum": opt_rel, "selected_tokens": opt_tokens, "selected_chunks": len(opt_idx)},
        {"name":"budget_pack_small", "relevance_sum": rel_of(sel_bp), "selected_tokens": sum(c.tokens for c in sel_bp), "selected_chunks": len(sel_bp)},
        {"name":"fcfs_small", "relevance_sum": rel_of(sel_fc), "selected_tokens": sum(c.tokens for c in sel_fc), "selected_chunks": len(sel_fc)},
        {"name":"random_small", "relevance_sum": rel_of(sel_rd), "selected_tokens": sum(c.tokens for c in sel_rd), "selected_chunks": len(sel_rd)},
    ]
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--corr", type=float, default=0.6)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    rows = eval_once(n=args.n, budget_ratio=args.budget, seed=args.seed, corr=args.corr)
    rows_q = quality_vs_optimal(n=min(200, args.n), budget_ratio=args.budget, seed=args.seed, corr=args.corr)

    print("\n=== Efficiency (n={}, budget={{:.0%}}) ===".format(args.n, args.budget))
    for r in rows:
        print("{name:12s} time={{time_ms:7.2f}}ms  save_ratio={{save_ratio:6.3f}}  tokens_saved={{tokens_saved:8d}}  rel_sum={{relevance_sum:8.3f}}".format(**r))

    print("\n=== Quality vs Optimal (subset) ===")
    for r in rows_q:
        print("{name:18s} rel_sum={{relevance_sum:8.3f}}  tokens={{selected_tokens:5d}} chunks={{selected_chunks:4d}}".format(**r))

    if pd is not None and args.save:
        pd.DataFrame(rows).to_csv("benchmarks/results_efficiency.csv", index=False)
        pd.DataFrame(rows_q).to_csv("benchmarks/results_quality.csv", index=False)
        print("Saved CSVs to benchmarks حضرتك.")

    if plt is not None and args.plot:
        # single-figure plots, no explicit colors
        x = [r["name"] for r in rows]
        y = [r["time_ms"] for r in rows]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar(x, y)
        plt.title("Packer Runtime (ms)")
        plt.xlabel("method")
        plt.ylabel("ms")
        plt.show()

if __name__ == "__main__":
    main()
