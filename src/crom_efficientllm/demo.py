"""
Demo & Metrics Server for CRoM-EfficientLLM
------------------------------------------
- `crom-demo demo`  : run sample pipeline
- `crom-demo serve` : start Flask + Prometheus metrics on :8000
"""
from __future__ import annotations

import argparse
from typing import List

from flask import Flask, Response
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

from crom_efficientllm.budget_packer.packer import budget_pack, pack_summary, Chunk
from crom_efficientllm.rerank_engine.rerank import hybrid_rerank
from crom_efficientllm.drift_estimator.estimator import DriftEstimator, DriftMode

# ---- Prometheus metrics ----
TOKENS_SAVED = Gauge("crom_tokens_saved", "Tokens saved by budget packer")
DRIFT_ALERTS = Counter("crom_drift_alerts_total", "Total drift alerts emitted")

class DummyEmbed:
    def encode(self, text, convert_to_numpy=False):
        vec = [ord(c) % 7 for c in str(text)[:16]]
        return vec

def run_demo() -> None:
    chunks: List[Chunk] = [
        Chunk(text="AI ethics is crucial", score=0.9, tokens=50),
        Chunk(text="Unrelated text", score=0.2, tokens=40),
        Chunk(text="Drift detection research", score=0.8, tokens=60),
    ]
    packed = budget_pack(chunks, budget=80)
    summary = pack_summary(packed)
    print("📦 Packed:", [c.text for c in packed], summary)

    docs = [{"text": "AI drift measurement"}, {"text": "Cooking recipes"}]
    reranked = hybrid_rerank("AI ethics", docs, DummyEmbed(), alpha=0.5)
    print("🔎 Reranked:", [d["text"] for d in reranked])

    de = DriftEstimator(threshold=0.5, mode=DriftMode.L2)
    print("⚙️ Drift state:", de.state())
    print("⚠️ Drift alert?", de.update([1, 2, 3]))
    print("⚠️ Drift alert?", de.update([10, 10, 10]))
    print("⚙️ Drift state:", de.state())

    # Update metrics
    TOKENS_SAVED.set(max(0, sum(c.tokens for c in chunks) - summary["tokens"]))
    alert1, *_ = de.update([1, 2, 3])
    alert2, *_ = de.update([10, 10, 10])
    if alert1:
        DRIFT_ALERTS.inc()
    if alert2:
        DRIFT_ALERTS.inc()

def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    return app

def main() -> None:
    parser = argparse.ArgumentParser(prog="crom-demo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run sample pipeline")

    pserve = sub.add_parser("serve", help="start metrics server on :8000")
    pserve.add_argument("--host", default="0.0.0.0")
    pserve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.cmd == "demo":
        run_demo()
        return

    if args.cmd == "serve":
        app = create_app()
        app.run(host=args.host, port=args.port)
        return

if __name__ == "__main__":
    main()
