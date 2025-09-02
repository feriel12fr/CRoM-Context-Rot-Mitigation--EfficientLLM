from crom_efficientllm.rerank_engine.rerank import hybrid_rerank

class Dummy:
    def encode(self, text, convert_to_numpy=False):
        return [ord(c) % 5 for c in str(text)[:8]]

def test_hybrid_rerank_returns_scores():
    docs = [{"text": "alpha"}, {"text": "beta"}]
    out = hybrid_rerank("alp", docs, Dummy(), alpha=0.5)
    assert len(out) == 2
    assert {"score_sparse", "score_dense", "score_final"} <= set(out[0].keys())
