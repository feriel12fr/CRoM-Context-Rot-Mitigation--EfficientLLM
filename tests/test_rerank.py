from crom_efficientllm.rerank_engine.rerank import hybrid_rerank

class Dummy:
    def encode(self, text_or_list, convert_to_numpy=False):
        if isinstance(text_or_list, list):
            return [self.encode(t) for t in text_or_list]
        vec = [ord(c) % 5 for c in str(text_or_list)[:8]]
        while len(vec) < 8:
            vec.append(0)
        return vec

def test_hybrid_rerank_returns_scores():
    docs = [{"text": "alpha"}, {"text": "beta"}]
    out = hybrid_rerank("alp", docs, Dummy(), alpha=0.5)
    assert len(out) == 2
    assert {"score_sparse", "score_dense", "score_final"} <= set(out[0].keys())
