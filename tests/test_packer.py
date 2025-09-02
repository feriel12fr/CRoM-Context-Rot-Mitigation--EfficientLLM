from crom_efficientllm.budget_packer.packer import budget_pack, Chunk

def test_budget_pack_respects_budget():
    chunks = [Chunk("a", 1.0, 60), Chunk("b", 0.9, 50), Chunk("c", 0.5, 20)]
    sel = budget_pack(chunks, budget=70)
    assert sum(c.tokens for c in sel) <= 70

def test_budget_pack_sorting_stable():
    chunks = [
        {"text": "x", "score": 0.9, "tokens": 30},
        {"text": "y", "score": 0.9, "tokens": 20},
        {"text": "z", "score": 0.8, "tokens": 10},
    ]
    sel = budget_pack(chunks, budget=60)
    assert [c.text for c in sel] == ["y", "x", "z"]
