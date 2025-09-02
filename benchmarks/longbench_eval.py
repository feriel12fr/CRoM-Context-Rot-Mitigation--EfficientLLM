"""
Benchmark script: LongBench-like evaluation.
Simulates context packing efficiency.
"""
from crom_efficientllm.budget_packer.packer import budget_pack

def evaluate():
    chunks = [{"text": f"chunk {i}", "score": i % 5, "tokens": 100} for i in range(20)]
    packed = budget_pack(chunks, budget=500)
    print("Selected:", len(packed), "chunks")

if __name__ == "__main__":
    evaluate()
