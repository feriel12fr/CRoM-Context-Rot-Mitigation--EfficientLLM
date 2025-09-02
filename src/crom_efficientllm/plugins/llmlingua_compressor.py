from __future__ import annotations

try:
    from llmlingua import PromptCompressor
except Exception as e:  # pragma: no cover
    raise RuntimeError("llmlingua not installed. Install extras: pip install .[plugins]") from e

def compress_prompt(text: str, target_ratio: float = 0.5) -> str:
    pc = PromptCompressor()
    out = pc.compress(text, target_ratio=target_ratio)
    return out["compressed_prompt"] if isinstance(out, dict) and "compressed_prompt" in out else str(out)
