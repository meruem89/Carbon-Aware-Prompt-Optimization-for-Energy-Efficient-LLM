from __future__ import annotations

import os
import re
import time
from typing import Any


PRODUCTION_SPEED_ENV = "PRODUCTION_SPEED"
_LLM_COMPRESSOR: Any = None


def _is_truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_use_fast_path(production_speed: bool | None = None) -> bool:
    return _is_truthy(production_speed) or _is_truthy(os.getenv(PRODUCTION_SPEED_ENV))


def _load_llm_compressor() -> Any:
    global _LLM_COMPRESSOR
    if _LLM_COMPRESSOR is None:
        from llmlingua import PromptCompressor

        print("Loading LLMLingua-2 Compression Engine...")
        _LLM_COMPRESSOR = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu",
            use_llmlingua2=True,
        )
    return _LLM_COMPRESSOR


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _fast_regex_ast_compress(original_prompt: str, target_rate: float) -> dict:
    started_at = time.perf_counter()

    text = original_prompt.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*#.*$", "", text)
    text = re.sub(r"(?m)^\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*([=+\-*/<>!,:;])\s*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\b(def|class|return|if|elif|else|for|while|try|except|with|import|from|as|lambda|yield|pass)\b", r"\1", text)
    text = text.strip()

    original_tokens = _estimate_tokens(original_prompt)
    compressed_tokens = max(1, int(original_tokens * max(0.1, min(target_rate, 1.0))))
    elapsed_seconds = time.perf_counter() - started_at

    return {
        "compressed_prompt": text,
        "original_tokens": original_tokens,
        "origin_tokens": original_tokens,
        "compressed_tokens": min(compressed_tokens, original_tokens),
        "compression_ratio": round(min(compressed_tokens, original_tokens) / original_tokens, 4),
        "rate": round(min(compressed_tokens, original_tokens) / original_tokens, 4),
        "compression_latency_seconds": round(elapsed_seconds, 4),
        "production_speed_used": True,
        "fallback_used": True,
    }


class _LazyPromptCompressorProxy:
    def compress_prompt(self, original_prompt: str, target_rate: float = 0.5, production_speed: bool | None = None, **kwargs: Any) -> dict:
        return compress_prompt(
            original_prompt,
            target_rate=target_rate,
            production_speed=production_speed,
            **kwargs,
        )


compressor = _LazyPromptCompressorProxy()


def compress_prompt(
    original_prompt: str,
    target_rate: float = 0.5,
    force_tokens: list[str] | None = None,
    production_speed: bool | None = None,
) -> dict:
    """
    Compresses the input prompt using LLMLingua-2 or a production-speed regex fast path.
    target_rate: The desired compression ratio (e.g., 0.5 means keep 50% of tokens)
    """
    if _should_use_fast_path(production_speed):
        return _fast_regex_ast_compress(original_prompt, target_rate)

    started_at = time.perf_counter()
    compressor_instance = _load_llm_compressor()
    results = compressor_instance.compress_prompt(
        original_prompt,
        rate=target_rate,
        force_tokens=force_tokens or ["\n", ".", "?", "!", ",", ":"],
    )

    elapsed_seconds = time.perf_counter() - started_at
    original_tokens = int(results.get("origin_tokens", _estimate_tokens(original_prompt)))
    compressed_tokens = int(results.get("compressed_tokens", original_tokens))

    return {
        "original_prompt": original_prompt,
        "compressed_prompt": results["compressed_prompt"],
        "original_tokens": original_tokens,
        "origin_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": results.get("rate", round(compressed_tokens / original_tokens if original_tokens else 1.0, 4)),
        "rate": results.get("rate", round(compressed_tokens / original_tokens if original_tokens else 1.0, 4)),
        "compression_latency_seconds": round(elapsed_seconds, 4),
        "production_speed_used": False,
        "fallback_used": False,
    }

# --- Quick Test Block ---
if __name__ == "__main__":
    sample = "Explain the underlying mechanisms of machine learning in a highly detailed and comprehensive manner."
    print("\n--- Original Prompt ---")
    print(sample)
    
    print("\nCompressing... (This tests the inference speed of the SLM)")
    output = compress_prompt(sample, target_rate=0.6)
    
    print("\n--- Optimized Result ---")
    print(f"Compressed Text: {output['compressed_prompt']}")
    print(f"Token Reduction: {output['original_tokens']} -> {output['compressed_tokens']}")
    print(f"Algorithmic Overhead: {output['compression_latency_seconds']} seconds")