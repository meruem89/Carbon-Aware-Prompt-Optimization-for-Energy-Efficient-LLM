import time
from llmlingua import PromptCompressor

# Initialize LLMLingua-2 (uses an encoder-based token classification approach)
# This model is small enough to run quickly on CPU or GPU
print("Loading LLMLingua-2 Compression Engine...")
compressor = PromptCompressor(
    model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
    device_map='cpu',
    use_llmlingua2=True
)
def compress_prompt(original_prompt: str, target_rate: float = 0.5) -> dict:
    """
    Compresses the input prompt using neural token classification.
    target_rate: The desired compression ratio (e.g., 0.5 means keep 50% of tokens)
    """
    start_time = time.perf_counter()
    
    # Execute the compression
    results = compressor.compress_prompt(
        original_prompt,
        rate=target_rate,
        force_tokens=["\n", ".", "?", "!", ",", ":"] # Guardrail: Preserve structural boundaries
    )
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return {
        "original_prompt": original_prompt,
        "compressed_prompt": results["compressed_prompt"],
        "original_tokens": results["origin_tokens"],
        "compressed_tokens": results["compressed_tokens"],
        "compression_ratio": results["rate"],
        "compression_latency_seconds": round(latency, 4)
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