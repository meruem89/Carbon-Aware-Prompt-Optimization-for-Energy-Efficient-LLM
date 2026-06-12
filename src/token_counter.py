from transformers import AutoTokenizer

# Load the specific tokenizer for Phi-3
# We use trust_remote_code=True if required by the model's architecture
TOKENIZER_NAME = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def get_token_metrics(prompt: str, response: str = "") -> dict:
    """
    Calculates token counts for a given prompt and an optional response.
    """
    # Count input tokens
    input_tokens = len(tokenizer.encode(prompt))
    
    # Count output tokens (if a response is provided)
    output_tokens = len(tokenizer.encode(response)) if response else 0
    
    # Calculate total
    total_tokens = input_tokens + output_tokens
    
    return {
        "Input Tokens": input_tokens,
        "Output Tokens": output_tokens,
        "Total Tokens": total_tokens
    }

# --- Quick Test Block ---
if __name__ == "__main__":
    sample_prompt = "Explain machine learning in a detailed manner."
    sample_response = "Machine learning is a subset of AI that uses data to learn."
    
    metrics = get_token_metrics(sample_prompt, sample_response)
    
    print("\n--- Token Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")