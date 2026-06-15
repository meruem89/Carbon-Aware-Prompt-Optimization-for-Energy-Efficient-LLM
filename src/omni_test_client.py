import requests

PROXY_URL = "http://127.0.0.1:8000/v1/chat/completions"

def test_omni_routing_pipeline():
    # Large chaotic data block with comments, uneven padding, and unformatted data
    chaotic_context_payload = """
    # Enterprise Data Target Metrics Log
    # Inefficient spaces burn millions of enterprise dollars at scale
    
    def process_data_stream(   input_frame   ):
        
        
        output_result = input_frame * 1.042
        
        
        return output_result
        
    Execute deep structure analytics across this target block.
    """
    
    # 🧪 Sequential validation sweep across completely different foundational AI models
    target_models = ["claude-3-5-sonnet", "gemini-1.5-pro", "grok-2", "gpt-4o"]
    
    for model in target_models:
        print(f"\n🚀 Routing unoptimized prompt matrix to provider model target: [{model}]...")
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": chaotic_context_payload}],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(PROXY_URL, json=payload, timeout=10)
            print(f"✅ Intercept Complete | Status: {response.status_code}")
            print(f"Response Payload text: {response.json()['choices'][0]['message']['content']}")
        except Exception as e:
            print(f"❌ Target Refused Connection: {e}")

if __name__ == "__main__":
    test_omni_routing_pipeline()