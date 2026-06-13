import json
from fastapi.testclient import TestClient
from src.server import app

# Create an in-memory testing channel bypassing Windows ports/firewalls completely
client = TestClient(app)

PAYLOAD = {
    "model": "llama3-70b",
    "target_rate": 0.5,
    "messages": [
        {
            "role": "user", 
            "content": "Refactor this architecture code: def parse_response(url='https://api.com'): return {key: val for key, val in data.items() if val >= 200}"
        }
    ]
}

def main():
    print("Initializing In-Memory Proxy Handshake...")
    print("Loading LLMLingua-2 Engine via Guardrails Layer...")
    
    # Fire the request directly into the endpoint matrix
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    
    print("\n--- PROXY RESPONSE RECEIVED ---")
    print(f"HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Gateway Overhead Latency: {data['usage']['gateway_processing_overhead_ms']} ms")
        print(f"Original Tokens Saved: {data['usage']['original_prompt_tokens_saved']}")
        print("\n[HEALED PROMPT OUTPUT FORWARDED TO TARGET API]:")
        print(data['choices'][0]['message']['content'])
    else:
        print(f"Error Log: {response.text}")

if __name__ == "__main__":
    main()