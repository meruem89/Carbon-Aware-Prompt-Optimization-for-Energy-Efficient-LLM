import time
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from src.compression_engine import compressor, compress_prompt
from src.guardrails import TokenHealingLayer
from src.evaluator import evaluate_semantic_fidelity
from src.carbon_estimator import measure_inference_carbon
from src.database import log_result

app = FastAPI(title="Carbon-Aware Prompt Optimization Engine Proxy Gateway", version="2.0")
healer = TokenHealingLayer()

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o")
    messages: list = Field(..., description="List of role/content chat entries")
    temperature: float = Field(default=0.7)
    target_rate: float = Field(default=0.5, ge=0.3, le=0.8)

def execute_background_telemetry(original: str, compressed: str, orig_tokens: int, comp_tokens: int, latency_saved: float):
    """
    Asynchronously pipelines evaluations and telemetry processing 
    outside the critical runtime client performance path.
    """
    # 1. Semantic Guardrail Evaluation
    # Since we are a mock pipeline locally, we simulate the output generation sequence
    ref_response = f"Processed response content for: {original[:30]}..."
    cand_response = f"Processed response content for: {compressed[:25]}..."
    f1_score = evaluate_semantic_fidelity(ref_response, cand_response)
    
    # 2. Hardware Carbon Fingerprinting
    carbon_metrics = measure_inference_carbon(time.sleep, latency_saved / 1000.0)
    
    # 3. Save directly to serverless database matrix
    token_delta = orig_tokens - comp_tokens
    simulated_cost_saved = (token_delta / 1000.0) * 0.0015
    
    log_result(
        original_prompt=original,
        optimized_prompt=compressed,
        token_delta=token_delta,
        latency_saved_ms=latency_saved,
        bert_f1_score=f1_score,
        simulated_cost=simulated_cost_saved,
        carbon_footprint_g=carbon_metrics.get("carbon_footprint_g", 0.0)
    )

@app.post("/v1/chat/completions")
async def handle_chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    t_start = time.perf_counter()
    
    # Extract structural text from standard payload array format
    if not request.messages or "content" not in request.messages[-1]:
        raise HTTPException(status_code=400, detail="Invalid OpenAI payload schema formatting structure.")
        
    user_payload_text = request.messages[-1]["content"]
    
    # Run Phase A: Token Healing Layer Execution
    protected_tokens = healer.extract_force_tokens(user_payload_text)
    
    # Execute Model-Driven Prompt Optimization
    loop = asyncio.get_running_loop()
    # Offload blocking compression loop execution context safely to thread pools
    compression_result = await loop.run_in_executor(
        None, 
        lambda: compressor.compress_prompt(
            user_payload_text, 
            rate=request.target_rate, 
            force_tokens=protected_tokens
        )
    )
    
    t_end = time.perf_counter()
    processing_overhead_ms = (t_end - t_start) * 1000.0
    
    # Extract results
    compressed_text = compression_result["compressed_prompt"]
    orig_tokens = compression_result["origin_tokens"]
    comp_tokens = compression_result["compressed_tokens"]
    
    # Calculate baseline proxy structural balances
    simulated_raw_latency_ms = 4500.0
    optimized_expected_latency_ms = simulated_raw_latency_ms * (comp_tokens / orig_tokens)
    latency_saved_ms = max(0.0, simulated_raw_latency_ms - optimized_expected_latency_ms - processing_overhead_ms)
    
    # Route profiling sequences to workers so the active user response stays immediate
    background_tasks.add_task(
        execute_background_telemetry,
        user_payload_text,
        compressed_text,
        orig_tokens,
        comp_tokens,
        latency_saved_ms
    )
    
    # Return structured OpenAI-compliant format block to client application
    return {
        "id": "chatcmpl-optimizedproxy",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"[PROXY COMPRESSION ENGAGED]\nOptimized Input Text Forwarded: {compressed_text}"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": comp_tokens,
            "completion_tokens": 50, # Standard placeholder
            "total_tokens": comp_tokens + 50,
            "original_prompt_tokens_saved": orig_tokens - comp_tokens,
            "gateway_processing_overhead_ms": round(processing_overhead_ms, 2)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)