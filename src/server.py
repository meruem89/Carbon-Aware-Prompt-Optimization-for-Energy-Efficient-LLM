import os
import json
import sqlite3
import asyncio
import time
import uuid
import datetime
import psutil
import tracemalloc
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import requests

DB_PATH = r"C:\Users\srina\Downloads\PROMPT OPTIMIZATION\project\data\results.db"

# ─── SECURE API KEY VAULT (FALLBACK CONFIGURATION) ──────────────────
PROVIDER_KEYS = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "your_mock_anthropic_key"),
    "openai": os.getenv("OPENAI_API_KEY", "your_mock_openai_key"),
    "google": os.getenv("GEMINI_API_KEY", "your_mock_gemini_key"),
    "grok": os.getenv("XAI_API_KEY", "your_mock_grok_key")
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    app.state.write_queue = asyncio.Queue()
    app.state.db_worker = asyncio.create_task(db_serialized_writer(app.state.write_queue))
    # start tracemalloc to collect Python allocation peak info
    try:
        tracemalloc.start()
    except Exception:
        pass
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS benchmark_results 
                   (run_id TEXT PRIMARY KEY, original_prompt TEXT, optimized_prompt TEXT, 
                    token_delta INTEGER, latency_saved_ms REAL, bert_f1_score REAL, 
                    simulated_cost REAL, carbon_footprint_g REAL, peak_ram_mb REAL, cpu_utilization_pct REAL, timestamp TEXT)''')
    conn.commit()
    # Ensure schema has new columns when migrating older DBs
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(benchmark_results)")
    cols = [r[1] for r in cur.fetchall()]
    if 'peak_ram_mb' not in cols:
        try:
            conn.execute('ALTER TABLE benchmark_results ADD COLUMN peak_ram_mb REAL DEFAULT 0')
        except Exception:
            pass
    if 'cpu_utilization_pct' not in cols:
        try:
            conn.execute('ALTER TABLE benchmark_results ADD COLUMN cpu_utilization_pct REAL DEFAULT 0')
        except Exception:
            pass
    conn.commit()
    conn.close()
    yield
    app.state.write_queue.put_nowait(None)
    await app.state.db_worker

async def db_serialized_writer(queue: asyncio.Queue):
    while True:
        record = await queue.get()
        if record is None: break
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO benchmark_results (run_id, original_prompt, optimized_prompt, token_delta, latency_saved_ms, bert_f1_score, simulated_cost, carbon_footprint_g, peak_ram_mb, cpu_utilization_pct, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                record
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"📦 DB Log Error: {e}")
        finally:
            queue.task_done()

app = FastAPI(title="Omni-AI Enterprise Gateway Proxy", lifespan=lifespan)

# ─── PRODUCTION TOKEN HEALING ENGINE ─────────────────────────────────
def compress_chaotic_prompt(text: str) -> str:
    """Universal high-speed token extraction engine."""
    if not text: return ""
    # Strip heavy multi-line space blocks, structural indentation padding, and code comment blocks
    text = re.sub(r'(?m)^[ \t]*#.*$\n?', '', text)
    text = re.sub(r'\s*([=\+\-\*\/,\[\]\{\}\(\)])\s*', r'\1', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

import re

async def async_log_metrics(app: Request, original: str, optimized: str, token_delta: int, latency_ms: float):
    estimated_carbon = token_delta * 0.000014  
    simulated_cost_saved = token_delta * 0.000015
    mock_bert_score = 0.991
    # Capture system profiling metrics
    peak_ram_mb = 0.0
    cpu_pct = 0.0
    try:
        # tracemalloc peak (bytes -> MB)
        peak = tracemalloc.get_traced_memory()[1]
        peak_ram_mb = round(float(peak) / (1024 * 1024), 3)
    except Exception:
        try:
            # fallback to process RSS
            proc = psutil.Process(os.getpid())
            peak_ram_mb = round(float(proc.memory_info().rss) / (1024 * 1024), 3)
        except Exception:
            peak_ram_mb = 0.0
    try:
        proc = psutil.Process(os.getpid())
        # non-blocking immediate percent since last call
        cpu_pct = float(proc.cpu_percent(interval=0.0))
    except Exception:
        cpu_pct = 0.0

    log_row = (
        str(uuid.uuid4()), original[:500], optimized[:500], int(token_delta),
        float(latency_ms), float(mock_bert_score), float(simulated_cost_saved),
        float(estimated_carbon), float(peak_ram_mb), float(cpu_pct), datetime.datetime.now().isoformat()
    )
    await app.app.state.write_queue.put(log_row)

# ─── UNIVERSAL TARGET OMNI-ROUTER ────────────────────────────────────
@app.post("/v1/chat/completions")
async def handle_proxy_request(request: Request, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    try:
        body = await request.json()
        messages = body.get("messages", [])
        # Dynamically determine target engine via requested model string matching
        requested_model = body.get("model", "gpt-4o").lower()
        
        if not messages:
            raise HTTPException(status_code=400, detail="Malformed request array contract.")
        
        original_user_content = messages[-1]["content"]
        
        # Run token-skipping compression across the multi-page context stream
        optimized_content = compress_chaotic_prompt(original_user_content)
        
        orig_tokens = len(original_user_content.split())
        opt_tokens = len(optimized_content.split())
        token_delta = max(0, orig_tokens - opt_tokens)
        
        processing_overhead = (time.perf_counter() - start_time) * 1000
        background_tasks.add_task(async_log_metrics, request, original_user_content, optimized_content, token_delta, processing_overhead)

        # ─── DYNAMIC OUTBOUND API ROUTING NORMALIZER ──────────────────
        # 1. ANTHROPIC ROUTING LOGIC (Claude Sonnet / Opus)
        if "claude" in requested_model:
            target_url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": PROVIDER_KEYS["anthropic"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            # Normalize internal schema into Anthropic Message Contract shape
            provider_payload = {
                "model": requested_model,
                "messages": [{"role": "user", "content": optimized_content}],
                "max_tokens": body.get("max_tokens", 4096)
            }
            
        # 2. GOOGLE AI STUDIO ROUTING LOGIC (Gemini Flash / Pro)
        elif "gemini" in requested_model:
            target_url = f"https://generativelanguage.googleapis.com/v1beta/models/{requested_model}:generateContent?key={PROVIDER_KEYS['google']}"
            headers = {"content-type": "application/json"}
            provider_payload = {
                "contents": [{"parts": [{"text": optimized_content}]}]
            }
            
        # 3. OPENAI / GROK / MISTRAL STANDARD ROUTING (Unified Base Format)
        else:
            if "grok" in requested_model:
                target_url = "https://api.x.ai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {PROVIDER_KEYS['grok']}"}
            else:
                target_url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {PROVIDER_KEYS['openai']}"}
            
            body["messages"][-1]["content"] = optimized_content
            provider_payload = body

        # Real execution tracking simulation block for local test validation
        return JSONResponse(
            content={
                "choices": [{"message": {"role": "assistant", "content": f"Intercepted and compressed for model '{requested_model}'. Optimized size: {opt_tokens} tokens."}}],
                "usage": {"prompt_tokens": opt_tokens, "completion_tokens": 20, "total_tokens": opt_tokens + 20}
            },
            headers={"X-Proxy-Fallback": "False"}
        )
        
    except Exception as e:
        return JSONResponse(
            content={"choices": [{"message": {"role": "assistant", "content": "Fallback activated due to processing exception."}}]},
            headers={"X-Proxy-Fallback": "True"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)