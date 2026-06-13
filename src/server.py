import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from src.carbon_estimator import measure_inference_carbon
from src.database import init_db
from src.guardrails import TokenHealingLayer


LOGGER = logging.getLogger(__name__)
DATABASE_PATH = Path(r"C:\Users\srina\Downloads\PROMPT OPTIMIZATION\project\data\results.db")
POOL_SIZE = 4


class SQLiteConnectionPool:
    def __init__(self, db_path: Path, pool_size: int = POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections: list[sqlite3.Connection] = []
        self._available: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._closed = False

        for _ in range(pool_size):
            connection = self._create_connection()
            self._connections.append(connection)
            self._available.put(connection)

    def _create_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=3000")
        return connection

    def acquire(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("SQLite connection pool is closed")
        return self._available.get()

    def release(self, connection: sqlite3.Connection) -> None:
        if self._closed:
            connection.close()
            return
        self._available.put(connection)

    def close(self) -> None:
        self._closed = True
        for connection in self._connections:
            try:
                connection.close()
            except Exception:
                LOGGER.exception("Failed to close SQLite connection cleanly")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db(DATABASE_PATH)
        app.state.db_pool = SQLiteConnectionPool(DATABASE_PATH)
    except Exception:
        LOGGER.exception("Database pool initialization failed; telemetry persistence will be disabled")
        app.state.db_pool = None

    yield

    db_pool = getattr(app.state, "db_pool", None)
    if db_pool is not None:
        db_pool.close()


app = FastAPI(title="Carbon-Aware Prompt Optimization Engine Proxy Gateway", version="2.0", lifespan=lifespan)
healer = TokenHealingLayer()

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o")
    messages: list = Field(..., description="List of role/content chat entries")
    temperature: float = Field(default=0.7)
    target_rate: float = Field(default=0.5, ge=0.3, le=0.8)


def _approximate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def _compress_with_fallback(
    original_prompt: str,
    target_rate: float,
    force_tokens: list[str],
    production_speed: bool,
) -> dict:
    started_at = time.perf_counter()

    try:
        from src.compression_engine import compress_prompt

        compression_result = compress_prompt(original_prompt, target_rate=target_rate, force_tokens=force_tokens, production_speed=production_speed)
        elapsed_seconds = time.perf_counter() - started_at

        return {
            "compressed_prompt": compression_result["compressed_prompt"],
            "original_tokens": compression_result["origin_tokens"],
            "compressed_tokens": compression_result["compressed_tokens"],
            "compression_ratio": compression_result.get("rate", 1.0),
            "compression_latency_seconds": round(elapsed_seconds, 4),
            "fallback_used": bool(compression_result.get("fallback_used", False)),
            "fallback_reason": "",
        }
    except Exception as exc:
        LOGGER.exception("Compression failed; returning the original prompt as a safe fallback")
        elapsed_seconds = time.perf_counter() - started_at
        original_tokens = _approximate_token_count(original_prompt)

        return {
            "compressed_prompt": original_prompt,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "compression_ratio": 1.0,
            "compression_latency_seconds": round(elapsed_seconds, 4),
            "fallback_used": True,
            "fallback_reason": str(exc),
        }


def _persist_background_telemetry(
    original_prompt: str,
    compressed_prompt: str,
    original_tokens: int,
    compressed_tokens: int,
    compression_latency_seconds: float,
    latency_saved_ms: float,
    fallback_used: bool,
    production_speed: bool,
) -> None:
    """
    Asynchronously pipelines evaluations and telemetry processing
    outside the critical runtime client performance path.
    """
    db_pool = getattr(app.state, "db_pool", None)
    if db_pool is None:
        return

    try:
        ref_response = f"Processed response content for: {original_prompt[:30]}..."
        cand_response = f"Processed response content for: {compressed_prompt[:25]}..."
        if production_speed:
            f1_score = 0.0
        else:
            from src.evaluator import evaluate_semantic_fidelity

            f1_score = evaluate_semantic_fidelity(ref_response, cand_response)
        carbon_metrics = measure_inference_carbon(time.sleep, max(0.0, latency_saved_ms) / 1000.0)
        token_delta = max(original_tokens - compressed_tokens, 0)
        simulated_cost_saved = (token_delta / 1000.0) * 0.0015
        payload_json = {
            "fallback_used": fallback_used,
            "fallback_mode": "original_prompt" if fallback_used else "compressed_prompt",
            "production_speed": production_speed,
        }

        connection = db_pool.acquire()
        try:
            connection.execute(
                """
                INSERT INTO benchmark_results (
                    created_at,
                    category,
                    original_prompt,
                    compressed_prompt,
                    original_tokens,
                    compressed_tokens,
                    compression_ratio,
                    compression_latency_seconds,
                    baseline_latency_seconds,
                    optimized_latency_seconds,
                    latency_saved_ms,
                    reference_response,
                    candidate_response,
                    bert_f1_score,
                    carbon_footprint_g,
                    simulated_cost,
                    token_savings,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    None,
                    original_prompt,
                    compressed_prompt,
                    original_tokens,
                    compressed_tokens,
                    1.0 if fallback_used else (compressed_tokens / original_tokens if original_tokens else 1.0),
                    compression_latency_seconds,
                    4500.0,
                    4500.0 if original_tokens == 0 else 4500.0 * (compressed_tokens / original_tokens),
                    latency_saved_ms,
                    ref_response,
                    cand_response,
                    f1_score,
                    carbon_metrics.get("carbon_footprint_g", 0.0),
                    simulated_cost_saved,
                    token_delta,
                    json.dumps(payload_json, ensure_ascii=True),
                ),
            )
            connection.commit()
        finally:
            db_pool.release(connection)
    except Exception:
        LOGGER.exception("Background telemetry persistence failed")

@app.post("/v1/chat/completions")
async def handle_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    production_speed: bool = Query(default=False, description="Enable the regex production-speed fast path."),
):
    t_start = time.perf_counter()
    
    # Extract structural text from standard payload array format
    if not request.messages or "content" not in request.messages[-1]:
        raise HTTPException(status_code=400, detail="Invalid OpenAI payload schema formatting structure.")
        
    user_payload_text = request.messages[-1]["content"]
    production_speed_enabled = production_speed or os.getenv("PRODUCTION_SPEED", "").strip().lower() in {"1", "true", "yes", "on"}
    
    # Run Phase A: Token Healing Layer Execution
    protected_tokens = healer.extract_force_tokens(user_payload_text)
    
    # Execute Model-Driven Prompt Optimization
    loop = asyncio.get_running_loop()
    compression_result = await loop.run_in_executor(
        None,
        _compress_with_fallback,
        user_payload_text,
        request.target_rate,
        protected_tokens,
        production_speed_enabled,
    )
    
    t_end = time.perf_counter()
    processing_overhead_ms = (t_end - t_start) * 1000.0
    
    # Extract results
    compressed_text = compression_result["compressed_prompt"]
    orig_tokens = compression_result["original_tokens"]
    comp_tokens = compression_result["compressed_tokens"]
    compression_latency_seconds = compression_result["compression_latency_seconds"]
    
    # Calculate baseline proxy structural balances
    simulated_raw_latency_ms = 4500.0
    optimized_expected_latency_ms = simulated_raw_latency_ms * (comp_tokens / orig_tokens) if orig_tokens else simulated_raw_latency_ms
    latency_saved_ms = max(0.0, simulated_raw_latency_ms - optimized_expected_latency_ms - processing_overhead_ms)

    if compression_result["fallback_used"]:
        response.headers["X-Proxy-Fallback"] = "True"
    
    # Route profiling sequences to workers so the active user response stays immediate
    background_tasks.add_task(
        _persist_background_telemetry,
        user_payload_text,
        compressed_text,
        orig_tokens,
        comp_tokens,
        compression_latency_seconds,
        latency_saved_ms,
        compression_result["fallback_used"],
        production_speed_enabled,
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
                "content": (
                    f"[PROXY FALLBACK ENGAGED]\nOriginal Input Text Forwarded: {compressed_text}"
                    if compression_result["fallback_used"]
                    else f"[PROXY COMPRESSION ENGAGED]\nOptimized Input Text Forwarded: {compressed_text}"
                )
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": comp_tokens,
            "completion_tokens": 50, # Standard placeholder
            "total_tokens": comp_tokens + 50,
            "original_prompt_tokens_saved": orig_tokens - comp_tokens,
            "gateway_processing_overhead_ms": round(processing_overhead_ms, 2),
            "compression_latency_seconds": compression_latency_seconds,
            "fallback_used": compression_result["fallback_used"],
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)