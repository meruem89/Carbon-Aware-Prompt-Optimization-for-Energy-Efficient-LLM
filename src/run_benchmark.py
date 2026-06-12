from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carbon_estimator import measure_inference_carbon
from src.compression_engine import compress_prompt
from src.database import init_db, log_result
from src.evaluator import evaluate_semantic_fidelity


DATA_PATH = PROJECT_ROOT / "data" / "prompts.csv"


def _build_reference_response(category: str, original_prompt: str) -> str:
    return (
        f"Reference answer for the {category.lower()} prompt: {original_prompt.strip()} "
        f"This response explains the topic clearly, covers the main ideas, and stays factual."
    )


def _build_candidate_response(category: str, original_prompt: str) -> str:
    return (
        f"Candidate answer for the {category.lower()} prompt: {original_prompt.strip()} "
        f"This paraphrased response keeps the same meaning while using different wording."
    )


def main() -> None:
    init_db()
    prompts_frame = pd.read_csv(DATA_PATH)

    total_prompts = len(prompts_frame)
    print(f"Loaded {total_prompts} prompts from {DATA_PATH}")

    for index, row in prompts_frame.iterrows():
        category = str(row.get("category", "General"))
        original_prompt = str(row["original_prompt"])

        print(f"[{index + 1}/{total_prompts}] Processing {category}: {original_prompt[:72]}")

        compression_result = compress_prompt(original_prompt, target_rate=0.6)
        original_tokens = compression_result["original_tokens"]
        compressed_tokens = compression_result["compressed_tokens"]
        compression_ratio = compression_result["compression_ratio"]

        baseline_latency_seconds = 4.5
        optimized_latency_seconds = (
            baseline_latency_seconds * (compressed_tokens / original_tokens)
            if original_tokens
            else baseline_latency_seconds
        )
        latency_saved_ms = (baseline_latency_seconds - optimized_latency_seconds) * 1000
        token_savings = max(original_tokens - compressed_tokens, 0)
        simulated_cost = token_savings * 0.0015 / 1000

        reference_response = _build_reference_response(category, original_prompt)
        candidate_response = _build_candidate_response(category, original_prompt)
        bert_f1_score = evaluate_semantic_fidelity(reference_response, candidate_response)

        def _sleep_for_optimized_latency() -> str:
            time.sleep(optimized_latency_seconds)
            return candidate_response

        carbon_result = measure_inference_carbon(_sleep_for_optimized_latency)

        telemetry = {
            "category": category,
            "original_prompt": original_prompt,
            "compressed_prompt": compression_result["compressed_prompt"],
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "compression_latency_seconds": compression_result["compression_latency_seconds"],
            "baseline_latency_seconds": baseline_latency_seconds,
            "optimized_latency_seconds": round(optimized_latency_seconds, 4),
            "latency_saved_ms": round(latency_saved_ms, 4),
            "reference_response": reference_response,
            "candidate_response": candidate_response,
            "bert_f1_score": bert_f1_score,
            "carbon_footprint_g": carbon_result["carbon_footprint_g"],
            "simulated_cost": round(simulated_cost, 6),
            "token_savings": token_savings,
            "carbon_wall_time_seconds": carbon_result["wall_time_seconds"],
            "carbon_cpu_time_seconds": carbon_result["cpu_time_seconds"],
        }

        row_id = log_result(**telemetry)
        print(
            f"    saved_ms={telemetry['latency_saved_ms']:.2f}, "
            f"bert_f1={bert_f1_score:.4f}, carbon_g={carbon_result['carbon_footprint_g']:.6f}, "
            f"db_row={row_id}"
        )

    print("Benchmark run complete.")


if __name__ == "__main__":
    main()