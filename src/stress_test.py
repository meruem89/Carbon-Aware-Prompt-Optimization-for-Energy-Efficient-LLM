import concurrent.futures
import sqlite3
import time
from pathlib import Path

import requests


PRIMARY_URL = "http://127.0.0.1:8000/v1/chat/completions"
PAYLOAD = {
    "model": "llama3-70b",
    "target_rate": 0.5,
    "messages": [
        {
            "role": "user",
            "content": "Refactor this architecture code: def parse_response(url='https://api.com'): return {key: val for key, val in data.items() if val >= 200}",
        }
    ],
}


def _post_request(url: str) -> requests.Response:
    return requests.post(url, json=PAYLOAD, timeout=120)


def _extract_gateway_metrics(response: requests.Response) -> tuple[str, str]:
    gateway_overhead = "N/A"
    fallback_flag = response.headers.get("X-Proxy-Fallback", "False")

    try:
        response_json = response.json()
        gateway_overhead = str(response_json.get("usage", {}).get("gateway_processing_overhead_ms", "N/A"))
    except ValueError:
        gateway_overhead = "N/A"

    return gateway_overhead, fallback_flag


def test_normal_request() -> dict:
    print("=== Normal Request Test ===")
    start_time = time.perf_counter()
    response = _post_request(PRIMARY_URL)
    elapsed_seconds = time.perf_counter() - start_time
    gateway_overhead, fallback_flag = _extract_gateway_metrics(response)

    print(f"HTTP Status Code: {response.status_code}")
    print(f"Gateway Processing Overhead (ms): {gateway_overhead}")
    print(f"X-Proxy-Fallback: {fallback_flag}")
    print("Response Headers:")
    for header_name, header_value in response.headers.items():
        print(f"  {header_name}: {header_value}")
    print(f"Round-Trip Time (s): {elapsed_seconds:.4f}")

    return {
        "status_code": response.status_code,
        "elapsed_seconds": elapsed_seconds,
        "fallback_flag": fallback_flag,
        "text": response.text,
    }


def _worker_request(worker_id: int) -> dict:
    started_at = time.perf_counter()
    try:
        response = _post_request(PRIMARY_URL)
        elapsed_seconds = time.perf_counter() - started_at
        gateway_overhead, fallback_flag = _extract_gateway_metrics(response)
        response_text = response.text
        database_locked_seen = "database is locked" in response_text.lower()

        return {
            "worker_id": worker_id,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed_seconds,
            "gateway_overhead": gateway_overhead,
            "fallback_flag": fallback_flag,
            "database_locked_seen": database_locked_seen,
            "error": "",
        }
    except requests.exceptions.RequestException as exc:
        elapsed_seconds = time.perf_counter() - started_at
        error_text = str(exc)
        return {
            "worker_id": worker_id,
            "status_code": 0,
            "elapsed_seconds": elapsed_seconds,
            "gateway_overhead": "N/A",
            "fallback_flag": "False",
            "database_locked_seen": "database is locked" in error_text.lower(),
            "error": error_text,
        }


def _database_row_count(database_path: str) -> int:
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute("SELECT COUNT(*) FROM benchmark_results")
        return int(cursor.fetchone()[0])


def simulate_load_concurrency() -> dict:
    print("\n=== Concurrent Load Test ===")
    database_path = Path(__file__).resolve().parents[1] / "data" / "results.db"
    initial_row_count = _database_row_count(str(database_path)) if database_path.exists() else 0

    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_worker_request, worker_id) for worker_id in range(1, 11)]
        worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
    total_elapsed_seconds = time.perf_counter() - started_at

    time.sleep(2.0)
    final_row_count = _database_row_count(str(database_path)) if database_path.exists() else 0

    successful_requests = sum(1 for result in worker_results if result["status_code"] == 200)
    average_round_trip_seconds = (
        sum(result["elapsed_seconds"] for result in worker_results) / len(worker_results)
        if worker_results
        else 0.0
    )
    database_locked_errors = any(result["database_locked_seen"] for result in worker_results)
    rows_logged_successfully = final_row_count > initial_row_count

    print(f"Successful Requests (200): {successful_requests}/10")
    print(f"Average Round-Trip Time (s): {average_round_trip_seconds:.4f}")
    print(f"Total Concurrent Test Duration (s): {total_elapsed_seconds:.4f}")
    print(f"Database Locked Errors Observed: {'Yes' if database_locked_errors else 'No'}")
    print(f"Transactions Logged: {'Yes' if rows_logged_successfully else 'No'}")

    return {
        "successful_requests": successful_requests,
        "average_round_trip_seconds": average_round_trip_seconds,
        "database_locked_errors": database_locked_errors,
        "rows_logged_successfully": rows_logged_successfully,
        "initial_row_count": initial_row_count,
        "final_row_count": final_row_count,
        "worker_results": worker_results,
    }


def main() -> None:
    normal_request_result = test_normal_request()
    concurrency_result = simulate_load_concurrency()

    print("\n=== Final Benchmark Summary ===")
    print(f"Total Successful Requests: {concurrency_result['successful_requests']}/10")
    print(f"Average Round-Trip Response Time: {concurrency_result['average_round_trip_seconds']:.4f} s")
    print(
        "Database Logging Healthy: "
        f"{'Yes' if concurrency_result['rows_logged_successfully'] and not concurrency_result['database_locked_errors'] else 'No'}"
    )
    print(f"Normal Request Fallback Flag: {normal_request_result['fallback_flag']}")


if __name__ == "__main__":
    main()