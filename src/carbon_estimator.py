from __future__ import annotations

import time
from typing import Any, Callable, TypeVar


T = TypeVar("T")

ESTIMATED_POWER_WATTS = 65.0
CARBON_INTENSITY_G_PER_KWH = 475.0


def measure_inference_carbon(func: Callable[..., T], *args: Any, **kwargs: Any) -> dict[str, Any]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = func(*args, **kwargs)
    wall_time_seconds = time.perf_counter() - wall_start
    cpu_time_seconds = time.process_time() - cpu_start

    effective_seconds = max(wall_time_seconds, cpu_time_seconds)
    energy_kwh = ESTIMATED_POWER_WATTS * effective_seconds / 3_600_000
    carbon_footprint_g = energy_kwh * CARBON_INTENSITY_G_PER_KWH

    return {
        "result": result,
        "wall_time_seconds": round(wall_time_seconds, 4),
        "cpu_time_seconds": round(cpu_time_seconds, 4),
        "carbon_footprint_g": round(carbon_footprint_g, 6),
    }