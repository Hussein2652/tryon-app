from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict


_lock = threading.Lock()
_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
_timers: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))


def increment(metric: str, label: str, value: int = 1) -> None:
    with _lock:
        _counters[metric][label] += value


def observe_latency(metric: str, label: str, duration_seconds: float) -> None:
    with _lock:
        _timers[metric]["sum"] += duration_seconds
        _timers[metric]["count"] += 1
        bucket_key = f"{label}"
        _timers.setdefault(metric + ":per_label", defaultdict(float))
        _timers[metric + ":per_label"][bucket_key] += duration_seconds


def snapshot() -> Dict[str, Dict[str, float]]:
    with _lock:
        counters_copy = {k: dict(v) for k, v in _counters.items()}
        timers_copy = {k: dict(v) for k, v in _timers.items()}
    return {"counters": counters_copy, "timers": timers_copy}


class Timer:
    def __init__(self, metric: str, label: str) -> None:
        self.metric = metric
        self.label = label
        self.start = time.perf_counter()

    def stop(self) -> None:
        duration = time.perf_counter() - self.start
        observe_latency(self.metric, self.label, duration)

    def __enter__(self) -> "Timer":  # pragma: no cover - context manager convenience
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # pragma: no cover
        self.stop()
