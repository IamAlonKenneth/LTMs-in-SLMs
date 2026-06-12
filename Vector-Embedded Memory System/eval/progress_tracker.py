"""
progress_tracker.py
===================
Simple progress tracking with timing information for evaluation runs.
"""

import time
from datetime import timedelta
from statistics import mean


class ProgressTracker:
    """Tracks progress with timing and ETA."""

    def __init__(self, total_items: int, name: str = "") -> None:
        self.total_items = total_items
        self.name        = name or "Progress"
        self.current     = 0
        self.start_time  = time.time()
        self.step_times: list[float] = []

    def update(self, step_name: str = "") -> None:
        self.current += 1
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)

        if self.total_items > 0 and self.current > 0:
            avg_step      = elapsed / self.current
            remaining     = (self.total_items - self.current) * avg_step
            eta_str       = str(timedelta(seconds=int(remaining)))
            pct           = self.current / self.total_items * 100
            label         = f" {step_name} |" if step_name else ""
            print(
                f"  [{self.name}] {self.current}/{self.total_items} "
                f"({pct:.0f}%) — {elapsed:.0f}s elapsed "
                f"{label} ETA {eta_str}",
                flush=True,
            )
        else:
            print(
                f"  [{self.name}] {self.current}/{self.total_items} "
                f"— {elapsed:.0f}s elapsed",
                flush=True,
            )

    def finish(self) -> None:
        elapsed = time.time() - self.start_time
        print(
            f"  [{self.name}] Complete in {timedelta(seconds=int(elapsed))}",
            flush=True,
        )
