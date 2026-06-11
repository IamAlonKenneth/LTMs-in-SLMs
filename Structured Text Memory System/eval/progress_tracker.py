"""
progress_tracker.py
===================
Simple progress tracking with timing information for evaluation runs.
"""

import time
from datetime import timedelta


class ProgressTracker:
    """Tracks progress with timing and ETA."""

    def __init__(self, total_items: int, name: str = ""):
        self.total_items = total_items
        self.name = name
        self.current = 0
        self.start_time = time.time()

    def update(self, step_name: str = ""):
        """Log progress of one item."""
        self.current += 1
        elapsed = time.time() - self.start_time

        # avg time per completed item (current-1 items done before this update call)
        completed = self.current - 1
        avg_time = (elapsed / completed) if completed > 0 else elapsed

        remaining = self.total_items - self.current
        eta_seconds = remaining * avg_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        percent = (self.current / self.total_items) * 100
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        status = f"[{self.name}] {self.current}/{self.total_items} ({percent:.1f}%) | "
        status += f"Elapsed: {elapsed_str} | ETA: {eta_str}"
        if step_name:
            status += f" | {step_name}"

        print(status)

    def finish(self):
        """Log completion."""
        total_time = time.time() - self.start_time
        total_str = str(timedelta(seconds=int(total_time)))
        print(f"[{self.name}] COMPLETE in {total_str}\n")
