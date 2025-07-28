import time
from typing import List
from loguru import logger

class PerfTimer:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.timings = {}
    
    def enable(self):
        self.debug = True
        
    def disable(self):
        self.debug = False
        self.timings = {}
        
    def start_timer(self, name: str):
        if self.debug:
            if name not in self.timings:
                self.timings[name] = {
                    "total": 0,
                    "count": 0,
                    "max": 0,
                    "min": float("inf"),
                }
            self.timings[name]["start"] = time.time()

    def stop_timer(self, name: str):
        if self.debug and name in self.timings and "start" in self.timings[name]:
            duration = time.time() - self.timings[name]["start"]
            data = self.timings[name]
            data["total"] += duration
            data["count"] += 1
            data["max"] = max(data.get("max", 0), duration)
            data["min"] = min(data.get("min", float("inf")), duration)
            del data["start"]

    def log_timings(self):
        if self.debug:
            logger.debug("--- Performance Timings ---")
            for name, data in self.timings.items():
                if data["count"] > 0:
                    avg_time = data["total"] / data["count"]
                    max_time = data["max"]
                    min_time = data["min"]
                    logger.debug(
                        f"{name}: Total={data['total']:.4f}s, Count={data['count']}, "
                        f"Avg={avg_time:.4f}s, Max={max_time:.4f}s, Min={min_time:.4f}s"
                    )
            logger.debug("---------------------------")
            self.timings = {}