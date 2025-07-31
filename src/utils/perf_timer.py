import time
from typing import Dict, Any, Optional, List
from loguru import logger


class PerfTimer:
    def __init__(self, debug: bool = False):
        """
        Initializes the performance timer.

        Args:
            debug: If True, the timer is enabled. Defaults to False.
        """
        self.debug = debug
        self.timings: Dict[str, Dict[str, Any]] = {}

    def enable(self):
        """Enables the timer."""
        self.debug = True

    def disable(self):
        """Disables the timer and clears any collected data."""
        self.debug = False
        self.timings = {}

    def start_timer(self, name: str, parent_name: Optional[str] = None):
        """
        Starts a timer for a given metric.

        Args:
            name: The name of the metric to start timing.
            parent_name: The optional name of the parent metric.
        """
        if self.debug:
            if name not in self.timings:
                self.timings[name] = {
                    "total": 0,
                    "count": 0,
                    "max": 0,
                    "min": float("inf"),
                    "parent": parent_name,
                }
            self.timings[name]["start"] = time.time()

    def stop_timer(self, name: str):
        """
        Stops a timer for a given metric and records the duration.

        Args:
            name: The name of the metric to stop.
        """
        if self.debug and name in self.timings and "start" in self.timings[name]:
            duration = time.time() - self.timings[name]["start"]
            data = self.timings[name]
            data["total"] += duration
            data["count"] += 1
            data["max"] = max(data.get("max", 0), duration)
            data["min"] = min(data.get("min", float("inf")), duration)
            del data["start"]

    def log_timings(self):
        """
        Logs the performance timings in a hierarchical structure and resets the timer.
        """
        if not self.debug or not self.timings:
            return

        logger.debug("--- Performance Timings ---")

        nodes = {name: data.copy() for name, data in self.timings.items()}

        for name in nodes:
            nodes[name]['children'] = []

        root_names: List[str] = []
        for name, node_data in nodes.items():
            parent_name = node_data.get("parent")
            if parent_name and parent_name in nodes:
                nodes[parent_name]['children'].append(name)
            else:
                root_names.append(name)

        def _print_hierarchy(name: str, prefix: str, children_prefix: str):
            node_data = nodes[name]
            if node_data["count"] > 0:
                avg_time = node_data["total"] / node_data["count"]
                max_time = node_data["max"]
                min_time = node_data["min"]
                logger.debug(
                    f"{prefix}{name}: Total={node_data['total']:.4f}s, Count={node_data['count']}, "
                    f"Avg={avg_time:.4f}s, Max={max_time:.4f}s, Min={min_time:.4f}s"
                )

            sorted_children = sorted(node_data.get('children', []))
            for i, child_name in enumerate(sorted_children):
                is_last = i == len(sorted_children) - 1
                connector = "└── " if is_last else "├── "
                new_children_prefix = children_prefix + ("    " if is_last else "│   ")
                _print_hierarchy(child_name, children_prefix + connector, new_children_prefix)

        for name in sorted(root_names):
            _print_hierarchy(name, "", "")

        logger.debug("---------------------------")
        self.timings = {}