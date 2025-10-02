 #!/usr/bin/env python3
"""
signal_controller.py

Signal Control & Timing module for AI Traffic Light Optimizer.

Provides:
- decide_signal class that:
  - loads config (or uses defaults)
  - computes green time per lane (base + factor * vehicle_count)
  - selects next lane to serve using a fairness/starvation boost
  - logs every cycle to CSV
- A simple demo loop when run as __main__ (simulates incoming counts)

How to integrate:
- Instantiate decide_signal(config_dict_or_path)
- Call controller.run_once(counts_dict) -> returns (lane, green_time)
- Or call controller.run_loop(poll_counts_fn, update_ui_fn) for a blocking demo loop.

"""

import json
import os
import time
from datetime import datetime
import random

try:
    import yaml
except Exception:
    yaml = None

DEFAULT_CONFIG = {
    "base_time": 8.0,            # minimum base green time (seconds)
    "factor": 1.5,               # seconds added per detected vehicle
    "min_time": 5.0,             # absolute minimum green (seconds)
    "max_time": 60.0,            # absolute maximum green (seconds)
    "control_interval": 8.0,     # how often controller recomputes (seconds)
    "starvation_weight": 0.8,    # boost weight for lanes not recently served
    "log_file": "logs/traffic_log.json",
    "service_order": "auto"      # 'auto' uses scoring, 'round_robin' uses fixed order
}


class decide_signal:
    def __init__(self, config=None, lanes=None):
        """
        config: dict or path to YAML config. If None uses DEFAULT_CONFIG.
        lanes: list of lane names (e.g., ['A','B','C','D']). If None, lanes are inferred dynamically.
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            if isinstance(config, str):
                self._load_config_from_file(config)
            elif isinstance(config, dict):
                self.config.update(config)
        self.control_interval = float(self.config["control_interval"])
        self.base_time = float(self.config["base_time"])
        self.factor = float(self.config["factor"])
        self.min_time = float(self.config["min_time"])
        self.max_time = float(self.config["max_time"])
        self.starvation_weight = float(self.config["starvation_weight"])
        self.log_file = self.config["log_file"]
        self.service_order = self.config.get("service_order", "auto")

        # Track lanes and last served times to prevent starvation
        self.lanes = list(lanes) if lanes else []
        now = time.time()
        self.last_served = {lane: now for lane in self.lanes}  # lane -> timestamp
        self._ensure_logfile()

        # round-robin pointer (used if service_order == 'round_robin')
        self.rr_index = 0

    def _load_config_from_file(self, path):
        if yaml is None:
            raise RuntimeError("PyYAML not installed: cannot load YAML config. Provide dict config or install pyyaml.")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            if data:
                self.config.update(data)

    def _ensure_logfile(self):
        logdir = os.path.dirname(self.log_file) or "."
        os.makedirs(logdir, exist_ok=True)

        # if file doesn't exist, create an empty array (for JSON logs)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("[]")  # start as empty JSON list

    def register_lanes(self, lanes):
        """Register lane names (list). Can be called at startup to set lanes."""
        for lane in lanes:
            if lane not in self.lanes:
                self.lanes.append(lane)
                self.last_served[lane] = time.time()

    def calculate_green_time(self, vehicle_count):
        """Compute green_time clamped to min/max using formula base + factor * count."""
        green_time = self.base_time + self.factor * float(vehicle_count)
        if green_time < self.min_time:
            green_time = self.min_time
        if green_time > self.max_time:
            green_time = self.max_time
        return float(green_time)

    def _score_for_lane(self, lane, count):
        """
        Score to pick lane to serve next.
        Uses vehicle count + starvation boost based on time since last served.
        """
        now = time.time()
        last = self.last_served.get(lane, 0.0)
        time_wait = max(0.0, now - last)
        # Normalize wait in units of control_interval, multiply by starvation_weight
        wait_boost = (time_wait / max(1.0, self.control_interval)) * self.starvation_weight
        return float(count) + wait_boost

    def select_lane(self, counts):
        """
        counts: dict of lane -> vehicle_count
        returns: selected lane name
        """
        # Register any new lanes found in counts
        self.register_lanes(list(counts.keys()))

        if self.service_order == "round_robin":
            # simple round robin that respects lane order
            lanes = self.lanes
            if not lanes:
                raise ValueError("No lanes registered.")
            selected = lanes[self.rr_index % len(lanes)]
            self.rr_index += 1
            return selected

        # default: score-based selection (counts + starvation boost)
        best_lane = None
        best_score = -1.0
        # deterministic tie-break by lane name
        for lane in sorted(counts.keys()):
            count = counts.get(lane, 0)
            score = self._score_for_lane(lane, count)
            # tie-breaking: use higher count then earlier last_served (handled by boost)
            if best_lane is None or score > best_score:
                best_score = score
                best_lane = lane
        return best_lane

    def run_once(self, counts):
        """
        Do one control decision based on counts dict.
        Returns (served_lane, green_time_seconds).
        Also writes a log row.
        """
        if not counts or len(counts) == 0:
            # nothing detected; default to round-robin or skip
            if self.lanes:
                # choose next lane in rr if lanes known
                served = self.lanes[self.rr_index % len(self.lanes)]
                self.rr_index += 1
            else:
                return (None, 0.0)
        else:
            served = self.select_lane(counts)

        vehicle_count = counts.get(served, 0)
        green_time = self.calculate_green_time(vehicle_count)
        # update last served timestamp
        self.last_served[served] = time.time()

        # log
        self._log_cycle(served, green_time, counts)
        return served, green_time

    def _log_cycle(self, served_lane, green_time, counts):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build the record
        record = {
            "timestamp": ts,
            "cycle": self._format_output(served_lane, green_time, counts)
        }

        # Load existing logs (if any)
        try:
            with open(self.log_file, "r") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []

        # Append new record
        logs.append(record)

        # Save updated list back to file
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
    def _format_output(self, served_lane, green_time, counts):
        """Return a dict with status, time, count, density for each lane."""

        output = {}
        for i, (lane, count) in enumerate(counts.items(), start=1):
            if lane == served_lane:
                status = "Green"
                time = int(green_time)
            else:
                status = "Red"
                time = int(green_time * 2)  # arbitrary, adjust if needed

            # density classification
            if count > 10:
                density = "High"
            elif count > 5:
                density = "Medium"
            else:
                density = "Low"

            output[f"Lane {i}"] = {
                "status": status,
                "time": time,
                "count": count,
                "density": density,
            }

        return output


    def run_loop(self, poll_counts_fn, update_ui_fn=None, stop_after_cycles=None):
        cycles = 0
        try:
            while True:
                counts = poll_counts_fn() or {}
                served, green_time = self.run_once(counts)

                # 🔹 build structured output
                output = {}
                for i, lane in enumerate(self.lanes, start=1):
                    count = counts.get(lane, 0)

                    # classify density
                    if count <= 5:
                        density = "Low"
                    elif count <= 12:
                        density = "Medium"
                    else:
                        density = "High"

                    if lane == served:
                        status = "Green"
                        time_val = green_time
                    else:
                        status = "Red"
                        time_val = max(green_time, 15)  # fallback red duration

                    output[f"Lane {i}"] = {
                        "status": status,
                        "time": time_val,
                        "count": count,
                        "density": density
                    }

                # 🔹 Add a Yellow transition for the served lane
                output[f"Lane {self.lanes.index(served)+1}"]["status"] = "Yellow"
                output[f"Lane {self.lanes.index(served)+1}"]["time"] = 5

                # print JSON-like structured output
                print(output)

                # 🔹 UI hook (if dashboard exists)
                if update_ui_fn:
                    try:
                        update_ui_fn(served, green_time)
                    except Exception as e:
                        print("Warning: update_ui_fn error:", e)

                # wait for green period but allow interruption
                waited = 0.0
                step = 0.2
                while waited < green_time:
                    time.sleep(step)
                    waited += step

                cycles += 1
                if stop_after_cycles and cycles >= stop_after_cycles:
                    print("Stopping after requested cycles.")
                    break
        except KeyboardInterrupt:
            print("Controller loop interrupted by user.")



# -----------------------
# Demo / example usage
# -----------------------
def _demo_poll_counts_random(lanes):
    """Return simulated random counts for demo/testing."""
    return {lane: random.randint(0, 20) for lane in lanes}


if __name__ == "__main__":
    # Simple CLI demo: serves four lanes with random counts
    demo_lanes = ["A", "B", "C", "D"]
    controller = decide_signal(config=None, lanes=demo_lanes)
    print("Demo decide_signal started. Press Ctrl+C to stop.")
    # Run loop with random counts; update_ui_fn is None (prints to console)
    controller.run_loop(lambda: _demo_poll_counts_random(demo_lanes), update_ui_fn=None, stop_after_cycles=12)