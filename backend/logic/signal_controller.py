#!/usr/bin/env python3
"""
Improved signal_controller.py

- decide_signal class (same external API: run_once / run_loop)
- More realistic defaults (green/yellow/all-red)
- EWMA smoothing of counts
- sqrt scaling for green time to avoid runaway durations
- Append-only CSV logging (much faster than reloading JSON each cycle)
- Two-phase run_loop (Green then Yellow)
"""

from __future__ import annotations
import os
import json
import time
import math
import random
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

DEFAULT_CONFIG = {
    # Green timing model (realistic defaults for urban/arterial intersections)
    "base_time": 15.0,         # base green (seconds)
    "factor": 2.0,             # multiplier for sqrt(vehicle_count)
    "min_time": 8.0,           # smallest allowed green (seconds)
    "max_time": 45.0,          # largest allowed green (seconds)

    # Transition / safety times
    "yellow_time": 4.0,        # yellow interval (seconds)
    "all_red_time": 1.0,       # short all-red clearance between phases (seconds)

    # Controller behavior
    "control_interval": 5.0,   # how often controller recalculates (seconds)
    "starvation_weight": 1.0,  # boost factor for starving lanes
    "max_starvation_multiplier": 3.0,  # bound the boost so it cannot dominate
    "service_order": "auto",   # 'auto' or 'round_robin'
    "smoothing_alpha": 0.4,    # EWMA alpha for smoothing counts (0-1)

    # Logging
    "log_file": "logs/traffic_log.json",

    #Parameters to guarantee high-density lanes aren't stuck too long
    "high_density_threshold": 12,         # smoothed count >= this => "High"
    "max_red_wait_high_density": 35.0,    # seconds: cap on red wait for high-density lanes
}


class decide_signal:
    def __init__(self, config: Optional[dict] = None, lanes: Optional[Iterable[str]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Timing & control params
        self.base_time = float(self.config["base_time"])
        self.factor = float(self.config["factor"])
        self.min_time = float(self.config["min_time"])
        self.max_time = float(self.config["max_time"])
        self.yellow_time = float(self.config["yellow_time"])
        self.all_red_time = float(self.config["all_red_time"])
        self.control_interval = float(self.config["control_interval"])

        # Behavior params
        self.starvation_weight = float(self.config["starvation_weight"])
        self.max_starvation_multiplier = float(self.config["max_starvation_multiplier"])
        self.service_order = str(self.config.get("service_order", "auto"))
        self.smoothing_alpha = float(self.config.get("smoothing_alpha", 0.4))

        # Logging
        self.log_file = str(self.config.get("log_file", "logs/traffic_log.json"))

        # high-density protection params
        self.high_density_threshold = float(self.config.get("high_density_threshold", 12))
        self.max_red_wait_high_density = float(self.config.get("max_red_wait_high_density", 35.0))

        # Lane bookkeeping
        self.lanes: List[str] = list(lanes) if lanes else ["lane1", "lane2", "lane3", "lane4"]
        now = time.time()
        self.last_served: Dict[str, float] = {lane: now for lane in self.lanes}
        # smoothed counts (EWMA) to avoid jitter
        self.smoothed_counts: Dict[str, float] = {lane: 0.0 for lane in self.lanes}
        self.rr_index = 0

        # ensure logging path
        self._ensure_logfile()

    # ---------- Logging (JSON array) ----------
    def _ensure_logfile(self):
        logdir = os.path.dirname(self.log_file) or "."
        os.makedirs(logdir, exist_ok=True)
        if not os.path.exists(self.log_file):
            # initialize as empty JSON array
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("[]")
    
    def _append_json_record(self, record: dict):
        """
        Append a record into the JSON array file.
        Implementation: read existing array (json.load), append, write back.
        This is straightforward and robust for moderate log sizes. If you expect
        extremely large files, switch to NDJSON or a DB.
        """
        try:
            # ensure file exists
            self._ensure_logfile()
            with open(self.log_file, "r+", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except Exception:
                    # file corrupted or empty -> reset to empty list
                    data = []
                data.append(record)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            # don't crash controller on logging failure
            print("Warning: failed to append JSON log:", e)

    def register_lanes(self, lanes: Iterable[str]):
        """Add lanes if new; initialize bookkeeping."""
        for lane in lanes:
            if lane not in self.lanes:
                self.lanes.append(lane)
                self.last_served[lane] = time.time()
                self.smoothed_counts[lane] = 0.0

    # ---------- Timing calculation ----------
    def calculate_green_time(self, vehicle_count: float) -> float:
        """
        Compute green time using sqrt scaling:
            green = base + factor * sqrt(count)
        Clamped to [min_time, max_time].
        sqrt scaling reduces runaway time when count is large.
        """
        v = max(0.0, float(vehicle_count))
        green = self.base_time + self.factor * math.sqrt(v)
        green = max(self.min_time, min(self.max_time, green))
        return float(green)

    # ---------- Selection ----------
    def _starvation_boost(self, lane: str) -> float:
        """Compute bounded starvation boost based on time since last served."""
        now = time.time()
        last = self.last_served.get(lane, 0.0)
        wait = max(0.0, now - last)
        multiplier = (wait / max(1.0, self.control_interval)) * self.starvation_weight
        # Bound the multiplier so starvation can't dominate completely
        return min(multiplier, self.max_starvation_multiplier)

    def _score_for_lane(self, lane: str) -> float:
        """Score = smoothed_count + starvation_boost"""
        count = self.smoothed_counts.get(lane, 0.0)
        return float(count) + self._starvation_boost(lane)

    def select_lane(self, counts: Dict[str, int]) -> str:
        """Select lane with prioritization for high-density lanes that waited too long."""
        self.register_lanes(counts.keys())
        if not self.lanes:
            raise ValueError("No lanes registered.")

        now = time.time()

        # 1) Forced selection: Any high-density lane that has waited >= max_red_wait_high_density?
        starving_high = []
        for lane in self.lanes:
            smoothed = float(self.smoothed_counts.get(lane, 0.0))
            if smoothed >= self.high_density_threshold:
                last = self.last_served.get(lane, 0.0)
                wait = now - last
                if wait >= self.max_red_wait_high_density:
                    starving_high.append((wait, lane))

        if starving_high:
            # Choose the high-density lane that waited the longest (tie-break by name).
            starving_high.sort(key=lambda x: (-x[0], x[1]))
            return starving_high[0][1]

        # 2) Normal operation
        if self.service_order == "round_robin":
            selected = self.lanes[self.rr_index % len(self.lanes)]
            self.rr_index += 1
            return selected

        best_lane = None
        best_score = -float("inf")
        for lane in sorted(self.lanes):  # deterministic tie-break
            score = self._score_for_lane(lane)
            if score > best_score:
                best_score = score
                best_lane = lane
        return best_lane

    # ---------- Public API ----------
    def run_once(self, counts: Dict[str, int]) -> Tuple[Optional[str], float]:
        """
        Make one decision.
        counts: raw detected counts (lane -> integer)
        Returns: (served_lane, green_time_seconds)
        """
        counts = counts or {}
        # register lanes and ensure smoothed_counts keys
        self.register_lanes(counts.keys())

        # Update smoothed counts using EWMA
        alpha = self.smoothing_alpha
        for lane in self.lanes:
            raw = float(counts.get(lane, 0))
            prev = self.smoothed_counts.get(lane, 0.0)
            self.smoothed_counts[lane] = alpha * raw + (1 - alpha) * prev

        # decide which lane to serve
        # if all counts zero, fallback to round robin quick cycle
        total_raw = sum(int(counts.get(l, 0)) for l in self.lanes)
        if total_raw == 0:
            # no detected traffic — short green to each lane in round-robin order
            if self.lanes:
                served = self.lanes[self.rr_index % len(self.lanes)]
                self.rr_index += 1
            else:
                return None, 0.0
        else:
            served = self.select_lane(counts)

        # compute green time using smoothed count for the served lane
        vehicle_count_for_served = self.smoothed_counts.get(served, 0.0)
        green_time = self.calculate_green_time(vehicle_count_for_served)

        # update last_served timestamp (we set it now so starvation resets)
        self.last_served[served] = time.time()

        # build record and append to JSON log
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cycle = self._format_cycle(served, green_time, counts)
        record = {"timestamp": ts, "cycle": cycle}
        self._append_json_record(record)

        return served, float(green_time)

    def _format_cycle(self, served_lane: Optional[str], green_time: float, counts: Dict[str, int]) -> Dict[str, dict]:
        """
        Build cycle dict matching dashboard structure:
        {
        "lane1": {"status": "Red"/"Green", "time": int},
        ...
        }
        Vehicle count and density should be returned separately via /vehicle-count
        """
        cycle = {}
        self.register_lanes(counts.keys())  # Ensure lanes are registered

        for i, lane in enumerate(self.lanes, start=1):
            count = int(counts.get(lane, 0))

            # Signal status and time
            if lane == served_lane:
                status = "Green"
                time_val = int(round(green_time))
            else:
                status = "Red"
                time_val = int(round(green_time + self.yellow_time + self.all_red_time))

            # Only include signal info here
            cycle[f"lane{i}"] = {
                "status": status,
                "time": time_val
            }

        return cycle

    def run_loop(
        self,
        poll_counts_fn: Callable[[], Dict[str, int]],
        update_ui_fn: Optional[Callable[[str, float], None]] = None,
        stop_after_cycles: Optional[int] = None
    ):
        """
        Blocking loop that polls counts and runs the controller.
        It emits a Green phase (served lane green_time), then a Yellow phase (yellow_time).
        """
        cycles = 0
        try:
            while True:
                counts = poll_counts_fn() or {}
                served, green_time = self.run_once(counts)

                # Print the same structure to console for visibility
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cycle = self._format_cycle(served, green_time, counts)
                print(json.dumps({"timestamp": ts, "cycle": cycle}, indent=2))

                if update_ui_fn:
                    try:
                        update_ui_fn(served, green_time)
                    except Exception as e:
                        print("Warning: update_ui_fn error:", e)

                # green phase -> yellow -> optional all red
                time.sleep(max(0.0, green_time))
                time.sleep(max(0.0, self.yellow_time))
                if self.all_red_time > 0:
                    time.sleep(max(0.0, self.all_red_time))

                cycles += 1
                if stop_after_cycles and cycles >= stop_after_cycles:
                    print("Stopping after requested cycles.")
                    break
        except KeyboardInterrupt:
            print("Controller loop interrupted by user.")

# --------------- Demo ---------------
def _demo_poll_counts_random(lanes):
    return {lane: random.randint(0, 20) for lane in lanes}

if __name__ == "__main__":
    demo_lanes = ["A", "B", "C", "D"]
    controller = decide_signal(config=None, lanes=demo_lanes)
    print("Demo decide_signal started. Press Ctrl+C to stop.")
    controller.run_loop(lambda: _demo_poll_counts_random(demo_lanes), update_ui_fn=None, stop_after_cycles=12)