#!/usr/bin/env python3
"""
Signal controller (round-robin planning; density only for green-time calculation)

Changes:
 - Removed starvation logic and special high-density preemption.
 - Lane order is purely round-robin (rotated by rr_index).
 - Density only affects per-lane green time via density_time_map.
 - EWMA smoothing, logging, red-duration tracking, mixed main+grace loop preserved.
"""
from __future__ import annotations
import os
import json
import time
import math
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import tempfile

DEFAULT_CONFIG = {
    # Legacy fallback timing (not used when density_time_map exists)
    "base_time": 15.0,
    "factor": 2.0,
    "min_time": 8.0,
    "max_time": 45.0,

    # Transition / safety times
    "yellow_time": 4.0,
    "all_red_time": 1.0,

    # Controller behavior
    "control_interval": 5.0,
    "service_order": "round_robin",
    "smoothing_alpha": 0.4,

    # Logging
    "log_file": "logs/traffic_log.json",

    # Density thresholds & map (smoothed counts -> density level)
    "density_thresholds": {
        "low": 0.0,
        "base": 4.0,
        "medium": 8.0,
        "high": 12.0
    },
    # Map density level -> explicit fixed green time (seconds)
    "density_time_map": {
        "low": 20.0,
        "base": 15.0,
        "medium": 25.0,
        "high": 30.0
    },

    # How many main round-robin repetitions to run per plan (~10-15)
    "main_rounds_per_plan": 12,

    # Grace loop per-lane green interval (seconds)
    "grace_green_time": 18.0,

    # Whether to use EWMA-smoothed counts as the basis for density
    "use_smoothed_for_density": True,
}


class decide_signal:
    def __init__(self, config: Optional[dict] = None, lanes: Optional[Iterable[str]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Logging path
        default_log = DEFAULT_CONFIG.get("log_file")
        if str(self.config.get("log_file", "")) == str(default_log):
            temp_log = os.path.join(tempfile.gettempdir(), "ai_traffic_log.json")
            self.config["log_file"] = temp_log

        # Timing & control params
        self.base_time = float(self.config["base_time"])
        self.factor = float(self.config["factor"])
        self.min_time = float(self.config["min_time"])
        self.max_time = float(self.config["max_time"])
        self.yellow_time = float(self.config["yellow_time"])
        self.all_red_time = float(self.config["all_red_time"])
        self.control_interval = float(self.config["control_interval"])

        # Behavior params
        self.service_order = str(self.config.get("service_order", "round_robin"))
        self.smoothing_alpha = float(self.config.get("smoothing_alpha", 0.4))

        # Logging
        self.log_file = str(self.config.get("log_file", "logs/traffic_log.json"))

        # Density config & map
        self.density_thresholds: Dict[str, float] = dict(self.config.get("density_thresholds", {}))
        self.density_time_map: Dict[str, float] = dict(self.config.get("density_time_map", {}))
        self.use_smoothed_for_density = bool(self.config.get("use_smoothed_for_density", True))

        # Main/grace parameters
        self.main_rounds_per_plan = int(self.config.get("main_rounds_per_plan", 12))
        self.grace_green_time = float(self.config.get("grace_green_time", 18.0))

        # Lane bookkeeping
        self.lanes: List[str] = list(lanes) if lanes else ["lane1", "lane2", "lane3", "lane4"]
        now = time.time()
        self.last_served: Dict[str, float] = {lane: now for lane in self.lanes}

        # smoothed counts (EWMA)
        self.smoothed_counts: Dict[str, float] = {lane: 0.0 for lane in self.lanes}
        # Round-robin index (points to next lane to start from when building plan)
        self.rr_index = 0

        # Red-duration tracking (monotonic time)
        mono_now = time.monotonic()
        self.red_start: Dict[str, float] = {lane: mono_now for lane in self.lanes}
        self.last_red_duration: Dict[str, float] = {lane: 0.0 for lane in self.lanes}

        self._ensure_logfile()

    # ---------- Logging ----------
    def _ensure_logfile(self):
        logdir = os.path.dirname(self.log_file) or "."
        os.makedirs(logdir, exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("[]")

    def _append_json_record(self, record: dict):
        try:
            self._ensure_logfile()
            with open(self.log_file, "r+", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except Exception:
                    data = []
                data.append(record)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print("Warning: failed to append JSON log:", e)

    def register_lanes(self, lanes: Iterable[str]):
        for lane in lanes:
            if lane not in self.lanes:
                self.lanes.append(lane)
                self.last_served[lane] = time.time()
                self.smoothed_counts[lane] = 0.0
                self.red_start[lane] = time.monotonic()
                self.last_red_duration[lane] = 0.0

    # ---------- Timing calculation (legacy fallback) ----------
    def calculate_green_time(self, vehicle_count: float) -> float:
        v = max(0.0, float(vehicle_count))
        green = self.base_time + self.factor * math.sqrt(v)
        green = max(self.min_time, min(self.max_time, green))
        return float(green)

    # ---------- Density helpers ----------
    def _density_level(self, value: float) -> str:
        # Determine density label based on thresholds (lower bounds)
        th = self.density_thresholds
        sorted_levels = sorted(th.items(), key=lambda kv: kv[1])
        level = "low"
        for lvl, bound in sorted_levels:
            if value >= float(bound):
                level = lvl
        return level

    # ---------- Plan generation: round-robin order + green times ----------
    def plan_from_counts(self, counts: Dict[str, int]) -> Tuple[List[str], Dict[str, float]]:
        """
        Produce:
         - ordered list of lanes to serve (round-robin order, rotated by rr_index)
         - mapping lane -> green_time (seconds) based on density_time_map
        Density only affects green_time. Ordering is pure round-robin.
        """
        self.register_lanes(counts.keys())

        # Update EWMA smoothing with the provided counts
        alpha = self.smoothing_alpha
        for lane in self.lanes:
            raw = float(counts.get(lane, 0))
            prev = self.smoothed_counts.get(lane, 0.0)
            self.smoothed_counts[lane] = alpha * raw + (1 - alpha) * prev

        # Compute green times per lane using density mapping (smoothed counts by default)
        green_times: Dict[str, float] = {}
        for lane in self.lanes:
            basis = self.smoothed_counts[lane] if self.use_smoothed_for_density else float(counts.get(lane, 0))
            density = self._density_level(basis)
            gtime = float(self.density_time_map.get(density, self.base_time))
            green_times[lane] = gtime

        # Build round-robin ordered list rotated by rr_index
        n = len(self.lanes)
        if n == 0:
            return [], green_times
        idx = int(self.rr_index) % n
        ordered = self.lanes[idx:] + self.lanes[:idx]
        return ordered, green_times

    # ---------- Format cycle for UI/logging ----------
    def _format_cycle(self, served_lane: Optional[str], green_time: float, counts: Dict[str, int]) -> Dict[str, dict]:
        """
        Build cycle dict using the provided green_time for the currently served lane.
        For red lanes we show an expected red interval (green_time + yellow + all_red).
        """
        cycle = {}
        self.register_lanes(counts.keys())

        for i, lane in enumerate(self.lanes, start=1):
            count = int(counts.get(lane, 0))
            if lane == served_lane:
                status = "Green"
                time_val = int(round(green_time))
            else:
                time_val = int(round(green_time + self.yellow_time + self.all_red_time))
                status = "Red"

            last_red = int(round(self.last_red_duration.get(lane, 0.0)))

            cycle[f"lane{i}"] = {
                "name": lane,
                "status": status,
                "time": time_val,
                "last_red_duration": last_red,
                "detected_count": count
            }

        return cycle

    # ---------- Public API: run once ----------
    def run_once(self, counts: Dict[str, int]) -> Tuple[Optional[str], float]:
        counts = counts or {}
        self.register_lanes(counts.keys())

        # Update EWMA
        alpha = self.smoothing_alpha
        for lane in self.lanes:
            raw = float(counts.get(lane, 0))
            prev = self.smoothed_counts.get(lane, 0.0)
            self.smoothed_counts[lane] = alpha * raw + (1 - alpha) * prev

        if not self.lanes:
            return None, 0.0

        # Choose next lane using round-robin (ignore density for ordering)
        served = self.lanes[self.rr_index % len(self.lanes)]
        # advance rr_index for next call
        self.rr_index = (self.rr_index + 1) % max(1, len(self.lanes))

        # Determine green_time from density mapping (smoothed counts)
        vehicle_count_for_served = self.smoothed_counts.get(served, 0.0)
        density = self._density_level(vehicle_count_for_served)
        green_time = float(self.density_time_map.get(density, self.base_time))

        # Update last_served timestamp
        self.last_served[served] = time.time()

        # compute last red duration for this lane (if any)
        now_m = time.monotonic()
        last_red = 0.0
        if served in self.red_start:
            last_red = now_m - self.red_start[served]
            self.last_red_duration[served] = last_red
            # reset red start mark — will be set when it turns red later
            self.red_start[served] = now_m

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cycle = self._format_cycle(served, green_time, counts)
        record = {"timestamp": ts, "cycle": cycle}
        self._append_json_record(record)

        return served, float(green_time)

    # ---------- Mixed loop (main rounds + grace loop) ----------
    def run_mixed_cycle(
        self,
        poll_counts_fn: Callable[[], Dict[str, int]],
        update_ui_fn: Optional[Callable[[str, float], None]] = None,
        stop_after_plans: Optional[int] = None
    ):
        """
        Loop:
         - compute plan (round-robin order rotated by rr_index + green times by density)
         - run main round-robin for main_rounds_per_plan times (each lane served in order each round)
         - run a grace loop: each lane gets grace_green_time sequentially; while grace runs, poll and recompute next plan
         - repeat
        """
        plan_count = 0
        try:
            latest_counts = poll_counts_fn() or {}

            while True:
                ordered, green_times = self.plan_from_counts(latest_counts)
                rounds = max(1, int(self.main_rounds_per_plan))
                print(f"Starting main rounds (plan #{plan_count+1}): {rounds} rounds, order: {ordered}")

                for r in range(rounds):
                    for lane in ordered:
                        # pick the planned green_time for this lane
                        green_time = float(green_times.get(lane, self.base_time))

                        # compute and record red duration when this lane turns green
                        now_m = time.monotonic()
                        if lane in self.red_start:
                            self.last_red_duration[lane] = now_m - self.red_start[lane]
                        else:
                            self.last_red_duration[lane] = 0.0

                        # Update last_served and red_start markers
                        self.last_served[lane] = time.time()
                        self.red_start[lane] = now_m

                        # Log / UI
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cycle = self._format_cycle(lane, green_time, latest_counts)
                        print(json.dumps({"timestamp": ts, "cycle": cycle}, indent=2))
                        self._append_json_record({"timestamp": ts, "cycle": cycle})
                        if update_ui_fn:
                            try:
                                update_ui_fn(lane, green_time)
                            except Exception as e:
                                print("Warning: update_ui_fn error:", e)

                        # While green is active, keep polling counts (to keep smoothed_counts updated)
                        green_start = time.monotonic()
                        poll_step = 0.5
                        next_counts = latest_counts
                        while (time.monotonic() - green_start) < max(0.0, green_time):
                            time.sleep(poll_step)
                            try:
                                latest = poll_counts_fn() or {}
                                if latest:
                                    next_counts = latest
                                    # update EWMA online
                                    alpha = self.smoothing_alpha
                                    for ln in self.lanes:
                                        raw = float(next_counts.get(ln, 0))
                                        prev = self.smoothed_counts.get(ln, 0.0)
                                        self.smoothed_counts[ln] = alpha * raw + (1 - alpha) * prev
                            except Exception:
                                pass

                        # Yellow + all-red phases (locked)
                        time.sleep(max(0.0, self.yellow_time))
                        if getattr(self, "all_red_time", 0.0) > 0:
                            self.red_start[lane] = time.monotonic()
                            time.sleep(self.all_red_time)

                        # advance rr_index so next plan will start from the next lane
                        if len(self.lanes) > 0:
                            # Find current lane index in lanes list and set rr_index to next
                            try:
                                cur_idx = self.lanes.index(lane)
                                self.rr_index = (cur_idx + 1) % len(self.lanes)
                            except ValueError:
                                # lane was removed concurrently; keep rr_index unchanged
                                pass

                        latest_counts = next_counts

                # After main rounds, run the grace loop while recomputing next plan progressively
                print(f"Main rounds completed for plan #{plan_count+1}. Entering grace loop (grace {self.grace_green_time}s per lane).")
                recomputed_counts = latest_counts.copy()
                for ln in self.lanes:
                    if ln not in self.red_start:
                        self.red_start[ln] = time.monotonic()

                for lane in self.lanes:
                    green_time = float(self.grace_green_time)
                    now_m = time.monotonic()
                    if lane in self.red_start:
                        self.last_red_duration[lane] = now_m - self.red_start[lane]
                    else:
                        self.last_red_duration[lane] = 0.0

                    self.last_served[lane] = time.time()
                    self.red_start[lane] = now_m

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cycle = self._format_cycle(lane, green_time, recomputed_counts)
                    print(json.dumps({"timestamp": ts, "cycle": cycle}, indent=2))
                    self._append_json_record({"timestamp": ts, "cycle": cycle})
                    if update_ui_fn:
                        try:
                            update_ui_fn(lane, green_time)
                        except Exception as e:
                            print("Warning: update_ui_fn error:", e)

                    # Poll during grace period and update smoothed counts for next plan computation
                    grace_start = time.monotonic()
                    step = 0.5
                    while (time.monotonic() - grace_start) < max(0.0, green_time):
                        time.sleep(step)
                        try:
                            latest = poll_counts_fn() or {}
                            if latest:
                                recomputed_counts = latest
                                alpha = self.smoothing_alpha
                                for ln in self.lanes:
                                    raw = float(recomputed_counts.get(ln, 0))
                                    prev = self.smoothed_counts.get(ln, 0.0)
                                    self.smoothed_counts[ln] = alpha * raw + (1 - alpha) * prev
                        except Exception:
                            pass

                    # Yellow + all-red
                    time.sleep(max(0.0, self.yellow_time))
                    if getattr(self, "all_red_time", 0.0) > 0:
                        self.red_start[lane] = time.monotonic()
                        time.sleep(self.all_red_time)

                    # After each lane in grace, use freshest counts to update recomputed_counts/latest_counts
                    latest_counts = recomputed_counts

                # compute next plan using freshest counts
                ordered, green_times = self.plan_from_counts(latest_counts)
                plan_count += 1
                print(f"Grace loop finished. Next plan #{plan_count+1} prepared with order: {ordered}")

                if stop_after_plans and plan_count >= int(stop_after_plans):
                    print("Stopping after requested plans count.")
                    break

        except KeyboardInterrupt:
            print("Controller interrupted by user.")