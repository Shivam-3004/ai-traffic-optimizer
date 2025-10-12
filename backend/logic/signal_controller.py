#!/usr/bin/env python3
"""
Improved signal_controller.py  (modified)

Changes made (high level):
- Introduced density levels (low, base, medium, high) mapped to fixed green times.
- Priority (order) is chosen by density level (high-first), tie-break by smoothed count and starvation.
- Implemented a "main" round-robin loop that runs N times (default 12; configurable 10-15).
- After main rounds, runs a "grace loop" where each lane receives a short fixed green (default 15-20s).
  While the grace loop runs, the controller keeps polling counts and computes the new plan (timings + order)
  so the next main round will use the updated plan.
- Added red-duration tracking for each lane (reports the last red duration when a lane turns green).
- Kept EWMA smoothing & logging behaviors.
"""

from __future__ import annotations
import os
import json
import time
import math
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import tempfile
import random

DEFAULT_CONFIG = {
    # Green timing model (fallback for old behavior; not used when density_time_map used)
    "base_time": 15.0,
    "factor": 2.0,
    "min_time": 8.0,
    "max_time": 45.0,

    # Transition / safety times
    "yellow_time": 4.0,
    "all_red_time": 1.0,

    # Controller behavior
    "control_interval": 5.0,
    "starvation_weight": 1.0,
    "max_starvation_multiplier": 3.0,
    "service_order": "auto",
    "smoothing_alpha": 0.4,

    # Logging
    "log_file": "logs/traffic_log.json",

    # Density thresholds & map (you can override in config)
    # These thresholds are applied to the smoothed counts
    "density_thresholds": {
        "low": 0.0,     # >= 0
        "base": 4.0,    # >= 4
        "medium": 8.0,  # >= 8
        "high": 12.0    # >= 12
    },
    # Map each density level to an explicit fixed green time (seconds).
    # You asked to "define how much time should be given to each density level".
    # Example defaults below follow your example ranges; override via config if needed.
    "density_time_map": {
        "low": 20.0,
        "base": 15.0,
        "medium": 25.0,
        "high": 30.0
    },

    # How many main round-robin repetitions to run per plan (choose a number in ~10-15)
    "main_rounds_per_plan": 12,

    # Grace loop per-lane green interval (seconds). We'll choose 15-20s; default 18.
    "grace_green_time": 18.0,

    # Whether to use EWMA-smoothed counts as the basis for density (recommended)
    "use_smoothed_for_density": True,

    # high-density protection (cap wait for high density lanes)
    "high_density_threshold": 12,
    "max_red_wait_high_density": 35.0,
}


class decide_signal:
    def __init__(self, config: Optional[dict] = None, lanes: Optional[Iterable[str]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            # shallow update; for nested dicts, user may need to pass full dict
            self.config.update(config)

        # Logging path adjustment if default used
        default_log = DEFAULT_CONFIG.get("log_file")
        if str(self.config.get("log_file", "")) == str(default_log):
            temp_log = os.path.join(tempfile.gettempdir(), "ai_traffic_log.json")
            self.config["log_file"] = temp_log

        # Timing & control params (legacy)
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

        # Density config & map
        self.density_thresholds: Dict[str, float] = dict(self.config.get("density_thresholds", {}))
        self.density_time_map: Dict[str, float] = dict(self.config.get("density_time_map", {}))
        self.use_smoothed_for_density = bool(self.config.get("use_smoothed_for_density", True))

        # Main/grace parameters
        self.main_rounds_per_plan = int(self.config.get("main_rounds_per_plan", 12))
        self.grace_green_time = float(self.config.get("grace_green_time", 18.0))

        # high-density protection
        self.high_density_threshold = float(self.config.get("high_density_threshold", 12))
        self.max_red_wait_high_density = float(self.config.get("max_red_wait_high_density", 35.0))

        # Lane bookkeeping
        self.lanes: List[str] = list(lanes) if lanes else ["lane1", "lane2", "lane3", "lane4"]
        now = time.time()
        self.last_served: Dict[str, float] = {lane: now for lane in self.lanes}
        # smoothed counts (EWMA)
        self.smoothed_counts: Dict[str, float] = {lane: 0.0 for lane in self.lanes}
        self.rr_index = 0

        # Red-duration tracking: when a lane became red (monotonic time)
        # Initialize as now because at startup lanes are assumed red until served
        mono_now = time.monotonic()
        self.red_start: Dict[str, float] = {lane: mono_now for lane in self.lanes}
        # last measured red duration when that lane next turned green
        self.last_red_duration: Dict[str, float] = {lane: 0.0 for lane in self.lanes}

        # ensure logging path
        self._ensure_logfile()

    # ---------- Logging (JSON array) ----------
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
        """
        Determine density label from numeric value (smoothed count recommended).
        Uses thresholds in ascending order: low < base < medium < high.
        """
        # Ensure thresholds exist and have numeric ordering
        # We interpret threshold values as lower bounds for that level.
        th = self.density_thresholds
        # Sort levels by their threshold value
        sorted_levels = sorted(th.items(), key=lambda kv: kv[1])
        level = "low"
        for lvl, bound in sorted_levels:
            if value >= float(bound):
                level = lvl
        return level

    # Ranking for sorting: higher density -> higher priority
    _density_rank_map = {"low": 1, "base": 2, "medium": 3, "high": 4}

    def _density_rank(self, level: str) -> int:
        return int(self._density_rank_map.get(level, 1))

    def plan_from_counts(self, counts: Dict[str, int]) -> Tuple[List[str], Dict[str, float]]:
        """
        Given a counts snapshot (raw or use smoothed), produce:
         - ordered list of lanes to serve (priority order),
         - mapping lane -> green_time (seconds) based on density_time_map.
        Uses smoothed_counts if configured.
        """
        self.register_lanes(counts.keys())

        # ensure EWMA is updated with the provided counts so plan considers freshest data
        alpha = self.smoothing_alpha
        for lane in self.lanes:
            raw = float(counts.get(lane, 0))
            prev = self.smoothed_counts.get(lane, 0.0)
            self.smoothed_counts[lane] = alpha * raw + (1 - alpha) * prev

        # Build per-lane density and green time
        green_times: Dict[str, float] = {}
        lane_info = []
        for lane in self.lanes:
            basis = self.smoothed_counts[lane] if self.use_smoothed_for_density else float(counts.get(lane, 0))
            density = self._density_level(basis)
            gtime = float(self.density_time_map.get(density, self.base_time))
            green_times[lane] = gtime
            # score for ordering: density rank, smoothed count, time since last served (to prefer starving high lanes)
            wait = time.time() - self.last_served.get(lane, 0.0)
            lane_info.append((lane, density, basis, wait))

        # First, enforce "starving high-density" protection if any high-density lane waited too long
        now = time.time()
        starving_high = []
        for lane, density, basis, wait in lane_info:
            if density == "high":
                if (now - self.last_served.get(lane, 0.0)) >= self.max_red_wait_high_density:
                    starving_high.append((now - self.last_served.get(lane, 0.0), lane))

        if starving_high:
            # Put the longest-waiting high density lane at front, then others by normal order
            starving_high.sort(key=lambda x: (-x[0], x[1]))
            priority_first = starving_high[0][1]
            # produce ordered list: first the priority, then remaining sorted by normal ordering
            remaining = [l for l in self.lanes if l != priority_first]
            # sort remaining by density rank then smoothed count descending then oldest last_served (longer wait higher)
            remaining_sorted = sorted(
                remaining,
                key=lambda l: (
                    -self._density_rank(self._density_level(self.smoothed_counts.get(l, 0.0))),
                    -self.smoothed_counts.get(l, 0.0),
                    -(time.time() - self.last_served.get(l, 0.0))
                )
            )
            ordered = [priority_first] + remaining_sorted
            return ordered, green_times

        # Normal ordering: sort by density rank desc, then smoothed count desc, then longest-wait desc
        ordered = sorted(
            self.lanes,
            key=lambda l: (
                -self._density_rank(self._density_level(self.smoothed_counts.get(l, 0.0))),
                -self.smoothed_counts.get(l, 0.0),
                -(time.time() - self.last_served.get(l, 0.0))
            )
        )

        return ordered, green_times

    # ---------- Starvation boost (legacy, still available) ----------
    def _starvation_boost(self, lane: str) -> float:
        now = time.time()
        last = self.last_served.get(lane, 0.0)
        wait = max(0.0, now - last)
        multiplier = (wait / max(1.0, self.control_interval)) * self.starvation_weight
        return min(multiplier, self.max_starvation_multiplier)

    # ---------- Format cycle for UI/logging (includes last red duration) ----------
    def _format_cycle(self, served_lane: Optional[str], green_time: float, counts: Dict[str, int]) -> Dict[str, dict]:
        """
        Build cycle dict:
        laneN: {"status": "Green"/"Red", "time": int, "last_red_duration": int}
        """
        cycle = {}
        self.register_lanes(counts.keys())

        for i, lane in enumerate(self.lanes, start=1):
            count = int(counts.get(lane, 0))
            if lane == served_lane:
                status = "Green"
                time_val = int(round(green_time))
            else:
                # The red duration expected while this lane is not served: we include green+yellow+all_red
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

    # ---------- Public API: run once (kept for compatibility) ----------
    def run_once(self, counts: Dict[str, int]) -> Tuple[Optional[str], float]:
        counts = counts or {}
        self.register_lanes(counts.keys())

        # Update EWMA
        alpha = self.smoothing_alpha
        for lane in self.lanes:
            raw = float(counts.get(lane, 0))
            prev = self.smoothed_counts.get(lane, 0.0)
            self.smoothed_counts[lane] = alpha * raw + (1 - alpha) * prev

        # fallback selection: round-robin if all zero
        total_raw = sum(int(counts.get(l, 0)) for l in self.lanes)
        if total_raw == 0:
            if self.lanes:
                served = self.lanes[self.rr_index % len(self.lanes)]
                self.rr_index += 1
            else:
                return None, 0.0
        else:
            # use plan_from_counts to choose ordered lanes, pick top
            ordered, green_times = self.plan_from_counts(counts)
            served = ordered[0]

        vehicle_count_for_served = self.smoothed_counts.get(served, 0.0)
        green_time = self.calculate_green_time(vehicle_count_for_served)

        # update last_served timestamp
        self.last_served[served] = time.time()

        # compute last red duration for this lane (if any)
        now_m = time.monotonic()
        last_red = 0.0
        if served in self.red_start:
            last_red = now_m - self.red_start[served]
            self.last_red_duration[served] = last_red
            # reset red start since this lane is now green
            self.red_start[served] = now_m  # will set new red start when it turns red again

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cycle = self._format_cycle(served, green_time, counts)
        record = {"timestamp": ts, "cycle": cycle}
        self._append_json_record(record)

        return served, float(green_time)

    # ---------- New high-level mixed loop (main rounds + grace loop + recompute while grace runs) ----------
    def run_mixed_cycle(
        self,
        poll_counts_fn: Callable[[], Dict[str, int]],
        update_ui_fn: Optional[Callable[[str, float], None]] = None,
        stop_after_plans: Optional[int] = None
    ):
        """
        New loop implementing:
         - compute plan (order + times) from counts
         - run main round-robin of that plan for `main_rounds_per_plan` times (each lane served in order each round)
         - after main rounds, run a grace loop: each lane gets grace_green_time seconds of green (sequentially)
             * while the grace loop runs, the controller continuously polls counts and computes the next plan
         - repeat (new plan becomes the active plan)
        Parameters:
          poll_counts_fn: callable returning {lane: count}
          update_ui_fn: hook (served_lane, green_time)
          stop_after_plans: optional int -> stop after this many full plans completed
        """
        plan_count = 0
        try:
            # initial snapshot
            latest_counts = poll_counts_fn() or {}

            while True:
                # compute plan from current counts
                ordered, green_times = self.plan_from_counts(latest_counts)

                # Stick to the user's instruction: run main loop ~10-15 times (configurable)
                rounds = max(1, int(self.main_rounds_per_plan))
                print(f"Starting main rounds (plan #{plan_count+1}): {rounds} rounds, order: {ordered}")
                for r in range(rounds):
                    for lane in ordered:
                        # Determine green_time for this lane from plan (fallback to calculate_green_time)
                        green_time = float(green_times.get(lane, self.base_time))
                        # --- Start green for 'lane' ---
                        served = lane
                        # compute and record red duration when this lane turns green
                        now_m = time.monotonic()
                        if lane in self.red_start:
                            red_dur = now_m - self.red_start[lane]
                            self.last_red_duration[lane] = red_dur
                        else:
                            self.last_red_duration[lane] = 0.0

                        # Update last_served
                        self.last_served[lane] = time.time()
                        # When lane goes green, it's no longer red: mark red_start to now (will be updated when it turns red later)
                        self.red_start[lane] = now_m

                        # Log / UI
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cycle = self._format_cycle(served, green_time, latest_counts)
                        print(json.dumps({"timestamp": ts, "cycle": cycle}, indent=2))
                        self._append_json_record({"timestamp": ts, "cycle": cycle})
                        if update_ui_fn:
                            try:
                                update_ui_fn(served, green_time)
                            except Exception as e:
                                print("Warning: update_ui_fn error:", e)

                        # While green is active, keep polling counts (to keep smoothed_counts updated) and allow early recompute
                        green_start = time.monotonic()
                        poll_step = 0.5
                        next_counts = latest_counts
                        while (time.monotonic() - green_start) < max(0.0, green_time):
                            time.sleep(poll_step)
                            try:
                                latest = poll_counts_fn() or {}
                                if latest:
                                    next_counts = latest
                                    # update EWMA online so subsequent planning uses freshest info
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
                            # When entering all-red, mark that lane becomes red -> set its red_start
                            now_m2 = time.monotonic()
                            self.red_start[lane] = now_m2
                            time.sleep(self.all_red_time)

                        # prepare for next lane using freshest polled counts
                        latest_counts = next_counts

                # Completed the main rounds for this plan
                # Now run the grace loop. While the grace loop runs, keep polling and compute next plan.
                print(f"Main rounds completed for plan #{plan_count+1}. Entering grace loop (grace {self.grace_green_time}s per lane).")
                # We'll compute the next plan incrementally while running the grace loop
                recomputed_counts = latest_counts.copy()
                # For safety, when grace loop begins, mark all lanes as red_start = now (they will be toggled as they get green)
                for ln in self.lanes:
                    # if a lane is considered red now, ensure its red_start is set
                    if ln not in self.red_start:
                        self.red_start[ln] = time.monotonic()

                # During grace loop, we still sequentially give green to each lane for grace_green_time,
                # and poll counts actively; after each lane in grace loop, we may recompute the next plan
                for lane in self.lanes:
                    served = lane
                    green_time = float(self.grace_green_time)
                    # compute red duration for this lane now turning green
                    now_m = time.monotonic()
                    if lane in self.red_start:
                        self.last_red_duration[lane] = now_m - self.red_start[lane]
                    else:
                        self.last_red_duration[lane] = 0.0
                    # mark last_served and flip red_start when it becomes red after green+y+all_red
                    self.last_served[lane] = time.time()
                    # mark lane as green right now (so red_start will be reset after it turns red)
                    self.red_start[lane] = now_m

                    # Log / UI
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cycle = self._format_cycle(served, green_time, recomputed_counts)
                    print(json.dumps({"timestamp": ts, "cycle": cycle}, indent=2))
                    self._append_json_record({"timestamp": ts, "cycle": cycle})
                    if update_ui_fn:
                        try:
                            update_ui_fn(served, green_time)
                        except Exception as e:
                            print("Warning: update_ui_fn error:", e)

                    # During grace green, poll counts frequently and update smoothed_counts
                    grace_start = time.monotonic()
                    step = 0.5
                    while (time.monotonic() - grace_start) < max(0.0, green_time):
                        time.sleep(step)
                        try:
                            latest = poll_counts_fn() or {}
                            if latest:
                                recomputed_counts = latest
                                # update EWMA online for immediate use
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
                        # mark lane turning red
                        self.red_start[lane] = time.monotonic()
                        time.sleep(self.all_red_time)

                    # After each lane in the grace loop, optionally recompute the next plan using the freshest smoothed counts
                    # We compute it but do not switch mid-grace; the computed plan will be used after grace ends.
                    ordered_next, green_times_next = self.plan_from_counts(recomputed_counts)
                    # store for after grace loop
                    latest_counts = recomputed_counts

                # finish grace loop: recomputed_counts & latest_counts hold freshest data -> compute new plan
                ordered, green_times = self.plan_from_counts(latest_counts)
                plan_count += 1
                print(f"Grace loop finished. Next plan #{plan_count+1} prepared with order: {ordered}")

                # stop condition
                if stop_after_plans and plan_count >= int(stop_after_plans):
                    print("Stopping after requested plans count.")
                    break

        except KeyboardInterrupt:
            print("Controller interrupted by user.")