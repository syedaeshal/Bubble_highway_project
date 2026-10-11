"""
controller.py - ENERGY ENGINEER 2, JOB B: the brain  [STAND-IN until EE2's version]
===================================================================================
Receives every 5 s: the entrance camera's reading (CameraSystem.for_controller)
and the separate current meter [TEAM decision].
Passes on: bubbles on/off, air strength, wave speed, which pipes fire.

1. On/off without flicker: on above on_level; off only after crowding has stayed
   below the lower off_level for off_hold_s (1 min); once on, stay on >= min_on_s (2 min).
2. Air from crowding: air = min + gain x crowding, kept between min and max.
3. Wave slower than the water: wave = ratio x (Bulson water speed + current),
   ratio starting at 1/3, always below wave_ratio_max. Same formula as EE1
   (bubble_physics.surface_speed), so the two parts never disagree.
4. Learn from the camera: jellyfish drifting much slower than the wave are
   falling behind -> slow the wave (x0.8); keeping up -> speed it up gently (x1.05).
5. Fire only where jellyfish are: each wave is a moving front that starts at the
   entrance while the camera sees jellyfish and travels at the wave speed; it is
   the controller's estimate of where the swarm it launched has got to. Only
   the pipe just behind each front fires. New fronts start every wave_gap_m.
6. Safe wall mode: strong current toward the intake OR camera down -> keep the
   pipes nearest the entrance on at full air as a fixed curtain.
Modes: "off" (never fires), "curtain" (all pipes whenever on), "wave".
"""
import numpy as np
import bubble_physics as bp


class Controller:
    def __init__(self, s):
        self.s = s
        self.pipes = bp.pipe_positions(s)
        self.on = False
        self.on_since = -1e9
        self.below_since = None
        self.adj = 1.0                 # learned correction to the wave ratio
        self.fronts = []               # wave front positions (m) at the last decision
        self.last = None
        self.records = []

    def _advance(self, t):
        if self.last is not None and self.fronts:
            dt = t - self.last["t"]
            self.fronts = [f + self.last["wave_speed"] * dt for f in self.fronts]
        L = self.s["corridor_length_m"] + self.s["pipe_spacing_m"]
        self.fronts = [f for f in self.fronts if f <= L]

    def decide(self, reading, current_meter, t):
        s = self.s
        self._advance(t)
        cam_ok = bool(reading.get("camera_ok")) and reading.get("crowding_per_m3") is not None
        crowd = reading["crowding_per_m3"] if cam_ok else 0.0

        # 1. on/off with hysteresis and minimum on-time
        if s["mode"] != "off" and cam_ok:
            if not self.on and crowd > s["on_level_per_m3"]:
                self.on, self.on_since, self.below_since = True, t, None
            elif self.on:
                if crowd < s["off_level_per_m3"]:
                    self.below_since = t if self.below_since is None else self.below_since
                    if t - self.below_since >= s["off_hold_s"] and t - self.on_since >= s["min_on_s"]:
                        self.on = False
                else:
                    self.below_since = None

        # 6. wall mode
        wall = s["mode"] != "off" and (current_meter < s["wall_current_mps"] or not cam_ok)

        # 2. air from crowding
        air = float(np.clip(s["air_min_lpm_per_m"] + s["air_gain"] * crowd,
                            s["air_min_lpm_per_m"], s["air_max_lpm_per_m"]))
        if wall:
            air = s["air_max_lpm_per_m"]

        # 3 + 4. wave speed below the water speed, corrected by the camera's drift
        water = float(bp.surface_speed(air, s)) + current_meter
        prev = self.last["wave_speed"] if self.last else None
        drift = reading.get("drift_along_mps") if cam_ok else None
        if s["mode"] == "wave" and self.on and drift is not None and prev:
            if drift < 0.7 * prev:
                self.adj *= s["lag_slowdown"]
            elif drift >= 0.9 * prev:
                self.adj *= s["keepup_speedup"]
        self.adj = float(np.clip(self.adj, 0.3, s["wave_ratio_max"] / s["wave_ratio"]))
        wave_speed = max(s["wave_ratio"] * self.adj * water, s["wave_speed_min_mps"])

        # 5. launch a new wave front while jellyfish are arriving
        if s["mode"] == "wave" and self.on and not wall and crowd > s["off_level_per_m3"]:
            if not self.fronts or min(self.fronts) >= s["wave_gap_m"]:
                self.fronts.append(0.0)

        # waves already launched keep carrying their jellyfish to the exit, even after
        # the controller has switched off (firing shrinks after arrivals stop)
        firing_on = self.on or wall or (s["mode"] == "wave" and bool(self.fronts))
        d = {"t": t, "on": firing_on, "controller_on": self.on, "mode": s["mode"], "wall": wall,
             "air_lpm_per_m": air, "wave_speed": wave_speed, "fronts": list(self.fronts),
             "crowding_seen": crowd if cam_ok else None, "current_meter": current_meter}
        d["n_pipes_now"] = len(pipes_firing(d, t, self.pipes, s))
        self.last = d
        self.records.append(d)
        return d


def pipes_firing(decision, t, pipes, s):
    """Which pipes fire at time t under a decision (the last decision stays in
    force between decisions; wave fronts keep moving at its wave speed)."""
    if decision is None or not decision["on"] or decision["mode"] == "off":
        return np.empty(0, int)
    if decision["mode"] == "curtain":
        return np.arange(len(pipes))
    idx = set()
    if decision["wall"]:
        idx.update(range(min(s["wall_pipes"], len(pipes))))
    dt = t - decision["t"]
    for f in decision["fronts"]:
        X = f + decision["wave_speed"] * dt
        i = int(np.floor(X / s["pipe_spacing_m"] + 1e-9))   # the pipe at or just behind the front
        if 0 <= i < len(pipes):
            idx.add(i)
    return np.array(sorted(idx), int)
