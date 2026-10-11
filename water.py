"""
water.py - ENERGY ENGINEER 1, JOB B: the bubbles and the water  [STAND-IN until EE1's version]
==============================================================================================
Receives: which pipes fire and with how much air (from the controller), 20 x per second.
Passes on: where every jellyfish is, at every moment (+ the scorecard records).

Step by step (numbers match the guide's Job B list):
 1. Pipes across the corridor every pipe_spacing_m (bubble_physics.pipe_positions).
 2. Firing order comes from the controller (controller.pipes_firing).
 3. Each firing pipe pushes the water: strongest ~d/2 from the pipe, fading after;
    optimistic = forward only, realistic = both ways (bubble_physics.push_shape).
    Pushes from several pipes add up, capped at one full pipe's push.
 4. Push linked to air by Bulson's cube-root law (bubble_physics.surface_speed).
 5. Push weakened with depth: only the top H/4 = 1.25 m feels it.
 6. Background current added (can help, oppose, or turn like a tide).
 7. Jellyfish catch up with the water gradually over the response time tau,
    bigger ones more slowly: tau = tau0 x (size / 0.375 m)^exponent.
 8. Small random swimming forward/back and slow sideways drift (random but
    smooth: each lasts ~swim_timescale_s).
 9. Status: past the exit = out_to_sea (success); back out of the entrance =
    back_to_intake (failure); out of the side = lost_sideways.
10. Roughness: how sharply the water speed changes from the front to the back
    of the body, per metre of body (1/s); the largest value each jellyfish felt.
Also (Job C step 4): time spent right above a firing pipe.
"""
import numpy as np
import bubble_physics as bp
from camera import STATUS_IN

FIELDS = ("id", "x", "y", "speed", "size", "depth", "status")
EXTRA = ("swim", "side", "t_enter", "rough_max", "t_above")


class Water:
    def __init__(self, s, rng):
        self.s, self.rng = s, rng
        self.pipes = bp.pipe_positions(s)
        self.J = {k: np.empty(0) for k in FIELDS + EXTRA}
        self.J["id"] = np.empty(0, int)
        self.J["status"] = np.empty(0, dtype="<U16")
        self.finished = {k: [] for k in ("id", "status", "t_enter", "t_exit", "rough_max",
                                         "t_above", "size", "depth")}

    # ------------------------------------------------------------------
    def add(self, new, t):
        n = len(new["x"])
        if n == 0:
            return
        new = dict(new)
        new.update(swim=np.zeros(n), side=np.zeros(n), t_enter=np.full(n, t),
                   rough_max=np.zeros(n), t_above=np.zeros(n))
        self.J = {k: np.concatenate([self.J[k], new[k]]) for k in self.J}

    def water_speed(self, x, depth, firing, air, t):
        """Water speed (m/s along the corridor) at positions x and depths."""
        s = self.s
        u = np.full(np.shape(x), bp.background_current(t, s), float)
        if len(firing) == 0 or np.size(x) == 0:
            return u
        dx = np.asarray(x)[:, None] - self.pipes[firing][None, :]
        push = np.clip(bp.push_shape(dx, s).sum(axis=1), -1.0, 1.0)
        return u + bp.surface_speed(air, s) * bp.depth_factor(depth, s) * push

    # ------------------------------------------------------------------
    def step(self, firing, air, t, dt):
        s, rng, J = self.s, self.rng, self.J
        n = len(J["x"])
        if n == 0:
            return
        # 3-6: the water speed where each jellyfish is
        u = self.water_speed(J["x"], J["depth"], firing, air, t)
        # 7: catch up gradually (exact solution of dv/dt = (u - v)/tau over dt)
        tau = s["response_time_s"] * (J["size"] / 0.375) ** s["response_size_exponent"]
        J["speed"] = u + (J["speed"] - u) * np.exp(-dt / tau)
        # 8: random swimming (smooth random processes)
        a = dt / s["swim_timescale_s"]
        J["swim"] += -J["swim"] * a + s["swim_noise_mps"] * np.sqrt(2 * a) * rng.normal(size=n)
        J["side"] += -J["side"] * a + s["side_drift_mps"] * np.sqrt(2 * a) * rng.normal(size=n)
        J["x"] = J["x"] + (J["speed"] + J["swim"]) * dt
        J["y"] = J["y"] + J["side"] * dt
        # 10: roughness = speed difference front-to-back / body length
        half = J["size"] / 2
        du = np.abs(self.water_speed(J["x"] + half, J["depth"], firing, air, t)
                    - self.water_speed(J["x"] - half, J["depth"], firing, air, t))
        J["rough_max"] = np.maximum(J["rough_max"], du / J["size"])
        # Job C step 4: time right above a firing pipe (bubbles rise through them)
        if len(firing):
            near = np.abs(J["x"][:, None] - self.pipes[firing][None, :]).min(axis=1)
            J["t_above"] += dt * (near <= s["above_pipe_halfwidth_m"])
        # 9: status
        st = np.full(n, STATUS_IN, dtype="<U16")
        st[J["x"] > s["corridor_length_m"]] = "out_to_sea"
        st[J["x"] < 0] = "back_to_intake"
        st[(J["y"] < 0) | (J["y"] > s["corridor_width_m"])] = "lost_sideways"
        done = st != STATUS_IN
        if done.any():
            f = self.finished
            f["id"] += J["id"][done].tolist()
            f["status"] += st[done].tolist()
            f["t_enter"] += J["t_enter"][done].tolist()
            f["t_exit"] += [t] * int(done.sum())
            for k in ("rough_max", "t_above", "size", "depth"):
                f[k] += J[k][done].tolist()
            self.J = {k: v[~done] for k, v in J.items()}

    # ------------------------------------------------------------------
    def jellyfish(self):
        """Where every jellyfish is now, in the agreed format - for the camera."""
        return {k: self.J[k] for k in FIELDS}

    def all_records(self, t_end):
        """Finished jellyfish plus those still in the corridor at the end."""
        f = {k: list(v) for k, v in self.finished.items()}
        J = self.J
        f["id"] += J["id"].tolist()
        f["status"] += ["in_corridor"] * len(J["x"])
        f["t_enter"] += J["t_enter"].tolist()
        f["t_exit"] += [t_end] * len(J["x"])
        for k in ("rough_max", "t_above", "size", "depth"):
            f[k] += J[k].tolist()
        return {k: np.array(v) for k, v in f.items()}
