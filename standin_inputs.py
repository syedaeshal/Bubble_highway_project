"""
standin_inputs.py - made-up inputs so the CV part can be tested ALONE
=====================================================================

Guide: "Until the real parts are ready, each person tests their own part with
made-up inputs." Nothing here is Energy Engineer 1's real physics; it only
produces jellyfish in the agreed format (id, x, y, speed, size, depth, status)
so the camera, detector and counter can be checked.

Three kinds of made-up input:
1. random_scene      - one random snapshot of jellyfish at a chosen crowding
                       (used for the training dataset and accuracy tests)
2. well_spaced_scene - 20 well-spaced jellyfish (the guide's "done" test)
3. StandinBloom      - a simple bloom that builds up, peaks and fades while
                       drifting down the corridor (used for Job D end-to-end tests)
"""
import numpy as np
from camera import STATUS_IN, view_x_range


def _depths(rng, n, mean_m, band_m):
    """Depths 'mostly near the surface' (EE1 Job A step 4): an exponential
    distribution, re-drawn if deeper than the depth band."""
    d = rng.exponential(mean_m, n)
    bad = d > band_m
    while bad.any():
        d[bad] = rng.exponential(mean_m, bad.sum())
        bad = d > band_m
    return d


def random_scene(settings, density_per_m3, rng, camera="entrance",
                 size_range=(0.30, 0.45), depth_mean_m=1.0, speed_mps=0.2):
    """Random jellyfish filling one camera's view.

    density_per_m3 is REAL jellyfish per cubic metre (the bloom's unit). Each
    simulated jellyfish stands for settings['represents'] real ones, so the
    number drawn is density x volume / represents. For camera accuracy tests
    we pass represents = 1, so every real jellyfish is drawn.
    """
    x0, x1 = view_x_range(settings, camera)
    width, band = settings["corridor_width_m"], settings["depth_band_m"]
    volume = (x1 - x0) * width * band
    n = rng.poisson(density_per_m3 * volume / settings["represents"])
    return {
        "id": np.arange(n),
        "x": rng.uniform(x0, x1, n),
        "y": rng.uniform(0.0, width, n),
        "speed": np.full(n, speed_mps),
        "size": rng.uniform(*size_range, n),
        "depth": _depths(rng, n, depth_mean_m, band),
        "status": np.full(n, STATUS_IN),
    }


def well_spaced_scene(settings, n_along=5, n_across=4, size_m=0.40, depth_m=0.3):
    """The guide's 'done' test: 20 well-spaced jellyfish (a 5 x 4 grid, ~4 m apart)."""
    x0, x1 = view_x_range(settings, "entrance")
    xs = np.linspace(x0 + 2.0, x1 - 6.0, n_along)   # room to drift forward
    ys = np.linspace(2.5, settings["corridor_width_m"] - 2.5, n_across)
    X, Y = np.meshgrid(xs, ys)
    n = X.size
    return {"id": np.arange(n), "x": X.ravel(), "y": Y.ravel(),
            "speed": np.zeros(n), "size": np.full(n, size_m),
            "depth": np.full(n, depth_m), "status": np.full(n, STATUS_IN)}


def moved(J, vx, vy, dt):
    """Copy of jellyfish J after drifting at (vx, vy) m/s for dt seconds."""
    out = {k: np.array(v, copy=True) for k, v in J.items()}
    out["x"] = out["x"] + vx * dt
    out["y"] = out["y"] + vy * dt
    out["speed"] = np.full(len(out["x"]), vx)
    return out


class StandinBloom:
    """A simple bloom drifting down the corridor - NOT Energy Engineer 1's model.

    * Crowding at the entrance follows a hill: builds up, peaks, fades.
    * Arrivals per second = crowding x current x width x depth band
      (the same formula as EE1 Job A step 2), with a random Poisson draw.
    * Jellyfish drift with the current plus a little random movement.
    """

    def __init__(self, settings, duration_s=1800.0, peak_density=0.5,
                 current_mps=0.2, seed=None):
        self.s = settings
        self.duration = duration_s
        self.peak = peak_density
        self.current = current_mps
        self.rng = np.random.default_rng(settings["random_seed"] if seed is None else seed)
        self.t = 0.0
        self.next_id = 0
        self.J = {k: np.empty(0) for k in ("x", "y", "speed", "size", "depth")}
        self.J["id"] = np.empty(0, int)
        self.J["status"] = np.empty(0, dtype="<U16")

    def density(self, t):
        """Hill-shaped crowding (real jellyfish per m^3): peak in the middle."""
        mid, spread = self.duration / 2.0, self.duration / 6.0
        return self.peak * np.exp(-0.5 * ((t - mid) / spread) ** 2)

    def step(self, dt=1.0):
        s, rng, J = self.s, self.rng, self.J
        # move everyone (forward with the current, a little random movement)
        n = len(J["x"])
        J["x"] = J["x"] + (self.current + rng.normal(0, 0.03, n)) * dt
        J["y"] = np.clip(J["y"] + rng.normal(0, 0.02, n) * dt, 0.0, s["corridor_width_m"])
        J["status"] = np.where(J["x"] > s["corridor_length_m"], "out_to_sea", J["status"])
        # new arrivals at the entrance
        rate = (self.density(self.t) * self.current * s["corridor_width_m"]
                * s["depth_band_m"] / s["represents"])
        k = rng.poisson(rate * dt)
        new = {
            "id": np.arange(self.next_id, self.next_id + k),
            "x": rng.uniform(0.0, self.current * dt, k),
            "y": rng.uniform(0.0, s["corridor_width_m"], k),
            "speed": np.full(k, self.current),
            "size": rng.uniform(0.30, 0.45, k),
            "depth": _depths(rng, k, 1.0, s["depth_band_m"]),
            "status": np.full(k, STATUS_IN),
        }
        self.next_id += k
        keep = J["status"] == STATUS_IN          # drop those already out to sea
        self.J = {f: np.concatenate([J[f][keep], new[f]]) for f in J}
        self.t += dt
        return self.J
