"""
bloom.py - ENERGY ENGINEER 1, JOB A: the arriving bloom  [STAND-IN until EE1's version]
=======================================================================================
1. Shape: a hill. Crowding builds up, peaks in the middle of the bloom and fades
   (a bell curve with its spread = duration / 6, so it is ~0 at both ends).
2. Arrivals per second = crowding x arrival speed x entrance width x depth band
   (guide example: 0.5 x 0.2 x 20 x 3 = 6 real jellyfish per second),
   divided by `represents` (1 simulated = 10 real, step 5).
3. Random like nature: a Poisson draw around that average each time step.
4. Each new jellyfish starts in the entrance zone (first arrival_zone_m metres, an
   ASSUMPTION), at a random spot across the width,
   moving at the local current, 30-45 cm, mostly near the surface.
6. Check: total over the bloom matches the average (see expected_total).
"""
import numpy as np
from camera import STATUS_IN


class Bloom:
    def __init__(self, s, rng):
        self.s, self.rng = s, rng
        self.next_id = 0

    def density(self, t):
        """Real jellyfish per m^3 arriving at the entrance at time t."""
        s = self.s
        mid, spread = s["bloom_duration_s"] / 2, s["bloom_duration_s"] / 6
        if t > s["bloom_duration_s"]:
            return 0.0
        return s["bloom_peak_density"] * np.exp(-0.5 * ((t - mid) / spread) ** 2)

    def rate(self, t):
        """Simulated jellyfish arriving per second."""
        s = self.s
        return (self.density(t) * s["arrival_speed_mps"] * s["corridor_width_m"]
                * s["depth_band_m"] / s["represents"])

    def expected_total(self, dt=1.0):
        return sum(self.rate(t) * dt for t in np.arange(0, self.s["bloom_duration_s"], dt))

    def arrivals(self, t, dt, current):
        """New jellyfish for this time step, in the agreed format (rule 5)."""
        s, rng = self.s, self.rng
        k = rng.poisson(self.rate(t) * dt)
        d = rng.exponential(s["depth_mean_m"], k)
        while (bad := d > s["depth_band_m"]).any():          # mostly near the surface
            d[bad] = rng.exponential(s["depth_mean_m"], bad.sum())
        new = {
            "id": np.arange(self.next_id, self.next_id + k),
            "x": rng.uniform(0.0, s["arrival_zone_m"], k),   # [ASSUME] spread over the entrance zone
            "y": rng.uniform(0.0, s["corridor_width_m"], k),
            "speed": np.full(k, current),
            "size": rng.uniform(*s["size_range_m"], k),
            "depth": d,
            "status": np.full(k, STATUS_IN, dtype="<U16"),
        }
        self.next_id += k
        return new
