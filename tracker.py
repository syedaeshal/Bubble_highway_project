"""
tracker.py - JOB C step 5: work out the drift
=============================================

Guide: "Match each blob with the nearest blob in the previous picture. How far
they moved in one second gives the drift speed and direction."

How it works
------------
1. Take the detections from the previous picture and this picture (in metres).
2. Pair each blob with its nearest blob in the previous picture. Each blob
   is used at most once (the Hungarian method finds the set of one-to-one
   pairs with the smallest total distance; with well-separated jellyfish this
   is exactly "nearest blob").
3. Reject pairs further apart than max speed x time between pictures
   (1 m/s x 1 s = 1 m). Those are new arrivals, leavers or mistakes.
4. Drift = the MEDIAN movement of all pairs, divided by the time step. The
   median ignores the odd wrong pairing that an average would not.
5. Direction in degrees: 0 = toward the sea (with the wave), 180 = back toward
   the intake, +90 = toward the far side of the corridor (larger y).

It also counts blobs crossing a counting line (used by the exit-vs-entrance
check in Job D).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class DriftTracker:
    def __init__(self, settings):
        self.dt = settings["camera_dt_s"]
        self.max_jump = settings["track_max_speed_mps"] * self.dt
        self.min_matches = settings["drift_min_matches"]
        self.prev = None

    def reset(self):
        """Forget the previous picture (e.g. after the camera was down)."""
        self.prev = None

    def update(self, detections, line_x=None):
        xy = np.array([[d["x"], d["y"]] for d in detections], float).reshape(-1, 2)
        out = {"n_matches": 0, "drift_along_mps": None, "drift_side_mps": None,
               "drift_speed_mps": None, "drift_direction_deg": None,
               "crossings_forward": 0, "crossings_back": 0}
        if self.prev is not None and len(xy) and len(self.prev):
            dist = cdist(self.prev, xy)
            cost = np.where(dist <= self.max_jump, dist, 1e6)   # forbid long jumps
            r, c = linear_sum_assignment(cost)
            ok = cost[r, c] < 1e6
            r, c = r[ok], c[ok]
            out["n_matches"] = int(len(r))
            if len(r) >= self.min_matches:
                v = np.median(xy[c] - self.prev[r], axis=0) / self.dt
                out["drift_along_mps"] = float(v[0])
                out["drift_side_mps"] = float(v[1])
                out["drift_speed_mps"] = float(np.hypot(v[0], v[1]))
                out["drift_direction_deg"] = float(np.degrees(np.arctan2(v[1], v[0])))
            if line_x is not None and len(r):
                before, after = self.prev[r, 0], xy[c, 0]
                weights = np.array([detections[j]["n"] for j in c])  # a clump of 3 counts 3
                out["crossings_forward"] = int(weights[(before < line_x) & (after >= line_x)].sum())
                out["crossings_back"] = int(weights[(before >= line_x) & (after < line_x)].sum())
        self.prev = xy
        return out
