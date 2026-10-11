"""
camera_system.py - the CV part as ONE object for Energy Engineer 2's joining program
=====================================================================================

Every second the joining program calls:

    reading = cam.step(jellyfish, t)

and one camera picture goes through the whole chain:

    true positions --(Job A)--> picture --(Job C 1-3)--> detections
        --(Job C 4)--> crowding --(Job C 5)--> drift --(Job C 6)--> score

What is passed on every second (guide: "the count, how crowded it is, the drift
speed and direction, and the true count (for the scorecard only)"):

    t                    time (s)
    camera               "entrance" or "exit"
    camera_ok            False if the camera is down -> controller falls back to wall mode
    count                jellyfish the camera counted (in SIMULATED jellyfish = blobs)
    crowding_per_m3      REAL jellyfish per m^3 (count x represents / volume) - the bloom's unit
    drift_along_mps      drift speed along the corridor (+ = toward the sea), None if unknown
    drift_side_mps       sideways drift (+ = toward larger y), None if unknown
    drift_speed_mps      overall drift speed, None if unknown
    drift_direction_deg  0 = toward the sea, 180 = toward the intake, None if unknown
    detector             which detector produced this reading
  scorecard only - the controller must NOT use these:
    true_count, true_crowding_per_m3, found, false_alarms, count_error, ...
"""
import numpy as np

from camera import render_frame, view_volume_m3
from detectors import make_detector, BlobDetector
from tracker import DriftTracker
from scorecard import score_frame

CONTROLLER_KEYS = ("t", "camera", "camera_ok", "count", "crowding_per_m3",
                   "drift_along_mps", "drift_side_mps", "drift_speed_mps",
                   "drift_direction_deg")


class CameraSystem:
    def __init__(self, settings, camera="entrance", detector=None):
        self.s = settings
        self.camera = camera
        # Own random stream per camera, from the shared seed (rule 7).
        self.rng = np.random.default_rng(settings["random_seed"] + (0 if camera == "entrance" else 1000))
        self.detector = detector if detector is not None else make_detector(settings)
        self.backup = self.detector if isinstance(self.detector, BlobDetector) else BlobDetector(settings)
        self.tracker = DriftTracker(settings)
        self.line_x = settings["count_line_entrance_m" if camera == "entrance" else "count_line_exit_m"]
        self.volume = view_volume_m3(settings)
        self.failed = False
        self.prev_true_x = {}
        self.records = []
        self.last_image, self.last_detections = None, []

    def set_failed(self, failed=True):
        """Simulate the camera breaking (lets EE2 test the safe wall mode)."""
        self.failed = failed
        if failed:
            self.tracker.reset()

    def _true_crossings(self, truth):
        """Scorecard only: how many TRUE jellyfish crossed the counting line."""
        fwd = back = 0
        for i, x in zip(truth["ids"].tolist(), truth["x"].tolist()):
            px = self.prev_true_x.get(i)
            if px is not None:
                fwd += px < self.line_x <= x
                back += x < self.line_x <= px
        self.prev_true_x = dict(zip(truth["ids"].tolist(), truth["x"].tolist()))
        return int(fwd), int(back)

    def step(self, jellyfish, t):
        s = self.s
        # Job A - take the picture
        image, truth = render_frame(jellyfish, s, self.rng, self.camera)
        true_f, true_b = self._true_crossings(truth)
        reading = {"t": float(t), "camera": self.camera,
                   "true_count": truth["count"],
                   "true_crowding_per_m3": truth["count"] * s["represents"] / self.volume,
                   "true_line_forward": true_f, "true_line_back": true_b}
        if self.failed:
            reading.update(camera_ok=False, count=None, crowding_per_m3=None,
                           drift_along_mps=None, drift_side_mps=None, drift_speed_mps=None,
                           drift_direction_deg=None, detector="none", line_forward=0,
                           line_back=0, found=0, false_alarms=0, count_error=None)
            self.records.append(reading)
            return reading

        # Job C steps 1-3 - find, filter, clumps (blob finder takes over if YOLO fails)
        try:
            dets = self.detector.detect(image, self.camera)
            used = self.detector.name
        except Exception as err:                     # model crashed -> backup
            print(f"[camera] {self.detector.name} failed at t={t}: {err} - using blob backup")
            dets = self.backup.detect(image, self.camera)
            used = "blob (backup)"
        count = int(sum(d["n"] for d in dets))

        # Job C step 4 - crowding, in REAL jellyfish per cubic metre
        crowding = count * s["represents"] / self.volume

        # Job C step 5 - drift (and line crossings for Job D)
        tr = self.tracker.update(dets, line_x=self.line_x)

        # Job C step 6 - score this picture
        sc = score_frame(dets, truth, s)

        reading.update(camera_ok=True, count=count, crowding_per_m3=crowding,
                       drift_along_mps=tr["drift_along_mps"], drift_side_mps=tr["drift_side_mps"],
                       drift_speed_mps=tr["drift_speed_mps"],
                       drift_direction_deg=tr["drift_direction_deg"],
                       drift_n_matches=tr["n_matches"], detector=used,
                       line_forward=tr["crossings_forward"], line_back=tr["crossings_back"], **sc)
        self.records.append(reading)
        self.last_image, self.last_detections = image, dets
        return reading

    @staticmethod
    def for_controller(reading):
        """Only the fields the controller may use (no truth)."""
        return {k: reading.get(k) for k in CONTROLLER_KEYS}
