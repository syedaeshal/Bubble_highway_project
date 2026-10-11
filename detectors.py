"""
detectors.py - JOB C steps 1-3: find the jellyfish, throw away fakes, handle clumps
=====================================================================================

Two detectors with the SAME output, so either can be used anywhere:

* YoloDetector - the trained AI model (main detector, trained in train_detector.py)
* BlobDetector - a simple OpenCV blob finder (backup "in case the model struggles",
                 and the comparison the guide asks for in the error-vs-crowding chart)

Each returns a list of detections; one detection is a dict:
    x, y      : position in metres (along, sideways)
    col, row  : position in pixels
    area_px   : size of the blob or box in pixels
    contrast  : how much lighter than the surrounding water (grey levels)
    n         : estimated number of jellyfish in it (1, or more for a clump)
    source    : "yolo" or "blob"

Both detectors only look at the picture. Neither ever sees the true positions.
"""
import os
import numpy as np
import cv2
from scipy.ndimage import percentile_filter

from camera import px_to_world


# ---------------------------------------------------------------------------
# Shared picture analysis: how much lighter than the water is each pixel?
# ---------------------------------------------------------------------------
def water_residual(image, settings):
    """Return (residual, smoothed residual, noise level).

    1. Lightness = average of the three colour channels.
    2. Local water brightness: jellyfish are only ever LIGHTER than water, so the
       water level is the DARK end of each neighbourhood. We take the 10th
       percentile in a ~2 m window (on a 4x smaller copy for speed). Unlike an
       average, this still works when jellyfish cover most of the picture.
    3. Residual = lightness - water brightness (0 for water, ~90 for a surface jellyfish).
    4. Noise level: water pixels scatter around 0; jellyfish only push pixels
       up, so pixels BELOW zero are pure noise. For a bell curve the root mean
       square of the negative half equals the noise level.
    """
    gray = image.astype(np.float32).mean(axis=2)
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    win = max(3, int(settings["det_background_px"]) // 4)
    water_small = percentile_filter(small, 10, size=win, mode="nearest")
    water = cv2.resize(water_small, (gray.shape[1], gray.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    resid = gray - water
    smooth = cv2.GaussianBlur(resid, (0, 0), 1.0)   # averages out the grain
    neg = smooth[smooth < 0]
    noise = float(np.sqrt(np.mean(neg ** 2))) if neg.size > 50 else 1.0
    return resid, smooth, max(noise, 0.5)


def _single_size(sizes, settings, default):
    """Job C step 3 - what does ONE jellyfish look like in this picture?

    If the picture has at least 5 blobs of normal size, use their median
    (this adapts automatically to smaller species, test 4). Otherwise fall
    back to the typical size from the settings."""
    clump_limit = settings["det_clump_factor"] * default
    singles = [a for a in sizes if a <= clump_limit]
    return float(np.median(singles)) if len(singles) >= 5 else default


def _clump_count(area, single_area, settings):
    """Job C step 3 - estimated number of jellyfish in one blob.
    A blob more than det_clump_factor times a single jellyfish is a clump;
    its count = its size divided by the size of one jellyfish."""
    if area <= settings["det_clump_factor"] * single_area:
        return 1
    return max(2, int(round(area / single_area)))


# ---------------------------------------------------------------------------
# Backup detector: simple OpenCV blob finder
# ---------------------------------------------------------------------------
class BlobDetector:
    name = "blob"

    def __init__(self, settings):
        self.s = settings
        m = settings["cam_m_per_px"]
        r_px = settings["det_typical_size_m"] / 2.0 / m
        self.typical_area = np.pi * r_px ** 2            # ~44 px for a 37.5 cm jellyfish
        self.min_area = np.pi * (settings["det_min_size_m"] / 2.0 / m) ** 2

    def detect(self, image, camera="entrance"):
        s = self.s
        resid, smooth, noise = water_residual(image, s)
        # Step 1 - find: pixels clearly lighter than the water, joined into blobs
        thr = max(s["det_threshold_k"] * noise, 0.5 * s["det_min_contrast"])
        mask = (smooth > thr).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            sub = labels[y:y + h, x:x + w] == i
            values = smooth[y:y + h, x:x + w][sub]
            contrast = float(np.median(values))
            peak = float(resid[y:y + h, x:x + w][sub].max())
            # Size measured at HALF the blob's own brightness: blur and the
            # detection threshold make faint and bright blobs look different
            # sizes at the threshold, but at half brightness a blob is about
            # as wide as the jellyfish's real bell, whatever its contrast.
            area = float(np.sum(values >= 0.5 * contrast))
            # Step 2 - throw away fakes
            if area < self.min_area:                     # too small: noise speck
                continue
            if contrast < s["det_min_contrast"]:         # too faint to trust
                continue
            if peak > s["det_glint_peak"] and area < 3 * self.typical_area:
                continue                                 # sun glint: brighter than any jellyfish
            if contrast > s["det_glint_peak"] and area > s["det_max_clump"] * self.typical_area:
                continue                                 # huge AND bright all over: glare patch
            blobs.append((cents[i][0], cents[i][1], float(area), contrast))
        # Step 3 - handle clumps
        single = _single_size([b[2] for b in blobs], s, self.typical_area)
        dets = []
        for col, row, area, contrast in blobs:
            xm, ym = px_to_world(col, row, s, camera)
            dets.append({"x": float(xm), "y": float(ym), "col": float(col), "row": float(row),
                         "area_px": area, "contrast": contrast,
                         "n": _clump_count(area, single, s), "source": self.name})
        return dets


# ---------------------------------------------------------------------------
# Main detector: the trained YOLO model
# ---------------------------------------------------------------------------
class YoloDetector:
    name = "yolo"

    def __init__(self, settings, weights=None):
        from ultralytics import YOLO                       # imported only when needed
        self.s = settings
        self.model = YOLO(weights or settings["yolo_weights"])
        m = settings["cam_m_per_px"]
        d_px = settings["det_typical_size_m"] / m
        self.typical_box = d_px ** 2                       # box area of one jellyfish
        self.min_side = settings["det_min_size_m"] / m

    def detect(self, image, camera="entrance"):
        s = self.s
        # Step 1 - find: the model draws a box around every jellyfish it sees
        result = self.model.predict(image, imgsz=s["yolo_imgsz"], conf=s["yolo_conf"],
                                    max_det=3000, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
        resid, smooth, _ = water_residual(image, s)
        rows, cols = smooth.shape
        kept = []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            col, row = (x1 + x2) / 2 - 0.5, (y1 + y2) / 2 - 0.5   # box edges -> pixel centre
            r0, r1 = int(max(row - 1, 0)), int(min(row + 2, rows))
            c0, c1 = int(max(col - 1, 0)), int(min(col + 2, cols))
            contrast = float(smooth[r0:r1, c0:c1].mean()) if r1 > r0 and c1 > c0 else 0.0
            peak = float(resid[int(max(y1, 0)):int(min(y2, rows)) + 1,
                               int(max(x1, 0)):int(min(x2, cols)) + 1].max())
            # Step 2 - throw away fakes (same rules as the blob finder)
            if max(w, h) < self.min_side:
                continue
            if contrast < s["det_min_contrast"]:
                continue
            if peak > s["det_glint_peak"] and w * h < 3 * self.typical_box:
                continue
            kept.append((col, row, float(w * h), contrast))
        # Step 3 - handle clumps: a box much bigger than one jellyfish
        single = _single_size([k[2] for k in kept], s, self.typical_box)
        dets = []
        for col, row, area, contrast in kept:
            xm, ym = px_to_world(col, row, s, camera)
            dets.append({"x": float(xm), "y": float(ym), "col": float(col), "row": float(row),
                         "area_px": area, "contrast": contrast,
                         "n": _clump_count(area, single, s), "source": self.name})
        return dets


def make_detector(settings, verbose=True):
    """Pick the detector named in the settings.
    'auto' uses YOLO if trained weights exist and ultralytics is installed,
    otherwise the blob finder (and says so)."""
    choice = settings["detector"]
    if choice == "blob":
        return BlobDetector(settings)
    if choice == "yolo":
        return YoloDetector(settings)
    if os.path.exists(settings["yolo_weights"]):
        try:
            return YoloDetector(settings)
        except ImportError:
            pass
    if verbose:
        print(f"[detectors] No trained YOLO weights at {settings['yolo_weights']!r} "
              "- using the OpenCV blob finder.")
    return BlobDetector(settings)
