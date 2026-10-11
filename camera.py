"""
camera.py - JOB A: make the pretend camera picture
===================================================

What it does
------------
Once per second it turns the true jellyfish positions (from Energy Engineer 1)
into a top-down camera picture, exactly what a real camera mounted above the
corridor entrance would see. Everything after this file (detector, tracker,
counter) only ever looks at the PICTURE, never at the true positions, just like
a real camera system. The true positions are returned separately ("truth")
and are used ONLY by the scorecard.

Map of the picture (guide rule 4)
---------------------------------
* Columns run ALONG the corridor: column 0 is the start of the camera's view
  (x = 0 m for the entrance camera), and the wave travels left -> right.
* Rows run ACROSS the corridor: row 0 is sideways position y = 0 m.
* 20 m x 20 m at 5 cm per pixel -> a 400 x 400 pixel picture.

Pixel-coordinate convention used everywhere in this project: pixel number c
covers x from c*0.05 m to (c+1)*0.05 m, so its CENTRE is at (c + 0.5)*0.05 m.

What each jellyfish carries (guide rule 5)
------------------------------------------
id, x (along, m), y (sideways, m), speed (m/s), size (bell diameter, m),
depth (m below the surface), status. The camera only needs x, y, size, depth
and status. Input can be a list of dicts or a dict of arrays (see as_arrays).
"""
import numpy as np
import cv2

STATUS_IN = "in_corridor"   # agreed status meaning "still in the corridor"
FIELDS = ("id", "x", "y", "speed", "size", "depth", "status")


# ---------------------------------------------------------------------------
# Small helpers: input format, picture size, coordinate conversion
# ---------------------------------------------------------------------------
def as_arrays(jellyfish):
    """Accept Energy Engineer 1's jellyfish in either common format and return
    a dict of numpy arrays, one array per field.

    * list of dicts:  [{"id": 0, "x": 1.2, "y": 4.0, ...}, ...]
    * dict of arrays: {"id": [...], "x": [...], ...}
    """
    if isinstance(jellyfish, dict):
        out = {k: np.asarray(v) for k, v in jellyfish.items()}
    else:
        jellyfish = list(jellyfish)
        out = {k: np.asarray([j[k] for j in jellyfish]) for k in FIELDS
               if jellyfish and k in jellyfish[0]}
    n = len(out["x"]) if "x" in out else 0
    defaults = {"id": np.arange(n), "x": np.zeros(n), "y": np.zeros(n),
                "speed": np.zeros(n), "size": np.full(n, 0.375),
                "depth": np.zeros(n), "status": np.full(n, STATUS_IN)}
    for k, v in defaults.items():
        out.setdefault(k, v)
    for k in ("x", "y", "speed", "size", "depth"):
        out[k] = out[k].astype(float)
    return out


def view_x_range(settings, camera="entrance"):
    """Which stretch of corridor (metres along) a camera sees.
    Entrance camera: 0-20 m. Exit camera (Job D step 2): the last 20 m."""
    length = settings["cam_view_length_m"]
    x0 = 0.0 if camera == "entrance" else settings["corridor_length_m"] - length
    return x0, x0 + length


def image_size(settings):
    """(rows, columns) of the picture. Rows = across the width, columns = along."""
    m = settings["cam_m_per_px"]
    rows = int(round(settings["corridor_width_m"] / m))
    cols = int(round(settings["cam_view_length_m"] / m))
    return rows, cols


def view_volume_m3(settings):
    """Volume of water in view: view length x corridor width x depth band.
    Used to turn a count into crowding (jellyfish per cubic metre)."""
    return (settings["cam_view_length_m"] * settings["corridor_width_m"]
            * settings["depth_band_m"])


def world_to_px(x, y, settings, camera="entrance"):
    """Metres -> pixel coordinates (column, row), pixel centres at whole numbers."""
    x0, _ = view_x_range(settings, camera)
    m = settings["cam_m_per_px"]
    return (np.asarray(x) - x0) / m - 0.5, np.asarray(y) / m - 0.5


def px_to_world(col, row, settings, camera="entrance"):
    """Pixel coordinates (column, row) -> metres (x along, y sideways)."""
    x0, _ = view_x_range(settings, camera)
    m = settings["cam_m_per_px"]
    return x0 + (np.asarray(col) + 0.5) * m, (np.asarray(row) + 0.5) * m


def visible_contrast(depth, settings):
    """Job A step 3 - deeper jellyfish look fainter.

    Light fades roughly exponentially with depth in water, so
        contrast = surface_lift * exp(-depth / visibility_depth).
    Murky water (turbidity) lowers contrast AND makes light fade faster."""
    turb = settings["cam_turbidity"]
    vis = settings["cam_visibility_depth_m"] * (1.0 - 0.6 * turb)
    surface = settings["cam_jelly_lift"] * (1.0 - 0.7 * turb)
    return surface * np.exp(-np.maximum(np.asarray(depth, float), 0.0) / vis)


def _stamp_disc(canvas, cx, cy, radius, value, softness):
    """Draw one soft-edged round blob into `canvas`.

    Inside the disc the value is `value`; it fades to 0 over about `softness`
    pixels at the edge. Where two jellyfish overlap we keep the brighter one
    (np.maximum): touching jellyfish merge into one blob of the same brightness,
    which is exactly what makes clumps hard to count (Job C step 3)."""
    h, w = canvas.shape
    pad = radius + 2 * softness + 1
    c0, c1 = max(int(np.floor(cx - pad)), 0), min(int(np.ceil(cx + pad)) + 1, w)
    r0, r1 = max(int(np.floor(cy - pad)), 0), min(int(np.ceil(cy + pad)) + 1, h)
    if c0 >= c1 or r0 >= r1:
        return
    yy, xx = np.mgrid[r0:r1, c0:c1]
    dist = np.hypot(xx - cx, yy - cy)
    alpha = np.clip((radius - dist) / softness + 0.5, 0.0, 1.0)
    np.maximum(canvas[r0:r1, c0:c1], alpha * value, out=canvas[r0:r1, c0:c1])


# ---------------------------------------------------------------------------
# The camera itself
# ---------------------------------------------------------------------------
def render_frame(jellyfish, settings, rng, camera="entrance"):
    """Draw one camera picture.

    Parameters
    ----------
    jellyfish : list of dicts or dict of arrays (see as_arrays)
    settings  : the shared settings dict
    rng       : numpy random Generator (seeded -> repeatable pictures, rule 7)
    camera    : "entrance" or "exit"

    Returns
    -------
    image : uint8 array (rows, cols, 3), colours in OpenCV's Blue-Green-Red order
    truth : dict with the TRUE jellyfish in view - for the scorecard ONLY
    """
    J = as_arrays(jellyfish)
    m = settings["cam_m_per_px"]
    x0, x1 = view_x_range(settings, camera)
    width = settings["corridor_width_m"]
    rows, cols = image_size(settings)
    alive = (J["status"] == STATUS_IN) & (J["depth"] <= settings["depth_band_m"])
    radius_m = J["size"] / 2.0

    # Step 1 - decide what the camera sees.
    # A jellyfish is DRAWN if any part of its bell is inside the view (a real
    # camera sees half a jellyfish at the picture's edge), but it only COUNTS
    # as "in view" for the scorecard if its centre is inside.
    drawn = alive & (J["x"] + radius_m >= x0) & (J["x"] - radius_m < x1) \
        & (J["y"] + radius_m >= 0) & (J["y"] - radius_m <= width)
    in_view = alive & (J["x"] >= x0) & (J["x"] < x1) & (J["y"] >= 0) & (J["y"] <= width)

    # Step 2 - blue water with each jellyfish as a lighter round blob, sized to
    # match its real size; Step 3 - deeper ones are fainter.
    lift = np.zeros((rows, cols), np.float32)
    col_all, row_all = world_to_px(J["x"], J["y"], settings, camera)
    contrast_all = visible_contrast(J["depth"], settings)
    for i in np.flatnonzero(drawn):
        _stamp_disc(lift, col_all[i], row_all[i], radius_m[i] / m,
                    contrast_all[i], settings["cam_edge_softness_px"])
    image = np.empty((rows, cols, 3), np.float32)
    image[:] = settings["cam_water_bgr"]
    image += lift[..., None]          # lighter in all three colour channels

    # Step 4 - real-world mess, each part scaled by the master "mess" knob.
    mess = settings["cam_mess"]
    blur = settings["cam_blur_sigma_px"] * mess
    if blur > 0.05:                   # (a) blur: water movement, focus, motion
        image = cv2.GaussianBlur(image, (0, 0), blur)
    n_glints = rng.poisson(settings["cam_glints_per_frame"] * mess) if mess > 0 else 0
    if n_glints:                      # (b) bright sunlight spots on the surface
        glints = np.zeros((rows, cols), np.float32)
        for _ in range(n_glints):
            _stamp_disc(glints, rng.uniform(0, cols), rng.uniform(0, rows),
                        rng.uniform(*settings["cam_glint_radius_px"]),
                        rng.uniform(*settings["cam_glint_lift"]), 0.7)
        image += glints[..., None]
    noise = settings["cam_noise_sigma"] * mess
    if noise > 0:                     # (c) grainy sensor noise
        image += rng.normal(0.0, noise, image.shape).astype(np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Truth for the scorecard (never given to the detector).
    idx = np.flatnonzero(in_view)
    truth = {
        "ids": J["id"][idx], "x": J["x"][idx], "y": J["y"][idx],
        "col": col_all[idx], "row": row_all[idx],
        "radius_px": radius_m[idx] / m, "contrast": contrast_all[idx],
        "visible": contrast_all[idx] >= settings["label_min_contrast"],
        "count": int(len(idx)),
    }
    # Every drawn jellyfish (including half-visible edge ones) - used for
    # training labels so the model is never punished for seeing a real blob.
    didx = np.flatnonzero(drawn)
    truth["drawn"] = {"col": col_all[didx], "row": row_all[didx],
                      "radius_px": radius_m[didx] / m,
                      "contrast": contrast_all[didx]}
    return image, truth
