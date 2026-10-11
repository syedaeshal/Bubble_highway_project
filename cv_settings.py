"""
cv_settings.py - the CV person's entries for the team's SHARED SETTINGS LIST
============================================================================

Guide rule 2: "Every number the simulation uses lives in one list. Nobody hides
numbers inside their own part."  So EVERY number the camera, detector, tracker and
scorecard use is defined here, with a note on where it came from.

How to use this in the team repository
--------------------------------------
Energy Engineer 2 owns the shared settings file. Ask them to paste the CV block
below into it (golden rule: never edit someone else's file yourself).

The first block ("SHARED") holds numbers the whole team agreed on. They are repeated
here ONLY so the CV part can be tested alone with made-up inputs. Once the real
shared list exists, combine them like this (the team's values win):

    settings = {**CV_SETTINGS, **team_settings}

Units (guide rule 3): metres, seconds, metres per second. Picture brightness is in
grey levels 0-255, the normal unit of an 8-bit camera.
"""
import copy

CV_SETTINGS = {
    # ------------------------------------------------------------------
    # SHARED - agreed by the whole team (copied here for solo testing)
    # ------------------------------------------------------------------
    "random_seed": 42,             # rule 7: same random starting point for everyone
    "corridor_length_m": 100.0,    # rule 4: entrance at 0 m, exit (open sea) at ~100 m
    "corridor_width_m": 20.0,      # confirmed: 20 m wide (the bloom example's opening)
    "depth_band_m": 3.0,           # depth band where jellyfish swim (bloom example: 3 m)
    "camera_dt_s": 1.0,            # rule 6: the camera takes one picture per second
    "represents": 10,              # EE1 Job A step 5: 1 simulated jellyfish = 10 real ones

    # ------------------------------------------------------------------
    # JOB A - the pretend camera picture
    # ------------------------------------------------------------------
    "cam_view_length_m": 20.0,     # guide: camera looks down on the first 20 m
    "cam_m_per_px": 0.05,          # guide: 1 pixel ~ 5 cm, so a 40 cm jellyfish ~ 8 px
    "cam_water_bgr": (120, 75, 25),  # blue water; OpenCV stores colours as Blue, Green, Red
    "cam_jelly_lift": 90.0,        # how much lighter (grey levels) a jellyfish at the surface looks
    "cam_visibility_depth_m": 2.0, # ASSUMPTION: brightness falls to 37 % at 2 m depth
    "cam_edge_softness_px": 1.0,   # blob edge fades over ~1 px instead of a hard cut
    # real-world mess, each with an adjustable strength (guide Job A step 4)
    "cam_mess": 1.0,               # master knob: 0 = perfect, 1 = normal day, 2+ = bad day
    "cam_noise_sigma": 8.0,        # grainy sensor noise (grey levels) at mess = 1
    "cam_blur_sigma_px": 0.8,      # blur (pixels) at mess = 1
    "cam_glints_per_frame": 4.0,   # average number of sun glints per picture at mess = 1
    "cam_glint_radius_px": (1.0, 3.0),
    "cam_glint_lift": (160.0, 240.0),  # glints are far brighter than any jellyfish
    "cam_turbidity": 0.0,          # murkiness 0 (clear) .. 1 (very murky)

    # ------------------------------------------------------------------
    # JOB C steps 1-3 - finding jellyfish, throwing away fakes, clumps
    # ------------------------------------------------------------------
    "detector": "auto",            # "yolo", "blob", or "auto" (YOLO if trained weights exist)
    "yolo_weights": "runs/jelly/weights/best.pt",
    "yolo_conf": 0.25,             # minimum confidence for a YOLO box
    "yolo_imgsz": 640,             # picture is 400 px; YOLO enlarges it so 8 px jellyfish become ~13 px
    "det_typical_size_m": 0.375,   # middle of the 30-45 cm size range (EE1 Job A step 4)
    "det_min_size_m": 0.10,        # smaller than 10 cm -> noise speck, not a jellyfish
    "det_min_contrast": 10.0,      # dimmer than this (grey levels above water) -> too faint to trust
    "det_glint_peak": 130.0,       # brighter than any jellyfish can look -> sun glint
    "det_clump_factor": 1.6,       # bigger than 1.6 single jellyfish -> treat as a clump
    "det_max_clump": 60,           # bigger than 60 jellyfish AND glint-bright all over -> glare patch, ignore
    "det_threshold_k": 3.5,        # blob finder: pixel must be 3.5 noise levels above the water
    "det_background_px": 41,       # window (pixels, ~2 m) used to estimate the local water brightness

    # ------------------------------------------------------------------
    # JOB B - training pictures (auto-labelled pretend camera pictures)
    # ------------------------------------------------------------------
    "label_min_contrast": 10.0,    # label only jellyfish a person could actually see
    "train_density_range": (0.02, 1.0),  # crowding (real /m^3); capped at 1 so a picture holds
                                         # <= ~1200 jellyfish and training fits in memory
    "train_mess_range": (0.0, 2.0),      # from perfect to twice-normal noise, blur and glints
    "train_turbidity_range": (0.0, 0.6),
    "train_small_species_share": 0.2,    # 20 % of pictures use 15-30 cm jellyfish (test 4)
    "train_batch": 8,                    # pictures per training step (fits a free Colab T4)

    # ------------------------------------------------------------------
    # JOB C steps 4-5 - crowding and drift
    # ------------------------------------------------------------------
    "track_max_speed_mps": 1.0,    # nothing drifts faster than 1 m/s -> max match distance per picture
    "drift_min_matches": 3,        # fewer matched pairs than this -> drift reported as unknown (None)
    "count_line_entrance_m": 10.0, # counting line in the middle of the entrance camera's view
    "count_line_exit_m": 90.0,     # counting line in the middle of the exit camera's view

    # ------------------------------------------------------------------
    # JOB D - scorecard
    # ------------------------------------------------------------------
    "score_match_px": 6.0,         # a detection within 6 px (30 cm) of a true jellyfish = "found"
    "score_peak_fraction": 0.10,   # "peak of the bloom" = the 10 % of pictures with most jellyfish
    "score_ok_recall": 0.90,       # guide: find more than 9 out of 10 jellyfish
    "score_ok_count_error": 0.10,  # ASSUMPTION: count within 10 % of the truth = "still working"
}


def get_settings(**overrides):
    """Return a fresh copy of the settings, with any values you want to change."""
    s = copy.deepcopy(CV_SETTINGS)
    s.update(overrides)
    return s
