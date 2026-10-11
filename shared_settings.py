"""
shared_settings.py - THE one settings list for the whole team (guide rule 2)
============================================================================
Owner: Energy Engineer 2. Every number the simulation uses lives here, with
where it came from. Tags used below:
  [GUIDE]   from the team implementation guide
  [ENG3]    from the eng3 document (preliminary design)
  [TEAM]    decided by the team in discussion
  [LIT]     from published literature (reference given)
  [ASSUME]  our assumption - must be stated openly in the slides

Units (rule 3): metres, seconds, m/s. Air flow in litres per minute per metre
of pipe (L/min/m) of free air, because that is how the references report it.
Energy is converted to kWh only for the final answer.
"""
import copy
from cv_settings import CV_SETTINGS

SETTINGS = dict(CV_SETTINGS)          # the CV person's block (camera, detector, scorecard)
SETTINGS.update({
    # ---------------- run control ----------------
    "random_seed": 42,                 # [GUIDE] rule 7
    "dt_physics_s": 0.05,              # [GUIDE] rule 6: water + jellyfish 20 times a second
    "camera_dt_s": 1.0,                # [GUIDE] rule 6: one picture per second
    "controller_dt_s": 5.0,            # [GUIDE] rule 6: one decision every 5 seconds
    "snapshot_dt_s": 2.0,              # [GUIDE] EE2 Job A step 3: snapshot every 2 s
    "extra_run_time_s": 3600.0,        # [ASSUME] up to 1 h after the bloom; the run stops early once the corridor is empty
    "mode": "wave",                    # [GUIDE] "off", "curtain" or "wave"
    "water_version": "realistic",      # [GUIDE] "optimistic" (forward only) or "realistic" (both ways)
    "camera_source": "picture",        # "picture" = CV part; "truth" = stand-in camera (guide EE2 Job A step 1)
    "use_exit_camera": True,           # [GUIDE] CV Job D step 2

    # ---------------- corridor map (rule 4) ----------------
    "corridor_length_m": 100.0,        # [GUIDE]
    "corridor_width_m": 20.0,          # [TEAM]
    "depth_band_m": 3.0,               # [GUIDE] bloom example: jellyfish swim in a 3 m band

    # ---------------- bloom (EE1 Job A) ----------------
    "bloom_duration_s": 7200.0,        # [GUIDE] try 2 hours
    "bloom_peak_density": 0.5,         # [GUIDE] 0.1-2 jellyfish per m^3 at the peak
    "arrival_speed_mps": 0.2,          # [GUIDE] bloom example: delivered to the entrance at 0.2 m/s
    "arrival_zone_m": 10.0,            # [ASSUME] the swarm appears across the first 10 m of the entrance
                                       #   (inside the camera's view); the guide does not say how it enters
    "represents": 10,                  # [GUIDE] 1 simulated jellyfish = 10 real ones
    "size_range_m": (0.30, 0.45),      # [GUIDE] 30-45 cm bells
    "depth_mean_m": 1.0,               # [ASSUME] "mostly near the surface": exponential, mean 1 m

    # ---------------- pipes and air ----------------
    "pipe_spacing_m": 2.0,             # [GUIDE] every 1-3 m -> 51 pipes over 100 m
    "pipe_depth_m": 5.0,               # [ENG3] diffuser depth 5 m (published trials)
    "air_nominal_lpm_per_m": 374.0,    # [TEAM] high-flow design: gives the guide's ~9 kW per pipe
    "air_min_lpm_per_m": 285.0,        # [TEAM] tested 8-15 L/min/m ratio (Circiumaru 2024) scaled up
    "air_max_lpm_per_m": 534.0,        # [TEAM] same scaling
    "plume_constant": 1.46,            # [LIT] Bulson (1961): Vmax = 1.46 (g q)^(1/3); 1.2 = sensitivity test
    "outflow_depth_fraction": 0.25,    # [LIT] surface current is H/4 deep, fastest at the surface (Bulson 1961)
    "push_peak_distance_fraction": 0.5,  # [LIT] fastest current about d/2 from the pipe (Ince guide)
    "push_decay_shape": "rise_then_exp",  # [ASSUME] shape of the push between those points

    # ---------------- background current (EE1 Job B step 6) ----------------
    "current_mps": -0.05,              # [ASSUME] weak drift TOWARD the intake (negative = toward intake)
    "current_reverse_at_s": None,      # tide test: time the current turns
    "current_after_reverse_mps": None, # tide test: speed after it turns
    "current_ramp_s": 600.0,           # [ASSUME] a tide turns over ~10 minutes
    "current_meter_noise_mps": 0.01,   # [ASSUME] separate current meter [TEAM decision], small error

    # ---------------- jellyfish response (EE1 Job B steps 7-8) ----------------
    "response_time_s": 3.0,            # [GUIDE] try 1-5 s; THE most important number; set by the key test
    "response_size_exponent": 1.0,     # [ASSUME] bigger jellyfish respond more slowly: tau ~ size^1
    "swim_noise_mps": 0.02,            # [ASSUME] small random swimming forward and back
    "side_drift_mps": 0.01,            # [ASSUME] slow sideways drift
    "swim_timescale_s": 10.0,          # [ASSUME] how long a random swimming direction lasts

    # ---------------- gentleness (EE1 Job C step 3) ----------------
    "roughness_limit_per_s": 1.0,      # [ASSUME] water-speed change across the body / body size; no published limit
    "above_pipe_halfwidth_m": 0.5,     # [ASSUME] "right above a firing pipe" = within 0.5 m

    # ---------------- controller (EE2 Job B) ----------------
    "on_level_per_m3": 0.05,           # [ASSUME] switch on above this crowding
    "off_level_per_m3": 0.02,          # [ASSUME] switch off below this ...
    "off_hold_s": 60.0,                # [GUIDE] ... for a minute
    "min_on_s": 120.0,                 # [GUIDE] stay on at least two minutes
    "air_gain": 250.0,                 # [ASSUME] extra L/min/m per jellyfish/m^3 (tuned in experiments)
    "wave_ratio": 1 / 3,               # [GUIDE] wave at about a third of the water speed
    "wave_ratio_max": 0.9,             # [ASSUME] the wave must stay slower than the water
    "wave_speed_min_mps": 0.02,        # [ASSUME]
    "wave_gap_m": 10.0,                # [ASSUME] launch a new wave when the last one is 10 m along
    "lag_slowdown": 0.8,               # [ASSUME] jellyfish falling behind -> wave x 0.8
    "keepup_speedup": 1.05,            # [ASSUME] keeping up -> wave x 1.05 (gently)
    "wall_current_mps": -0.15,         # [ASSUME] current this strongly toward the intake -> wall mode
    "wall_pipes": 3,                   # [ASSUME] the 3 pipes nearest the entrance form the wall

    # ---------------- energy (EE3 Job A) ----------------
    "water_density": 1025.0,           # [ENG3] seawater
    "gravity": 9.81,
    "p_atm_pa": 101325.0,
    "hose_loss_pa": 0.0,               # [TEAM] water pressure only
    "compressor_efficiency": 0.6,      # [GUIDE] about 60 %
    "gamma_air": 1.4,                  # adiabatic compression of air
    "electricity_price_aed_per_kwh": None,  # optional extra: look up the current UAE business tariff
})


def get(**overrides):
    """A fresh copy of the settings with any values changed - change one thing at a time."""
    s = copy.deepcopy(SETTINGS)
    s.update(overrides)
    return s
