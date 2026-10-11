"""
key_test.py - ENERGY ENGINEER 1: THE KEY TEST ("do jellyfish ride the wave?")
==============================================================================
Guide: run 100 jellyfish with no camera and no controller. Try response times
from 1 to 10 s and wave speeds from very slow to fast, in both water versions.
Make a colour map of how many made it out for each combination. If the
realistic version shows NO combination that works, tell the team the same day:
the design needs to change, not the settings.

Set-up: 100 jellyfish start 2-6 m inside the entrance (random across the width,
depths 'mostly near the surface'); one wave front leaves the entrance at the
chosen speed with nominal air; the background current stays at its usual value.

Also the three simple checks:
  A. no bubbles -> jellyfish just drift with the current;
  B. optimistic -> a slow wave carries them, a fast wave leaves them behind;
  C. realistic with all pipes on -> jellyfish in the middle barely move.

    python key_test.py            # saves results/key_test.json and the colour map
"""
import json
import os

import numpy as np

import shared_settings
from camera import STATUS_IN
from water import Water

RESPONSE_TIMES = [1, 2, 3, 5, 7, 10]
WAVE_SPEEDS = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45]


def place(s, rng, n=100, x_range=(2.0, 6.0)):
    d = rng.exponential(s["depth_mean_m"], n)
    while (bad := d > s["depth_band_m"]).any():
        d[bad] = rng.exponential(s["depth_mean_m"], bad.sum())
    return {"id": np.arange(n), "x": rng.uniform(*x_range, n),
            "y": rng.uniform(2, s["corridor_width_m"] - 2, n),
            "speed": np.full(n, s["current_mps"]), "size": rng.uniform(*s["size_range_m"], n),
            "depth": d, "status": np.full(n, STATUS_IN, dtype="<U16")}


def one_wave(s, wave_speed, seed=0, n=100):
    """Share of n jellyfish that one wave front carries out to sea."""
    rng = np.random.default_rng(seed)
    w = Water(s, rng)
    w.add(place(s, rng, n), 0.0)
    dt, air = s["dt_physics_s"], s["air_nominal_lpm_per_m"]
    t_max = s["corridor_length_m"] / wave_speed + 600
    t = 0.0
    while t < t_max and len(w.J["x"]):
        i = int(np.floor(wave_speed * t / s["pipe_spacing_m"]))
        firing = np.array([i]) if i < len(w.pipes) else np.empty(0, int)
        w.step(firing, air, t, dt)
        t += dt
    st = np.array(w.finished["status"] + ["in_corridor"] * len(w.J["x"]))
    return float(np.mean(st == "out_to_sea"))


def colour_map(s, versions=("optimistic", "realistic"), seed=0):
    out = {}
    for v in versions:
        grid = np.zeros((len(RESPONSE_TIMES), len(WAVE_SPEEDS)))
        for i, tau in enumerate(RESPONSE_TIMES):
            for j, w in enumerate(WAVE_SPEEDS):
                grid[i, j] = one_wave(dict(s, water_version=v, response_time_s=tau), w, seed)
            print(f"  {v:<10} tau={tau:>2}s: " + " ".join(f"{100*g:4.0f}" for g in grid[i]))
        out[v] = grid.tolist()
    return {"response_times": RESPONSE_TIMES, "wave_speeds": WAVE_SPEEDS, "share_out": out}


def sanity_checks(s, seed=0):
    rng = np.random.default_rng(seed)
    dt, air, res = s["dt_physics_s"], s["air_nominal_lpm_per_m"], {}
    # A. no bubbles: average speed should equal the current
    w = Water(dict(s, water_version="realistic"), rng)
    w.add(place(s, rng, 100, (40, 60)), 0.0)
    x0 = w.J["x"].copy()
    for k in range(int(60 / dt)):
        w.step(np.empty(0, int), air, k * dt, dt)
    res["A_no_bubbles_mean_speed"] = float(np.mean(w.J["x"] - x0[: len(w.J["x"])]) / 60)
    res["A_current"] = s["current_mps"]
    # B. optimistic: slow vs fast wave
    so = dict(s, water_version="optimistic")
    res["B_optimistic_slow_wave_out"] = one_wave(so, 0.05, seed)
    res["B_optimistic_fast_wave_out"] = one_wave(so, 0.6, seed)
    # C. realistic, all pipes on: jellyfish mid-corridor barely move (surface ones)
    sr = dict(s, water_version="realistic", depth_mean_m=0.3)
    w = Water(sr, np.random.default_rng(seed))
    w.add(place(sr, np.random.default_rng(seed), 100, (45, 55)), 0.0)
    x0 = w.J["x"].copy()
    allp = np.arange(len(w.pipes))
    for k in range(int(60 / dt)):
        w.step(allp, air, k * dt, dt)
    res["C_realistic_all_on_mean_move_m_in_60s"] = float(np.mean(np.abs(w.J["x"] - x0[: len(w.J["x"])])))
    return res


if __name__ == "__main__":
    s = shared_settings.get()
    os.makedirs("results", exist_ok=True)
    print("Sanity checks:", sanity_checks(s))
    print("Colour map (% out to sea; columns = wave speeds", WAVE_SPEEDS, "m/s):")
    km = colour_map(s)
    km["sanity"] = sanity_checks(s)
    json.dump(km, open("results/key_test.json", "w"), indent=1)
    import charts
    charts.key_test_map(km, "results/key_test_colour_map.png")
    best = max(max(r) for r in km["share_out"]["realistic"])
    if best < 0.5:
        print(f"\n!!! Realistic version: best combination moves only {100*best:.0f}% out. "
              "Tell the team today: the DESIGN needs to change, not the settings.")
