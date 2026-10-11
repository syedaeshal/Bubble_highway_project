"""
experiments.py - ENERGY ENGINEER 2, JOB C: run all the experiments
===================================================================
Rules for fair results (guide):
 * change ONE setting at a time, everything else stays the same;
 * every test compares Bubble Highway, an ordinary curtain and no system;
 * run each test with three random seeds and show the spread;
 * if Bubble Highway loses anywhere, show it (the charts do not hide it).
Every result is saved together with the settings that produced it.

Tuning order: 1 response time (key_test.py), 2 wave/water speed ratio,
3 controller gain, 4 camera noise (CV noise sweep; in-loop with --full).
Six adaptability tests: current, tide turning, bloom size, other species,
size of the system, optimistic vs realistic. Plus: plume constant 1.2 (sensitivity).

    python experiments.py                # quick: 20-min blooms, stand-in camera, 3 seeds
    python experiments.py --seeds 1      # faster check
    python experiments.py --full         # 2-hour blooms with the real camera (hours)
"""
import argparse
import copy
import json
import os
import time

import numpy as np

import charts
import shared_settings
from run_simulation import run

OPTIONS = ["wave", "curtain", "off"]
BLOOM = 1200.0

TUNING = {
    "wave_ratio": [0.1, 0.2, 1 / 3, 0.5, 0.7],
    "air_gain": [0.0, 125.0, 250.0, 500.0],
    "wave_gap_m": [10.0, 20.0, 30.0, 50.0],   # added after the key test: back-push from the next front
}
TESTS = {
    "1_current": ("current_mps", [-0.3, -0.2, -0.1, -0.05, 0.0, 0.1, 0.2], "Background current (m/s, + = toward sea)"),
    "2_tide": ("tide", ["with->against", "against->with"], "Tide turning halfway"),
    "3_bloom_size": ("bloom_peak_density", [0.1, 0.5, 1.0, 2.0], "Bloom peak crowding (jellyfish/m^3)"),
    "4_species": ("species", ["standard", "small 15-30 cm", "deep swimmers"], "Species"),
    "5a_pipe_spacing": ("pipe_spacing_m", [1.0, 2.0, 3.0], "Pipe spacing (m)"),
    "5b_corridor_length": ("corridor_length_m", [60.0, 100.0, 140.0], "Corridor length (m)"),
    "6_water_version": ("water_version", ["optimistic", "realistic"], "Water version"),
    "S_plume_constant": ("plume_constant", [1.2, 1.46], "Plume constant C (Bulson 1.46)"),
}


def condition(key, value, s):
    """Apply one change to the settings."""
    s = copy.deepcopy(s)
    if key == "tide":
        a, b = (0.1, -0.1) if value == "with->against" else (-0.1, 0.1)
        s.update(current_mps=a, current_after_reverse_mps=b, current_reverse_at_s=s["bloom_duration_s"] / 2)
    elif key == "species":
        if value.startswith("small"):
            s.update(size_range_m=(0.15, 0.30))
        elif value.startswith("deep"):
            s.update(depth_mean_m=2.0)
    else:
        s[key] = value
    return s


def one(s):
    r = run(s, verbose=False, keep_snapshots=False)
    j, e = r["jellyfish"], r["energy"]
    return {"share_out": j["share_out_to_sea"], "share_back": j["share_back_to_intake"],
            "share_in": j["share_in_corridor"], "moved_real": j["moved_out_real"],
            "energy_kwh": e["energy_kwh"], "kwh_per_1000": e["kwh_per_1000_moved"],
            "rough_max": j["roughness_max_per_s"], "runtime_s": r["runtime_s"]}


def sweep(name, key, values, base, seeds, options):
    out = []
    for v in values:
        for opt in options:
            for sd in seeds:
                s = condition(key, v, dict(base, mode=opt, random_seed=sd))
                res = one(s)
                res.update(test=name, key=key, value=v, option=opt, seed=sd,
                           settings={k: (list(x) if isinstance(x, tuple) else x) for k, x in s.items()})
                out.append(res)
            m = np.mean([o["share_out"] for o in out if o["value"] == v and o["option"] == opt])
            print(f"  {name:<20} {str(v):<16} {opt:<8} out to sea {100*m:5.1f}%", flush=True)
    return out


def main(seeds=(42, 43, 44), full=False, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    base = shared_settings.get(camera_source="picture" if full else "truth", use_exit_camera=False)
    if not full:
        base["bloom_duration_s"] = BLOOM
    t0, rows = time.time(), []
    print("Tuning (Bubble Highway only):")
    for k, vals in TUNING.items():
        rows += sweep(f"T_{k}", k, vals, base, seeds, ["wave"])
    print("Six adaptability tests (+ sensitivity):")
    for name, (key, vals, _) in TESTS.items():
        rows += sweep(name, key, vals, base, seeds, OPTIONS)
    json.dump(rows, open(os.path.join(out_dir, "experiments.json"), "w"), default=str)
    make_charts(rows, out_dir)
    print(f"Done in {(time.time()-t0)/60:.1f} min; results in {out_dir}/")
    return rows


def make_charts(rows, out_dir):
    def pick(test):
        d = {}
        for r in rows:
            if r["test"] == test:
                d.setdefault(r["option"], {}).setdefault(r["value"], []).append(r["share_out"])
        return d
    for name, (key, vals, label) in TESTS.items():
        d = pick(name)
        if not d:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            charts.adaptability(d, key, label, os.path.join(out_dir, f"test_{name}.png"),
                                f"Test {name}: share moved out to sea")
        else:   # categorical: one bar group per value
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.2))
            w = 0.27
            for i, opt in enumerate([o for o in OPTIONS if o in d]):
                ms = [100 * np.mean(d[opt][v]) for v in vals]
                ax.bar(np.arange(len(vals)) + (i - 1) * w, ms, w, color=charts.COLOURS[opt], label=charts.NAMES[opt])
            ax.set_xticks(range(len(vals)), vals)
            ax.set(ylabel="Jellyfish moved out to sea (%)", ylim=(0, 105), title=f"Test {name}: {label}")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"test_{name}.png"), dpi=150); plt.close(fig)
    # Chart 1 + Chart 2 from the realistic baseline condition (test 6, realistic)
    base_rows = [r for r in rows if r["test"] == "6_water_version" and r["value"] == "realistic"]
    if base_rows:
        share = {o: [r["share_out"] for r in base_rows if r["option"] == o] for o in OPTIONS}
        charts.chart1_share_out(share, os.path.join(out_dir, "chart1_share_out.png"),
                                "Realistic water: share of jellyfish moved out to sea")
        en = {o: [r["energy_kwh"] for r in base_rows if r["option"] == o] for o in ["wave", "curtain"]}
        mv = {o: [r["moved_real"] for r in base_rows if r["option"] == o] for o in ["wave", "curtain"]}
        charts.chart2_energy(en, mv, os.path.join(out_dir, "chart2_energy.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    main(tuple(range(42, 42 + a.seeds)), a.full)
