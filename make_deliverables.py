"""
make_deliverables.py - the hand-in items in one go ("What we hand in at the end")
==================================================================================
Must-haves: the animation (curtain vs Bubble Highway side by side), Chart 1
(share moved out), Chart 2 (energy with the saving marked), the headline number
(kWh per 1,000 jellyfish moved, assumptions in a footnote).
Strong extras come from key_test.py (colour map), experiments.py (adaptability)
and the CV part (camera accuracy chart).

    python make_deliverables.py              # 20-min bloom, real camera (YOLO or blob)
    python make_deliverables.py --full       # 2-hour bloom
    python make_deliverables.py --camera truth
"""
import argparse
import json
import os

import charts
import energy
import shared_settings
from run_simulation import run


def main(full=False, camera="picture", out="results"):
    os.makedirs(out, exist_ok=True)
    over = dict(water_version="realistic", camera_source=camera)
    if not full:
        over["bloom_duration_s"] = 1200.0
    res = {m: run(shared_settings.get(mode=m, **over)) for m in ("wave", "curtain", "off")}
    s = res["wave"]["settings"]

    charts.chart1_share_out({m: [r["jellyfish"]["share_out_to_sea"]] for m, r in res.items()},
                            f"{out}/main_chart1_share_out.png", "Realistic water, main run: share moved out to sea")
    charts.chart2_energy({m: [res[m]["energy"]["energy_kwh"]] for m in ("wave", "curtain")},
                         {m: [res[m]["jellyfish"]["moved_out_real"]] for m in ("wave", "curtain")},
                         f"{out}/main_chart2_energy.png")
    charts.animation(res["curtain"]["snapshots"], res["wave"]["snapshots"], s, f"{out}/animation.gif")

    e = res["wave"]["energy"]
    summary = {
        "energy_check": {k: float(v) for k, v in energy.check_against_guide(s).items()},
        "main_runs": {m: {"jellyfish": r["jellyfish"], "energy": r["energy"],
                          "camera_accuracy": r.get("camera_accuracy"),
                          "exit_vs_entrance": r.get("exit_vs_entrance")} for m, r in res.items()},
        "headline_kwh_per_1000_moved": e["kwh_per_1000_moved"],
        "headline_footnote": ("Realistic water; Bulson plume theory (C=1.46), 5 m pipes every 2 m, "
                              "374 L/min/m nominal air, 60 % compressor efficiency, no hose loss, "
                              f"{s['bloom_duration_s']/60:.0f}-min bloom peaking at {s['bloom_peak_density']} /m^3, "
                              f"background current {s['current_mps']} m/s; see shared_settings.py for all assumptions."),
    }
    json.dump(summary, open(f"{out}/main_summary.json", "w"), indent=1, default=float)
    print(f"\nSaved charts, animation and main_summary.json in {out}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--camera", default="picture", choices=["picture", "truth"])
    a = ap.parse_args()
    main(a.full, a.camera)
