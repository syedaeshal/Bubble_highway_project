"""
run_simulation.py - ENERGY ENGINEER 2, JOB A: joining all the parts into one program
=====================================================================================
One loop, three speeds of time (rule 6):
    every 0.05 s : new arrivals (bloom) -> water + jellyfish move (water)
    every 1 s    : entrance and exit cameras take a picture (CV part)
    every 5 s    : the controller decides (controller); the last decision stays in force
    every 2 s    : snapshot of all jellyfish for the animation
Records kept (Job A step 3): every controller decision, every camera reading,
the snapshots, and every jellyfish's fate.
Run modes (step 4): mode = off / curtain / wave; water_version = optimistic /
realistic - any combination with one setting change.
Swapping in a teammate's real part (step 5): replace the import of that one
module; nothing else changes as long as its inputs and outputs stay the same.

    python run_simulation.py                     # one main run, prints the scorecard
    python run_simulation.py --mode curtain --quick
"""
import argparse
import time

import numpy as np

import bubble_physics as bp
import energy
import jelly_scorecard
import shared_settings
from bloom import Bloom
from camera_system import CameraSystem
from controller import Controller, pipes_firing
from detectors import make_detector
from scorecard import summarise_counting, exit_vs_entrance
from water import Water


class TruthCamera:
    """Stand-in camera (guide EE2 Job A step 1): reports the true count and drift."""
    def __init__(self, s, camera="entrance"):
        self.s, self.camera, self.records = s, camera, []
        self.volume = s["cam_view_length_m"] * s["corridor_width_m"] * s["depth_band_m"]
        self.x0 = 0.0 if camera == "entrance" else s["corridor_length_m"] - s["cam_view_length_m"]

    def step(self, J, t):
        inview = (J["x"] >= self.x0) & (J["x"] < self.x0 + self.s["cam_view_length_m"])
        n = int(inview.sum())
        r = {"t": t, "camera": self.camera, "camera_ok": True, "count": n, "true_count": n,
             "crowding_per_m3": n * self.s["represents"] / self.volume,
             "drift_along_mps": float(J["speed"][inview].mean()) if n >= 3 else None,
             "drift_side_mps": 0.0, "drift_speed_mps": None, "drift_direction_deg": None}
        self.records.append(r)
        return r

    @staticmethod
    def for_controller(r):
        return r


def run(s, verbose=True, keep_snapshots=True, detector=None):
    t0 = time.time()
    rng = np.random.default_rng(s["random_seed"])
    bloom, water, ctrl = Bloom(s, rng), Water(s, rng), Controller(s)
    if s["camera_source"] == "picture":
        det = detector or make_detector(s, verbose=verbose)
        entrance = CameraSystem(s, "entrance", det)
        exit_cam = CameraSystem(s, "exit", det) if s["use_exit_camera"] else None
    else:
        entrance, exit_cam = TruthCamera(s), (TruthCamera(s, "exit") if s["use_exit_camera"] else None)

    dt = s["dt_physics_s"]
    every_cam = int(round(s["camera_dt_s"] / dt))
    every_ctrl = int(round(s["controller_dt_s"] / dt))
    every_snap = int(round(s["snapshot_dt_s"] / dt))
    n_steps = int((s["bloom_duration_s"] + s["extra_run_time_s"]) / dt)
    decision, reading, fired_sum, fired_n, snaps = None, None, 0, 0, []

    k_bloom_end = int(s["bloom_duration_s"] / dt)
    for k in range(n_steps):
        t = k * dt
        if k > k_bloom_end and k % every_ctrl == 0 and len(water.J["x"]) == 0:
            break                                                # corridor empty: stop early
        u_c = bp.background_current(t, s)
        water.add(bloom.arrivals(t, dt, u_c), t)
        if k % every_cam == 0:                                   # cameras: once a second
            J = water.jellyfish()
            reading = entrance.step(J, t)
            if exit_cam:
                exit_cam.step(J, t)
        if k % every_ctrl == 0:                                  # controller: every 5 s
            if decision is not None:
                decision["n_pipes_avg"] = fired_sum / max(fired_n, 1)
            meter = u_c + rng.normal(0, s["current_meter_noise_mps"])
            decision = ctrl.decide(entrance.for_controller(reading), meter, t)
            fired_sum = fired_n = 0
        firing = pipes_firing(decision, t, water.pipes, s)
        fired_sum += len(firing)
        fired_n += 1
        water.step(firing, decision["air_lpm_per_m"], t, dt)
        if keep_snapshots and k % every_snap == 0:               # snapshot: every 2 s
            st = water.finished["status"]
            snaps.append({"t": t, "x": water.J["x"].astype(np.float32),
                          "y": water.J["y"].astype(np.float32),
                          "firing": firing.tolist(),
                          "out": st.count("out_to_sea"), "back": st.count("back_to_intake")})
    decision["n_pipes_avg"] = fired_sum / max(fired_n, 1)
    t_end = t

    rec = water.all_records(t_end)
    jelly = jelly_scorecard.summarise(rec, s)
    res = {"settings": s, "jellyfish": jelly, "records": rec, "decisions": ctrl.records,
           "camera_entrance": entrance.records, "camera_exit": exit_cam.records if exit_cam else [],
           "snapshots": snaps, "runtime_s": time.time() - t0,
           "energy": energy.summarise(ctrl.records, s, jelly["moved_out_real"])}
    if s["camera_source"] == "picture":
        res["camera_accuracy"] = summarise_counting(entrance.records, s)
        if exit_cam:
            res["exit_vs_entrance"] = exit_vs_entrance(entrance.records, exit_cam.records)
    if verbose:
        print_summary(res)
    return res


def print_summary(res):
    s, j, e = res["settings"], res["jellyfish"], res["energy"]
    print(f"\n=== {s['mode']} / {s['water_version']} | seed {s['random_seed']} | "
          f"{res['runtime_s']:.0f} s to run ===")
    print(f"jellyfish (real): {j['jellyfish_real']}  out to sea {100*j['share_out_to_sea']:.1f}%  "
          f"back to intake {100*j['share_back_to_intake']:.1f}%  sideways "
          f"{100*j['share_lost_sideways']:.1f}%  still in corridor {100*j['share_in_corridor']:.1f}%")
    if j["median_transit_s"]:
        print(f"median transit {j['median_transit_s']:.0f} s; roughest {j['roughness_max_per_s']:.2f}/s "
              f"(limit {j['roughness_limit_per_s']}/s, {100*j['share_over_roughness_limit']:.1f}% over); "
              f"time above pipes mean {j['time_above_pipes_mean_s']:.1f} s")
    if e["energy_kwh"]:
        sv = e["saving_vs_curtain"]
        print(f"energy {e['energy_kwh']:.1f} kWh vs curtain-equivalent {e['curtain_equivalent_kwh']:.1f} kWh "
              f"(saving {100*sv:.0f}%); {e['kwh_per_1000_moved'] or float('nan'):.1f} kWh per 1000 moved")
    if "camera_accuracy" in res and res["camera_accuracy"]["overall"]:
        c = res["camera_accuracy"]["overall"]
        print(f"camera: found {100*c['found_share']:.1f}%, count error {100*c['relative_count_error']:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="wave", choices=["off", "curtain", "wave"])
    ap.add_argument("--water", default="realistic", choices=["optimistic", "realistic"])
    ap.add_argument("--camera", default="picture", choices=["picture", "truth"])
    ap.add_argument("--quick", action="store_true", help="20-minute bloom instead of 2 hours")
    a = ap.parse_args()
    over = dict(mode=a.mode, water_version=a.water, camera_source=a.camera)
    if a.quick:
        over.update(bloom_duration_s=1200.0)
    run(shared_settings.get(**over))
