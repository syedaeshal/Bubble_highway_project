"""
jelly_scorecard.py - ENERGY ENGINEER 1, JOB C: what happened to the jellyfish
==============================================================================
1. Share that made it out to sea, went back toward the intake, drifted out
   sideways, or were still in the corridor when the run ended.
2. Typical (median) time to get through the corridor, for those that made it.
3. Gentleness: the roughest water change any jellyfish felt, against a limit we
   choose and state openly (roughness_limit_per_s - no published limit exists).
4. Time spent right above firing pipes, where bubbles rise through them.
Shares are the same for simulated and real jellyfish; counts are scaled by
`represents` to real jellyfish.
"""
import numpy as np


def summarise(rec, s):
    n = len(rec["status"])
    out = {"jellyfish_simulated": int(n), "jellyfish_real": int(n * s["represents"])}
    for st in ("out_to_sea", "back_to_intake", "lost_sideways", "in_corridor"):
        out[f"share_{st}"] = float(np.mean(rec["status"] == st)) if n else 0.0
    ok = rec["status"] == "out_to_sea"
    out["moved_out_real"] = int(ok.sum() * s["represents"])
    transit = rec["t_exit"][ok] - rec["t_enter"][ok]
    out["median_transit_s"] = float(np.median(transit)) if ok.any() else None
    r = rec["rough_max"]
    out["roughness_max_per_s"] = float(r.max()) if n else 0.0
    out["roughness_p95_per_s"] = float(np.percentile(r, 95)) if n else 0.0
    out["roughness_limit_per_s"] = s["roughness_limit_per_s"]
    out["share_over_roughness_limit"] = float(np.mean(r > s["roughness_limit_per_s"])) if n else 0.0
    out["time_above_pipes_mean_s"] = float(rec["t_above"].mean()) if n else 0.0
    out["time_above_pipes_max_s"] = float(rec["t_above"].max()) if n else 0.0
    return out
