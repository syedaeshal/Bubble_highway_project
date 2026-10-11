"""
scorecard.py - JOB C step 6 and JOB D: the CV person's half of the scorecard
=============================================================================

score_frame        - Job C step 6: score one picture against the truth
summarise_counting - Job D step 1: accuracy overall and at the peak of the bloom
exit_vs_entrance   - Job D steps 2-3: the system checking its own success
noise_sweep        - Job D step 4: "how murky can the water get before it struggles"
crowding_sweep     - "done" check: counting error against crowding, YOLO vs blob finder
plot_*             - the charts (titles carry a one-line message, as EE3 asks)

This is the ONLY file that uses the true positions; it never feeds them back
into detection.
"""
import copy
import numpy as np

from camera import render_frame
from standin_inputs import random_scene

# Agreed chart colours (EE3 Job B step 1 asks for one consistent look).
COLOURS = {"yolo": "#1f77b4", "blob": "#ff7f0e", "truth": "#555555"}



# ---------------------------------------------------------------------------
# Job C step 6 - score one picture
# ---------------------------------------------------------------------------
def score_frame(detections, truth, settings):
    """Compare what the camera found with what was really there.

    A true jellyfish is 'found' if a detection lies within score_match_px of
    it. A clump estimated at n jellyfish may claim up to n true jellyfish
    within its own radius. Detections that claim nothing are false alarms.
    """
    T = np.c_[truth["col"], truth["row"]] if truth["count"] else np.zeros((0, 2))
    matched = np.zeros(len(T), bool)
    false_alarms = 0
    gate = settings["score_match_px"]
    # single jellyfish first, then clumps take what is left nearby
    for d in sorted(detections, key=lambda d: d["n"]):
        if not len(T):
            false_alarms += d["n"]
            continue
        radius = gate + (np.sqrt(d["area_px"] / np.pi) if d["n"] > 1 else 0.0)
        dist = np.hypot(T[:, 0] - d["col"], T[:, 1] - d["row"])
        cand = np.flatnonzero(~matched & (dist <= radius))
        take = cand[np.argsort(dist[cand])][: d["n"]]
        matched[take] = True
        false_alarms += d["n"] - len(take)
    count = int(sum(d["n"] for d in detections))
    return {
        "true_count": int(truth["count"]),
        "true_visible": int(np.sum(truth["visible"])),
        "found": int(matched.sum()),
        "found_visible": int(np.sum(matched & truth["visible"])),
        "false_alarms": int(false_alarms),
        "count_error": count - int(truth["count"]),   # + means over-counted
    }


# ---------------------------------------------------------------------------
# Job D step 1 - how accurate was the counting?
# ---------------------------------------------------------------------------
def _stats(recs):
    true = np.array([r["true_count"] for r in recs], float)
    count = np.array([r["count"] for r in recs], float)
    found = np.array([r["found"] for r in recs], float)
    fa = np.array([r["false_alarms"] for r in recs], float)
    tot = max(true.sum(), 1.0)
    return {
        "pictures": len(recs),
        "true_jellyfish_seen": int(true.sum()),
        "found_share": float(found.sum() / tot),             # how many we found
        "false_alarms_per_100": float(100 * fa.sum() / tot),  # false alarms per 100 real
        "mean_abs_count_error": float(np.mean(np.abs(count - true))),
        "relative_count_error": float(np.abs(count - true).sum() / tot),
        "count_bias": float((count - true).sum() / tot),     # + over-counts, - under-counts
    }


def summarise_counting(records, settings):
    """Accuracy over the whole run, and at the peak of the bloom
    (the top score_peak_fraction of pictures by true number in view)."""
    recs = [r for r in records if r["camera_ok"] and r["count"] is not None]
    if not recs:
        return {"overall": None, "peak": None}
    true = np.array([r["true_count"] for r in recs])
    cutoff = np.quantile(true, 1 - settings["score_peak_fraction"])
    peak = [r for r, t in zip(recs, true) if t >= cutoff and t > 0]
    return {"overall": _stats(recs), "peak": _stats(peak) if peak else None}


# ---------------------------------------------------------------------------
# Job D steps 2-3 - exit camera vs entrance camera
# ---------------------------------------------------------------------------
def exit_vs_entrance(entrance_records, exit_records):
    """Net jellyfish crossing each camera's counting line, measured by the
    cameras and by the truth. exit / entrance = share that made it through,
    measured WITHOUT anyone going into the water."""
    def net(recs, key_f, key_b):
        return int(sum(r[key_f] - r[key_b] for r in recs))
    cam_in = net(entrance_records, "line_forward", "line_back")
    cam_out = net(exit_records, "line_forward", "line_back")
    true_in = net(entrance_records, "true_line_forward", "true_line_back")
    true_out = net(exit_records, "true_line_forward", "true_line_back")
    return {
        "camera_in": cam_in, "camera_out": cam_out,
        "camera_success_share": cam_out / cam_in if cam_in else None,
        "true_in": true_in, "true_out": true_out,
        "true_success_share": true_out / true_in if true_in else None,
    }


# ---------------------------------------------------------------------------
# Sweeps on random scenes (represents = 1, so every real jellyfish is drawn)
# ---------------------------------------------------------------------------
def _run_scenes(detector, settings, density, n_scenes, seed, **scene_kw):
    """Render n_scenes random pictures and add up their scores.
    The SAME scenes (same seed) are used at every level of a sweep, so the
    only thing that changes is the setting being tested (fair-results rule)."""
    tot = {"true": 0, "found": 0, "fa": 0, "abs_err": 0, "err": 0}
    for k in range(n_scenes):
        J = random_scene(settings, density, np.random.default_rng(seed + k), **scene_kw)
        img, truth = render_frame(J, settings, np.random.default_rng(seed + 10_000 + k))
        dets = detector.detect(img)
        sc = score_frame(dets, truth, settings)
        tot["true"] += sc["true_count"]
        tot["found"] += sc["found"]
        tot["fa"] += sc["false_alarms"]
        tot["abs_err"] += abs(sc["count_error"])
        tot["err"] += sc["count_error"]
    t = max(tot["true"], 1)
    return {"found_share": tot["found"] / t, "false_alarms_per_100": 100 * tot["fa"] / t,
            "relative_count_error": tot["abs_err"] / t, "count_bias": tot["err"] / t,
            "true_jellyfish": tot["true"]}


def noise_sweep(detector, settings, mess_levels=None, density=0.05, n_scenes=20, seed=None):
    """Job D step 4 - raise the picture mess step by step (noise, blur and
    glints together) and find where results noticeably worsen.

    'Tolerance' = the highest mess level at which every level up to it still
    finds >= score_ok_recall of jellyfish AND keeps the count within
    score_ok_count_error."""
    mess_levels = np.round(np.arange(0, 4.01, 0.25), 2) if mess_levels is None else mess_levels
    seed = settings["random_seed"] + 20_000 if seed is None else seed
    rows, tolerance = [], None
    still_ok = True
    for mess in mess_levels:
        s = copy.deepcopy(settings)
        s.update(cam_mess=float(mess), represents=1)
        res = _run_scenes(detector, s, density, n_scenes, seed)
        res["mess"] = float(mess)
        rows.append(res)
        ok = (res["found_share"] >= settings["score_ok_recall"]
              and res["relative_count_error"] <= settings["score_ok_count_error"])
        if ok and still_ok:
            tolerance = float(mess)
        still_ok = still_ok and ok
    return {"rows": rows, "tolerance_mess": tolerance}


def crowding_sweep(detectors, settings, densities=None, n_scenes=10, seed=None):
    """Counting error against crowding for each detector (dict name -> detector).
    Shows where clumps start to fool the software."""
    densities = np.round(np.geomspace(0.02, 2.0, 9), 3) if densities is None else densities
    seed = settings["random_seed"] + 30_000 if seed is None else seed
    s = copy.deepcopy(settings)
    s.update(represents=1)
    out = {}
    for name, det in detectors.items():
        out[name] = []
        for dens in densities:
            res = _run_scenes(det, s, float(dens), n_scenes, seed)
            res["density"] = float(dens)
            out[name].append(res)
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_crowding(results, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, rows in results.items():
        d = [r["density"] for r in rows]
        ax[0].plot(d, [100 * r["count_bias"] for r in rows], "o-", color=COLOURS.get(name), label=name)
        ax[1].plot(d, [100 * r["found_share"] for r in rows], "o-", color=COLOURS.get(name), label=name)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xscale="log", xlabel="Crowding (jellyfish per m³)", ylabel="Count error (% of true count)",
              title="Count error vs crowding (+ = over-count)")
    ax[1].axhline(90, color="k", lw=0.8, ls="--")
    ax[1].set(xscale="log", xlabel="Crowding (jellyfish per m³)", ylabel="Jellyfish found (%)",
              ylim=(0, 102), title="Share of jellyfish found")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle("Counting stays accurate until clumps form in heavy blooms")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise(sweeps, path):
    """sweeps: dict detector name -> noise_sweep() result.
    The y axis is capped at 105 % so a detector that falls apart does not squash
    the others; points above the cap are drawn at the top with an arrow marker."""
    plt = _plt()
    cap = 105
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, sw in sweeps.items():
        col = COLOURS.get(name)
        m = np.array([r["mess"] for r in sw["rows"]])
        found = np.array([100 * r["found_share"] for r in sw["rows"]])
        err = np.array([100 * r["relative_count_error"] for r in sw["rows"]])
        tol = sw["tolerance_mess"]
        ax.plot(m, found, "o-", color=col, label=f"{name}: found (works up to mess {tol})")
        ax.plot(m, np.minimum(err, cap - 2), "s--", color=col, alpha=0.6, label=f"{name}: count error")
        off = err > cap - 2
        if off.any():
            ax.plot(m[off], np.full(off.sum(), cap - 2), "^", color=col, ms=9)
            ax.annotate("off the chart", (m[off][0], cap - 2), textcoords="offset points",
                        xytext=(-30, -18), color=col, fontsize=8)
        if tol is not None:
            ax.axvline(tol, color=col, lw=1, ls=":")
    ax.axhline(90, color="k", lw=0.8, ls="--")
    ax.axhline(10, color="grey", lw=0.8, ls="--")
    ax.set(xlabel="Picture mess (1 = normal: noise + blur + sun glints)", ylabel="%",
           ylim=(0, cap), title="The trained model copes with 2.5x normal mess; the blob finder with 1x")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bloom_run(entrance_records, path):
    plt = _plt()
    t = [r["t"] for r in entrance_records]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, [r["true_count"] for r in entrance_records], color=COLOURS["truth"], label="true")
    ax[0].plot(t, [r["count"] if r["count"] is not None else np.nan for r in entrance_records],
               color=COLOURS.get(entrance_records[0]["detector"].split()[0], "C0"), alpha=0.8, label="camera")
    ax[0].set(ylabel="Jellyfish in view", title="Camera count follows the bloom")
    ax[0].legend()
    drift = [r["drift_along_mps"] if r["drift_along_mps"] is not None else np.nan for r in entrance_records]
    ax[1].plot(t, drift, ".", ms=2)
    ax[1].set(xlabel="Time (s)", ylabel="Drift along corridor (m/s)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
