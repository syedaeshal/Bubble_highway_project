"""
charts.py - ENERGY ENGINEER 3, JOB B: the final charts and the animation
=========================================================================
1. One look everywhere: the same colour for "wave", "curtain" and "no system".
2. Every chart is built from saved results, so it can be redrawn when numbers change.
3. Side-by-side animation from EE2's snapshots: jellyfish drifting, firing pipes
   lighting up, live counts on screen (curtain left, Bubble Highway right).
Every chart has labels, units and a one-line message as its title.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"wave": "#0072B2", "curtain": "#D55E00", "off": "#7F7F7F"}
NAMES = {"wave": "Bubble Highway", "curtain": "Ordinary curtain", "off": "No system"}


def _mean_spread(vals):
    v = np.array(vals, float)
    return v.mean(), (v.min(), v.max())


def chart1_share_out(summary, path, title=None):
    """summary: {option: [share_out per seed]}"""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, opt in enumerate(["wave", "curtain", "off"]):
        if opt not in summary:
            continue
        m, (lo, hi) = _mean_spread(summary[opt])
        ax.bar(i, 100 * m, color=COLOURS[opt], yerr=[[100 * (m - lo)], [100 * (hi - m)]], capsize=6)
        ax.text(i, 100 * m + 2, f"{100*m:.0f}%", ha="center")
    ax.set_xticks(range(3), [NAMES[o] for o in ["wave", "curtain", "off"]])
    ax.set(ylabel="Jellyfish moved out to sea (%)", ylim=(0, 105),
           title=title or "Share of jellyfish moved out to sea")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def chart2_energy(energy, moved, path):
    """energy/moved: {option: [values per seed]}; never energy on its own (EE3 step 7)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    opts = [o for o in ["wave", "curtain"] if o in energy]
    for i, opt in enumerate(opts):
        m, _ = _mean_spread(energy[opt])
        mv, _ = _mean_spread(moved[opt])
        ax.bar(i, m, color=COLOURS[opt])
        ax.text(i, m * 1.02, f"{m:.0f} kWh\n{mv:,.0f} jellyfish moved", ha="center", va="bottom", fontsize=9)
    if len(opts) == 2:
        e_w, e_c = np.mean(energy["wave"]), np.mean(energy["curtain"])
        if e_c > 0:
            ax.set_title(f"Bubble Highway uses {100*(1-e_w/e_c):.0f}% less energy than an ordinary curtain")
    ax.set_xticks(range(len(opts)), [NAMES[o] for o in opts])
    ax.set(ylabel="Compressor energy over the bloom (kWh)")
    ax.set_ylim(0, max(np.mean(energy[o]) for o in opts) * 1.3 + 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def key_test_map(km, path):
    vs = list(km["share_out"])
    fig, axes = plt.subplots(1, len(vs), figsize=(6 * len(vs), 4.4))
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, vs):
        g = 100 * np.array(km["share_out"][v])
        im = ax.imshow(g, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(km["wave_speeds"])), [f"{w:g}" for w in km["wave_speeds"]])
        ax.set_yticks(range(len(km["response_times"])), km["response_times"])
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g[i, j]:.0f}", ha="center", va="center",
                        color="w" if g[i, j] < 60 else "k", fontsize=8)
        ax.set(xlabel="Wave speed (m/s)", ylabel="Jellyfish response time (s)", title=f"{v} water")
    fig.colorbar(im, ax=axes, label="Jellyfish moved out to sea (%)")
    best = max(np.max(km["share_out"][v]) for v in vs if v == "realistic") if "realistic" in vs else None
    fig.suptitle("Key test: when do jellyfish ride the bubble wave?" +
                 (f"  (realistic best: {100*best:.0f}%)" if best is not None else ""))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def adaptability(results, x_key, x_label, path, title):
    """results: {option: {x_value: [share_out per seed]}}"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for opt, by_x in results.items():
        xs = sorted(by_x)
        m = [100 * np.mean(by_x[x]) for x in xs]
        lo = [100 * np.min(by_x[x]) for x in xs]
        hi = [100 * np.max(by_x[x]) for x in xs]
        ax.plot(xs, m, "o-", color=COLOURS[opt], label=NAMES[opt])
        ax.fill_between(xs, lo, hi, color=COLOURS[opt], alpha=0.15)
    ax.set(xlabel=x_label, ylabel="Jellyfish moved out to sea (%)", ylim=(0, 105), title=title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def animation(snaps_left, snaps_right, s, path, seconds=40, fps=12):
    """Side-by-side animation (EE3 Job B step 3): curtain left, Bubble Highway right."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from bubble_physics import pipe_positions
    pipes = pipe_positions(s)
    n = min(len(snaps_left), len(snaps_right))
    frames = np.linspace(0, n - 1, seconds * fps).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6))
    arts = []
    for ax, label, opt in zip(axes, ["Ordinary curtain", "Bubble Highway"], ["curtain", "wave"]):
        ax.set(xlim=(-2, s["corridor_length_m"] + 2), ylim=(0, s["corridor_width_m"]),
               xlabel="Along the corridor (m): intake 0 -> open sea", ylabel="Across (m)")
        ax.set_facecolor("#dcebf5")
        lines = [ax.axvline(p, color="white", lw=1.5, alpha=0.6) for p in pipes]
        dots = ax.scatter([], [], s=6, color="#5b2a86")
        txt = ax.text(0.01, 1.03, "", transform=ax.transAxes, fontsize=9)
        ax.set_title(label, color=COLOURS[opt], loc="right", fontsize=10)
        arts.append((lines, dots, txt))
    fig.tight_layout()

    def update(fi):
        k = frames[fi]
        for (lines, dots, txt), snaps in zip(arts, [snaps_left, snaps_right]):
            sn = snaps[k]
            on = set(sn["firing"])
            for i, ln in enumerate(lines):
                ln.set_color("#ffffff" if i not in on else "#00a2ff")
                ln.set_linewidth(1 if i not in on else 4)
            dots.set_offsets(np.c_[sn["x"], sn["y"]] if len(sn["x"]) else np.empty((0, 2)))
            txt.set_text(f"t = {sn['t']/60:5.1f} min   in corridor {len(sn['x'])*s['represents']:>5}   "
                         f"out to sea {sn['out']*s['represents']:>6}   back to intake {sn['back']*s['represents']:>6}")
        return []

    FuncAnimation(fig, update, frames=len(frames)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
