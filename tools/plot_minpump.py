#!/usr/bin/env python3
"""Figure for tools/minpump.py: the shortest pump that doesn't lose height.

    python3 tools/plot_minpump.py [runs/minpump] [out.png]     # default out: <dir>/fig/minpump.png

Reads every lap_T*.npy under <dir>/*/ (SLSQP runs; min-D unless the directory is named gmax_*, which hold max-Δy
laps) and <dir>/final/lap_T*.npy (coordinate-polished, then SLSQP to make Δy = 0 exact; <dir>/cpol holds the
polish alone, on its 0.25-degree grid). Panels: forward distance per lap against lap length T, the most
a T-tick lap can climb, the shortest lap's pitches, and its flight path beside pumpdp's lambda -3.9 lap.
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minpump as m  # noqa: E402

INK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#8a8984"
OK = 1e-6   # closure and Δy slack that still counts as a pump


def laps(pattern):
    """{T: [report, x]} for every lap file matching pattern."""
    out = {}
    for f in glob.glob(pattern):
        x = np.load(f)
        out.setdefault(len(x) - 2, []).append((m.report(x), x))
    return out


def best_pump(found):
    """Per T, the feasible lap with least D."""
    best = {}
    for T, rs in found.items():
        ok = [(r, x) for r, x in rs if r["close"] < OK and r["dy"] > -1e-3]
        if ok:
            best[T] = min(ok, key=lambda rx: rx[0]["D"])
    return best


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/minpump"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "fig/minpump.png")
    slsqp, gmax = {}, {}
    for d in glob.glob(os.path.join(root, "*/")):
        name = os.path.basename(d.rstrip("/"))
        if name in ("cpol", "final"):
            continue
        tgt = gmax if name.startswith("gmax_") else slsqp
        for T, rs in laps(os.path.join(d, "lap_T*.npy")).items():
            tgt.setdefault(T, []).extend(rs)
    sl = best_pump(slsqp)
    po = best_pump(laps(os.path.join(root, "final/lap_T*.npy")))
    gm = {T: max(r["dy"] for r, x in rs if r["close"] < OK) for T, rs in gmax.items()
          if any(r["close"] < OK for r, x in rs)}
    Tb = min(po, key=lambda T: po[T][0]["D"])
    rb, xb = po[Tb]

    # the most a lap can climb crosses 0 between two sampled T: interpolate
    gT = sorted(gm)
    cross = next((gT[i] + (gT[i + 1] - gT[i]) * -gm[gT[i]] / (gm[gT[i + 1]] - gm[gT[i]])
                  for i in range(len(gT) - 1) if gm[gT[i]] < 0 <= gm[gT[i + 1]]), None)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    a = ax[0, 0]
    Ts = sorted(t for t in sl if 125 <= t <= 200)
    a.plot(Ts, [sl[t][0]["D"] for t in Ts], ".-", color=MUTED, lw=1, ms=4, label="SLSQP, best of all seeds")
    Tp = sorted(po)
    a.plot(Tp, [po[t][0]["D"] for t in Tp], "o-", color=INK[0], lw=2, ms=4, label="coordinate-polished, then SLSQP")
    a.plot([Tb], [rb["D"]], "o", ms=9, mfc="none", mec=INK[1], mew=2)
    a.annotate(f"shortest: T {Tb}, {rb['D']:.2f} blocks", (Tb, rb["D"]), xytext=(14, 22),
               textcoords="offset points", color="#333")
    if cross:
        a.axvspan(Ts[0] - 2, cross, color="#f1efe9", zorder=0)
        a.axvline(cross, color=MUTED, lw=1)
        a.text(cross + 1, max(sl[t][0]["D"] for t in Ts), f"← no lap shorter than T {cross:.1f}\n   climbs at all",
               ha="left", va="top", color=MUTED)
    a.set_xlim(Ts[0] - 2, Ts[-1] + 2)
    a.set_xlabel("lap length T (ticks)")
    a.set_ylabel("forward distance per lap (blocks)")
    a.set_title("min forward distance per closed lap, s.t. Δy ≥ 0", loc="left")
    a.legend(frameon=False, loc="lower right")

    a = ax[0, 1]
    a.plot(gT, [gm[t] for t in gT], "o-", color=INK[2], lw=2, ms=4)
    a.axhline(0, color=MUTED, lw=1, ls="--")
    if cross:
        a.axvline(cross, color=MUTED, lw=1)
        a.annotate(f"T ≈ {cross:.1f}", (cross, 0), xytext=(6, 10), textcoords="offset points", color="#333")
    a.set_xlabel("lap length T (ticks)")
    a.set_ylabel("max Δy per closed lap (blocks)")
    a.set_title("the most a T-tick lap can climb", loc="left")

    a = ax[1, 0]
    a.plot(np.arange(Tb), xb[:-2], color=INK[0], lw=1.5, drawstyle="steps-post")
    a.invert_yaxis()
    a.set_xlabel("tick")
    a.set_ylabel("pitch (deg)")
    vy, vz = m.fly(xb)
    a.set_title(f"the shortest lap: T {Tb}, v0 ({xb[-2]:.3f}, {xb[-1]:.3f}) b/t, "
                f"vz {vz.min():.2f}–{vz.max():.2f}", loc="left")

    a = ax[1, 1]
    ref, v0 = m.seed_lap("runs/fwddp/probe_disc3000_lam-3.9")
    for x, c, lab in [(xb, INK[0], f"shortest pump, {rb['D']:.1f} blocks"),
                      (np.concatenate([ref, v0]), MUTED, "pumpdp λ −3.9 lap, %.1f blocks" % m.report(np.concatenate([ref, v0]))["D"])]:
        vy, vz = m.fly(x)
        z = np.concatenate([[0], np.cumsum(vz[1:])])
        y = np.concatenate([[0], np.cumsum(vy[1:])])
        a.plot(z, y, color=c, lw=2 if c != MUTED else 1.2, label=lab)
        a.plot(z[-1], y[-1], "o", color=c, ms=4)
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("forward z (blocks)")
    a.set_ylabel("height y (blocks)")
    a.set_title("one lap's flight path, from the lap's start", loc="left")
    a.legend(frameon=False, loc="lower right")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
