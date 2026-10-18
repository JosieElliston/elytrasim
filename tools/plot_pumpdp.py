#!/usr/bin/env python3
"""The stationary pump policies of src/bin/pumpdp.rs, drawn on the velocity plane.

    python3 tools/plot_pumpdp.py [run ...] [--out=DIR]

Each run is a directory runs/pumpdp/<name>/ (or a path); with none given, every finished run
there (one with a spec.txt) whose name doesn't start with test_, or the test runs if that is
all there is. Figures go to runs/pumpdp/fig/:

  <run>_chart.png, <run>_zoom.png   (a) policy pitch, (b) V or h with contours, (c) log10 of the
                                    margin, (d) the one-tick field G; elytra-vario's chart
                                    window and a zoom around the cycles
  <run>_chatter.png                 the same, around the run of tick-to-tick pitch flips on the
                                    lap, if it has one
  compare_pitch_chart.png, compare_pitch_zoom.png   every run's pitch map, one scale and window
  compare_cycle_pitch.png           pitch vs tick over one lap of each limit cycle, from min vz
  <run>_bias_check.png              avg runs only: TE + h(v) - rho t along the flown cycle

The flown limit cycle is the end of `fly`; its lap and exact period are found by return
distance (see find_period) and printed. The reference cycle is field_data.cycle(), a person's, period 300.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_data as fd
from plot_field_replay import AXIS_A, FIELD_ZERO, GRID_A, PITCH_LIMIT, field_rgb
from plot_field_troughs import CYCLE_INK, NOSE_DOWN, NOSE_UP, PITCH_CMAP, grid_step

TPS = 20
ROOT = fd.ROOT
RUNS = os.path.join(ROOT, "runs", "pumpdp")
BG, INK, DIM, SPINE = "#14171a", "#e6e9ed", "#b9c0c9", "#4a525c"
FLOWN_INK = "#f5ae68"      # the policy's own limit cycle: off the reference's green
MARGIN_FLOOR = 1e-5        # blocks; margins at or below it are drawn as it


def halo(lw):
    return [matplotlib.patheffects.withStroke(linewidth=lw + 2.2, foreground=FIELD_ZERO,
                                              alpha=.85)]


class Run:
    def __init__(self, d):
        self.dir = d
        self.name = os.path.basename(d.rstrip("/"))
        load = lambda n: np.load(os.path.join(d, f"{n}.npy"))
        self.vy, self.vz = load("vy"), load("vz")
        self.V, self.pitch, self.margin, self.q1 = (load(n) for n in ("V", "pitch", "margin", "q1"))
        self.fly = load("fly")
        self.spec = open(os.path.join(d, "spec.txt")).read()
        self.avg = self.spec.startswith("pumpdp avg")
        m = re.search(r"\(([-\d.]+) b/s\)", self.spec)
        self.rate_bps = float(m.group(1)) if m else float("nan")
        m = re.search(r"horizon ([\d.]+)", self.spec)
        self.horizon = float(m.group(1)) if m else None
        self.period, self.ret, self.exact, self.exact_ret = find_period(self.fly[:, 1],
                                                                        self.fly[:, 2])
        # Drawn on the velocity maps: the whole exact period up to four laps, so lap-to-lap
        # variation shows; the last row returns (near) the first.
        self.cyc = self.fly[-min(self.exact, 4 * self.period) - 1:]
        self.lap = self.fly[-self.period - 1:]

    def label(self):
        kind = "average reward" if self.avg else f"discounted, H = {self.horizon:g}"
        return f"{self.name} ({kind}): flown {self.rate_bps:.4f} b/s"

    def h_at(self, vy, vz):
        """V (h in avg) read bilinearly, clamped onto the grid, as pumpdp.rs reads it."""
        dy, dz = self.vy[1] - self.vy[0], self.vz[1] - self.vz[0]
        fy = np.clip((vy - self.vy[0]) / dy, 0, len(self.vy) - 1 - 1e-9)
        fz = np.clip((vz - self.vz[0]) / dz, 0, len(self.vz) - 1 - 1e-9)
        j, k = fy.astype(int), fz.astype(int)
        a, b = fy - j, fz - k
        V = self.V
        return ((1 - a) * (1 - b) * V[j, k] + (1 - a) * b * V[j, k + 1]
                + a * (1 - b) * V[j + 1, k] + a * b * V[j + 1, k + 1])


def return_distance(vy, vz, tail=600, lo=20, hi=3000):
    """(p, mean |v(t) - v(t - p)| over the last `tail` ticks) for p in [lo, hi)."""
    n = len(vy)
    ps = np.arange(lo, min(hi, n - tail - 1))
    d = np.array([np.mean(np.hypot(vy[n - tail:] - vy[n - tail - p:n - p],
                                   vz[n - tail:] - vz[n - tail - p:n - p])) for p in ps])
    return ps, d


def find_period(vy, vz, lap_tol=0.02):
    """(lap, its return distance, exact period, its return distance), blocks/tick.

    The exact period is the smallest p the flight repeats on (within 1e-9). The lap is the
    first local minimum of the return distance below lap_tol (0.4 b/s): the flight can close
    only after several laps that differ slightly (the policy is read on a quarter-degree grid,
    and laps can alternate), and the lap is what a pitch-vs-tick plot should span."""
    ps, d = return_distance(vy, vz)
    best = d.min()
    k = int(np.argmax(d <= best + 1e-9))
    lmin = np.r_[False, (d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]), False] & (d < lap_tol)
    j = int(np.argmax(lmin)) if lmin.any() else k
    return int(ps[j]), float(d[j]), int(ps[k]), float(d[k])


def reference():
    vy, vz = fd.cycle()
    pitches = np.loadtxt(fd.REFERENCE, ndmin=1)
    return vy, vz, pitches


def chrome(ax, title, window_bps):
    zlo, zhi, ylo, yhi = window_bps
    ax.set_facecolor(BG)
    for lo, hi, line, ticks in ((zlo, zhi, ax.axvline, ax.set_xticks),
                                (ylo, yhi, ax.axhline, ax.set_yticks)):
        step = grid_step(hi - lo)
        at = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        for v in at:
            line(v, color="w", lw=.8, alpha=AXIS_A if abs(v) < 1e-9 else GRID_A, zorder=1)
        ticks(at)
    ax.set_xlim(zlo, zhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("vz (blocks/second)", color=INK)
    ax.set_ylabel("vy (blocks/second)", color=INK)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.set_title(title, color=INK, fontsize=11)


def colorbar(fig, im, ax, label, invert=False):
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label(label, color=INK)
    cb.ax.tick_params(colors=DIM)
    cb.outline.set_edgecolor(SPINE)
    if invert:
        cb.ax.invert_yaxis()
    return cb


def draw_path(ax, vy, vz, color, lw=1.6, dots=20):
    x, y = vz * TPS, vy * TPS
    ax.plot(x, y, color=color, lw=lw, zorder=5, path_effects=halo(lw))
    ax.plot(x[::dots], y[::dots], "o", color=color, ms=3.0, mec=FIELD_ZERO, mew=.8, zorder=5)
    for i in range(dots // 2, len(x) - 1, 3 * dots):
        ax.annotate("", (x[i + 1], y[i + 1]), (x[i], y[i]), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=13))


def draw_cycles(ax, run, ref):
    draw_path(ax, ref[0], ref[1], CYCLE_INK)
    if len(run.cyc) > len(run.lap):             # earlier laps, thin, so lap-to-lap drift shows
        ax.plot(run.cyc[:, 2] * TPS, run.cyc[:, 1] * TPS, color=FLOWN_INK, lw=.7, alpha=.6,
                zorder=4.9)
    draw_path(ax, run.lap[:, 1], run.lap[:, 2], FLOWN_INK)


def cycle_handles(run):
    line = matplotlib.lines.Line2D
    return [line([], [], color=FLOWN_INK, lw=1.6, marker="o", ms=3, mec=FIELD_ZERO,
                 label=f"policy's limit cycle, lap {run.period} ticks"
                 + (f" (repeats exactly after {run.exact})" if run.exact != run.period else "")
                 + ", dot every 20"),
            line([], [], color=CYCLE_INK, lw=1.6, marker="o", ms=3, mec=FIELD_ZERO,
                 label="reference cycle (a person), period 300")]


def extent(run):
    hz, hy = run.vz[1] - run.vz[0], run.vy[1] - run.vy[0]
    return ((run.vz[0] - hz / 2) * TPS, (run.vz[-1] + hz / 2) * TPS,
            (run.vy[0] - hy / 2) * TPS, (run.vy[-1] + hy / 2) * TPS)


def chart_window(run):
    zlo, zhi, ylo, yhi = fd.CHART_BPS
    return (max(zlo, run.vz[0] * TPS), min(zhi, run.vz[-1] * TPS),
            max(ylo, run.vy[0] * TPS), min(yhi, run.vy[-1] * TPS))


def zoom_window(runs, ref, pad=0.12):
    vy = np.concatenate([ref[0]] + [r.cyc[:, 1] for r in runs]) * TPS
    vz = np.concatenate([ref[1]] + [r.cyc[:, 2] for r in runs]) * TPS
    sz, sy = vz.max() - vz.min(), vy.max() - vy.min()
    zlo, zhi, ylo, yhi = fd.CHART_BPS
    return (max(0.0, zlo, vz.min() - pad * sz), min(zhi, vz.max() + pad * sz),
            max(ylo, vy.min() - pad * sy), min(yhi, vy.max() + pad * sy))


def chatter_window(run, jump=20.0, least=6, pad=0.25):
    """Around the ticks of the lap where pitch jumps more than `jump` degrees and back, if
    there are at least `least` of them: the policy riding a switching surface. Else None."""
    c = run.lap
    dp = np.abs(np.diff(c[:, 3]))
    i = np.nonzero(dp > jump)[0]
    if len(i) < least:
        return None
    # The snap is one big jump; chatter is a run of them. Keep the longest run whose jumps
    # are at most 4 ticks apart.
    runs, start = [], 0
    for k in range(1, len(i) + 1):
        if k == len(i) or i[k] - i[k - 1] > 4:
            runs.append(i[start:k])
            start = k
    i = max(runs, key=len)
    if len(i) < least:
        return None
    vz, vy = c[i[0]:i[-1] + 2, 2] * TPS, c[i[0]:i[-1] + 2, 1] * TPS
    s = max(vz.max() - vz.min(), vy.max() - vy.min(), 1.0)
    return (max(0.0, vz.min() - pad * s), vz.max() + pad * s,
            vy.min() - pad * s, vy.max() + pad * s)


def in_window(run, w):
    zs = (run.vz * TPS >= w[0] - 1) & (run.vz * TPS <= w[1] + 1)
    ys = (run.vy * TPS >= w[2] - 1) & (run.vy * TPS <= w[3] + 1)
    return np.ix_(ys, zs)


def panel_pitch(fig, ax, run, ref, w):
    im = ax.imshow(run.pitch, extent=extent(run), origin="lower", interpolation="nearest",
                   cmap=PITCH_CMAP, vmin=-PITCH_LIMIT, vmax=PITCH_LIMIT, zorder=0)
    chrome(ax, "(a) policy pitch", w)
    colorbar(fig, im, ax, "pitch (degrees, − is nose up)", invert=True)
    draw_cycles(ax, run, ref)


def panel_value(fig, ax, run, ref, w):
    sub = run.V[in_window(run, w)]
    lo, hi = np.percentile(sub, [1, 99])
    im = ax.imshow(run.V, extent=extent(run), origin="lower", interpolation="nearest",
                   cmap="cividis", vmin=lo, vmax=hi, zorder=0)
    chrome(ax, "(b) " + ("bias h, 0 at the reference v0" if run.avg
                         else "V − V(reference v0)"), w)
    colorbar(fig, im, ax, "blocks")
    step = grid_step(hi - lo) / 2
    levels = np.arange(np.floor(lo / step) * step, hi + step, step)
    sel = run.V[in_window(run, w)]
    levels = levels[(levels > sel.min()) & (levels < sel.max())]
    cs = ax.contour(run.vz * TPS, run.vy * TPS, run.V, levels=levels, colors="w",
                    linewidths=.6, alpha=.55, zorder=2)
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7, colors=INK)
    draw_cycles(ax, run, ref)


def panel_margin(fig, ax, run, ref, w):
    lm = np.log10(np.maximum(run.margin, MARGIN_FLOOR))
    hi = np.percentile(lm[in_window(run, w)], 99.5)
    im = ax.imshow(lm, extent=extent(run), origin="lower", interpolation="nearest",
                   cmap="magma", vmin=np.log10(MARGIN_FLOOR), vmax=hi, zorder=0)
    chrome(ax, "(c) margin over the best pitch >10° away (dark: switching)", w)
    colorbar(fig, im, ax, "log10 margin (blocks)")
    draw_cycles(ax, run, ref)


def panel_field(fig, ax, run, ref, w, F):
    ax.imshow(field_rgb(F.G), extent=F.ext(), origin="lower", interpolation="nearest", zorder=0)
    chrome(ax, "(d) one-tick field G (best pitch), elytra-vario colors", w)
    ax.contour(F.vz * TPS, F.vy * TPS, F.G, levels=[0.0], colors="w", linewidths=.8, alpha=.45,
               zorder=1)
    draw_cycles(ax, run, ref)


def run_figure(run, ref, F, w, tag, out):
    zs, ys = w[1] - w[0], w[3] - w[2]
    pw = 6.4
    fig, axs = plt.subplots(2, 2, figsize=(2 * pw + 1.6, 2 * pw * ys / zs + 1.8))
    fig.patch.set_facecolor(BG)
    panel_pitch(fig, axs[0, 0], run, ref, w)
    panel_value(fig, axs[0, 1], run, ref, w)
    panel_margin(fig, axs[1, 0], run, ref, w)
    panel_field(fig, axs[1, 1], run, ref, w, F)
    for ax in axs.flat:
        ax.set_aspect("equal")
    axs[0, 0].legend(handles=cycle_handles(run), loc="upper left", facecolor="#1b1f24",
                     edgecolor=SPINE, labelcolor=INK, fontsize=8, framealpha=.92).set_zorder(10)
    fig.suptitle(run.label(), color=INK, fontsize=13)
    fig.tight_layout()
    path = os.path.join(out, f"{run.name}_{tag}.png")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return path


def compare_maps(runs, ref, w, tag, out):
    n = len(runs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    zs, ys = w[1] - w[0], w[3] - w[2]
    pw = 5.2
    fig, axs = plt.subplots(rows, cols, figsize=(cols * pw + 1.2, rows * pw * ys / zs + 1.4),
                            squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, run in zip(axs.flat, runs):
        im = ax.imshow(run.pitch, extent=extent(run), origin="lower", interpolation="nearest",
                       cmap=PITCH_CMAP, vmin=-PITCH_LIMIT, vmax=PITCH_LIMIT, zorder=0)
        chrome(ax, f"{run.name}: {run.rate_bps:.4f} b/s", w)
        ax.set_aspect("equal")
        draw_cycles(ax, run, ref)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    axs.flat[0].legend(handles=[
        matplotlib.lines.Line2D([], [], color=FLOWN_INK, lw=1.6, label="policy's limit cycle"),
        matplotlib.lines.Line2D([], [], color=CYCLE_INK, lw=1.6, label="reference cycle")],
        loc="upper left", facecolor="#1b1f24", edgecolor=SPINE, labelcolor=INK, fontsize=8,
        framealpha=.92).set_zorder(10)
    fig.suptitle("policy pitch, flown rate in each title", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, .93, 1))
    cax = fig.add_axes((.94, .15, .012, .7))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pitch (degrees, − is nose up)", color=INK)
    cb.ax.tick_params(colors=DIM)
    cb.outline.set_edgecolor(SPINE)
    cb.ax.invert_yaxis()
    path = os.path.join(out, f"compare_pitch_{tag}.png")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return path


def from_min_vz(vz, pitch):
    """Rotate one period (no repeated endpoint) to start at its minimum vz."""
    i = int(np.argmin(vz))
    return np.roll(pitch, -i), i


def style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.grid(color="w", alpha=GRID_A, lw=.8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def compare_cycles(runs, ref, out):
    rvz, rp = ref[1][:300], ref[2][:300]
    rp, _ = from_min_vz(rvz, rp)
    tmax = max([300] + [r.period for r in runs])
    fig, axs = plt.subplots(len(runs), 1, figsize=(11, 1.9 * len(runs) + 1), sharex=True,
                            squeeze=False)
    axs = axs[:, 0]
    fig.patch.set_facecolor(BG)
    for ax, run in zip(axs, runs):
        ax.plot(np.arange(300), rp, color=CYCLE_INK, lw=1.2, alpha=.8)
        laps = min(4, max(1, run.exact // run.period))
        for k in range(laps - 1, -1, -1):          # earlier laps faint, behind the last
            end = len(run.fly) - k * run.period
            c = run.fly[end - run.period:end]
            p, _ = from_min_vz(c[:, 2], c[:, 3])
            ax.plot(np.arange(run.period), p, color=FLOWN_INK, lw=1.4 if k == 0 else .9,
                    alpha=1 if k == 0 else .4)
        ax.set_ylim(PITCH_LIMIT + 5, -PITCH_LIMIT - 5)
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_xlim(0, tmax)
        style(ax, f"{run.label()}, lap {run.period}"
              + (f", repeats after {run.exact}" if run.exact != run.period else ""))
        ax.set_ylabel("pitch (°)", color=INK)
    axs[-1].set_xlabel("ticks since min vz", color=INK)
    axs[0].legend(handles=[
        matplotlib.lines.Line2D([], [], color=FLOWN_INK, lw=1.4, label="policy's limit cycle"),
        matplotlib.lines.Line2D([], [], color=CYCLE_INK, lw=1.2,
                                label="reference cycle (a person)")],
        loc="lower right", facecolor="#1b1f24", edgecolor=SPINE, labelcolor=INK, fontsize=8)
    fig.suptitle("pitch over one lap, from min vz (earlier laps faint)", color=INK, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, "compare_cycle_pitch.png")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return path


def bias_check(run, out, laps=4):
    """TE + h(v) - rho t over the last few laps, rounded up to whole exact periods so rho
    closes the line; flat on the optimal flight."""
    span = max(run.exact, run.period * laps)
    span = -(-span // run.exact) * run.exact if run.exact_ret < 1e-12 else run.period * laps
    n = min(len(run.fly), span + 1)
    c = run.fly[-n:]
    t = c[:, 0] - c[0, 0]
    te = c[:, 4] - c[0, 4]
    h = run.h_at(c[:, 1], c[:, 2])
    rho = (c[-1, 4] - c[0, 4]) / t[-1]      # over whole periods, so the line closes
    a = te + h - h[0] - rho * t
    b = te - rho * t
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    fig.patch.set_facecolor(BG)
    axs[0].plot(t, a, color=FLOWN_INK, lw=1.3)
    style(axs[0], f"TE + h(v) − ρt   (spread {a.max() - a.min():.3f} blocks)")
    axs[0].set_ylabel("blocks", color=INK)
    axs[1].plot(t, b, color=CYCLE_INK, lw=1.3)
    style(axs[1], f"TE − ρt, for contrast   (spread {b.max() - b.min():.3f} blocks)")
    axs[1].set_ylabel("blocks", color=INK)
    axs[1].set_xlabel(f"ticks (laps of {run.period}, ending at the flight's end)",
                      color=INK)
    fig.suptitle(f"{run.name}: along the flown limit cycle, ρ = {rho * TPS:.4f} b/s "
                 f"(h read bilinearly; both start at 0)", color=INK, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, f"{run.name}_bias_check.png")
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    return path, a.max() - a.min(), b.max() - b.min()


def finished(d):
    return os.path.exists(os.path.join(d, "spec.txt"))


def pick_runs(names):
    if names:
        dirs = [n if os.path.isdir(n) else os.path.join(RUNS, n) for n in names]
    else:
        all_ = sorted(d for d in os.listdir(RUNS) if d != "fig" and
                      os.path.isdir(os.path.join(RUNS, d)) and finished(os.path.join(RUNS, d)))
        main = [d for d in all_ if not d.startswith("test_")]
        dirs = [os.path.join(RUNS, d) for d in (main or all_)]
    runs = [Run(d) for d in dirs if finished(d)]
    # avg first, then by horizon; same objective, by name (so avg before avg_fine).
    return sorted(runs, key=lambda r: (not r.avg, r.horizon or 0, r.name))


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
               os.path.join(RUNS, "fig"))
    os.makedirs(out, exist_ok=True)
    runs = pick_runs(names)
    if not runs:
        sys.exit("no finished runs")
    ref = reference()
    F = fd.load()
    zoom = zoom_window(runs, ref)
    paths = []
    for run in runs:
        print(f"{run.name}: lap {run.period} (return {run.ret:.1e} b/t), exact period "
              f"{run.exact} (return {run.exact_ret:.1e}), "
              f"rate {run.rate_bps:.4f} b/s")
        cw = chart_window(run)
        paths.append(run_figure(run, ref, F, cw, "chart", out))
        paths.append(run_figure(run, ref, F, zoom, "zoom", out))
        cw = chatter_window(run)
        if cw:
            paths.append(run_figure(run, ref, F, cw, "chatter", out))
        if run.avg:
            p, sa, sb = bias_check(run, out)
            print(f"  TE + h - rho t spread {sa:.4f} blocks; TE - rho t spread {sb:.4f}")
            paths.append(p)
    paths.append(compare_maps(runs, ref, chart_window(runs[0]), "chart", out))
    paths.append(compare_maps(runs, ref, zoom, "zoom", out))
    paths.append(compare_cycles(runs, ref, out))
    for p in paths:
        print(os.path.abspath(p))


if __name__ == "__main__":
    main()
