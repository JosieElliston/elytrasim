#!/usr/bin/env python3
"""Every pump policy's pitch along every policy's own limit cycle.

    python3 tools/plot_pump_policies.py [--out=PNG] run ...

Each run is a pumpdp output directory (V.npy, fly.npy, spec.txt). One column per run: its limit
cycle, one lap from its snap, read off the end of pumpdp's own flight. On that lap's states, every
run's policy is evaluated as pumpdp flies it: the quarter degree maximizing the tick's reward plus
gamma V(v'), V read as the run's --extract reads it, physics with the run's --conv-lift-exp. A run's own policy on its own
cycle must reproduce the flown pitch, which is checked. Below, the lap's velocity.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 0.08
DRAG_Y = float(np.float32(0.98))
DRAG_Z = float(np.float32(0.99))
SNAP = -80   # a lap starts at the first tick pitch reaches this, as in pump_explorer
INK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


class Run:
    def __init__(self, d):
        self.dir = d
        spec = open(os.path.join(d, "spec.txt")).read()
        self.spec = spec
        m = re.search(r" w ([\d.]+)", spec)
        self.w = float(m.group(1)) if m else 0.0
        m = re.search(r" lambda ([\d.]+)", spec)
        self.lam = float(m.group(1)) if m else 0.0
        m = re.search(r"conv_lift_exp ([\d.]+)", spec)
        self.k = float(m.group(1)) if m else 1.0
        h = float(re.search(r"horizon ([\d.]+)", spec).group(1))
        self.gamma = 1.0 - 1.0 / h
        self.limit = float(re.search(r"limit ([\d.]+)", spec).group(1))
        self.fine = float(re.search(r"fine ([\d.]+)", spec).group(1))
        m = re.search(r"extract (\w+)", spec)
        self.read = m.group(1) if m else "Linear"
        self.period = int(re.search(r"period (\d+)", spec).group(1))
        self.vy = np.load(os.path.join(d, "vy.npy"))
        self.vz = np.load(os.path.join(d, "vz.npy"))
        self.V = np.load(os.path.join(d, "V.npy"))
        self.fly = np.load(os.path.join(d, "fly.npy"))

    def label(self):
        return f"λ = {self.lam:g}"

    def rough(self):
        """Grid noise on the nose-down glide of the own lap: the RMS, in degrees, of pitch minus an
        11-tick local quadratic fit, over glide ticks at least 6 from any step over 8 degrees (the
        real pullouts), and the largest such deviation. The policy's quarter-degree output step
        alone is 0.07 RMS."""
        from scipy.signal import savgol_filter
        p = self.lap()[:, 3]
        ok = p > 5
        jump = np.flatnonzero(np.abs(np.diff(p)) > 8)
        for j in jump:
            ok[max(0, j - 5):j + 7] = False
        r = (p - savgol_filter(p, 11, 2))[ok]
        return float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())

    def lap(self):
        """One lap off the flight's end, from its snap: rows (t, vy, vz, pitch, TE)."""
        lap = self.fly[-self.period:]
        p = lap[:, 3]
        s = np.flatnonzero((p <= SNAP) & (np.roll(p, 1) > SNAP))
        # a period of two near-identical laps is drawn as its first
        return np.roll(lap, -s[0], axis=0)[:(s[1] - s[0]) if len(s) > 1 else None] if len(s) else lap

    def policy(self, vy, vz):
        """pumpdp's flown pitch at each state (arrays): its rank() over the fine pitch grid."""
        n = int(round(2 * self.limit / self.fine))
        ps = np.minimum(-self.limit + self.fine * np.arange(n + 1), self.limit)
        lean = np.radians(ps)
        lift = np.cos(lean) ** 2
        up = np.where(lean < 0, -np.sin(lean), 0.0)
        outer = lift if self.k == 1.0 else lift ** self.k
        vy, vz = np.asarray(vy)[:, None], np.asarray(vz)[:, None]
        hor = np.abs(vz)
        y = vy + G * (-1 + lift * 0.75)
        z = vz + 0 * y
        conv = np.where(y < 0, y * -0.1 * outer, 0.0)
        y, z = y + conv, z + conv
        u = hor * up * 0.04
        y, z = y + u * 3.2, z - u
        z = z + (hor - z) * 0.1
        y, z = y * DRAG_Y, z * DRAG_Z
        dv = self.vy[1] - self.vy[0]
        ny, nz = len(self.vy), len(self.vz)
        cy = np.clip(y, self.vy[0], self.vy[-1])
        cz = np.clip(z, 0.0, self.vz[-1])
        xj, xk = (cy - self.vy[0]) / dv, cz / dv
        j = np.minimum(np.floor(xj).astype(int), ny - 2)
        k = np.minimum(np.floor(xk).astype(int), nz - 2)
        a = (xj - j).astype(np.float32).astype(float)
        b = (xk - k).astype(np.float32).astype(float)
        V = self.V
        if self.read in ("Cubic", "Clipped"):
            def w(t):
                return [(-t**3 + 2 * t**2 - t) / 2, (3 * t**3 - 5 * t**2 + 2) / 2, (-3 * t**3 + 4 * t**2 + t) / 2, (t**3 - t**2) / 2]
            wa, wb = w(a), w(b)
            v = sum(wa[p] * wb[q] * V[np.clip(j + p - 1, 0, ny - 1), np.clip(k + q - 1, 0, nz - 1)] for p in range(4) for q in range(4))
            if self.read == "Clipped":
                c = np.stack([V[j, k], V[j, k + 1], V[j + 1, k], V[j + 1, k + 1]])
                v = np.clip(v, c.min(axis=0), c.max(axis=0))
        else:
            v = (V[j, k] * (1 - b) + V[j, k + 1] * b) * (1 - a) + (V[j + 1, k] * (1 - b) + V[j + 1, k + 1] * b) * a
        ke = lambda p, q: (p * p + q * q) / (2 * G)
        q = y + ke(cy, cz) - ke(vy, vz) + self.w * cz + self.gamma * v
        self.last_q, self.last_ps = q, ps   # every pitch's Q at those states, for plot_pump_cycles
        return ps[np.argmax(q, axis=1)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--out=")]
    out = next((a[6:] for a in sys.argv[1:] if a.startswith("--out=")), os.path.join(ROOT, "runs/fwddp/fig/policies_on_cycles.png"))
    runs = [Run(d) for d in args]
    fig, axes = plt.subplots(2, len(runs), figsize=(6.4 * len(runs), 7.2), sharex="col",
                             gridspec_kw={"height_ratios": [2, 1]}, squeeze=False)
    for c, cyc in enumerate(runs):
        lap = cyc.lap()
        t = np.arange(len(lap))
        top, bot = axes[0, c], axes[1, c]
        for i, pol in enumerate(runs):
            p = pol.policy(lap[:, 1], lap[:, 2])
            if pol is cyc:
                bad = np.flatnonzero(p != lap[:, 3])
                print(f"{cyc.label()}: own policy vs flown pitch, {len(bad)} of {len(lap)} ticks differ"
                      + (f", max {np.abs(p - lap[:, 3]).max():.2f} deg" if len(bad) else "")
                      + ";  glide noise %.2f deg RMS, max %.1f" % cyc.rough())
            else:
                d = np.abs(p - lap[:, 3])
                print(f"{pol.label()} on {cyc.label()}'s cycle: mean |dpitch| {d.mean():.1f} deg, max {d.max():.1f}, ticks within 1 deg {np.mean(d <= 1):.0%}")
            own = pol is cyc
            top.step(t, p, where="post", color=INK[i], lw=2.0 if own else 1.4,
                     label=f"{pol.label()} policy" + (" (flown)" if own else ""), zorder=3 if own else 2)
        top.invert_yaxis()
        top.set_ylabel("pitch (deg)")
        top.set_title(f"{cyc.label()} limit cycle: {len(lap)} ticks, k = {cyc.k:g}", loc="left", fontsize=11)
        top.legend(frameon=False, loc="upper right")
        bot.plot(t, lap[:, 2], color="#555", lw=1.6, label="vz (forward)")
        bot.plot(t, lap[:, 1], color="#999", lw=1.6, ls="--", label="vy (up)")
        bot.axhline(0, color="#ccc", lw=0.8)
        bot.set_ylabel("velocity (b/t)")
        bot.set_xlabel("tick from snap")
        bot.legend(frameon=False, loc="upper right")
        for ax in (top, bot):
            ax.grid(color="#e6e6e6", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, len(lap))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
