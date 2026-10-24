#!/usr/bin/env python3
"""Every pump policy along every policy's own limit cycle, as pitch over the lap and on the velocity plane.

    python3 tools/plot_pump_policies.py [--out=PNG] [--every=N] run ...

Each run is a pumpdp output directory (V.npy, fly.npy, spec.txt). One column per run: its limit
cycle, one lap off the end of pumpdp's flight, starting at the detrended-TE peak (TE minus the
lap's mean climb per tick: the end of the gain, just before the dive). On that lap's states, every
run's policy is evaluated as pumpdp flies it: the quarter degree maximizing the tick's reward plus
gamma V(v'), V read as the run's --extract reads it, physics with the run's --conv-lift-exp. Top,
every run's pitch along the lap. Bottom, the lap in (vz, vy) at equal scale, with every run's
pitch drawn as an arrow every N ticks (nose up points up); the other runs' cycles faint.

Prints, per cycle: that its own policy reproduces the flown pitch (checked), its glide noise (see
Run.rough), and per other policy the pitch gap and the one-tick cost of flying the other's pitch:
the drop in the cycle's own Q = r + gamma V(v'), blocks (plus w times forward distance when
lambda > 0).
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
SNAP = -80   # a snap is the first tick pitch reaches this, as in pump_explorer
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
        p = self.lap()[0][:, 3]
        p = np.roll(p, -int(np.argmax((p <= SNAP) & (np.roll(p, 1) > SNAP))))   # from the snap, so no glide straddles the seam
        ok = p > 5
        jump = np.flatnonzero(np.abs(np.diff(p)) > 8)
        for j in jump:
            ok[max(0, j - 5):j + 7] = False
        r = (p - savgol_filter(p, 11, 2))[ok]
        return float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())

    def lap(self):
        """One lap off the flight's end, rows (t, vy, vz, pitch, TE), from its detrended-TE peak,
        and the lap's mean climb per tick. A period of two near-identical laps (snap to snap) is
        cut to its first."""
        lap = self.fly[-self.period:]
        p = lap[:, 3]
        s = np.flatnonzero((p <= SNAP) & (np.roll(p, 1) > SNAP))
        if len(s) > 1:
            lap = np.roll(lap, -s[0], axis=0)[:s[1] - s[0]]
        climb = (self.fly[-1, 4] - self.fly[-1 - self.period, 4]) / self.period
        h = lap[:, 4] - climb * np.arange(len(lap))
        return np.roll(lap, -int(np.argmax(h)), axis=0), climb

    def q(self, vy, vz):
        """pumpdp's Q at each state (arrays) for every pitch of the fine grid: (Q [state, pitch], pitches)."""
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
        return y + ke(cy, cz) - ke(vy, vz) + self.w * cz + self.gamma * v, ps

    def policy(self, vy, vz):
        """pumpdp's flown pitch at each state (arrays): its rank() over the fine pitch grid, the first best."""
        q, ps = self.q(vy, vz)
        return ps[np.argmax(q, axis=1)]

    def q_loss(self, vy, vz, pitch):
        """How much less this run's Q is at each state for `pitch` than for its own best, blocks."""
        q, ps = self.q(vy, vz)
        i = np.abs(ps[None, :] - np.asarray(pitch)[:, None]).argmin(axis=1)
        return q.max(axis=1) - q[np.arange(len(i)), i]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    out = opt.get("out", os.path.join(ROOT, "runs/fwddp/fig/policies_on_cycles.png"))
    every = int(opt.get("every", 6))
    runs = [Run(d) for d in args]
    laps = [r.lap() for r in runs]
    fig, axes = plt.subplots(2, len(runs), figsize=(6.6 * len(runs), 12.5),
                             gridspec_kw={"height_ratios": [1, 1.35]}, squeeze=False)
    # one velocity window for every column, so the cycles compare by eye
    vys = np.concatenate([l[:, 1] for l, _ in laps]); vzs = np.concatenate([l[:, 2] for l, _ in laps])
    pad = 0.15
    win = (vzs.min() - pad, vzs.max() + pad, vys.min() - pad, vys.max() + pad)
    span = max(win[1] - win[0], win[3] - win[2])
    alen = span / 28   # arrow length, b/t
    for c, (cyc, (lap, climb)) in enumerate(zip(runs, laps)):
        t = np.arange(len(lap))
        top, bot = axes[0, c], axes[1, c]
        for o, (other, (ol, _)) in enumerate(zip(runs, laps)):
            if other is not cyc:
                bot.plot(np.append(ol[:, 2], ol[0, 2]), np.append(ol[:, 1], ol[0, 1]), color=INK[o], lw=1, alpha=0.35,
                         label=f"{other.label()} cycle", zorder=1)
        bot.plot(np.append(lap[:, 2], lap[0, 2]), np.append(lap[:, 1], lap[0, 1]), color="#333", lw=1.4,
                 label=f"{cyc.label()} cycle (this)", zorder=2)
        bot.plot(lap[0, 2], lap[0, 1], "o", color="#333", ms=6, zorder=4)
        bot.annotate("t = 0", (lap[0, 2], lap[0, 1]), xytext=(6, 4), textcoords="offset points", fontsize=9, color="#333")
        for m in range(50, len(lap), 50):
            bot.plot(lap[m, 2], lap[m, 1], "o", color="#333", ms=3, zorder=4)
            bot.annotate(str(m), (lap[m, 2], lap[m, 1]), xytext=(4, -10), textcoords="offset points", fontsize=8, color="#666")
        own = cyc.policy(lap[:, 1], lap[:, 2])
        bad = np.flatnonzero(own != lap[:, 3])
        print(f"{cyc.label()}: own policy vs flown pitch, {len(bad)} of {len(lap)} ticks differ"
              + (f", max {np.abs(own - lap[:, 3]).max():.2f} deg" if len(bad) else "")
              + ";  glide noise %.2f deg RMS, max %.1f" % cyc.rough())
        for i, pol in enumerate(runs):
            p = own if pol is cyc else pol.policy(lap[:, 1], lap[:, 2])
            mine = pol is cyc
            top.step(t, p, where="post", color=INK[i], lw=2.0 if mine else 1.4,
                     label=f"{pol.label()} policy" + (" (flown)" if mine else ""), zorder=3 if mine else 2)
            s = t[::every]
            ang = np.radians(-p[s])   # pitch is nose-down positive; the arrow points where the nose does
            bot.quiver(lap[s, 2], lap[s, 1], alen * np.cos(ang), alen * np.sin(ang), color=INK[i],
                       angles="xy", scale_units="xy", scale=1, width=0.0042 if mine else 0.0032,
                       headwidth=3.5, headlength=4, alpha=0.95 if mine else 0.8, zorder=5 if mine else 3,
                       label=f"{pol.label()} pitch")
            if not mine:
                loss = cyc.q_loss(lap[:, 1], lap[:, 2], p)
                d = np.abs(p - own)
                print(f"{pol.label()} on {cyc.label()}'s cycle: |dpitch| mean {d.mean():.1f} max {d.max():.1f} deg; "
                      f"one-tick loss in {cyc.label()}'s Q: sum {loss.sum():.4f} b/lap, max {loss.max():.4f} at t={int(np.argmax(loss))}")
        top.invert_yaxis()
        top.set_ylabel("pitch (deg)")
        top.set_xlabel("tick from the detrended-TE peak")
        top.set_xlim(0, len(lap))
        top.set_title(f"{cyc.label()} limit cycle: {len(lap)} ticks, climb {climb * 20:.3f} b/s, k = {cyc.k:g}",
                      loc="left", fontsize=11)
        top.legend(frameon=False, loc="upper right")
        bot.set_aspect("equal")
        bot.set_xlim(win[0], win[1]); bot.set_ylim(win[2], win[3])
        bot.axhline(0, color="#ccc", lw=0.8, zorder=0)
        bot.set_xlabel("vz, forward (b/t)")
        bot.set_ylabel("vy, up (b/t)")
        bot.legend(frameon=False, loc="upper right", fontsize=9)
        for ax in (top, bot):
            ax.grid(color="#e6e6e6", lw=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
