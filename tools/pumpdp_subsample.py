#!/usr/bin/env python3
"""How coarse a pumpdp V can be shipped: keep every s-th node, read it as the run's --extract
reads it, and fly that policy (the run's physics, from the run's own lap start).

    python3 tools/pumpdp_subsample.py run [s ...]       default s: 1 2 4 8

Per s: the flown climb and forward speed over whole laps, glide noise (see
plot_pump_policies.Run.rough), and the pitch gap to the full-resolution policy along the full
policy's lap. s = 1 must reproduce pumpdp's own numbers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pump_policies import DRAG_Y, DRAG_Z, G, SNAP, Run

LAPS_FLOWN = 12


def step(run, vy, vz, p):
    lean = np.radians(p)
    lift = np.cos(lean) ** 2
    up = -np.sin(lean) if lean < 0 else 0.0
    y, z = vy + G * (-1 + lift * 0.75), vz
    if y < 0:
        c = y * -0.1 * lift ** run.k
        y, z = y + c, z + c
    u = abs(vz) * up * 0.04
    y, z = y + u * 3.2, z - u
    z += (abs(vz) - z) * 0.1
    return y * DRAG_Y, z * DRAG_Z


def coarsen(run, s):
    """The run with every s-th node, aligned so the grid keeps vz = 0 and its vy origin."""
    c = Run.__new__(Run)
    c.__dict__.update(run.__dict__)
    c.vy, c.vz, c.V = run.vy[::s], run.vz[::s], run.V[::s, ::s]
    return c


def fly(run, v0, n):
    vy, vz = v0
    rows = []
    for _ in range(n):
        p = run.policy([vy], [vz])[0]
        rows.append((vy, vz, p))
        vy, vz = step(run, vy, vz, p)
    return np.array(rows)


def main():
    full = Run(sys.argv[1])
    ss = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8]
    lap = full.lap()
    n = LAPS_FLOWN * len(lap)
    print(f"{full.dir}: dv {full.vy[1] - full.vy[0]:g}, read {full.read}, lap {len(lap)} ticks")
    for s in ss:
        c = coarsen(full, s)
        f = fly(c, lap[0, 1:3], n)
        p = f[:, 2]
        snaps = np.flatnonzero((p[1:] <= SNAP) & (p[:-1] > SNAP)) + 1
        a, b = snaps[len(snaps) // 2], snaps[-1]   # whole laps, past the transient
        y = np.cumsum([step(c, *r) [0] for r in f[a:b]])
        ke = lambda r: (r[0] ** 2 + r[1] ** 2) / (2 * G)
        climb = (y[-1] + ke(f[b]) - ke(f[a])) / (b - a)
        fwd = np.mean([step(c, *r)[1] for r in f[a:b]])
        c.fly = np.column_stack([np.arange(b - a), f[a:b, 0], f[a:b, 1], f[a:b, 2], np.zeros(b - a)])
        c.period = b - a
        noise = c.rough()
        gap = np.abs(c.policy(lap[:, 1], lap[:, 2]) - lap[:, 3])
        print(f"  s {s}: dv {c.vy[1] - c.vy[0]:g}  climb {climb:.6f} b/t  fwd {fwd:.6f} b/t  "
              f"glide noise {noise[0]:.2f} RMS max {noise[1]:.1f}  |pitch - full| on full lap: "
              f"mean {gap.mean():.2f} p95 {np.percentile(gap, 95):.2f} max {gap.max():.1f}")


if __name__ == "__main__":
    main()
