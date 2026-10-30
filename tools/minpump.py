#!/usr/bin/env python3
"""The shortest pump that doesn't lose height: min forward distance over one closed lap, subject to Δy >= 0.

    python3 tools/minpump.py --seed runs/fwddp/probe_disc3000_lam-3.9 --T 120 [--out runs/minpump]

A lap is T pitches flown from v0 that end back at v0 (so it repeats), scored as
    D = sum of v_z over the lap (forward blocks),   Δy = sum of v_y over the lap (height; on a closed lap, ΔTE).
Solved per T as a nonlinear program in (pitches, v0): minimize D subject to v_T = v0 and Δy >= 0, with SLSQP and
exact gradients by complex step; `--polish` then runs a coordinate descent that steps over the pitch-0 kink SLSQP
stops on (0.3 blocks at T 145). `--objective dy` instead maximizes Δy, the feasibility frontier in T. Physics is pumpdp's `step` (field's f64 libm kernel, yaw 0, pitch clamped to 89).
The seed is a pumpdp run: its flight's last lap (snap to snap) is time-stretched to T ticks as the starting guess.
A Lagrangian over laps can't do this: a lap with D - ν Δy < 0 repeats to -inf, so the dual only reads its sign.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

G = 0.08
DRAG_Y = float(np.float32(0.98))
DRAG_Z = float(np.float32(0.99))
LIMIT = 89.0
H = 1e-30   # complex-step size


def step(vy, vz, p):
    """pumpdp's `step` at pitch p degrees (negative nose up); complex-safe, branches on real parts."""
    lean = p * (np.pi / 180.0)
    c = np.cos(lean)
    lift = c * c
    up_sin = np.where(np.real(lean) < 0, -np.sin(lean), 0.0)
    move_hor = vz   # v_z >= 0 throughout
    y = vy + G * (-1.0 + lift * 0.75)
    z = vz
    conv = np.where(np.real(y) < 0, y * -0.1 * lift, 0.0)
    y = y + conv
    z = z + conv
    up = move_hor * up_sin * 0.04
    y = y + up * 3.2
    z = z - up
    z = z + (move_hor - z) * 0.1
    return y * DRAG_Y, z * DRAG_Z


def fly(x):
    """x = (p_0..p_{T-1}, vy0, vz0), any leading batch shape; returns vy, vz of shape (..., T+1)."""
    p, vy, vz = x[..., :-2], x[..., -2], x[..., -1]
    ys, zs = [vy], [vz]
    for t in range(p.shape[-1]):
        vy, vz = step(vy, vz, p[..., t])
        ys.append(vy)
        zs.append(vz)
    return np.stack(ys, -1), np.stack(zs, -1)


def outcomes(x):
    """(D, Δy, closure_y, closure_z) of the lap x."""
    vy, vz = fly(x)
    return np.stack([vz[..., 1:].sum(-1), vy[..., 1:].sum(-1), vy[..., -1] - vy[..., 0], vz[..., -1] - vz[..., 0]], -1)


class Lap:
    """Caches one complex-step pass per x: all four outcomes and their gradients."""

    def __init__(self):
        self.x = None

    def eval(self, x):
        if self.x is not None and np.array_equal(x, self.x):
            return
        n = len(x)
        xc = np.tile(x.astype(complex), (n, 1)) + 1j * H * np.eye(n)
        out = outcomes(xc)
        self.x = x.copy()
        self.f = out[0].real
        self.g = out.imag.T / H   # (4, n)

    def val(self, i):
        return lambda x: (self.eval(x), self.f[i])[1]

    def grad(self, i):
        return lambda x: (self.eval(x), self.g[i])[1]


def seed_lap(run):
    """The last snap-to-snap lap of a pumpdp run's flight: (pitches, v0)."""
    f = np.load(os.path.join(run, "fly.npy"))[-5000:]
    p = f[:, 3]
    snaps = np.where((p[1:] <= -80) & (p[:-1] > -80))[0] + 1
    a, b = snaps[-2], snaps[-1]
    return p[a:b].copy(), f[a, 1:3].copy()


def stretch(p, T):
    """Resample a pitch schedule to T ticks, piecewise constant."""
    return p[np.minimum((np.arange(T) * len(p) / T).astype(int), len(p) - 1)]


def steady_v0(p, v0, iters=2000):
    """Fly the schedule back to back until v0 settles: the lap's own fixed point (or wherever it got)."""
    x = np.concatenate([p, v0])
    for _ in range(iters):
        vy, vz = fly(x)
        new = np.array([vy[-1], vz[-1]])
        if np.abs(new - x[-2:]).max() < 1e-12:
            break
        x[-2:] = new
    return x[-2:]


# The solver's variables are x / SCALE: pitch tenths of a degree in, v0 hundredths of a block/tick out. why: per
# unit, D and Δy move ~0.02 per degree of pitch and ~50 per block/tick of v0, and SLSQP's BFGS starts at the
# identity; unscaled it never converged in 3000 iterations.
def scale(T):
    return np.concatenate([np.full(T, 10.0), [0.01, 0.01]])


def solve(x0, maxiter=500, dymin=0.0, objective="D"):
    lap = Lap()
    T = len(x0) - 2
    s = scale(T)
    f = lambda i: (lambda u: lap.val(i)(u * s))
    g = lambda i: (lambda u: lap.grad(i)(u * s) * s)
    cons = [{"type": "eq", "fun": f(2), "jac": g(2)},
            {"type": "eq", "fun": f(3), "jac": g(3)}]
    if objective == "D":
        obj, jac = f(0), g(0)
        cons.append({"type": "ineq", "fun": lambda u: f(1)(u) - dymin, "jac": g(1)})
    else:   # max Δy: the most a closed T-tick lap can climb, whatever its distance
        obj, jac = (lambda u: -f(1)(u)), (lambda u: -g(1)(u))
    lo = np.concatenate([np.full(T, -LIMIT), [-2.0, 0.0]]) / s
    hi = np.concatenate([np.full(T, LIMIT), [2.8, 3.6]]) / s
    r = minimize(obj, x0 / s, jac=jac, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                 options={"maxiter": maxiter, "ftol": 1e-10})
    r.x = r.x * s
    return r


GRID = np.arange(-LIMIT, LIMIT + 1e-9, 0.25)


def fixed_point_maps(x):
    """(D, Δy), their gradients in v0, and Φ = dv_T/dv0, at a closed lap x."""
    lap = Lap()
    lap.eval(x)
    gv = lap.g[:, -2:]
    return lap.f[:2], gv[:2], gv[2:] + np.eye(2)


def reduced_grad(x):
    """d(D, Δy)/dp with v0 riding the schedule's fixed point."""
    lap = Lap()
    lap.eval(x)
    gp, gv = lap.g[:, :-2], lap.g[:, -2:]
    return gp[:2] - gv[:2] @ np.linalg.solve(gv[2:], gp[2:])


def coord_pass(x, nu, tol=1e-7):
    """One pass of coordinate descent on L = D - nu Δy, v0 at the schedule's fixed point, in place.

    Each tick in random order takes the best pitch on GRID: every candidate flown to the lap's end at once, its
    closure error carried onto v0 by the linearized fixed point, and an accepted move re-solves v0 exactly and is
    kept only if the exact L improved. why: SLSQP stops on the pitch-0 kink (nose-down has no up term), parking
    0-holds mid-climb and mid-dive; a move that sees the whole pitch range steps over it. Returns moves made.
    """
    T = len(x) - 2
    (D, dy), gv, phi = fixed_point_maps(x)
    moved = 0
    for t in np.random.permutation(T):
        vy, vz = fly(x)
        cy, cz = np.full(GRID.shape, vy[t]), np.full(GRID.shape, vz[t])
        Dc, Yc = np.full(GRID.shape, vz[1:t + 1].sum()), np.full(GRID.shape, vy[1:t + 1].sum())
        for k in range(t, T):
            cy, cz = step(cy, cz, GRID if k == t else x[k])
            Dc += cz
            Yc += cy
        dv0 = np.linalg.solve(np.eye(2) - phi, np.stack([cy - vy[0], cz - vz[0]]))
        Lc = Dc + gv[0] @ dv0 - nu * (Yc + gv[1] @ dv0)
        i = np.argmin(Lc)
        if Lc[i] > D - nu * dy - tol or GRID[i] == x[t]:
            continue
        old, oldv = x[t], x[-2:].copy()
        x[t] = GRID[i]
        x[-2:] = steady_v0(x[:-2], x[-2:])
        (D2, dy2), gv2, phi2 = fixed_point_maps(x)
        if D2 - nu * dy2 < D - nu * dy - tol:
            D, dy, gv, phi = D2, dy2, gv2, phi2
            moved += 1
        else:
            x[t], x[-2:] = old, oldv
    return moved


def polish(x, passes=30, outer=12, verbose=False):
    """Coordinate-polish a closed lap toward min D st Δy = 0: descent on D - nu Δy, nu by secant between rounds,
    started at the lap's KKT multiplier."""
    x = x.copy()
    T = len(x) - 2
    x[-2:] = steady_v0(x[:-2], x[-2:])
    gD, gY = reduced_grad(x)
    free = (np.abs(x[:T]) < LIMIT - 0.1) & (np.abs(x[:T]) > 0.5)
    nu = float(gD[free] @ gY[free] / (gY[free] @ gY[free]))
    hist = []
    for _ in range(outer):
        for _ in range(passes):
            if coord_pass(x, nu) == 0:
                break
        dy = float(outcomes(x)[1])
        if verbose:
            print(f"  nu {nu:.4f} {report(x)}", flush=True)
        hist.append((nu, dy))
        if abs(dy) < 1e-4:
            break
        if len(hist) >= 2 and hist[-1][1] != hist[-2][1]:
            (n0, y0), (n1, y1) = hist[-2:]
            nu = n1 - y1 * (n1 - n0) / (y1 - y0)
        else:
            nu *= 1.05 if dy < 0 else 0.95
    return x, nu


def report(x):
    D, dy, cy, cz = outcomes(x)
    return {"T": len(x) - 2, "D": float(D), "dy": float(dy), "close": float(np.hypot(cy, cz)),
            "v0": [float(x[-2]), float(x[-1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="runs/fwddp/probe_disc3000_lam-3.9")
    ap.add_argument("--T", type=int, nargs="+", required=True)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--chain", action="store_true",
                    help="seed each T after the first from the previous T's solution instead of the seed run")
    ap.add_argument("--init", default=None, help="a saved lap (.npy, pitches then v0) to seed from instead of --seed")
    ap.add_argument("--objective", choices=["D", "dy"], default="D",
                    help="D: min distance st Δy >= 0 (the point); dy: max Δy, the feasibility frontier")
    ap.add_argument("--polish", action="store_true",
                    help="after SLSQP, coordinate-polish the lap (objective D only); see coord_pass")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.init:
        x = np.load(a.init)
        p, v0 = x[:-2], x[-2:]
    else:
        p, v0 = seed_lap(a.seed)
    print("seed lap", report(np.concatenate([p, v0])))
    for T in a.T:
        q = stretch(p, T)
        x0 = np.concatenate([q, steady_v0(q, v0)])
        r = solve(x0, a.maxiter, objective=a.objective)
        rep = report(r.x)
        rep.update(objective=a.objective, status=int(r.status), msg=r.message, nit=int(r.nit), seed=report(x0))
        if a.polish and a.objective == "D":
            rep["slsqp"] = {"D": rep["D"], "dy": rep["dy"]}
            r.x, nu = polish(r.x)
            rep.update(report(r.x), nu=nu)
        print(json.dumps(rep), flush=True)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            np.save(os.path.join(a.out, f"lap_T{T}.npy"), r.x)
        if a.chain and rep["close"] < 1e-6 and (rep["dy"] > -1e-6 or a.objective == "dy"):
            p, v0 = r.x[:-2], r.x[-2:]


if __name__ == "__main__":
    main()
