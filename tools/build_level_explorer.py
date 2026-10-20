#!/usr/bin/env python3
"""Build the level-flight explorer: tools/level_explorer.html with pumpdp's lambda-priced maps embedded.

    python3 tools/build_level_explorer.py [out.html]      # default runs/fwddp/level_explorer.html

Reads every finished run runs/fwddp/probe_disc3000_lam<lambda>/ (pumpdp --mode disc --discount
0.9996666666666667 --evals 300 --lambda <lambda>; see src/bin/pumpdp.rs for what lambda prices). Each
map is the policy's pitch at every node on pumpdp's quarter-degree grid, embedded exactly as int16
quarter-degrees, zlib-compressed (the page inflates it with DecompressionStream). Each map's
limit-cycle numbers are read here off pumpdp's own flight (fly.npy), the exact one-tick lookahead
on V. The page flies that same lookahead by fetching the run's V.npy, so it must be served from
runs/fwddp/ (the pitch map is its fallback).
The runs are under runs/, which is gitignored, so the built page is too.
"""

import base64
import glob
import json
import os
import re
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 0.08
SNAP = -80   # a snap is the first tick pitch reaches this, as in pump_explorer
WINDOW = 50000   # pumpdp's default --window: its rates are read over the flight's last this many ticks


def cycle(fly):
    """Numbers off pumpdp's flight, rows (t, vy, vz, pitch, TE) at each tick's start, over its last WINDOW ticks."""
    vy, vz, p, te = fly[-WINDOW:, 1], fly[-WINDOW:, 2], fly[-WINDOW:, 3], fly[-WINDOW:, 4]
    n = len(te) - 1
    climb = (te[-1] - te[0]) / n
    y = te - (vy * vy + vz * vz) / (2 * G)
    h = y - climb * np.arange(n + 1)
    d = np.diff(p)
    rev = np.sum((d[:-1] * d[1:] < 0) & (np.abs(d[:-1]) > 2) & (np.abs(d[1:]) > 2))
    return {"swing": float(h.max() - h.min()), "vz": [float(vz.min()), float(vz.max())],
            "rev": float(rev / n * 1000), "laps": int(np.sum((p[1:] <= SNAP) & (p[:-1] > SNAP)))}


# pumpdp's default --v0: the hand-flown reference cycle's start, (vy, vz) blocks/tick.
V0 = (0.167467, 0.200887)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs/fwddp/level_explorer.html")
    maps, grid = [], None
    for d in glob.glob(os.path.join(ROOT, "runs/fwddp/probe_disc3000_lam*")):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "spec.txt")):
            continue
        spec = open(os.path.join(d, "spec.txt")).read()
        lam = float(re.search(r" lambda ([\d.]+)", spec).group(1)) if " lambda " in spec else 0.0
        assert abs(lam - float(d.rsplit("lam", 1)[1])) < 1e-12, d
        vy, vz = np.load(os.path.join(d, "vy.npy")), np.load(os.path.join(d, "vz.npy"))
        g = {"vy0": float(vy[0]), "dv": float(vy[1] - vy[0]), "ny": len(vy), "nz": len(vz)}
        assert abs(vz[0]) < 1e-12 and abs((vz[1] - vz[0]) - g["dv"]) < 1e-12
        assert grid is None or all(grid[k] == g[k] for k in g), f"{d}: grid differs"
        grid = grid or g
        p = np.load(os.path.join(d, "pitch.npy"))
        q = np.round(p * 4)
        assert np.abs(q / 4 - p).max() < 1e-9, f"{d}: pitch is off the quarter-degree grid"
        num = lambda pat: float(re.search(pat, spec).group(1))
        period = re.search(r"period (\d+) ticks", spec)
        gamma = float(re.search(r"discount ([\d.]+)", spec).group(1))
        assert grid.get("gamma", gamma) == gamma, f"{d}: discount differs"
        grid["gamma"] = gamma
        maps.append({"lam": lam, "w": lam * 21.5 / 330.0,   # opt::w_of_lambda
                     "climb": num(r"flown rate over last \d+ ticks (-?[\d.]+) b/tick"),
                     "fwd": num(r"flown forward speed over that window (-?[\d.]+) b/tick"),
                     "period": int(period.group(1)) if period else None,
                     **cycle(np.load(os.path.join(d, "fly.npy"))), "dir": os.path.basename(d),
                     "pitch": base64.b64encode(zlib.compress(q.astype("<i2").tobytes(), 9)).decode()})
    if not maps:
        sys.exit("no finished runs under runs/fwddp/probe_disc3000_lam*")
    maps.sort(key=lambda m: m["lam"])
    data = {"grid": grid, "maps": maps, "v0": V0}
    html = open(os.path.join(ROOT, "tools/level_explorer.html")).read()
    assert "/*@DATA@*/null" in html
    html = html.replace("/*@DATA@*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(f"{out}: {len(maps)} maps (lambda {maps[0]['lam']:g}..{maps[-1]['lam']:g}), {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
