#!/usr/bin/env python3
"""Build the ablated DP explorer: tools/dp_explorer.html with pumpdp's value functions embedded.

    python3 tools/build_dp_explorer.py [out.html]      # default runs/pumpdp/ablate/dp_explorer.html

Reads every finished pumpdp run runs/pumpdp/ablate/curv<f> and runs/pumpdp/ablate_exp/exp<k> (made
with `--mode disc --discount 0.9996666666666667 --evals 300` and `--conv-curv <f>` or
`--conv-lift-exp <k>`, the rest default; see src/bin/pumpdp.rs). k = 1 is the game, the same run as
f = 1, so it is skipped. The page
re-derives the policy, its flights and Q from V, so it embeds V alone: cropped to vy -1..2,
vz 0..2.7 (the cycles stay inside vy -0.8..1.83, vz 0.19..2.47), as float32. Against pumpdp's
fly.npy (as of 2026-10-18 21:27), float32 moves the flown pitch on 48 of the first 3000 ticks at
f = 1 (up to 3°, where Q is nearly flat in pitch) and the flight ends on the same state; at
f = 0.5 and 1.2 it ends within 3e-4 b/t. The bytes
are split into planes and gzipped, which halves them; the page inflates them with
DecompressionStream. runs/ is gitignored, so the built page is too.
"""

import base64
import gzip
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs/pumpdp/ablate")
# (directory, name pattern, spec key, which knob) for the two ablations
FAMILIES = [("runs/pumpdp/ablate", r"curv([\d.]+)", "conv_curv", "f"), ("runs/pumpdp/ablate_exp", r"exp([\d.]+)", "conv_lift_exp", "k")]
DISCOUNT = 0.9996666666666667
# The crop, in node indices of pumpdp's default grid (vy -2..2.8, vz 0..3.6, dv 0.01).
J0, J1, K1 = 100, 400, 270


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RUNS, "dp_explorer.html")
    runs = []
    for (fam_dir, pat, key, knob), name in [(fam, n) for fam in FAMILIES for n in os.listdir(os.path.join(ROOT, fam[0]))]:
        m = re.fullmatch(pat, name)
        d = os.path.join(ROOT, fam_dir, name)
        if not m or not os.path.exists(os.path.join(d, "spec.txt")):
            continue
        x = float(m.group(1))
        if knob == "k" and x == 1:
            continue
        spec = open(os.path.join(d, "spec.txt")).read()
        assert f"discount {DISCOUNT} " in spec and "grid dv 0.01 vy -2..2.8 vz 0..3.6" in spec and "limit 89" in spec, f"{name}: not the standard run"
        assert (x == 1) == ("conv_" not in spec) and (x == 1 or f"{key} {m.group(1)}" in spec), f"{name}: spec disagrees on {knob}"
        f, k = (x, 1.0) if knob == "f" else (1.0, x)
        vy = np.load(os.path.join(d, "vy.npy"))
        assert abs(vy[J0] + 1) < 1e-12
        V = np.load(os.path.join(d, "V.npy"))[J0:J1 + 1, :K1 + 1].astype("<f4")
        planes = np.ascontiguousarray(V).view(np.uint8).reshape(-1, 4).T.copy().tobytes()
        runs.append({"f": f, "k": k, "rate": float(re.search(r"\(([\d.-]+) b/s\)", spec).group(1)),
                     "period": int(re.search(r"period (\d+)", spec).group(1)) if "period none" not in spec else None,
                     "V": base64.b64encode(gzip.compress(planes, 9)).decode()})
    runs.sort(key=lambda r: (r["k"] != 1, r["f"], -r["k"]))
    assert any(r["f"] == 1 and r["k"] == 1 for r in runs), "need the game's run, curv1"
    data = {"J0": J0, "ny": J1 - J0 + 1, "nz": K1 + 1, "vy0": -2.0, "dv": 0.01, "gam": DISCOUNT,
            "ref": [0.167467, 0.200887], "runs": runs}
    html = open(os.path.join(ROOT, "tools/dp_explorer.html")).read()
    assert "/*@DATA@*/null" in html
    html = html.replace("/*@DATA@*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(f"{out}: f = {', '.join(f'{r['f']:g}' for r in runs if r['k'] == 1)}; "
          f"k = {', '.join(f'{r['k']:g}' for r in runs if r['k'] != 1)}; {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
