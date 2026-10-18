#!/usr/bin/env python3
"""Build the pump policy explorer: tools/pump_explorer.html with pumpdp's policy maps embedded.

    python3 tools/build_pump_explorer.py [out.html]      # default runs/pumpdp/pump_explorer.html

Reads every run under runs/pumpdp/ named in RUNS that has finished (has spec.txt); see
src/bin/pumpdp.rs for what a run is. Each map is the policy's pitch at every node, which pumpdp
writes on a quarter-degree grid, so it is embedded exactly as int16 quarter-degrees, plus the v0
the page reads each blur's limit cycle from (the hand-flown reference's, which pumpdp flies from
too). The runs are under runs/, which is gitignored, so the built page is too.
"""

import base64
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("disc3000", "discounted, horizon 3000"), ("avg", "average reward"),
        ("disc10000", "discounted, horizon 10000"), ("disc1000", "discounted, horizon 1000"),
        ("disc300", "discounted, horizon 300"), ("disc100", "discounted, horizon 100"),
        ("disc3000_fine", "discounted, horizon 3000, fine grid"), ("avg_fine", "average reward, fine grid")]
# pumpdp's default --v0: the hand-flown reference cycle's start, (vy, vz) blocks/tick.
V0 = (0.167467, 0.200887)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs/pumpdp/pump_explorer.html")
    maps = []
    for name, label in RUNS:
        d = os.path.join(ROOT, "runs/pumpdp", name)
        if not os.path.exists(os.path.join(d, "spec.txt")):
            continue
        vy, vz = np.load(os.path.join(d, "vy.npy")), np.load(os.path.join(d, "vz.npy"))
        q = np.round(np.load(os.path.join(d, "pitch.npy")) * 4)
        assert np.abs(q / 4 - np.load(os.path.join(d, "pitch.npy"))).max() < 1e-9, f"{name}: pitch is off the quarter-degree grid"
        spec = open(os.path.join(d, "spec.txt")).read()
        m = re.search(r"\(([\d.]+) b/s\)", spec)
        maps.append({"name": name, "label": label, "vy0": float(vy[0]), "dv": float(vy[1] - vy[0]),
                     "ny": len(vy), "nz": len(vz), "solverRate": float(m.group(1)) if m else None,
                     "pitch": base64.b64encode(q.astype("<i2").tobytes()).decode()})
        assert abs(vz[0]) < 1e-12 and abs((vz[1] - vz[0]) - (vy[1] - vy[0])) < 1e-12
    if not maps:
        sys.exit("no finished runs under runs/pumpdp/")
    data = {"maps": maps, "v0": V0}
    html = open(os.path.join(ROOT, "tools/pump_explorer.html")).read()
    assert "/*@DATA@*/null" in html
    html = html.replace("/*@DATA@*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(f"{out}: {len(maps)} maps ({', '.join(m['name'] for m in maps)}), {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
