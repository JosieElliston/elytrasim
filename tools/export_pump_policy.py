#!/usr/bin/env python3
"""Pack a pumpdp V for elytra-vario's PumpPolicy, and write test vectors for it.

    python3 tools/export_pump_policy.py RUN OUT.bin [--every=S] [--vectors=CSV]

Keeps every S-th node (default 4: dv 0.0025 -> 0.01; tools/pumpdp_subsample.py measures what that
costs, nothing as of 2026-10-22). V is quantized to 1e-6 blocks, as int32 second differences
along vz per vy row (each row's first value raw, then its first difference), then deflated:
0.8 MB a table against 1.3 MB for deflated float32, with the same pitch on every lap tick.

Layout, big-endian (Java's DataInputStream), then a zlib stream of ny * nz int32:
    magic 'EVPV', int version 1, int ny, int nz,
    double vy0, dv, gamma, w, conv_lift_exp, quantum, pitch_limit, pitch_step

--vectors writes, from the table as decoded here, the pitch pumpdp's lookahead picks at every
tick of the run's own lap: rows vy, vz, pitch. PumpPolicyTest replays them.
"""

import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pump_policies import Run
from pumpdp_subsample import coarsen

QUANTUM = 1e-6


def pack(run, s):
    c = coarsen(run, s)
    q = np.round(c.V / QUANTUM).astype(np.int64)
    d = q.copy()
    d[:, 1:] = np.diff(q, axis=1)
    d[:, 2:] = np.diff(d[:, 1:], axis=1)
    assert np.abs(d).max() < 2**31
    ny, nz = q.shape
    head = b"EVPV" + struct.pack(">iii", 1, ny, nz) + struct.pack(
        ">8d", c.vy[0], c.vy[1] - c.vy[0], c.gamma, c.w, c.k, QUANTUM, c.limit, c.fine)
    return head + zlib.compress(d.astype(">i4").tobytes(), 9)


def unpack(blob, like):
    """The table as PumpPolicy will hold it, as a Run like `like`."""
    ny, nz = struct.unpack(">ii", blob[8:16])
    vy0, dv, _, _, _, quantum, _, _ = struct.unpack(">8d", blob[16:80])
    d = np.frombuffer(zlib.decompress(blob[80:]), dtype=">i4").astype(np.int64).reshape(ny, nz)
    first = np.cumsum(d[:, 1:], axis=1)   # the first differences, back from the second
    q = np.concatenate([d[:, :1], d[:, :1] + np.cumsum(first, axis=1)], axis=1)
    c = Run.__new__(Run)
    c.__dict__.update(like.__dict__)
    c.vy, c.vz, c.V = vy0 + dv * np.arange(ny), dv * np.arange(nz), q * quantum
    return c


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    run, out = Run(args[0]), args[1]
    s = int(opts.get("every", 4))
    blob = pack(run, s)
    shipped = unpack(blob, run)
    ref = coarsen(run, s).V
    assert np.abs(shipped.V - ref).max() <= QUANTUM / 2 + 1e-9, "round trip"
    with open(out, "wb") as f:
        f.write(blob)
    print(f"{out}: {len(blob) / 1e6:.2f} MB, {shipped.V.shape}, dv {shipped.vy[1] - shipped.vy[0]:g}, "
          f"vy {shipped.vy[0]:g}..{shipped.vy[-1]:g}, vz 0..{shipped.vz[-1]:g}, lambda {run.lam:g}")
    if "vectors" in opts:
        lap = run.lap()
        p = shipped.policy(lap[:, 1], lap[:, 2])
        same = np.mean(p == lap[:, 3])
        np.savetxt(opts["vectors"], np.column_stack([lap[:, 1], lap[:, 2], p]), fmt="%.17g",
                   delimiter=",", header=f"{os.path.basename(out)} on its lap (lambda {run.lam:g}): vy,vz,pitch",
                   comments="# ")
        print(f"{opts['vectors']}: {len(lap)} ticks, {same:.0%} equal to the full-resolution pitch")


if __name__ == "__main__":
    main()
