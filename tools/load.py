"""Read a sweep corpus. Standard library only, so it runs anywhere the files land.

    from load import Profile, load_dir

    ps = load_dir("sweep")
    p  = ps[0]
    p.header["lambda"], p.pitches          # what it was optimized for, and the answer
    y, z, vy, vz = p.replay()              # re-derive the trajectory from the pitches alone

The point of a profile is that it is checkable: the header states the utility function and the
initial conditions completely, so `replay()` reproduces every number in the header from the
pitches. `verify_header()` does exactly that and is the first thing to run on a corpus that
came off a different machine.
"""

import math
import os

GRAVITY = 0.08


class Profile:
    __slots__ = ("path", "header", "pitches")

    def __init__(self, path, header, pitches):
        self.path, self.header, self.pitches = path, header, pitches

    # ---- header access

    @property
    def n(self):
        return int(self.header["n"])

    @property
    def lam(self):
        return float(self.header["lambda"])

    @property
    def w(self):
        return float(self.header["w"])

    @property
    def v0(self):
        vy, vz = self.header["v0"].split()[:2]
        return float(vy), float(vz)

    @property
    def trig(self):
        return self.header["trig"]

    # ---- physics

    def replay(self):
        """Positions and velocities per tick, from the pitches alone.

        This is a Python transcription of `update_fall_flying_movement` with yaw pinned to
        zero. It uses the platform's trig, so it will not be bit-identical to a profile
        written under `trig = mth_lut`; it is for analysis, not for certification. Use
        `sweep verify --optimality` when the exact optimum matters.
        """
        vy, vz = self.v0
        y = z = 0.0
        ys, zs, vys, vzs = [], [], [], []
        # the drag constants are f32 in the sim, and 0.99f32 is not 0.99
        drag_y, drag_z = 0.9800000190734863, 0.9900000095367432
        for pitch in self.pitches:
            lean = math.radians(pitch)
            look_z = math.cos(lean)          # with yaw pinned, the look vector is (0, ., cos)
            look_hor = abs(look_z)
            # captured before any of this tick's updates, exactly as the sim does
            move_hor = abs(vz)
            lift = math.cos(lean) ** 2
            vy += GRAVITY * (-1.0 + lift * 0.75)
            if vy < 0.0 and look_hor > 0.0:                    # descent -> forward
                conv = vy * -0.1 * lift
                vy += conv
                vz += look_z * conv / look_hor
            if lean < 0.0 and look_hor > 0.0:                  # forward -> up, nose above level
                conv = move_hor * -math.sin(lean) * 0.04
                vy += conv * 3.2
                vz -= look_z * conv / look_hor
            if look_hor > 0.0:                                 # turning
                vz += (look_z / look_hor * move_hor - vz) * 0.1
            vy *= drag_y
            vz *= drag_z
            y += vy
            z += vz
            ys.append(y); zs.append(z); vys.append(vy); vzs.append(vz)
        return ys, zs, vys, vzs

    def energy(self, vy, vz, y):
        """Total energy in blocks: the height this state could reach."""
        return (vy * vy + vz * vz) * 0.5 / GRAVITY + y

    def verify_header(self, tol=1e-3):
        """Re-derive dy and dz from the pitches and compare with the header.

        Returns (ok, message). A profile written under `mth_lut` will disagree slightly,
        because this file uses the platform's trig rather than Minecraft's table -- that is
        the expected size of the disagreement, not an error, so `tol` is loose by default.
        """
        ys, zs, _, _ = self.replay()
        out = []
        for key, got in (("dy", ys[-1]), ("dz", zs[-1])):
            want = float(self.header[key])
            if abs(want - got) > tol * max(1.0, abs(want)):
                out.append(f"{key}: header {want:.6f}, replay {got:.6f}")
        return (not out, "; ".join(out) or "ok")


def parse(text, path="<memory>"):
    header, pitches = {}, []
    for line in text.splitlines():
        body, _, _ = line.partition("#")
        if line.lstrip().startswith("#"):
            # "# key   value    # trailing note"
            rest = line.lstrip()[1:].strip()
            if not rest:
                continue
            key, _, value = rest.partition(" ")
            value = value.partition("#")[0].strip()
            if value:
                header[key] = value
            continue
        pitches.extend(float(t) for t in body.split())
    return Profile(path, header, pitches)


def load(path):
    with open(path) as f:
        return parse(f.read(), path)


def load_dir(root, pattern=".pitches"):
    """Every profile under `root`, sorted by (n, lambda, vy0, vz0)."""
    out = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith(pattern):
                out.append(load(os.path.join(dirpath, name)))
    out.sort(key=lambda p: (p.n, p.lam) + p.v0)
    return out


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    ps = load_dir(root)
    print(f"{len(ps)} profiles under {root}")
    print(f"{'n':>5} {'lambda':>8} {'vy0':>7} {'vz0':>7} {'dy':>9} {'dz':>9} {'check':>7}")
    for p in ps[:20]:
        ok, msg = p.verify_header()
        print(f"{p.n:>5} {p.lam:>8.3f} {p.v0[0]:>7.3f} {p.v0[1]:>7.3f} "
              f"{float(p.header['dy']):>9.3f} {float(p.header['dz']):>9.2f} "
              f"{'ok' if ok else 'DIFFERS':>7}")
    if len(ps) > 20:
        print(f"... and {len(ps) - 20} more")
