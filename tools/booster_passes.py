"""One line per booster pass in a `booster_log` session: the motion packets it sent, their speed,
their largest component, and the look the server read.

The question it is for: does a booster faster than 3.9 b/t lose speed when aimed along a world
axis? Motion packets before 1.21.9 (and Spigot/Paper's setVelocity) clamp each component to
+-3.9 b/t, so a launch along an axis is capped at 3.9 there while a diagonal one is not. A clamped
pass shows `max|c|` at 3.9 and `|v|` below the booster's speed.

    python3 tools/booster_passes.py <log dir>/<stamp>_<tag>   # without .csv / .packets.txt
"""
import csv
import math
import re
import sys

LAG = 4      # ticks between the look the server reads and the packet (README-booster.md)
GAP = 10     # packets further apart than this are separate passes
CLAMP = 3.9

base = sys.argv[1]
player_id = None
packets = []
for line in open(base + ".packets.txt"):
    tick, name, rest = line.rstrip("\n").split(",", 2)
    if name != "ClientboundSetEntityMotionPacket":
        continue
    pid = int(re.search(r"id=(-?\d+)", rest).group(1))
    v = tuple(float(x) for x in re.search(r"movement=\(([^)]*)\)", rest).group(1).split(","))
    packets.append((int(tick), pid, v))

# The player is the id with the most motion packets; other entities' packets are noise here.
ids = {}
for _, pid, _ in packets:
    ids[pid] = ids.get(pid, 0) + 1
player_id = max(ids, key=ids.get)
packets = [(t, v) for t, pid, v in packets if pid == player_id]

rows = {int(r["game_tick"]): r for r in csv.DictReader(open(base + ".csv"))}

passes = []
for t, v in packets:
    if passes and t - passes[-1][-1][0] <= GAP:
        passes[-1].append((t, v))
    else:
        passes.append([(t, v)])

print("pass,first_tick,n,speed,max_abs_comp,clamped,look_yaw,look_pitch,launch_yaw,launch_pitch,"
      "speed_next_tick")
for i, ps in enumerate(passes):
    t, v = ps[-1]                       # the last packet sets the launch
    speed = math.sqrt(sum(c * c for c in v))
    big = max(abs(c) for c in v)
    look = rows.get(t - LAG)
    after = rows.get(t + 1)
    # Minecraft yaw: 0 is +z, 90 is -x. Pitch negative nose-up.
    lyaw = math.degrees(math.atan2(-v[0], v[2]))
    lpitch = -math.degrees(math.asin(v[1] / speed))
    nxt = math.sqrt(sum(float(after[k]) ** 2 for k in ("vx", "vy", "vz"))) if after else float("nan")
    print(f"{i},{ps[0][0]},{len(ps)},{speed:.5f},{big:.5f},{int(big > CLAMP - 0.01)},"
          f"{float(look['yaw']) if look else float('nan'):.2f},"
          f"{float(look['pitch']) if look else float('nan'):.2f},"
          f"{lyaw:.2f},{lpitch:.2f},{nxt:.5f}")
