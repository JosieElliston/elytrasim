# Boosters on EMC, and the climb out of them

Measured on the elytramc.net ("EMC") server on 2026-09-27, from one logged session on the
course *goldrush* (28 passes through one booster) and two HUD screenshots on *Monster*, and on
2026-10-17 from one logged session on *classic_urban_slop* (a 5 b/t booster, approached both along
and across the course). Physics throughout is `sim` with `mth_lut` trig, yaw pinned,
`|pitch| <= 89`.

Every angle here is a Minecraft pitch, as F3 shows it: degrees, **negative is nose up**, −90
straight up. A launch's flight path is `−pitch` above horizontal. The code, the run data and the
explorer use the same convention.

## How an EMC booster works

A booster is a server-side velocity set, nothing else:

- **It sends `ClientboundSetEntityMotionPacket` for the player, with a fixed speed per booster.**
  On goldrush all 84 booster packets have `|v|` between 1.64980 and 1.65002 b/t (33.00 b/s), so
  the speed is a setting of the booster, not a function of how you arrive.
- **Each world-axis component of that velocity is clamped to ±3.9 b/t**, so above 3.9 b/t
  the speed you get depends on which way you look. Motion packets before Minecraft 1.21.9 clamp
  this way (and so does Spigot/Paper's `setVelocity`); EMC sits behind a proxy that accepts 1.8.
  On *classic_urban_slop* every one of the 55 booster packets is `clamp(5.0 * look(t - lag), ±3.9)`
  to within 2e-4 b/t, while `|v| * look` misses by up to 0.25. 42 of the 55 had a component at
  3.8999, and their `|v|` ran from 3.94 to 4.99. Looks near a world axis got 3.9–4.4 b/t,
  depending on pitch. A look far enough between the axes, with no component over 3.9, gets the
  full 5.
  This is why a strong booster is faster approached from the side: a course built along an axis
  puts a straight boost on one component.
- **The direction is your look direction as the server last saw it, 3–4 ticks earlier.** Matching
  each packet's direction against the logged look at `k` ticks before it arrived, the median
  angle between them is 0 at every lag from 1 to 4, but the p90 falls from 33° at lag 0 to 3.4°
  at lag 4 and rises again past it. The lag is round-trip ping plus a server tick; the HUD showed
  108 and 196 ms of latency during the two sessions. **Turning in the last ~200 ms before the
  boost does not count.**
- **It re-sends every tick you are inside the booster volume**, 2–5 packets per pass on goldrush.
  Normal elytra physics runs between packets, and the last packet sets the launch.
- So a pass through a booster is: `v := clamp(|v_b| * look(t - lag), ±3.9)` per component,
  repeated while inside. The launch pitch and yaw are a free choice, made by where you look ~4
  ticks before you leave the booster. Below 3.9 b/t the clamp never binds.

What is not a booster:

- A course reset is a `ClientboundPlayerPositionPacket` to spawn.
- Eight other motion packets (0.15–0.91 b/t) appeared in the goldrush session. All of them point
  exactly along a world axis (yaw ±90 or 180, or straight down), and some land mid-boost.
  *classic_urban_slop* had 13 (0.28–3.9 b/t), all within ~10 ticks after a pass; each has at
  least one component exactly 0, but some have two nonzero (e.g. `(0, −1.16, 3.44)`), so "a
  world axis" is really "a world-axis plane". They are unexplained; collision handling zeroing
  the velocity into a wall is the guess.

**The sim matches EMC's flight physics.** Predicting each logged tick's velocity from the one
before, with that tick's logged pitch, misses by 4e-6 b/t median over 229 gliding ticks. The large
misses are all wall hits. The pitch on a log row is the pitch that tick's physics used.

`|v|` for *Monster*'s booster is inferred, not logged: back-solving the HUD's post-boost velocity
(+7.64 b/s up, 58.30 b/s horizontal) through one tick of flight gives 2.99 b/t, so **3.0 b/t
(60 b/s)**, launched at a look pitch of about −5.5. More ticks between boost and screenshot give 3.04–3.12.

## The best approach pitch

The "approach pitch" is the look pitch the booster reads, which sets the launch direction. For
each booster speed and approach pitch, the pitch schedule that maximizes apex height was solved
(`examples/booster.rs`, `apex_opt`):

| booster `|v|`, b/t | best approach | apex, blocks | time to apex |
|---|---|---|---|
| ≤ 0.6 | −90 (straight up) | ≤ 6.2 | ≤ 1.15 s |
| 1.0 | −44.0 | 17.8 | 2.30 s |
| 1.65 (goldrush) | ~−30.5 | ~58 | ~3.8 s |
| 2.0 | −27.9 | 86.4 | 4.45 s |
| 3.0 (Monster) | −24.4 | 184.5 | 5.65 s |
| 4.0 | −23.0 | 296.4 | 6.50 s |
| 5.0 | −22.2 | 415.9 | 7.15 s |

**Every result in this section ignores the ±3.9 clamp**: `examples/booster.rs` launches at
`|v| * look`, unclamped. Up to 4.0 b/t the table holds at any yaw, because no component of these
launches reaches 3.9. At 5.0 b/t the best launch has 4.63 b/t horizontal, which the clamp leaves
alone only with the look at least 32.6° off the nearest world axis. Along an axis it is capped at
3.9, so the row overstates the apex there, and the best approach is a different pitch, and is
probably steeper. That optimum has not been solved, and neither has the best yaw. The
asymptote below is the unclamped problem by construction. Under the clamp, no launch exceeds
3.9√3 ≈ 6.75 b/t.

The optimum is flat: from 1 b/t up, missing the best approach by 3° costs under 0.4% of the apex
and by 5° under 1.5% (at 3.0 b/t, 0.5 and 1.0 blocks).

**The best approach pitch tends to −19.63 as `|v|` grows.** Every term of the elytra velocity
update is linear in velocity except gravity's `g(0.75 cos²p − 1)`, so scaling `|v|` by `λ` is the
same flight as dividing `g` by `λ`: the best approach is a function of `g/|v|` alone, and its
limit is the gravity-free problem. Solved out to 50 b/t (`runs/booster/asymptote/`, 1° grid over
pitch −14 to −24, quadratic refine):

| `|v|`, b/t | 3 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|
| best approach pitch | −24.43 | −22.24 | −20.85 | −20.22 | −20.02 | −19.86 |

Fitting a quadratic in `g/|v|` over `|v|` ≥ 6 gives

    p* ≈ −19.63° − 11.4°/|v| − 8.0°/|v|²        (|v| in b/t)

with residuals of 0.001°. The fitted limit moves by under 0.02° across fit windows (≥ 3, 6, 10)
and between the quadratic and a linear fit on the upper range. It is an extrapolation: the
gravity-free problem was not solved directly.

**Flying the climb:** after the launch, the HUD's gain marker (`argmax ΔTE` held 20 ticks) and
the closed-form gain law (`s(s + A) = K v_z`, `K = 0.771`) both fly within 0.3 blocks of the
optimal apex from every level or upward launch at every speed. They fail on downward launches: at
pitch +5 they lose up to 3 blocks, because the optimum first holds pitch 0 until `v_z` stops rising
(the cycle's snap, `vz_peaked`) and neither marker does. Lookahead 1 is not a climb marker after a
shallow boost: it stays level whenever `v_y/v_z < 0.036/0.128 ≈ 0.28`, because the first nose-up
tick loses energy there.

**Racing to a height instead of the apex.** A target reached sooner than the apex wants a more
nose-up climb than the gain marker, whose 20-tick horizon is sized for the whole climb. The
Monster ceiling, ~180 blocks above the booster and only 4.5 blocks under the best apex, is not that
case: the fastest schedule (approach pitch −25) reaches it at tick 92, and the apex climb and both
markers at tick 93.

**Never aim exactly -90.** Vanilla's `Mth.cos(-90°)` is exactly 0, which skips every lift and
conversion term, and the mouse stop pins pitch there. The sim's libm `cos` is -4.4e-8, which
points the look backwards and costs ~17% of forward speed a tick. `gain_law_pitch` returns exactly
-90 above `v_z ≈ 2.03`, so clamp it (everything here uses 89).

## Reproducing

- `tools/minescript/booster_log.pyj` — the in-game logger (Minescript 5, Pyjinn). Copy it into the
  profile's `minescript/` and run `\booster_log <tag>`; it writes one row per client tick
  (position, velocity, look, and elytra-vario's own markers) and every velocity packet with its
  arrival tick to `minescript/booster_logs/`.
- `tools/booster_passes.py` — one line per pass from a `booster_log` session: the packets' speed,
  their largest component (3.8999 means clamped), the look the server read, and the launch.
- `examples/booster.rs` — every analysis above (unclamped); its header lists the modes
  (`explore`, `cell`, `launch`, `ceiling`, `race`, `replay`, `flights`, `lag`, `trace`). Set
  `TRIG=mth_lut LIM=89`; `FLIGHT=algebraic` is identical and 3.6x faster.
- `tools/booster_sweep.sbatch` with `tools/booster_solve.sh` (explorer cells) or
  `tools/booster_cell.sh` (apex only) — the cluster runs, one array task per node, work list
  `$RUN/work.txt` of `<speed> <pitch>` lines. The 700-cell explorer took 3.3 minutes on 16 nodes.
- `tools/plot_booster_explorer.py` — `runs/booster/explorer/booster-explorer.html`, the interactive
  explorer (speed slider; every approach's profile at that speed, the best-apex one highlighted;
  apex, time, distance and forward speed against approach; best approach against speed; marker
  losses). Written by Codex from `runs/booster/explorer/BRIEF.md`, which predates the switch to
  pitch: its angles are degrees above horizontal and its cells are named `v<speed>_g<angle>`.
- Data: `runs/booster/` (gitignored). `explorer/cells/` is the 700-cell grid (`|v|` 0.2–5.0 by 0.2;
  approach pitch +5, 0 to −90 by 5, −20 to −30 by 1); `asymptote/` is the high-speed grid. Both
  are unclamped, so a cell is right on EMC only for a yaw that keeps every component under 3.9.
