# The optimal-profile corpus

A grid of optimal elytra pitch schedules, one file per cell, over
`(num_ticks, lambda, vy0, vz0)`.

Each file states the utility function and initial conditions it was optimized under, and its
claim is checkable by re-running the optimizer's own stopping test against the file alone:

```
sweep verify --optimality runs/corpus/vy+0.0000_vz+0.0000/n0300_lam+0.000000.pitches
```

Plain `sweep verify` is the cheap check: it replays the file and compares the end state with
the rollout its header records (`dJ`, `dte`, `dy`, `dz`, `v_end`), with no search involved.

How a schedule was reached is deliberately not recorded. Given `(v0, pitches)` the trajectory is
fully determined, so every derived quantity is recomputable and the optimality claim is
falsifiable without trusting the pipeline that produced it.

## The objective

```
KE = |v|^2 / (2 * GRAVITY)     PE = y     TE = KE + PE        (blocks)
J  = TE(s_n) + w * z_n                                        (terminal velocity free)
w  = lambda * Y_REF / Z_REF    Y_REF = 21.5   Z_REF = 330
```

`lambda` normalizes the reference cycle's 21.5 blocks of climb against its 330 blocks of
distance, so at `lambda = 1` the two terms are the same scale. Negative `lambda` punishes
distance, which is the elytra-parkour side and matters at least as much as the positive side.

Pitch is `f32` because the sim casts it to `f32`, so `f32` is the exact control alphabet -- the
text in the file is the schedule that will be flown, not a rounding of it.

## Degeneracy, and why stopping time is the regularizer

Polished to convergence, the exact objective genuinely prefers a schedule no human can fly. At
pitch 90 degrees `cos(lean_angle)` is zero, so lift vanishes *and* `look_hor_length` vanishes,
gating off both the descent-to-forward conversion and the turning term; bang-bang between "full
aero" and "no aero" beats any fixed pitch on the singular arc. This is not a convergence
failure: 200 passes reaches residual 9.4e-6, a real coordinate optimum.

**Total variation cannot detect it**, because a real flick is also a large move -- the cycle
drops to 0 and covers ~88 degrees in about six ticks. What separates them is whether
neighboring ticks move *together*. `lag1`, the lag-1 correlation of the per-tick changes, is
the statistic that does: the reference cycle reads +0.48, the same schedule polished 200 passes
reads -0.74, and the sign flip is the diagnosis. It is in every header.

Things that were tried and measured:

| lever | result |
|---|---|
| slew-rate limit | no. Any limit loose enough to keep the real flicks keeps the chatter. |
| per-tick pitch jitter | no. Held fixed it made chatter *worse* (TV 2406 -> 3493); resampled it cut TV 29% for 2.3 blocks of climb. The chattering end is pitch-insensitive -- `lift_force = cos^2(lean_angle)` is flat at 90 degrees. |
| initial-velocity jitter, polished hard | no. At 60 forced passes TV is 925, 871, 838, 826, 884 for sigma 0, 0.05, 0.1, 0.2, 0.4 -- flat across an 8x range, while `dz` falls monotonically 345 -> 337. |
| initial-velocity jitter, stopped early | **yes**, but only while stopped early -- which leaves the answer depending on when the optimizer was interrupted. Superseded. |
| stopping time | **yes**, and it was what bounded it before the curvature price. Same objection: the answer is wherever the polish was cut off. |
| l1 curvature price (`Rough::mu`) | **yes, and this is what is used now.** A price rather than a stopping rule, so the polish runs to convergence and the result is a real optimum that `verify --optimality` can re-check from the header. `runs/atlas` at `mu = 1e-4`, `runs/antichatter` at `mu = 1e-3`, both with `limit = 85` and `jitter 0`. |

lag-1 against pass count from the reference cycle at `n = 300`, `sigma = 0`:

```
passes    0     1     2     3     4     8    16    32    60   200
lag-1  +0.48 -0.09 +0.01 +0.05 +0.13 +0.27 +0.29 +0.07 -0.27 -0.74
```

It is *not monotone*. The dip in the first few passes is the entry dive forming, which is
legitimate structure; the decay after ~16 passes is the chatter. Under `sigma = 0.1` the early
dip stays above 0.2 (+0.32, +0.42, +0.57, +0.53), which is what makes an absolute floor usable
as a stopping rule -- it catches the degeneracy without firing on the transient.

Nearly all the honest gain is early: `dJ` runs 21.50, 21.54, 21.58, 21.77 at 0, 4, 32 and 200
passes. The last 0.2 blocks is bought entirely with chatter.

### Continuation compounds

A warm-started cell inherits its neighbor's polish, so a per-cell pass budget accumulates along
the continuation path. Measured on `n = 300 -> 310 -> 320` at 8 passes each:

```
lag-1  +0.73  +0.59  -0.07
TV      293    361    600
```

Degenerate by the second hop. A pass budget cannot bound this; a property of the schedule can,
which is what `--lag1-floor` is for. It stops a polish that crosses the floor and *discards the
crossing pass* -- on that same `n = 320` cell it stops at 2 passes with lag-1 +0.217, TV 505,
and `dy` 19.02 against the unguarded 18.91. The discarded passes were buying chatter, not
height.

## The entry is under-determined

Splicing three very different entries (ticks 0-24) onto one shared tail moves `J` by 0.047
blocks total -- reference 21.942, an extreme dip 21.989, a gentler one 21.947 -- and the choice
of tail changes nothing (3.5e-5). The entry is a flat plateau, consistent with the 5.8e-7
curvature and 2.0-degree half-width measured in `README-myopic.md`. So an entry that looks
alarming is not necessarily wrong, and imposing smoothness there costs almost nothing.

More generally: the dive's score is 15x flatter than the gain's and the entry's flatter still,
while the snap is ~3000x more pitch-sensitive. **Any analysis reading pitch differences between
cells as signal has to check the `J` difference first.** Convergence is tested on `J`, never on
pitch.

## Running it

```
sweep polish --n 300 --lambda 0 --vy 0 --vz 0 --jitter 0.10 --draws 8 \
             --passes 8 --tol 0 --lag1-floor 0.2 --flight algebraic \
             --anchor <file> --out <file>
sweep run    --out <dir> --ns <list> --lams <list> --vys <list> --vzs <list> [--shard i,j]
sweep verify [--optimality] <file>...
sweep fingerprint
```

`--mu <price>` is the curvature price and is the current chatter regularizer; `--penalty` sets its
shape, `l2:2` by default since 2026-09-24 and l1 before. `runs/atlas` was built at
`--mu 1e-4 --limit 85` under l1, so extending or re-checking it now needs `--penalty l1`. The `--jitter` shown above is the older generation's
regularizer and current runs set it to 0 -- see the table further up.

A shard is one `(vy0, vz0)` cell, so it is a directory and an independent job. Inside a shard,
continuation runs breadth-first over the `(n, lambda)` plane from the cell nearest `n = 300`,
`lambda = 0`; `--anchor` seeds that one cell from a file. This matters: at `v0 = 0` the built-in
policy seed lands degenerate (lag-1 -0.19 at 8 passes) where the reference cycle gives +0.73,
and the anchor propagates to the whole shard. Once the `v0 = 0` shard exists, anchor the
nonzero-velocity shards on its *solved* `n = 300, lambda = 0` cell rather than on the raw
reference file: it is the nearest solved point, so continuation starts closer and is less
likely to pick a bad branch.

Resume is by file existence -- a cell whose file matches `(n, lambda, v0, trig, flight)` is not redone
but still seeds its neighbors, so a killed job costs at most one cell.

`--flight reference|algebraic` selects the movement kernel. `reference` remains the default;
`algebraic` routes yaw-zero movement through the collapsed equations and falls back to the
reference path for nonzero yaw. Profiles and manifests record the route, and resume also
requires it to match.

**A seed does not carry its route, so a mismatched one is refused.** `--init` and `--anchor`
take only the pitches; the route stays whatever `--trig`/`--flight` say, defaults included.
Seeding from an `mth_lut`/`algebraic` profile while the process sits on `libm`/`reference`
therefore solves a different problem than the seed solved, and nothing downstream can see it --
every header truthfully records the route it was actually run under, so the files come out
self-consistent and quietly wrong about which question they answer. It is not a rounding
detail: re-polishing one `runs/atlas` cell under the defaults took 186 passes and landed on
fixed point (0.415215, 0.330157) where its own route takes 156 and lands on
(0.415055, 0.330120).

So it is a hard error, with no override flag: matching the seed is what `--trig`/`--flight` are
for, and cross-route seeding has no legitimate use. The escape hatch for the rare case that
wants it already exists -- a headerless pitch list claims no route and is accepted as-is. On an Apple arm64 test machine, a representative `dp` transition-table
build fell from 0.121 seconds to 0.057 seconds (2.12x); the generated value table and schedule
were byte-identical.

That 2.12x was measured before pitch trigonometry was cached, and caching has since taken most
of it back -- `PitchTrig` hoists the sin/cos out of *both* routes, and eliminating that trig was
most of what the algebraic route was winning. What is left is three divisions and a sqrt per
tick, and whether removing them pays depends on the machine. Measured on `sweep polish`, one
n=300 cell at 30 passes, single core:

```
                          reference   algebraic
  Apple M5 (arm64)           7.15 s      7.98 s     algebraic 1.12x SLOWER
  Intel i7-6700 (Skylake)   14.11 s      6.93 s     algebraic 2.04x faster
```

On Skylake f64 division is long-latency and poorly pipelined, so dropping three of them per
tick is worth 2x; on the M5 division is cheap and the extra branches the collapsed form needs
(`vel.x == 0.0`, `is_sign_negative`) cost more than they save. So pick the route for the
machine: `reference` on arm64, `algebraic` on the cluster. Both are recorded in the header, and
`verify` replays under the route the file claims, so a corpus may mix them without ambiguity --
though a single corpus is easier to reason about if it does not.

Check a corpus with `python3 tools/check.py <dir>`; read profiles from Python with
`tools/load.py`.

**What `runs/corpus` actually is.** It was built in four phases -- the `n` axis at stride 10,
then `lambda` at `n` stride 50, then the velocity corners, then `lambda` at those corners --
over `n = 100..600` and `lambda = -4..4` stride 0.25, with `--trig libm` and
`--passes 8 --tol 0 --jitter 0.10 --draws 8`. **The lag-1 floor was off** (`--lag1-floor -2`,
below the -1 a correlation can reach), so every cell ran all 8 passes; all 509 profiles read
`(8 passes)`, which is how you can confirm it from the corpus alone. That is worth stating
because the synopsis above shows `--lag1-floor 0.2` and the section on degeneracy argues for
it: the guard is the recommendation, it is not what produced this corpus, and a re-run with it
on would not be comparable. The per-phase manifests record the cells of each phase:
`manifest_a_n.json`, `manifest_b_lambda.json`, `manifest_c_vel.json`.

## Steady state

`--steady` optimizes the cycle you would fly *back to back*. It re-solves `v0` to the schedule's
own fixed point -- the fixed point of `v -> (replay the schedule from v).vel` -- after every
pass, so at convergence `v_final == v0` identically and no constraint or penalty is needed to
make the schedule repeatable.

It does that by moving `v0` and nothing else. The objective stays `J = TE(s_n) + w*z_n` with
`v_n` free, and the roughness price is not wrapped across the cut either -- pricing the
`p[n-1] -> p[0]` jump was tried and reverted at `fad211d`, for moving utility by at most 0.016
blocks. Both are deliberate: a steady profile is a *repeatable open-horizon optimum*, not the
optimum of the periodic problem, and the two differ wherever the free terminal velocity has
something to say. Two places it shows up, both worth knowing before measuring against this
corpus:

* **The seam.** The pitch jump at the cut is a move a repeated schedule asks a hand to make and
  nothing charges for it. Re-solving `v0` closes it anyway up to about `n = 376`, past which the
  schedule instead ends parked at the pitch limit and the seam stays wide;
  `tools/plot_steady_seam.py` draws both regimes, at `lambda = 0` on `runs/atlas/nsweep`. It also
  widens with the price on distance, which that plot's single lambda cannot show: over the 1233
  cyclic cells of `runs/steady/nlamsweep`, median `|p[0] - p[n-1]|` runs 7.5 deg at
  `lambda = -2` to 24.9 deg at `lambda = +7`, against a typical per-tick step of 0.28 deg falling
  to 0.03 deg across the same range.
* **The costate.** `myopic adjoint` solves a *periodic* `mu` -- the price vector of the periodic
  problem, not of the one these cells solve, whose terminal price at the cut is `dJ/dv_n`. It is
  a good stand-in at `lambda = 0` and a worse one as the price rises: the closed-form stationary
  pitch reproduces the `n = 300` cell's own climb to 2.4 deg RMS at `lambda = 0` and 8.2 deg at
  `lambda = 4`. How much of that gap is the periodic/open mismatch and how much is where the cut
  happens to fall in the climb is not separated anywhere yet.

Off by default. The single-cycle problem is the simpler object and is still the one most
questions are about.

**`--vy/--vz` change meaning under it.** They seed the fixed-point iteration rather than naming
the answer, so the profile's `v0` is whatever the schedule converged to and *that* is what the
header states.

So `v0` stops being a grid axis. A path names the coordinates that determine the answer, and a
seed that barely influences it is not one, so `run` writes steady cells under `steady/` instead
of a `vy.../vz...` shard and takes a single `--vys/--vzs` pair -- more than one is refused
rather than silently collapsed, since every shard would solve the same problems and race to
write the same files. Read `v0` from the header, as always.

Resume needs the same care: comparing objectives would never match, because a steady file's
`v0` is its own fixed point and not the cell's seed, and the whole grid would re-solve on every
resume while reporting success. It instead matches on horizon, lambda and price and then checks
that the file's stated `v0` really is the fixed point of its own pitches -- a property of the
file, so it is checked rather than trusted.

What it is worth, over 13 `runs/atlas/nsweep` cells at `lambda 0`, `mu 1e-4`, `limit 85`,
scoring each schedule at its own `v0` for one lap. The gain is U-shaped in `n`, not monotone:

```
n     150    176    200    226    250    276    300    326    350    376    400    426    450
gain +0.94  +0.80  +0.64  +0.38  +0.24  +0.13  +0.05  -0.00  -0.00  +0.01  +0.06  +0.18  +0.18
base  1.47   7.05  11.87  16.21  18.99  20.79  21.54  21.62  21.02  19.73  17.24  14.70  13.51
```

The minimum sits exactly where single-cycle `dTE` peaks. The schedules differ only in the
entry, the first 20-30 ticks, and the fixed point flips sign with horizon: `v0y` +0.415 at
`n = 150`, +0.030 at `n = 326`, -0.352 at `n = 426`. So steady state buys most where the
single-cycle answer is worst, and nothing where it is already good.

A `--steady` profile closes exactly, which is checkable from the file alone and is the point:
`examples/repeat` on one reads `sum|dv| 0.0000` on every lap.

## Horizon effects

A one-lap polish leaves `v_n` free, so it is greedy at the end of the lap. It spends the next
lap's setup on height. That bias is large and depends on where the lap is cut. Any comparison
of one-lap `dTE` against a long-horizon answer (a `pumpdp` cycle, a steady state, or another
lap length) has to remove it first.

These measurements are as of 2026-10-18. They use `libm` trig, `--limit 89` and no price,
against the `pumpdp` disc3000 cycle (253 ticks, 19.7497 blocks per lap).

**Where to cut.**
- Cut at the cycle's detrended-TE peak: the last 0-degree tick before the pushover. There a
  one-lap polish gains only +0.16 blocks over the cycle.
- Cut mid-climb and it gains up to +40 blocks, measured every 12 ticks of phase.
- So never compare one-lap numbers across cuts.

**What the end effect is.**
- Start from the same state 50 ticks before the end. From there the free end holds the nose up
  to 12 degrees higher than the long-horizon tail. The long-horizon tail eases to 0 instead,
  which turns off the forward-to-up conversion.
- The free end finishes 0.18 blocks higher. Its final velocity is rotated upward
  (`vz` -0.022, `vy` +0.020) at about the same speed (-2%).
- So it trades the horizontal speed that the next pushover needs for height.

**Removing it.**
- **Unroll.** Polish two laps back to back and keep the first. Its last 50 ticks match the
  `pumpdp` cycle within 1.4 degrees, and it ends 0.006 b/t from its start. The second lap takes
  the end effect.
- **Tie.** `examples/tied` keeps one lap's pitches and scores them flown `k` times. This only
  dilutes the end effect: the last-50 pitch gap is 5.5, 2.3 and 0.54 degrees at `k` = 2, 4 and
  8. Every copy still pays for the last copy's greed, so prefer unrolling when the question is
  about the schedule itself.
- **`--steady` does not remove it.** It closes the lap but keeps the free-end objective: it
  flies 19.60 blocks per lap, against 19.75 for the cycle it was seeded from.

**Compare late effects from a shared state.** Two schedules that each start from `v0` also
differ earlier in the lap. The free polish smooths the DP's grid aliasing in the dive, and that
alone moves 0.4-0.9 blocks between height and KE by mid-lap. Read per-tick TE differences
against a common state, or that difference gets credited to the horizon.

For scale, the real gain over the DP cycle, after smoothing its aliasing, is small. The `k` = 8
tied lap, flown repeatedly, settles at 19.7622 blocks per lap (1.5622 b/s), against 19.7497
(1.5612 b/s) for the cycle.

## Portability

Vanilla's `Mth` sin LUT is ported and selectable with `--trig`, which removes the f32 trig
divergence between platforms (LLVM fuses standalone `sinf`/`cosf` into `___sincosf_stret` on
arm64, differing by 1 ULP on ~2.6% of inputs). `lift_force` still goes through f64 `cos`, so
bit-identity across platforms is **not** guaranteed; `sweep fingerprint` makes any drift a
measured number rather than a silent one. Compare it between machines before trusting output
from a new one.
