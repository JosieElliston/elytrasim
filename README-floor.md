# Flight over a floor

How long, and how far, can you fly from height `y0` and velocity `v0` before you first go under
`y = 0`? The floor is a plane, not Minecraft's collision: nothing happens at contact except that
the flight is over. `src/bin/floor.rs` is the tool; `Floor` in `src/opt.rs` is the price the
fixed-horizon polish pays for it.

Conventions. A replay starts at `y = 0` like every other schedule in this crate, so the floor sits
at `y = -y0`. The *clearance* at tick `t` is `h_t = y_t + y0`. A schedule *survives* `k` ticks when
`h_1 .. h_k >= 0`. Endurance is the most ticks any schedule survives; range is the most `z` at
the moment of exit. Physics is `mth_lut` trig and `reference` flight throughout.

## What the optimum looks like, `v0 = 0`, `y0 = 1..8`

```
            best hold          optimized, endurance     optimized, range      hold 0      hold -13
 y0   ticks  pitch       z   t*       ticks  z(t*)     t*      ticks z(t*)   ticks  z    ticks  z
  1     12   -10.2   0.490   12.661   12    0.529      12.64   12   0.530    12  0.466   11  0.403
  2     20    -8.0   1.622   20.511   20    1.663      20.47   20   1.671    20  1.603   18  1.299
  3     27   -10.6   3.263   28.191   28    3.387      28.08   28   3.414    27  3.232   26  2.962
  4     35   -10.7   5.636   36.448   36    5.787      36.18   36   5.856    34  5.446   34  5.192
  5     44   -10.3   8.737   45.867   45    9.008      45.32   45   9.167    40  7.790   43  8.204
  6     53   -13.7  12.510   56.956   56   13.229      55.87   55  13.575    47 11.015   53 11.971
  7     65   -12.8  17.197   69.813   69   18.499      67.76   67  19.264    54 14.738   64 16.429
  8     77   -14.8  22.609   83.827   83   24.557      80.13   80  26.212    60 18.299   77 21.929
```

The table is the first run (commit 8a19cdd, l1 price in score units); with the price in energy
units the optimized cells move by at most 0.04. "Best hold" is the best constant pitch for each column separately (0.1-degree grid), so its `z` is
not flown at its ticks' pitch. `t*` is the interpolated exit tick; see below. Every optimized cell
was reached from three different starts -- hold -13, hold 0, and a dive-then-pull -- that agree to
0.003 ticks in `t*` and 0.003 blocks in `z`.

What the optimizer buys over the best constant pitch is nothing at `y0 <= 2`, one tick at 3 to 5,
and six ticks (8%) and 3.6 blocks of range (16%) at `y0 = 8`. The gain grows with height.

**Below `y0` about 10-14 there is no dive and no climb:** `v_y` never goes positive.

* **Endurance** is one ramp: pitch 0 at the first tick, then nose-up at a near-constant rate --
  about -0.25 deg/tick at `y0 = 8` -- to about -22 at the exit.
* **Range** is a hold then a pull: pitch -0.0055 for 36 ticks at `y0 = 8`, then nose-up at
  0.6-1 deg/tick to -23, held to the exit. The hold sits just past the corner at 0, where the
  forward-to-up branch switches on.

The "look down" half of a mini pump is **pitch 0, not nose-down**. A steep dive keeps *more*
energy per block fallen -- from rest, 8 blocks down, `TE` is -4.34 at hold 60 against -5.42 at
hold 0 -- but it stores it as `v_y`, and `v_y` only turns into forward speed through the
`v_y < 0` conversion, 10% x `cos^2(pitch)` per tick, while still sinking. Pitch 0 is where that
conversion (and the lift) is largest, so the pull-out from a steep dive costs height a low floor
does not have. With the `minipump` seed (below), the fitted dive angle is 0 for every
endurance cell but two (5 degrees at `y0 = 22, 31`), 0-5 degrees for range up to `y0 = 27`, and
10-15 degrees for range at `y0 = 28..32`. At `y0 = 4` and `8` every dive angle from 0 to 45 lands on the
same optimum.

**Above that the mini pump appears.** The first climb (`v_y > 0`) shows up at `y0 = 14` for
endurance and `y0 = 10` for range: hold 0 while sinking, pull hard (about -44) and relax, then glide
or climb again. Endurance does exactly one climb from 14 to 32; range does two from 18 and three
from 27. From `v0 = 0`, cluster corpus of 2026-09-24 15:20 EDT (`runs/floor/v2-cluster`):

```
 y0   endurance t*  first climb   range z*   climbs
 10      112.47          -          43.10       1
 14      169.11         100         83.97       1
 16      211.19         114        111.45       1
 20      301.23         141        176.86       2
 24      386.12         168        249.65       2
 28      462.92         194        347.44       3
 32      538.23         208        476.92       3
```

A cold start from a hold never finds it. At `y0 = 19` the best of hold -13 and a tiled steady
cycle exits at 239.6 ticks; the `minipump` seed finds 279.0.

Range and endurance separate. At `y0 = 8` the range optimum exits 3.7 ticks earlier than the
endurance optimum and goes 1.66 blocks further.

## How it is solved, and what worked

The polish only knows a fixed horizon and a terminal objective, and the question is about a
*first exit* -- a discontinuous, path-dependent quantity. What follows is every trick that was
tried, what it measured, and whether it stayed. Timings are wall-clock on an Apple M5, all
cores. Measured at the commit that added this file.

### 1. The bubble: a soft floor in the fixed-horizon polish (`Floor`, kept)

`J = TE(s_n) + w*z_n`, less a price on every post-tick state:

```
pen = weight * (max(0, margin - h) / margin)^2  +  wall * max(0, -h)
```

A bubble of thickness `margin` that costs `weight` at contact, and a wall that is linear in depth
under the floor. The wall is not a barrier on purpose: the coordinate search samples the whole
pitch range at every tick and most samples crash, and a log barrier scores them all `-inf`, so
the sweep cannot tell a crash at tick 80 from one at tick 3. A finite wall keeps them ranked.

For **endurance** it does not matter what the bubble is. Four shapes, from `margin 0.02,
weight 0.1` to `margin 1, weight 10`, all put the edge at `y0 = 4` between `n = 36` (feasible,
clearance +0.02 to +0.05) and `n = 37` (clearance -0.0630 in every case). The floor only binds at
the last tick, so the bubble only decides how far off it the answer stands.

For **range** the bubble *is* the answer. A range optimum skims the floor for many ticks and pays
the bubble on each of them, so `z` is set by the ratio `weight / lambda`, not by `lambda`. At
`y0 = 8` and `n = 83`:

```
lambda   weight   wall     z
  20        1      100    25.44
  20       10     1000    24.21
 100       10     1000    24.78
 100      100    10000    24.14
1000      100    10000    24.78
1000     1000   100000    24.14
```

Equal ratios, equal `z`, to 0.01. **A bubble is a bias wherever the optimum lives near the floor.**

### 2. Feasibility is read off the replay, never off the price (kept)

`Floor::clearance` is the minimum `h` over the replay. That, not a small penalty, is what decides
feasibility, since a finite bubble stands the optimum off the floor by an amount that depends on
the price.

### 3. The safety value as a feasibility oracle (`floor safety`, kept as a check)

Coordinate ascent on the soft minimum clearance, `-tau ln sum exp(-h_t/tau)` with `tau = 0.01`:
the discrete HJ safety value `max_u min_t h_t`, which is `>= 0` exactly when the horizon is
feasible. It is what a feasibility question should be optimized on -- energy only picks among
feasible schedules. At `y0 = 4`, `n = 37`, it reaches -0.06303 from hold -13, hold 0 and a
dive-then-pull alike, the same number the energy polish reached. That is the evidence that
`n = 37` really is infeasible, not merely that the energy objective declined it.

### 4. Walking the horizon (`floor endure`, superseded by 6)

Endurance is the largest feasible `n`. Walk `n` up from the best hold's survival; polish each
from the previous answer (last pitch repeated) *and* from every cold start; keep the best feasible
one; send a horizon nobody made feasible to the safety oracle before calling it a miss; stop
after three misses. It finds the right endurance -- the same ticks as 6 at every `y0` -- but
costs 17 s at `y0 = 8`, and for range it is path-dependent: at `y0 = 8` with the bubble annealed
(5), `lambda = 20` found 26.14 blocks and `lambda = 100` found 25.75, where a larger price on
distance can only have helped.

### 5. Annealing the bubble (`--anneal`, `--shrink`; superseded by 6)

The interior-point schedule: polish, then shrink `margin` and `weight` by `shrink` (0.3) and
polish again, keeping the last stage that is still feasible. It is what makes range work under
the bubble: `y0 = 8` range rose from 25.44 to 26.14 blocks, and showed for the first time that
the range optimum exits *earlier* than the endurance one (79 ticks against 83). Three stages and
six gave the same answer; each costs one more polish per start per horizon, 17 s -> 48 s.

### 6. The interpolated first exit (`floor exit`, **kept -- this is the solver**)

Drop the horizon search. Fly a long fixed cap and score the schedule on where it first goes
under, interpolated along the secant between the two states that straddle the floor:

```
k   = first tick with h_k < 0
t*  = (k - 1) + h_{k-1} / (h_{k-1} - h_k)          endurance
z*  = z_{k-1} + (t* - (k-1)) (z_k - z_{k-1})        range
```

The integer exit tick is a step function of every pitch; `t*` is continuous as the crossing
slides from one tick to the next (at `h_k = 0` both readings give `t* = k`). Ticks after the exit
are dead: the search skips them and the replay stops at the crossing, which is most of the speed.
Against the walk at `y0 = 8`:

```
                     walk (4, 5)             exit (6)
endurance            83 ticks, 17 s          83 ticks (t* 83.83), 0.5 s
range                26.14 blocks, 48 s      26.21 blocks, 0.7 s
```

Same endurance at every `y0` in the table, better range, 30-70x faster, and no bubble, so no
bias. What it does not smooth is a **touch-and-go**: a dip under the floor that comes back up makes
an earlier pair the first crossing, and `t*` drops discontinuously. Below `y0 = 8` nothing dips
twice, so this never bites. Above it, it does -- see the next section.

A replay that survives the whole cap scores the cap plus a value-to-go for the energy it still
holds -- `(TE + y0) / 0.0708` ticks at min sink, or `10.10 (TE + y0)` blocks at the best glide
ratio. Crude, so the cap is not allowed to bind. It starts at `--n 1200`, far past any exit up
to `y0 = 32`, which is nearly free: a dead tick costs one replay step, not a search. If the answer
still survives, `floor exit` doubles the cap (up to `--nmax 4800`) and re-ascends, and a file whose header
says `SURVIVES the cap` holds a guess, not an exit. why? on the first `y0 <= 32` corpus, ten
cells at `y0 >= 27` (range) and `y0 >= 29` (endurance) survived a fixed 450-tick cap, and their
scores were the guess.

The l1 curvature price is quoted in blocks of energy, as in `polish`, and converted into the
score's units: one block of energy is worth `1 / 0.0708 = 14.1` ticks or `10.10` blocks of `z`.
why? charged unconverted, `mu = 1e-4` was 14x (endurance) or 10x (range) weaker than the same
number in `polish`. Dead ticks are reset to the last live pitch after every pass, so the price
never sees the seed's leftover tail either.

The conversion did *not* remove the one-tick nose-down spikes above `y0 = 20`, but they are not
"paid for" either, as this section first claimed. Flattening a spike to its neighbors' mean costs
0.1-3.9 ticks, but that test removes the nose-down itself. Spreading the same total lift dump
(`sum sin^2 p`) over nine ticks scores *better* (`y0 = 16` endurance: score 212.016 -> 212.082;
spreading both lumps and re-ascending, 212.217 with roughness 202 -> 86). The proposed reason: at
pitch >= 0 the up-conversion branch is off and a tick depends on pitch only through `cos^2 p`, so
near 0 the physics is flat (quadratic in `p`) while the l1 price is linear, and a small one-tick
step from 0 always loses. Measured 2026-09-24 17:45 EDT by an independent agent on single cells at
`y0 = 15, 16`. But rounding the price's corner (Huber, 6b) leaves the spikes, while a quadratic
price removes them, so the corner is at best not the whole story.

### 6a. Terminal energy, tail shifts, and a growing cap (kept; measured 2026-09-24 17:00 EDT)

All at `mu = 1e-4`, the price every earlier run used; sums are over the 32 cells `y0 = 1..32`.

* **`--ke c`** adds `c` times the energy left at the crossing (all kinetic there), in score units.
  `c = +1` wrecks both utilities -- it hoards speed instead of flying (range -24%). `c < 0`,
  charging for reaching the floor fast, does nothing for endurance (which already lands slow, 0.85
  blocks of KE) and helps range: `c = -0.3` gave +79 blocks and 6 spikes instead of 15.
* **Tail shifts (`--shift`).** A per-pitch search crawls under the curvature price: moving one pitch
  off a straight line pays for three kinks, so a ramp that should start a tick later advances by
  slivers. At `mu = 1e-3`, `y0 = 16` took 263 passes. A move that shifts every pitch from `t` on
  changes only two second differences; with it the same cell converges in 48 passes. At
  `mu = 1e-4`, `y0 = 16`, it found t* 213.1 against 211.0 with a third of the roughness.
* **A growing cap (`--n 150`).** The ascent returns after the first pass whose flight outlasts the
  cap, and the doubled cap continues from it -- a continuation in horizon, optimizing the early
  flight first. With `c = -0.3`: endurance +272 ticks over the fixed 1200 cap.

Together (`--n 150 --shift --ke -0.3`): endurance +327 ticks (+4.2%) with 6 spikes instead of 18,
range +229 blocks (+4.5%). Range comes out rougher (summed |second difference| 8939 against 6214),
partly because it finds more pumps -- one more climb at `y0 = 24` and `32`. The cost is 1.7x the
core time of growing alone; the slowest cell took 840 s on one core. Data: `runs/floor/v3-ke`
(c sweep), `v4-mu`, `v5-grow`, `v6-shift`.

### 6b. The curvature price's shape, and multi-pitch moves (measured 2026-09-24 19:00 EDT)

`--penalty l1|huber:<d>|l2:<d>` (`l2:2` by default since 2026-09-24 23:45 EDT; `l1`
before, including every run in this section) sets the shape of the price on each second difference `x`: `|x|`,
`x^2/2d` inside `d` degrees and `|x| - d/2` outside, or `x^2/2d` everywhere, all times `mu`.
`--moves tick,box:<k>:..,ramp,shift` sets which sweeps a pass makes: `box:k` adds one change to `k`
consecutive pitches, and `ramp` adds `d*(s-t)` from `t` on, which is coordinate search in the second
differences (see `Move` in `src/bin/floor.rs`). All runs below: `mu = 1e-4`, 30 passes, the
minipump seed, `--n 150`, no bubble; sums over `y0 = 1..32`; chatter counts sign flips of the first
difference with both steps over 2 degrees. `l1` with `tick` reproduces `v7-30pass` exactly.

| variant | endurance Σt* | Σ\|d2\| | chatter | range Σz | Σ\|d2\| | chatter | core-h |
|---|---|---|---|---|---|---|---|
| l1 | 8104 | 4829 | 55 | 5263 | 8759 | 57 | 0.60 |
| huber:0.5 | +3 | 4955 | 47 | +12 | 9460 | 65 | 0.59 |
| huber:2 | +4 | 4677 | 35 | +5 | 9396 | 61 | 0.59 |
| l2:0.125 | -13 | 900 | 0 | 0 | 1426 | 0 | 0.62 |
| l2:0.5 | +22 | 1122 | 1 | -102 | 2259 | 0 | 0.61 |
| l2:2 | +58 | 1839 | 8 | +258 | 3893 | 31 | 0.64 |
| l1, tick+box:3:9:17:33 | -31 | 3487 | 65 | +170 | 4990 | 50 | 1.36 |
| l2:0.5, tick+box | -36 | 1405 | 0 | +311 | 3625 | 4 | 1.40 |
| l1, ramp only | -1050 | 5059 | 18 | -1145 | 4894 | 11 | 0.27 |

* **The quadratic price removes the chatter and, on the sums, costs nothing.** Rounding only the
  corner (Huber) barely changes anything, which argues against the corner mechanism in 6 (a small
  first step losing to a linear price) and points at the large second differences instead: a
  spike's price grows linearly in its height under l1 and quadratically under l2. Not tested
  further than this. Under l1 the minipump seed's 2.5-degree dive plateau at `y0 = 13`
  range stays at exactly 2.5 through every pass, box moves included; under `l2:0.5` it moves to
  3.1-3.7 and `z` improves (72.19 -> 72.31). The plateau was only coordinatewise optimal.
* **Range at `y0 >= 22` is basin luck.** Cells swing by 20% between variants (`y0 = 32`: 485 under
  l1, 608 under `l2:0.5` with box moves), and no variant wins every cell, so the range sums above
  measure which basins each variant fell into as much as the variant itself. Endurance at every `y0`,
  and range below 22, agree across variants within about 2.5%.
* **Ramp-only search is badly conditioned.** One ramp moves every later pitch, so 30 passes get
  nowhere near the per-tick optimum, and it cannot make a sharp pull-up. Its low chatter is that of
  a schedule that barely moved.
* Box moves cost 2.3x the core time.
* About 20 of the 32 optima per mode rest a mid-flight state within 1e-9 blocks of the floor, and
  two of them (`dist_y27_hub0.5`, `dist_y27_l2_2`) go under by 1e-14 when replayed on the laptop
  rather than on the cluster that solved them, exiting 100-200 ticks early. The vis skips them.

Data: `runs/floor/v10-pen` (`buggy-moves/` holds the box and ramp runs from before a fix to how a move
prices a candidate that exits earlier; they differ by at most 0.07).

### 6c. Gradient ascent (`--method grad`; measured 2026-09-25 00:06 EDT, commit 9136906)

`floor exit --method grad` replaces the coordinate sweep with L-BFGS ascent on the exact gradient
of the same score, every pitch moving at once. The default is still `--method tick` (`ascend`);
`grad+tick` runs the coordinate sweep from the gradient's answer at every cap. Seed, growing cap,
dead-tail flattening, the `f32` round and the header's score are the same as `ascend`'s.

**The gradient** (`src/adjoint.rs`) is one replay and one backward sweep, O(n). The tick's
partials (`fall_flying_partials` in `src/sim/entity.rs`) differentiate trig as if it were smooth
-- under `mth_lut` it is a 65536-cell staircase whose derivative is 0 almost everywhere -- and
count each branch of the flight kernel only where the replay took it, with no smoothing. At the
exit the score is differentiated through the secant, `dt*/dh[k-1] = -b/(a-b)^2` and
`dt*/dh[k] = a/(a-b)^2` (plus `1 - f` and `f` on the two `z`s for range), with the crossing's
index held; a survivor is differentiated through its value-to-go. The l2 and Huber prices are
differentiated exactly and l1 by its subgradient (0 at the corner). Checked against central
differences under `libm` in `cargo test`: worst relative error 4e-5 through the crossing (both
modes, with and without `--ke`) and 5e-6 on the whole priced objective.

**The optimizer.** L-BFGS, 10 pairs, a first step of at most 1 degree and after that at most 5
per pitch per step (`--max-step`), and an Armijo line search on the true score: the replay, not
the gradient, decides whether a step is kept. why? the score has a corner at pitch 0, a
staircase at 0.0055 degrees and whole-tick cliffs at a touch-and-go; a fixed-rate method (Adam)
would step across all three blind. `--iters` (2000 per cap) never binds: every run stops at a
stall, after at most 1209 iterations at one cap.

**What the plain gradient cannot see, and what it cost before it was handled.** Both measured on
the laptop, 2026-09-24.

* **Dips.** A pre-exit minimum of the clearance is a constraint the score feels only by falling
  off a cliff. Unhandled, the ascent walks a dip onto the floor and every trial step then crosses
  it: the first version, with neither fix in this list, stalled on `y0 = 16` endurance at
  t* 186.2 (coordinate ascent: 214.9) with a dip 1e-5 blocks off the floor. Each dip under
  one block now adds its own gradient (one more backward sweep) and the step is projected onto
  `dh_j . d >= margin - h_j`. A margin of 1e-4 was not enough: under `mth_lut` a dip parked there
  is crossed by the table's noise at every step size (`y0 = 24` endurance stalled at 298 ticks),
  so the line search also rejects a step that sinks a dip under half the margin, and the margin
  steps down `0.01, 0.003, 0.001` blocks (`--graze`), one stage per stall. The optima therefore
  stand 5e-4 to 5e-3 blocks off the floor at their dips, where coordinate ascent's rest within
  1e-9 -- and every gradient file replays on the laptop to within 1e-3 of the header the cluster
  wrote, where `dist_y27_l2_2` still goes under by 1e-14 (6b).
* **Pitch 0.** The forward-to-up branch is off at exactly 0, and the off branch's pitch-derivative
  there is 0 (lift goes as `cos^2`), so the seed's zeros never move. At a pitch on that switch
  the ascent takes both one-sided derivatives (`climb_switch_partials`) and moves the way that
  climbs. This is the other branch's derivative, used only at pitches sitting on the switch, and
  no smoothing. Without it, over `y0 = 4, 8, .., 32`: endurance 1824 against 2319 ticks, range
  1161 against 1769 blocks.

**Results.** Cluster, `y0 = 1..32`, minipump seed, `--n 150`, `--penalty l2:2`, no bubble;
coordinate ascent is `--moves tick --passes 30 --tol -1e9`, rerun on the same build as the
gradient runs (it reproduces v10's `l2:2` sums). Roughness is Σ|second difference| and chatter
the sign flips of the first difference with both steps over 2 degrees, over live pitches.
Core time is the gradient runs' own `wall` (which leaves out the seed fit, a fraction of a
second) and `times.tsv` for coordinate ascent, which lost 2 of its 64 lines to interleaved
appends (both `y0 <= 8`, a few seconds each).

| variant | endurance Σt* | Σ\|d2\| | chatter | range Σz | Σ\|d2\| | chatter | core time |
|---|---|---|---|---|---|---|---|
| coordinate, 30 passes | 8162 | 1839 | 8 | 5521 | 3893 | 31 | 2130 s |
| grad | +203 | 1156 | 3 | +675 | 4263 | 40 | 18 s |
| grad, `--max-step 2` | +193 | 1118 | 3 | +1337 | 3676 | 20 | 20 s |
| grad+tick, 30 passes | +183 | 1104 | 5 | +538 | 3449 | 31 | 2346 s |

```
 y0   endurance t*: coord    grad  step 2     range z(t*): coord    grad  step 2
  1                  12.66   12.66   12.66                    0.53    0.53    0.53
  4                  36.45   36.45   36.45                    5.86    5.86    5.86
  8                  83.83   83.82   83.82                   26.21   26.21   26.21
 10                 112.48  112.48  112.48                   43.14   43.13   43.13
 12                 140.80  140.79  140.79                   61.51   62.27   62.31
 13                 153.40  154.40  154.43                   72.26   72.99   72.96
 14                 173.35  173.28  173.40                   84.12   84.61   84.50
 16                 214.86  215.32  215.37                  111.24  111.69  111.82
 18                 263.90  262.24  260.43                  142.88  145.50  145.28
 20                 306.40  318.10  317.77                  177.27  186.58  183.44
 22                 349.05  361.72  363.79                  219.28  239.41  239.65
 24                 397.47  413.22  413.31                  275.96  303.74  305.33
 26                 456.36  462.47  462.43                  324.42  419.54  415.00
 28                 500.12  510.93  509.14                  408.07  473.91  447.24
 30                 537.00  562.33  560.25                  481.50  527.77  648.90
 32                 570.43  605.23  598.28                  537.76  577.44 1055.25
```

* **It is about 120 times cheaper and, above `y0` about 12, better.** The slowest single solve
  took 4.9 s, against up to 128 s for coordinate ascent. Endurance gains grow with height, to
  +35 ticks (6%) at `y0 = 32`, flying the same single climb better (one dip each, 606 against
  571 ticks). It loses at `y0 = 14` and `18` (by 1.7 ticks at 18), and with step 2 also at 15
  and 17.
* **Below `y0` about 12 it stops short by up to 0.011 ticks.** There are no dips there; the line
  search stalls when its predicted rise falls under the table's noise, where the coordinate
  scan, which prices every candidate on the real table, still resolves. `grad+tick` closes it.
* **Range is basin luck above `y0 = 22`, more so than in 6b.** The gradient runs find more and
  longer pump laps (`y0 = 26`: 4 dips and 614 ticks, against 3 and 460), but the step size alone
  moves `y0 = 32` from 577 to 1055 blocks, and a laptop run of the same binary found 1308 with
  step 2 and 611 with 5 (the likeliest reason: vanilla's lift `cos` is the platform's libm,
  see `src/sim/mth.rs`).
  Only `y0` below about 18, in either mode, compares optimizers (see the next item).
* **Endurance above `y0` about 18 is far from optimal under every method.** All of them stay in
  the minipump seed's one-climb basin. Seeded instead from the same `y0`'s best range schedule,
  `--method grad` reaches t* 1483.8 at `y0 = 32` (against 605.1; z 1111, above every range run),
  and gains at every `y0 >= 20`: +16 at 21, +70 at 24, +266 at 27, +480 at 30. The reverse (range
  seeded from endurance) loses at every `y0 >= 18`. Range runs do grow extra laps from the one-climb seed under
  `--method grad` (the `y0 = 32` `grad_s2` range run grew four as its cap doubled from 150 to
  2400); endurance runs have not, and why is open. The cross-seeded results are non-monotone in `y0` (27: 751, 28: 690),
  so they are lower bounds too, and the infinite-flight threshold then quoted (35-36 blocks) was
  lowered to 29.25 by the K-climb seed (6f, Infinite flight) and to (28.75, 29.0] by the
  backward DP (6g). Measured 2026-09-25 00:30 EDT on the laptop,
  `runs/floor/v11-grad/cross-seed`.
* **Chatter.** Endurance: fewer flips (3 against 8) and 37% less roughness. Range: not
  uniformly. Most counted flips in every variant are the tops of pull-ups (-52 then -49) and the
  peaks of nose-down lumps; the gradient's own are 3-8 degree zigzags at the glide-to-dive entry
  of a late pump at `y0 = 30, 32` (`dist_y30_grad`, ticks 477-483), pitches at or above 0 where
  the physics sees pitch only through `cos^2`. The first explanation proposed here (a flat
  score, and the ascent stalling on table noise) was tested and is wrong; see 6e.
* `grad+tick` is worse than `grad` alone on both sums: its coordinate passes at the smaller caps
  move the schedule the next cap's gradient ascent starts from, and it lands in other basins.

Data: `runs/floor/v11-grad` (`l2_2`, `grad`, `grad_s2`, `gradtick`), drawn by
`tools/plot_floor_profiles.py`.

### 6d. The smooth bubble in place of the graze (measured 2026-09-25 18:28 EDT on the laptop)

`--method grad` now takes `--bubble <margin>,<weight>` (`bubble_cost` in `src/bin/floor.rs`, differentiated
exactly in `utility_grad`: each state inside the bubble is seeded on its height, and the crossing
term `f * weight` through the same `df/dh` as `t*`), and `--graze off` turns the dip projection
and its rejection rule off. `--shrink r --anneal k` gives at least `k + 1` stall stages, with margin
and weight both times `r` at each; like the graze, the stages restart at every cap. The
finite-difference test covers the bubble in both modes, with and without `--ke`, and for a
survivor: worst relative error 4.3e-5, with the bubble 0.7 to 1.2 of the gradient at the checked ticks.

All runs: minipump seed, `--n 150`, `--penalty l2:2`, `y0 = 1..32`, code as committed with this section. Deltas
are against the graze baseline (`grad`, graze `0.01,0.003,0.001`: it reproduces 6c's cluster sums
to 3 ticks in endurance). *Parked* counts cells, over all `y0`, whose lowest pre-exit dip ends within
1e-4 blocks of the floor, which is how an ascent stalls early here (see 6c). *Wobble* counts
turning points of the live pitch with a swing over 5 degrees on both sides, pull-up bottoms under
-20 excluded; unlike chatter it sees multi-tick wobbles. Every file replays to within 5e-5 of its
header.

| variant | end Δ y0<18 | range Δ y0<18 | end Δ 18+ | range Δ 18+ | Σ\|d2\|/tick end, range | chatter | wobble | min dip y0<18 | parked | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| graze (baseline) | 1808.30 | 765.84 | 6554 | 5551 | 0.14, 0.39 | 4, 20 | 49, 132 | 6e-4, 7e-4 | 0, 0 | 14 s |
| graze off, no bubble | -21.1 | -36.5 | -1928 | -2709 | 0.45, 0.66 | 26, 17 | 26, 14 | 8e-6, 8e-6 | 20, 22 | 0.5 s |
| bubble 0.05, 1e-3 | -14.2 | -37.3 | -844 | -2454 | 0.34, 0.54 | 32, 23 | 76, 26 | 1e-6, 4e-6 | 20, 21 | 1 s |
| bubble 0.05, 1e-2 | -1.4 | -0.9 | +28 | -1135 | 0.18, 0.62 | 7, 71 | 51, 196 | 1e-4, 9e-3 | 9, 13 | 2 s |
| bubble 0.05, 1e-1 | -349.1 | -77.8 | -2183 | -2405 | 0.14, 0.57 | 0, 7 | 24, 64 | 0.19, 0.045 | 0, 0 | 5 s |
| bubble 0.01, 1e-3 | -10.2 | -24.6 | -894 | -2376 | 0.35, 0.54 | 31, 27 | 72, 35 | 3e-8, 4e-8 | 20, 21 | 1 s |
| bubble 0.01, 1e-2 | -0.0 | -1.5 | +0 | -240 | 0.18, 0.48 | 9, 34 | 52, 165 | 6e-3, 8e-3 | 0, 5 | 4 s |
| bubble 0.01, 3e-2 | -0.4 | -26.2 | +62 | -458 | 0.18, 0.64 | 3, 40 | 55, 196 | 9e-3, 9e-3 | 0, 0 | 3 s |
| bubble 0.01, 1e-1 | -357.1 | -83.2 | -2694 | -2416 | 0.35, 0.67 | 5, 44 | 47, 90 | 0.11, 0.010 | 0, 0 | 4 s |
| bubble 0.003, 3e-3 | -0.8 | -1.1 | -32 | -948 | 0.19, 0.47 | 10, 43 | 58, 145 | 2e-3, 2e-3 | 0, 0 | 2 s |
| bubble 0.003, 1e-2 | -0.7 | -1.9 | +76 | -649 | 0.24, 0.56 | 14, 71 | 71, 187 | 3e-3, 3e-3 | 0, 0 | 2 s |
| **bubble 0.01, 1e-2, `--shrink 0.3 --anneal 2`** | **+0.3** | **+0.1** | +3 | -233 | 0.15, 0.46 | 6, 26 | 50, 164 | 5e-4, 6e-4 | 0, 2 | 4 s |
| bubble 0.05, 1e-1, `--shrink 0.3 --anneal 3` | -2.5 | -13.6 | -730 | -9 | 0.19, 0.68 | 7, 49 | 53, 190 | 9e-4, 1e-3 | 0, 0 | 16 s |
| bubble 0.01, 1e-2 + graze | -0.5 | -1.4 | -5 | +608 | 0.14, 0.40 | 4, 24 | 49, 166 | 7e-3, 7e-3 | 0, 0 | 28 s |

Wall is the sum of the headers' `wall` over 64 solves (seed fit excluded); the whole grid of 896
solves took about a minute at `-P 4`; every shrink run reached its last stage.

* **A bubble has two jobs, and its weight is squeezed between them.** It must be steep enough at
  contact to hold a dip: its slope there is `2 weight / margin` blocks of energy per block of
  height, and at 0.04 and 0.2 (weight 1e-3) dips park exactly as with no bubble at all, while at
  2 (0.01, 1e-2) they stand at 6-8e-3. And it must not tax the exit: the crossing term costs
  `weight * per_block` per unit of `f`, which `t*` values at 1 and `z(t*)` at the exit's `v_z`
  (0.33-0.40 blocks per tick here). At weight 1e-1 that is 1.41 against 1 in endurance, so flying
  on past the exit tick *lowers* the score: at `y0 = 1` the ascent's first gradient is already 0
  and every flight from `y0 = 4` ends early (`y0 = 8`: 66-67 against 84 ticks). At 3e-2 in range it
  is 0.30 against 0.40: the graze optimum scores higher *under the 3e-2 objective* than what the
  ascent found (`y0 = 12`: 61.79 against 59.50), so the loss there is a stall in a worse local
  maximum, not a moved optimum. Rule: keep `weight * per_block` well under the exit's value per
  tick -- in practice weight <= 1e-2 -- and pick the margin so `2 weight / margin` is about 2 or more.
  why? below that slope the utility's pull on a dip beats the bubble, and above that weight the
  exit tax cancels most of the reason to fly on.
* **A fixed bubble's stand-off is a small bias.** Holding dips at 7e-3 instead of the graze's 9e-4
  costs 0.05-0.33 blocks per range cell at `y0 = 11..17`, and the crossing tax costs up to 0.06
  blocks below 11 where there are no dips. Endurance does not feel either (-0.02 ticks in total).
* **Shrinking both by one factor removes it.** `r` times margin and weight keeps `2 weight /
  margin`, the holding strength, fixed while the stand-off and the exit tax shrink by `r`: two
  stages at 0.3 end at (9e-4, 9e-4), with dips at 5-6e-4 as the graze leaves them, and tie or beat
  the graze below `y0 = 18` in both modes. Starting the same schedule from a too-strong bubble
  (0.05, 1e-1) does not recover: the first stage has already shortened the flights.
* **Chatter and wobble below `y0 = 18` are unchanged** (endurance chatter 2 against 3, wobble 8
  against 9; range 3 against 2, 6 against 7); range Σ|d2| per tick rises from 0.12 to 0.16-0.19
  there with no extra turning points. The large full-range differences in the table are the
  high-`y0` basins (6c): more laps, more wobble. No bubble and no graze is the rough one (0.43-0.47
  per tick below 18): its ascents stall with a dip parked on the floor and a gradient still
  0.03-0.3, before the price has smoothed anything.
* **The bubble is about half the wall time of the graze** (4 s against 14 s): no extra backward
  sweep per dip and no projection. Bubble and graze together cost the most (28 s) and buy
  nothing below 18.

Recommended when a dip mechanism is needed: `--graze off --bubble 0.01,1e-2 --shrink 0.3 --anneal 2`.
The default is still the graze, unchanged. Data: `runs/floor/v12-bubble-grad` (`out/`, solver
stop lines in `logs/`, `analyze.py` for the table), drawn by `tools/plot_floor_profiles.py`.

### 6e. Why nose-down dives wobble: the flight is convex there (measured 2026-09-25 19:30 EDT)

The multi-tick wobble in nose-down dives (10-25 degrees, period about 7 ticks; `dist_y32_grad_s2`
ticks 420-540) is not an optimizer artifact. The flight's own score is **convex** in how the
nose-down is spread across ticks, and the l2 price is too weak to cancel that at periods of about
7-20 ticks.

* **Measured curvature.** Along a sinusoidal wobble of period `P` over a 60-tick dive window
  (pitch 16-41 degrees; `y0 = 32` range, smooth `libm` trig so finite differences are clean),
  `v'Hv/|v|^2` of the flight score against that of the price (`l2:2`, `mu = 1e-4`), per deg^2:

  ```
  P        2        3        4        5        7        10       14       20       30
  flight  +8.1e-4  +8.0e-4  +8.0e-4  +7.9e-4  +7.7e-4  +7.1e-4  +6.1e-4  +4.1e-4  -1.0e-4
  price    8.1e-3   4.5e-3   2.0e-3   9.7e-4   2.9e-4   7.7e-5   2.2e-5   5.9e-6   1.5e-6
  ```

  The flight's curvature is positive and nearly independent of `P` up to about 14 ticks: a
  per-tick effect. Lift goes as `cos^2 p`, so the lift a nose-down tick sheds goes as `sin^2 p`,
  whose slope `sin 2p` still rises below 45 degrees: nose-down pays more where there already is
  more of it. The price's curvature on second differences falls as `(2 - 2 cos(2 pi / P))^2`,
  about `P^-4`. It wins at `P <= 5`, loses at 7-20, and past about 30 the flight turns concave
  (the trajectory couples the ticks). So one-tick chatter is suppressed and the wobble settles at
  the shortest unstable period.
* **Not table noise, not the dip mechanism.** On `y0 = 24, 28, 30, 32` range, wobble (turning
  points with a 5-degree swing, as a share of ticks) is 0.6-3.0% with the table, 0.9-1.9% with
  smooth trig, and 0.7-3.1% with the bubble of 6d in place of the graze.
* **The price weight is the lever.** At `mu = 1e-2` wobble falls to 0.1-0.5% and roughness to a
  quarter, for range 1-3% lower at `y0 = 24` and `28` (the higher cells are basin noise). To
  stabilize period `P` the price's curvature must exceed about 8e-4: `mu` of about 3e-3 covers
  `P <= 10`, about 1e-2 covers 14.
* **The same convexity explains the earlier findings.** Under `l1` it makes lumps (6), and at
  `mu = 0` it makes bang-bang chatter: the `v0 = 0, n = 300` cycle polished at `mu = 0` alternates
  0 and 85 degrees at its dive bottom and scores 19.836 against 19.704 at `mu = 1e-4`
  (`runs/pencycle/mu`). The unregularized optimum *is* chattering; the price decides how much.
* The dives sit at a floor graze (6 dips at 0.0006-0.0019 blocks in the laptop run), so smoothing
  a whole lap's dive by even 0.18 degrees at most moves the next graze under the floor
  (t* 1721 -> 697). The wobble cannot be judged by perturbing a finished flight.

Data: `runs/floor/v14-wobble` (stage runs `cap*.pitches`, the matrix `m_<variant>_y<y0>.pitches`,
the gradient dump `g.txt`). The `libm` switch and the `gradump` subcommand used here live only in
a scratch worktree and were not merged.

### 6f. A K-climb seed (`--init pumps:<K>`; measured 2026-09-25 18:30 EDT, commit 93aa9da; `--after` in the commit after it)

`pumps:<K>` is a floor-grazing pump flown by events rather than tick counts: from rest, dive at
`d` nose-down until the clearance is under `lvl`, hold 0 until it is under `pull`, pull at `a`
while `v_y` rises, then relax linearly in `v_y` to `a2` at the apex. After each climb but the
last, glide at 0 for `g` ticks and dive again; after the last, hold `end`. The seven parameters
not given as `key=value` are fitted to the seed's own exit score, a coarse joint grid then
coordinate passes, in about a second. why events: a lap's timing depends on how fast it
arrives, so a tick count fitted on one lap or one `y0` is wrong on the next. The laps are copied
from the best v11 `y0 = 32` range schedule, which does exactly this five times. The fitted
values barely move above `y0` about 24: `d` 20-27.5, `lvl` about 3, `pull` 0.25-0.3,
`a` -34 to -38, `a2` 0, `g` 35-90. Below that, `d` falls to 0-5, as with `minipump`.

Two things were needed that the first version lacked:

* **The relax has to be a ramp.** Held at a constant pitch, every seed lap lost about a block at
  `y0 = 32` (peak energy 29.0, 28.3, 27.2, 25.4), so the fit never used more than six. The
  optimized climbs relax about linearly to -10 at the apex, and the ramp raised the `K = 6` seed
  from 1119 to 1362 ticks. The seed's laps still lose energy; the gradient makes them gain.
* **The cap cannot start below the seed's flight.** Growing from `--n 150` keeps only the
  seed's first 150 pitches and repeats the last one, which erases every lap after the first. With
  `pumps`, the ascent starts at the first doubling of `--n` that the seed does not outlast.

**Results.** Laptop, `y0 = 1..32`, both modes, `K = 1..8` (and 10, 12, 16 from `y0 = 20`),
`--method grad --penalty l2:2 --n 150`, the v11 flags. 590 solves took 41 s of wall clock. The
table gives the best `K`, then in parentheses the climbs the answer actually flies (a peak more
than 2 blocks above the dip before it). "v11 best" is the best of `grad`, `grad_s2` and the
cross-seed (6c). "+laps" is four more rounds, each continuing the best `K` from its last apex
with 2 more seed laps (`--after`, below), then grad. Below `y0 = 12` every `K` ties v11 to 0.01.

| y0 | K (climbs) | t* | v11 grad | v11 best | vs best | +laps | K (climbs) | z(t*) | v11 grad | v11 best | vs best | +laps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 2 (0) | 140.8 | 140.8 | 140.8 | +0.0% | - | 1 (0) | 62.3 | 62.3 | 62.3 | +0.0% | - |
| 13 | 1 (0) | 154.4 | 154.4 | 154.4 | -0.0% | - | 1 (0) | 73.0 | 73.0 | 73.0 | +0.0% | - |
| 14 | 2 (1) | 173.3 | 173.3 | 173.4 | -0.1% | - | 2 (1) | 84.6 | 84.6 | 84.6 | +0.0% | - |
| 15 | 1 (1) | 193.6 | 193.6 | 193.6 | +0.0% | - | 1 (1) | 97.6 | 97.5 | 97.6 | -0.0% | - |
| 16 | 2 (1) | 215.2 | 215.3 | 215.4 | -0.1% | - | 1 (1) | 111.8 | 111.7 | 111.8 | -0.0% | - |
| 17 | 1 (1) | 238.3 | 239.2 | 239.2 | -0.4% | - | 1 (1) | 127.4 | 127.3 | 127.7 | -0.3% | - |
| 18 | 3 (1) | 260.1 | 262.2 | 265.2 | -1.9% | 261.8 | 2 (1) | 145.2 | 145.5 | 145.5 | -0.2% | 145.2 |
| 19 | 1 (1) | 287.2 | 292.0 | 292.0 | -1.6% | 287.3 | 1 (1) | 160.9 | 164.9 | 165.1 | -2.6% | 160.8 |
| 20 | 2 (1) | 313.8 | 318.1 | 321.9 | -2.5% | 313.8 | 2 (1) | 180.5 | 186.6 | 186.6 | -3.2% | 180.5 |
| 21 | 2 (2) | 359.6 | 342.4 | 359.1 | +0.2% | 359.7 | 2 (2) | 209.7 | 209.7 | 211.6 | -0.9% | 209.8 |
| 22 | 2 (2) | 401.0 | 361.7 | 398.5 | +0.6% | 401.2 | 2 (2) | 238.8 | 239.4 | 239.6 | -0.4% | 240.5 |
| 23 | 3 (2) | 448.1 | 388.8 | 433.9 | +3.3% | 448.5 | 2 (2) | 275.9 | 269.2 | 270.2 | +2.1% | 276.0 |
| 24 | 2 (2) | 490.1 | 413.2 | 483.2 | +1.4% | 502.6 | 2 (3) | 316.7 | 303.7 | 305.3 | +3.7% | 317.6 |
| 25 | 4 (2) | 542.7 | 436.5 | 559.8 | -3.1% | 579.8 | 2 (3) | 369.1 | 360.1 | 360.1 | +2.5% | 369.4 |
| 26 | 3 (3) | 644.3 | 462.5 | 629.2 | +2.4% | 655.8 | 2 (3) | 422.2 | 419.5 | 419.5 | +0.6% | 428.5 |
| 27 | 4 (3) | 721.7 | 485.1 | 751.6 | -4.0% | 788.2 | 3 (4) | 489.6 | 474.1 | 474.1 | +3.3% | 521.1 |
| 28 | 3 (3) | 799.2 | 510.9 | 690.8 | +15.7% | 968.8 | 5 (4) | 610.7 | 473.9 | 473.9 | +28.9% | 610.7 |
| 29 | 5 (4) | 1022.9 | 536.7 | 900.7 | +13.6% | 1456.1 | 5 (6) | 874.4 | 515.8 | 581.4 | +50.4% | 1018.3 |
| 30 | 5 (5) | 1302.6 | 562.3 | 1043.1 | +24.9% | unbounded | 6 (5) | 913.3 | 527.8 | 648.9 | +40.7% | unbounded |
| 31 | 6 (6) | 1685.3 | 579.7 | 832.7 | +102.4% | unbounded | 5 (7) | 1327.7 | 561.7 | 598.2 | +121.9% | unbounded |
| 32 | 8 (7) | 2645.8 | 605.2 | 1483.8 | +78.3% | unbounded | 10 (9) | 2065.4 | 577.4 | 1055.2 | +95.7% | unbounded |

* **The number of climbs rises with `y0`, and from `y0` about 21 the seed beats every v11
  run.** Endurance takes 2 climbs from 21, 3 from 26, 4-7 from 29. Range usually takes one more
  climb than endurance from `y0 = 24` (not at 26 or 30). Endurance gains up to 102% over v11's best, and 337% over
  plain `grad`, whose one-climb basin is the whole gap. Range gains up to 122% over v11's best.
* **From `y0` about 29.25 there is no best `K`** (see Infinite flight; the DP's laps gain from
  `y0 = 29.0`, 6g): every lap gains
  energy, so more laps is always longer and further. The `y0 >= 30` rows measure how many laps
  the seed can fly, not an optimum. Continuing a `y0 = 30` flight in laps reached t* 21007 after
  8 rounds and was still gaining.
* **`K` is an upper bound, not the answer's lap count.** The climbs flown plateau however large
  `K` is (at `y0 = 27` endurance, 3 for every `K >= 3`), and many cells tie across `K`. The
  seed's laps lose energy, so past a few the fitted seed crashes before its `K`-th climb, and
  endurance grad never adds a lap. Range grad sometimes does: at `y0 = 32` the `K = 10` seed flies
  8 climbs and its solve flies 9.
* **It loses at `y0 = 17..20`, by up to 3.2%.** There the optimum has one climb. The proposal,
  not tested: this seed's single lap pulls at a clearance, near the floor, while `minipump` pulls
  at a fitted tick after a short dive, and that is a better start. Continuing in laps barely
  helps (+1.7 ticks at 18). Take the better of the two seeds there.
* **Continuing in laps (`--after <file>`) is worth up to 42%** below the threshold (`y0 = 29`
  endurance: 1023 -> 1456, and 1590 from another start). It keeps a solved schedule to its last
  apex and fits `K` fresh seed laps after it, so each round starts from laps the gradient has
  already made to gain.
* **Grad did not stall at the floor.** Every dip stands 5e-4 to 2e-3 blocks off it, as the
  graze stages intend (6c). No run reached the 2000-iteration limit (the most was 1672 at one
  cap). The slowest continuation, 37 climbs over 21000 ticks, took 73 s.

Why endurance grad does not add laps where range grad does (6c): not tested. The proposal is that
a range glide runs at pitch about 0 and 1.4 blocks/tick, where the up-conversion (proportional to
speed) makes a small pull pay at once. An endurance glide runs near min sink, -13 to -22 and
about 0.36 blocks/tick, where it pays almost nothing, and a new lap first needs a dive that costs
time.

Data: `runs/floor/v13-pumps` (`out/` the K table, `cont-k/` its continuations, `inf/` and `cont/`
the infinite-flight scan, `v1-constant-relax/` the first seed; `summary.py`, `final.py`,
`infsum.py` and `budget.py` print this section's and Infinite flight's tables from them).

### 6g. Backward DP (`floordp`; measured 2026-09-26 17:30 EDT on the cluster, commit 9525ae6)

`floordp` computes the value of every state instead of improving one schedule. The state is
`(h, v_y, v_z)`: the kernel reads only velocity and pitch, and the floor only the clearance.
Sweep `n` of backward induction is `V_n(s) = max_p [r + V_{n-1}(s')]`, the best flight capped
at `n` ticks. `r` is 1 (endurance) or the tick's `dz` (range). A step under the floor earns only
the fraction `h / (h - h')` of `r`, exactly `exit_score`'s interpolated exit. `V` lives on a grid
read by trilinear interpolation, with heights `hmax (i / (nh-1))^1.5`, packed toward the floor.
171 pitches, 1 degree apart. The **policy** is then flown in the exact simulator from rest: each
tick, the pitch on a 0.25-degree grid that maximizes `r + V(s')`. `V` is the grid's opinion; the
flown policy is a real flight. Gauss-Seidel in order of rising energy was tried first and gained
nothing: a glide moves about 0.07 blocks a tick against cells of 0.02-0.4, so a backup mostly
reads its own cell.

**The grid limits it, and velocity spacing most of all.** At `y0 = 16`, against the best
earlier answer (`runs/floor/v16-dp/known-before-dp.tsv`, the maximum over every earlier run):

| grid | h nodes | dv | pitch step | endurance V / best, flown / best | range V / best, flown / best |
|---|---|---|---|---|---|
| hi_v05 | 133 to 33 | 0.05 | 1 | 1.312, 0.9731 | 1.521, 0.9445 |
| hi_v025 | 133 to 33 | 0.025 | 1 | 1.051, 0.9969 | 1.073, 0.9961 |
| lo_v025 | 69 to 17 | 0.025 | 1 | 1.051, 0.9956 | 1.075, 0.9948 |
| lo_v025_dp05 | 69 to 17 | 0.025 | 0.5 | 1.051, 0.9957 | - |
| lo_v025_nh137 | 137 to 17 | 0.025 | 1 | 1.044, 0.9976 | - |
| lo_v0125 | 69 to 17 | 0.0125 | 1 | 1.014, 0.9981 | 1.025, 0.9973 |

* `V` is optimistic, and converges from above as `dv` shrinks (31%, 5%, 1.4% over at
  `y0 = 16`). Interpolation lets a state borrow value from neighbors it cannot reach. It is not
  an upper bound either: at `dv = 0.0125` it reads 0.9997 of the best at `y0 = 10`. On the
  coarser grids it never converges above about `y0 = 17`, still rising after 1700 sweeps, so
  there it is not an estimate of anything.
* The pitch step does not matter (1 against 0.5 degrees agree to 1e-4), and doubling the height
  nodes helps a little.
* **Below `y0` about 12 the DP confirms the earlier answers.** Its flown policy matches the best
  to 1e-4 in both modes at every grid, and on the finest grid it is within 0.2% (0.9% at
  `y0 = 13`) up to 16.

**Above that it finds better flights.** On the `dv = 0.025`, 33-block grid, the flown policy
alone beats every earlier answer at `y0 = 26..28`. Polished by `floor exit --method grad`
(defaults, the policy as the seed), it beats or ties them from 16 up:

| y0 | best t* | DP flown | DP polished | vs best | best z | DP flown | DP polished | vs best |
|---|---|---|---|---|---|---|---|---|
| 16 | 216.1 | 215.4 | 216.4 | +0.1% | 111.8 | 111.4 | 111.8 | +0.0% |
| 18 | 265.2 | 264.8 | 266.0 | +0.3% | 145.5 | 144.0 | 145.4 | -0.1% |
| 19 | 292.1 | 285.5 | 288.3 | -1.3% | 165.1 | 163.6 | 164.9 | -0.1% |
| 20 | 321.9 | 319.7 | 321.8 | -0.0% | 186.7 | 185.1 | 186.7 | +0.0% |
| 22 | 401.2 | 398.1 | 402.9 | +0.4% | 240.5 | 239.3 | 242.4 | +0.8% |
| 24 | 502.6 | 499.0 | 506.4 | +0.8% | 317.6 | 314.9 | 319.8 | +0.7% |
| 25 | 579.8 | 575.0 | 582.3 | +0.4% | 369.4 | 367.6 | 374.5 | +1.4% |
| 26 | 655.8 | 663.8 | 676.5 | +3.2% | 428.5 | 437.4 | 447.1 | +4.3% |
| 27 | 788.2 | 803.5 | 817.4 | +3.7% | 521.1 | 543.4 | 555.4 | +6.6% |
| 28 | 968.8 | 1056.9 | 1093.0 | +12.8% | 610.7 | 735.4 | 760.8 | +24.6% |
| 29 | 1456.1 | no exit | - | sustained | 1018.3 | no exit | - | sustained |

(`runs/floor/v16-dp/readme_tables.py` prints every `y0`.) The best-before at 24-29 came from
`pumps` seeds continued lap by lap (6f), and still the DP wins, by keeping more energy per lap.
At `y0 = 28` both pump with dips of 0.001, but the DP flights' peak energies fall 25.0, 23.9,
22.0, 18.9, 14.3 (endurance) against 24.8, 23.4, 21.4, 17.0, 10.7. In range the old flight
glides 170 ticks between its third and fourth laps and ends with 5 peaks; the DP pumps
steadily and has 6. The DP has no curvature price, so its policies are rough; the gradient
polish, which has the l2 price, is what the table's polished column scores.

**From rest at `y0 = 29` the DP policy flies forever** (both modes, 20000 ticks, 127-130 dips,
the lowest 1e-4 blocks above the floor; replayed independently by `floor laps`). Every earlier
flight died at 29.0 (Infinite flight). Its laps gain energy from the first: peaks 26.7, 26.8,
26.9, 27.2, 27.5, 28.1, .. (endurance), where the `pumps` laps break even only at about 27. The
first dive costs the same, `29 - 26.7`. The peaks level off at 34.7, with the peak height at
33.0-33.1, the top of the grid, which values nothing higher: that plateau is the grid's, not
the physics'.

**The DP's threshold is in (28.75, 29.0]** (same grid and 1700 sweeps, rerun with `V` saved at
`y0` steps of 0.25 from 27.5; finished 2026-09-26 19:52 EDT on the cluster, commit 9525ae6). At 28.75
the flown policy lands, in both modes (t* 1660.8, z 1205.0). Polished, it still lands
(t* 1851.1, z 1524.6): the polish has a finite cap and no reason to add laps it cannot finish.
The first peak decides it. At 28.75 it is 26.27 (endurance) and the peaks fall from there
(26.10, 25.82, 25.38, ..); at 29.0 it is 26.67 and they rise. So the DP's laps break even at a
first peak between 26.3 and 26.7. t* grows steeply as `y0` approaches the threshold: the
polished endurance t* is 1093.0, 1205.8, 1401.2, 1851.1 at 28, 28.25, 28.5, 28.75, and range
z(t*) is 760.8, 859.5, 1028.2, 1524.6. This is a threshold for this grid's policy,
**not a proof**: a finer grid could lower it, as the finer grids did at 16.

**The flown policy chatters, in the descents** (same grid and `V`; measured 2026-09-28 13:40 EDT on the laptop, commit 25099d2 plus `--trace`).
Chatter here means sign flips of the first difference with both steps over 2°. None below
`y0 = 14` (10 in range), then 11 at 16, 113 at 24 and 243 at 28 (endurance; range is similar). Polished, the
same flights have 0-3 below 28. About 95% of the flips come while falling (`v_y < 0`), and they
swing 10-45° between nose-down pitches from 0 to about 45. The pitches compared there are
close: at a flip, the best pitch more than 10° from the choice is behind it by a median of
0.04-0.07 (ticks or blocks), against 0.08-0.14 over all ticks, and every pitch from 0 to about
50 is within a few tenths. That is far below `V`'s own error: `V(start)` is 5% over the flight at `y0 = 16` and 42% (209
ticks) at 24, so
interpolation noise makes the choice. On the policy map the descent runs along the edge
between the pitch-0 region and the nose-down one, and one tick crosses it each way. That is a
feedback policy sliding on a switching surface. The polish replaces the alternation with the
steady pitch between them (about 28° at `y0 = 24`, ticks 60-110), and gains (499.0 to 506.4).
`floordp --trace` writes each flight's per-tick state, choice, runner-up and the value of every
pitch. `tools/plot_floor_dp.py` draws them, with the polish and the best earlier flight, into
`runs/floor/fig/floor-dp.html`. Its policy map is computed in the page, at the exact height of the
probed flight, by a JS port of the backup (`tools/floor_dp_policy.js`) reading a crop of the saved
`V`. The build replays every traced DP tick through the port and refuses a single different
choice: as of 2026-09-28 13:55 EDT, 63146 ticks agree, values to the bit. (`--map` writes the
same field at fixed heights, for use outside the page.)

Caveats. The policy is feedback on a grid, so the flight it gives is only as good as the grid
near the states it visits. `V` above about `y0 = 17` is not a trustworthy estimate even at
`dv = 0.025` (still 5-80% over and rising). What is trustworthy is every flown or polished number
here: each is a replay.

Data: `runs/floor/v16-dp` (`<run>.txt` each run's table, `<run>/` its policies as pitch files,
`<run>.err` its sweep log; `pol/` the polished policies; `known-before-dp.tsv` the comparison;
`compare.py`, `polcmp.py`, `readme_tables.py`; `thresh/` the threshold rerun, with its saved
`V_time.bin` and `V_dist.bin` for `floordp --load` and its polishes in `thresh/pol/`; `viz/` the
traces and maps from that `V`). Cost on one cluster node (4 cores): 1.3 s per
sweep at 0.8M states, 5.1 s at 3.2M, 10 s at 6.7M.

### 6h. The penalty method on a fixed horizon (`floor penalty`; measured 2026-10-02 18:00 EDT on the cluster)

A different problem from the rest of this file: a fixed horizon `n`, `J = TE(s_n)` (lambda 0),
from rest, every post-tick state kept at or above the floor. `floor penalty` polishes (coordinate
ascent, `polish`) under `quad * max(0, -h)^2` per tick, for each `quad` in `--schedule` in turn,
warm-starting each stage from the last; `--weight 0 --wall 0` turns the bubble and wall off. Every
run here was seeded with the free `n = 300` optimum at its `mu` (`runs/pencycle/mu`), stretched to
`n`, and every schedule starts with a `c = 0` stage, which is the free optimum at that `n`.
Schedules: `g10` (1, 10, .. 1e6), `g100` (1, 1e2, 1e4, 1e6), `g3` (1, 3, 10, .. 1e6), `lo` (1e-2,
1e-1, .. 1e6), and the abrupt `jump3` / `jump6` (straight to 1e3 / 1e6); against one polish under
the exact l1 wall (`wall10`, `wall100`) and the default bubble. Up to 400 passes a stage.

| n | y0 | mu | free J (flips) | gradual best (flips) | jump best | wall best | bubble |
|---|---|---|---|---|---|---|---|
| 200 | 40 | 0 | 8.57 (14) | 4.53 (2) `lo` | 0.95 (16) | 0.60 (16) | 3.93 (3) |
| 300 | 40 | 0 | 19.84 (33) | 0.92 (1) `g3` | -9.64 (37) | -10.21 (35) | 0.53 (1) |
| 300 | 60 | 0 | 19.84 (33) | 15.19 (15) `lo` | 5.82 (35) | 5.55 (34) | 13.55 (31) |
| 400 | 40 | 0 | 17.40 (39) | -7.75 (0) `g3` | -13.92 (16) | -20.15 (17) | -11.74 (1) |
| 400 | 60 | 0 | 17.40 (39) | 6.23 (1) `g3` | -5.13 (43) | -4.21 (29) | 4.33 (10) |
| 300 | 40 | 1e-4 | 19.70 (0) | 0.71 (0) `g3` | -10.43 (1) | -10.97 (0) | 0.48 (0) |

Flips are chatter as in 6b. The last stage of every run is within 1.2e-3 blocks of feasible
(5e-7 at `c = 1e6`). `y0 = 10` is infeasible at every `n` here and `y0 = 20` at `n = 300` is barely
feasible (the endurance there is about 320 ticks), so those cells only measure violation; `n = 200,
y0 = 60` does not bind. `runs/penalty/v1/summarize.py` prints every cell.

* **Raise the price gradually.** Every gradual schedule beats the bubble, the exact wall and the
  abrupt jumps on every binding cell, by 10-20 blocks of `TE` over the jumps at `n = 300, 400`.
  Starting at 1e-2 or stepping by 3 is a little better than by 10 or 100 (`g3` or `lo` win each
  cell). The jumps and the wall land on the same answer as each other: from the infeasible free
  optimum, a steep price makes the first stage a feasibility problem, and its nearest feasible
  point keeps the free dive's shape -- a flat glide, then a fast dive onto the floor -- where the
  gradual schedules move the dive earlier and shallower (`fig/profiles_*.png`).
* **Mostly, the floor removes the dive chatter by capping the sink rate.** At `mu = 0`, outside
  `n = 300, y0 = 60`, the gradual answers fall at most 0.33-0.57 blocks/tick and have 0-7 flips;
  the jumps and the wall fall at 0.63-0.79 and keep 14-43, the free optimum's bang-bang 0-to-85
  alternation (`fig/chatter_vs_sink.png`). That matches the free cycle's two-tick sink threshold
  (about -0.6 b/t). No run chatters at the floor: every optimum here touches it at one or two ticks
  at the dive's bottom and does not ride it, so this problem says nothing about chatter on a
  constrained arc.
* **Two exceptions, not explained.** (1) `n = 300, y0 = 60` keeps 15-31 flips under every method:
  +-3 to +-7 degree zigzags around 35-45 nose-down, 18-25 blocks above the floor, falling at
  0.42-0.55 blocks/tick when they start. It is not a stalled search: polished again from its own
  5-tick running mean, it grows back (smaller, to a J 0.04 lower). It sits where 6e found the
  flight convex in how nose-down is spread (lift dump ~ `sin^2 p`), which would not need the sink
  threshold. (2) `n = 400, y0 = 40` under `jump3`/`jump6` alternates 85 and 16 degrees at 0.26-0.48
  blocks/tick, 6-15 blocks up. Neither was isolated.
* **A large `c` makes the per-pitch score a staircase.** At `c = 1e6` a 0.05-degree pitch change moves
  the score by thousands of blocks, because the `mth_lut` stairs move the replay's under-floor depth
  and `c` multiplies it. The certificate's residual then grows with `c` (3e-3, 0.35, 35 at 1e2, 1e4,
  1e6 on a `lambda = 4, y0 = 20` run stalled 3.3 blocks under the floor), and it measures the stairs,
  not progress: read feasibility off the replay, as everywhere else here.
* Cost: `g3` took 69 core-minutes over its 36 cells, `jump6` 16. The first priced stage often hits
  the 400-pass cap (it starts 40-60 blocks under the floor); later stages stop in 12-30 passes.

Data: `runs/penalty/v1` (`out/<cell>_<schedule>_s<stage>.{pitches,trace}`, `.tsv` per run;
`make_work.py`, `summarize.py`, `check_claims.py` for the bullets), drawn by `plot.py` into `runs/penalty/fig/penalty.html` (every
stage of a schedule, and the last stage of every method, per cell) and by `figs.py`.
`tools/penalty.sbatch` ran it; `examples/penalty_probe.rs` and `examples/smooth_test.rs` are the
staircase and zigzag checks.

### 7. Exact `f32` output (a correctness fix, kept)

An exit optimum skims the floor at zero margin. Written to four decimals, one `y0 = 30` schedule
replayed 0.004 blocks through the floor it had cleared. Pitches are written as the shortest
decimal that round-trips their `f32`.

### Multi-start, and the collapse onto a hold

At `y0 <= 8` every start lands on the same optimum, so the hold -13 collapse was not seen there.
It is real higher up. At `y0 = 30`, cap 450: from hold -13 the exit ascent finds a plain glide and
exits at 395 ticks. Seeded with a steady 150-tick cycle tiled end to end, it survives the cap,
doing two full pumps that each bottom out at exactly the floor (clearance 0.000). No
per-tick move gets from a glide to a pump, so **past the heights where pumping pays, the search has
to be seeded with a pump.** The tiled cycle is a poor one: it pumps from the first tick. The
`minipump` seed -- `d` nose-down for `k` ticks, 0 for 10, -40 while `v_y` rises, then 0, with `k`
fitted to the best exit (and `d` on a 2.5-degree grid unless given) -- wins every endurance cell
from `y0 = 14` and most range cells, by up to 16% (`y0 = 19`, 279.0 against 239.7 ticks).
Range above `y0 = 22` still has several local optima: warming each cell from its neighbors' winners
beat every cold seed at `y0 = 29, 31, 32`, and the curve is ragged there.

**Running the corpus.** `tools/floor.sbatch` + `tools/floor_job.sh` run a work list of
`<y0> <time|dist> <name> <init spec>` lines, one array task per node, eight single-threaded solves
at a time, against the tree rsync'd to `~/elytra-atlas` on `cif-cpu` (the same one `sweep` uses); `tools/floor_pick.py <out>
--install runs/floor` keeps the best per cell. The v2 corpus was three cold seeds per cell, then
two rounds of warm starts from the neighbors' winners: 341 solves, 7.8 core-hours, about 25 minutes of wall clock on the rack.

## Infinite flight

**Superseded in part by 6g (2026-09-26):** the backward DP's policy flies forever from rest at
`y0 = 29.0` and not from 28.75, with laps that gain energy from a peak of 26.7, so the threshold below is too high and
the break-even of about 27 is the `pumps` laps', not the physics'. The rest of this section
is about those laps.

From rest, flight was sustained from **`y0 = 29.25`**. At 29.0 no run sustained it: 8 runs from
the `pumps` seed (two starts, two step sizes, 2 or 4 laps per round) all died after 8-9 climbs.
That brackets the threshold in (29.0, 29.25] **as found, not proved**: a better lap or a better
first dive could lower it. Measured 2026-09-25 18:30 EDT on the laptop, commit 93aa9da plus
`--after`, `runs/floor/v13-pumps/{inf,cont}`.

The test is the one below: total energy rising lap over lap once the first dive is done.
Surviving a cap is not the test. Each flight is endurance (`--mode time --method grad`) seeded
with `pumps:<K>`, then continued in laps. Each round keeps the solved schedule to its last apex,
flies 2 or 4 more seed laps from there (`--after`), and re-solves the whole flight, for 6-8 rounds
with the cap allowed to reach 38400. At 29.25 and above this grows until stopped: 28-39 climbs and
9600-22800 ticks, peak energy climbing into the hundreds. At 29.0 and below, no round adds a lap
that survives. Energy above the floor (`TE + y0`, blocks) at each peak and dip, `floor laps`:

```
 y0     t*       peak energy, lap by lap (the best run at each y0)
 28.5   1109     25.6 24.8 23.3 21.4 17.4 11.6 3.3
 29.0   1590     26.6 26.4 25.8 25.9 24.8 23.2 20.3 15.8 8.7
 29.25  13528    27.0 27.2 27.3 28.2 29.1 30.8 33.4 38.3 45.5 56.3 ..  (28 climbs)
 29.5   19618    27.4 27.9 28.6 30.3 32.7 36.7 42.2 51.8 63.0 78.1 ..  (36 climbs, 322 at the last)
```

**What sets it: an unstable fixed point of the lap map, plus the first dive's extra loss.** Split
each lap at the tick the pull starts, about 0.4-0.5 blocks above the floor. The dip then bottoms
out 5e-4 to 2e-3 blocks off it (the graze margin).

```
 lap from peak E   dive loss   climb gain   net     speed at the dip   (y0 = 29.5, then 29.0)
     27.43          -12.44       12.90      +0.46       1.462
     28.59          -12.86       14.50      +1.64       1.485
     32.72          -14.76       18.68      +3.92       1.630
     42.17          -19.16       28.74      +9.59       1.808
     26.41          -11.91       11.30      -0.61       1.427
     24.80          -11.48        9.90      -1.58       1.391
     23.21          -10.82        7.92      -2.90       1.323
 first dive, from rest at y0 = 29.25:  -14.64, then a normal lap from 26.98
```

* **The dive loses a near-constant fraction of the peak energy,** 45-48% from `E = 23` to `63`,
  mostly to drag on the way down.
* **The climb's gain grows faster than that with the speed at the dip.** It is 1.7 blocks at 1.10
  blocks/tick (`y0 = 20`), 9.9 at 1.39, 12.4 at 1.45, 18.7 at 1.63 and 35.9 at 1.98. The energy is
  made by the pull: nose-up, the kernel moves `cu = 0.04 |v_z| sin(-pitch)` per tick from `v_z`
  into `v_y` at 3.2 to 1 (`fall_flying_partials`). That creates energy once `v_y` is large, and it
  scales with the horizontal speed the dive delivered.
* So the net per lap crosses zero at a **peak energy of about 27 blocks** (+0.14 at 26.98,
  -0.16 at 26.57). It is unstable: above it the gain grows every lap (+15 by `E = 63`), and below
  it the loss does, which is why 29.0 dies in eight laps instead of hovering.
* **The first dive from rest costs about 2.3 blocks more than a later lap from the same energy**
  (-14.6 from 29.25 against -12.3 from 27.0), and it reaches the same speed at the dip, 1.46. The
  threshold is therefore about 27 + 2.3. Why the first dive loses more was not isolated.
* **The floor enters through the dip.** Every lap's pull starts 0.4-0.5 blocks above the floor
  and bottoms out on it, so a lap's whole dive is its peak's height above the floor, and the dip
  speed is what that height buys. With no floor the free first dive from rest goes 42.75 blocks
  deep (below), so every floor in this range clips it.

**The earlier claim, 35-36 blocks, was an upper bound and is superseded.** It came from one steady
150-tick `lambda = 0` cycle tiled four times (`runs/steady/nlamsweep/out/n0150_lamP0/tight_t0100.pitches`),
`floor exit --mode time --n 600`, coordinate ascent. With the same test, total energy at
`t = 150, 300, 450, 600`, that run found:

```
 y0    TE at t = 150    300      450      600     first dive bottoms at
 32         -6.01     -10.83   -19.92   -30.40    the floor
 33         -5.54      -8.90   -15.68   -25.52    the floor
 34         -5.17      -7.33   -11.94   -19.58    the floor
 35         -4.82      -5.56    -7.13   -10.29    the floor
 36         -4.48      -4.19    -3.59    -2.16    the floor      sustained
 40         -4.23      -2.51     0.14     3.93    the floor      sustained
 44         -3.55      -0.56     2.35     6.00    -42.75, free   sustained
 50         -3.55      -0.56     2.35     6.00    -42.75, free   sustained
```

From 44 up the floor never binds and the answer stops depending on `y0`. The seed was one
150-tick cycle, and the shallowest break-even steady cycle in `runs/steady/nlamsweep` (1233
cyclic cells, `dy >= 0`, `n >= 150`) dives 33.4 blocks below its own start. A per-tick search
cannot restructure such a cycle into a shorter lap (see the collapse above). The `pumps` laps last 155-195 ticks, peak about 27 blocks above the floor,
and graze it at every dip.

**A steady cycle under a floor** is still the honest measurement: maximize per-lap gain subject
to `min y >= -D`, and find the least `D` at which it is `>= 0`. That would measure the fixed
point (about 27 blocks of peak energy) directly, without a first dive. It is not easy today:
`polish` asserts that a floored polish takes neither `--steady` nor `--block` (`src/opt.rs`,
"a floored polish supports neither"), because neither line search prices the floor. A cheaper
check is to repeat one lap from these runs as a `tile:` seed with `--vy/--vz` set to its peak
velocity.

## Tricks not tried

* **Lexicographic ranking** (Deb's rules: feasible beats infeasible, then objective, then least
  violation). A drop-in for the per-tick comparison with no weight to tune; not needed once the
  exit objective replaced the bubble.
* **Smoothing a touch-and-go** by scoring a soft minimum over the pre-exit clearance alongside
  `t*`, so that a near-touch is ranked before it becomes a crossing.
* **Velocity jitter** (`Jitter`), which averages the knife edges over starting states, for when
  the uncertainty is in `v0`. A raised floor is not a separate trick: a floor `delta` higher is
  the same problem as `y0 - delta`.

## Running it

```
floor probe
floor exit   --y0 8 --mode time|dist [--init hold:-13|minipump|pumps:<K>] [--after <file>] [--ke <c>] [--shift] [--n 150] [--tol 1e-3] [--out <file>]
             [--method tick|grad|grad+tick] [--iters 2000] [--max-step 5] [--graze 1e-2,3e-3,1e-3]
             [--bubble <margin>,<weight> [--shrink <r>] [--anneal <k>]] [--graze off]
floor endure --y0 8 [--lambda 20 --anneal 3] [--out <file>]
floor safety --y0 4 --n 37 [--init <spec>]
floor penalty --y0 40 --n 300 --weight 0 --wall 0 --schedule 0,1,3,10,..,1e6 [--mu 0] [--init <spec>] [--out <prefix>]
floor solve  --y0 4 --n 36 [--init <spec>]
floor depth  --file <pitches> [--vy --vz] [--every 50]
floor laps   --file <pitches> --y0 <h>
```

Init specs: `hold:<p>`, `pump:<p_down>,<k>,<p_up>`, `tile:<file>` (a cycle repeated), `stretch:<file>`
(resampled to the horizon), a file, or
(`exit` only) `minipump[:<d>[,<k>]]` and `pumps:<K>[,d=..,lvl=..,pull=..,a=..,a2=..,g=..,end=..]` (6f).
`--mu` and `--limit` are the usual curvature price and pitch limit, defaulting to `1e-4` and `85`
as in `runs/atlas`. `runs/floor/` holds the best schedule per cell (`exit_{time,dist}_y<y0>.pitches`, `y0 = 1..32`),
`runs/floor/v7-30pass/` the with/without-tail-shift ascents, `v8-bubble/` and `v9-bubble-cont/` the floor-bubble ones and
`v10-pen/` the curvature-price shapes and search moves, `v11-grad/` gradient against coordinate ascent, `v12-bubble-grad/` the gradient with a bubble, `v13-pumps/` the K-climb seed and the infinite-flight scan with it, `v14-wobble/` the wobble study, `v16-dp/` the backward DP; `tools/plot_floor_profiles.py` draws the no-shift v7 run and all of v8-v12,
`runs/floor/v2-cluster/` every solve behind the best schedules,
and `runs/floor/inf/` the first infinite-flight scan (tiled cycle, superseded).
