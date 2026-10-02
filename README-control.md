# The control space, and where the weird answers come from

`README-sweep.md` records that polishing the exact objective to convergence produces a schedule
no human can fly, and that the way out was to stop the optimizer early -- a pass budget, or the
`--lag1-floor` guard. That works, and it is unsatisfying for a specific reason: it makes the
answer a property of when the optimizer was interrupted rather than a property of the problem.
It also invites the reading "the optimum really is weird, we just cannot reach it", which would
mean the corpus is a record of an optimizer's stamina.

This document is the follow-up. It says what the degenerate answers actually are, shows that a
stronger optimizer finds them faster rather than avoiding them, and replaces the stopping rule
with two things that go in the header and leave the polish free to converge: a *price* on how
much the schedule asks of a wrist, and a *margin* between the pitch and the ends of its range.

The short version:

* Adam at 20000 passes -- a hundred times more passes than coordinate ascent gets, for less wall
  clock -- chatters harder than coordinate ascent does and scores higher from the same seed. The
  degeneracy is not something a better optimizer avoids.
* There are **two** degeneracies. The chatter is worth 0.7%. The other one -- parking a pitch a
  hundredth of a degree from +-90, where `look_hor_length` underflows and the aerodynamics
  switch off -- is worth five to seventeen blocks of fragility, is invisible to `lag1`, and is in
  91% of the cells of the corpus we have.
* Total variation cannot separate a flick from chatter (a factor of 2.3). The *second* difference
  can (380 at the 95th percentile), in an l1 norm, because a real schedule's curvature is sparse.
* Priced and margined, polished to convergence with no stopping rule at all: at the reference
  cycle's own curvature budget, `dJ` **21.83 against a person's 21.49** for a single cycle from
  the reference's own starting velocity, with a *lower* peak angular acceleration and no
  fragility left.
* **21.83 is the answer to the question the sweep asks**, which is: best schedule of `n` ticks
  from the swept `v0`, terminal velocity free. Every cell of the rebuilt corpus solves exactly
  that and nothing else.
* *Under a different question* -- if the schedule has to be **repeatable**, flown over and over
  the way a person flies the reference -- most of that gain is a first-cycle transient. Fly it
  again from where it ended and it settles at 21.53, because it ends 0.07 blocks/tick slower
  than it started and books the difference as climb; iterating `v0 := v_end` to a self-consistent
  cell sustains 21.574 against a person's 21.494, +0.37%. **That is a condition I added, not one
  the objective states**, so it is scoped to its own section and does not apply to the corpus.
  See the scope note under "Most of the gain is a first-cycle transient".

Everything below is at the cell every chatter number in `README-sweep.md` was measured on:
`n = 300`, `lambda = 0`, `v0 = (0.167467, 0.200887)`. `J` is `J(s_n)`; `dJ = J - 0.4275`.

## The optimizer is not too weak

`examples/gd.rs` is finite-difference gradient ascent with Adam, sharing the objective and the
jitter with `polish`. A gradient pass costs two tail replays per tick; a coordinate-ascent pass
cost a few hundred when this was measured (721 sweep points plus 70 ternary steps on a global
pass, 321 plus 70 on a local one). Measured on this cell: 1.8 ms against 192 ms per pass, a
factor of 108, so 20000 gradient passes cost 36 seconds against 20 seconds for 104
coordinate-ascent passes. Since 2026-10-02 the per-tick search is `opt::coordinate_argmax`
(2-degree samples on global passes, 0.5 on local ones, golden-section refinement, and a scan of
the `mth_lut` stairs around the answer), which spends 1.7-4.2x fewer simulated ticks per polish
at the same or better J. That narrows the factor but does not change the comparison below.

| | J | TV | lag1 | max abs d2p | residual |
|---|---|---|---|---|---|
| reference cycle (flown by a person) | 21.929 | 248 | +0.48 | 38 | - |
| coordinate ascent, no price, from the reference | 22.180 | 2121 | -0.73 | 180 | 3.8e-5 |
| coordinate ascent, no price, from a chattering seed | 22.346 | 2667 | -0.80 | 180 | 4.3e-5 |
| **Adam, 20000 passes, from the reference** | **22.198** | **3751** | **-0.75** | **180** | **6.5e-4** |
| Adam, 7000 passes, from a chattering seed | 22.367 | 4326 | -0.89 | 179 | - |

Gradient ascent chatters *harder* than coordinate ascent and scores higher doing it. Plain SGD
diverges at every step size that moves at all (`lr` 30 leaves `dJ` at 17.7 with a gradient norm
of 375; `lr` 300 goes to -17.2). So the degeneracy is not a Gauss-Seidel artifact and it is not
a budget problem. Any optimizer that is actually good at this objective finds it.

## What the degenerate answers are, exactly

Two separate things, and only one of them is the chatter.

### 1. Chatter: tick-rate alternation of the aerodynamics

At pitch 0 the elytra is fully engaged; at 90 `lift_force = cos^2` is zero to fifteen digits, so
the tick is very nearly ballistic. Alternating between them beats holding any fixed pitch. It is
localized: ticks ~160-192, the end of the dive, and nowhere else.

Things that do not detect it, with the numbers:

| statistic | reference cycle | chattering optimum | ratio |
|---|---|---|---|
| total variation, deg | 248 | 2407 | 9.7 |
| max abs first difference, deg/tick | 38.3 | 90.0 | **2.3** |
| max abs second difference, deg/tick^2 | 37.9 | 180.0 | 4.7 |
| 95th pct abs second difference | 0.4 | 153.2 | 380 |
| sum abs second difference | 147.1 | 4211.1 | 29 |

The first difference is why a slew-rate limit cannot work: the reference cycle's snap really does
cover 38 degrees in one tick, so any budget that admits it admits an alternation. The *second*
difference separates them, and the reference's is **sparse** -- a straight line almost everywhere
with a handful of corners. So the norm to charge is l1, which is l1 trend filtering: it buys a
corner wherever one earns its keep and charges nothing for the straight runs between.

Two other things were tried and did not work, and are recorded so they are not tried again:

* **Sub-tick phase.** Read the schedule as samples of a piecewise linear control and let the game
  sample it at `t + phi` instead of `t` (`examples/fragility.rs`). Averaged over phase the
  chattering schedule scores `dJ` 21.85 against the reference's 21.54, and its worst phase is
  21.78. Being unable to phase-lock your mouse to the tick clock does not cost you the exploit.
* **Initial-velocity jitter**, re-measured here: `E[dJ]` at sigma 0.1 is *higher* than the
  unperturbed value for every schedule in the table, chattering or not, and the 5th percentile
  is about 1.9 blocks down for all of them equally. It does not discriminate.

### 2. Boundary parking: sitting a hundredth of a degree from a cliff

This one is larger than the chatter and a curvature price does not touch it.

`look_hor_length` is `|cos(pitch)|`, and the descent-to-forward conversion, the forward-to-up
conversion and the turning term are all behind `if look_hor_length > 0.0`. Both ends of the pitch
range are that gate rather than a limit, and which cliff you get depends on the trig:

* **`mth_lut`, which is vanilla.** `Mth::cos(-90 deg)` indexes `SIN[0]`, exactly `0.0`, so the
  gate *fails*: pitch -90 is a dead tick with no lift, no conversion and no turning. At +90 the
  same table gives +9.6e-5 and the gate passes -- the two ends are not symmetric. The flick wants
  the forward-to-up conversion at its maximum, which is at -90, so the optimizer parks at
  **-89.98902**, one table cell (0.0055 deg) short of the cliff. Overshoot by 0.05 degrees and
  that tick's next `v_y` is 0.331 instead of 0.611 and the schedule loses **5.2 blocks**.
* **`libm`, the current default.** `cos(90 deg)` in f32 is `-4.4e-8`, *negative*, so
  `look_angle.z / look_hor_length` flips to -1: forward reverses and the turning term becomes
  `-0.2 * v_z`, a fifth of your speed every tick. Scanned at tick 169 (`examples/razor.rs`) the
  cliff sits between pitch 89.9999924 (`dJ` 21.918) and 90.0000000 (`dJ` -3.395): 25 blocks
  across 7.6e-6 of a degree. The optimizer parks 6e-5 degrees from it.

Worst loss from nudging one single tick by 0.05 degrees (`examples/whichtick.rs`, vanilla trig):

```
reference cycle (flown by a person)     0.0002 blocks
polished under a curvature price        5.2353 blocks   tick 209, pitch -89.98902
polished with no price at all          16.9060 blocks   tick 188, pitch -90.00000
```

Minecraft delivers rotation in steps of about `0.15 * sensitivity` degrees. Nobody holds a pitch
a hundredth of a degree off a cliff on purpose, so the pitches beyond a margin are not worse than
the ones inside it -- they are unavailable, which makes this a statement about the control space
and not a term in the objective.

**This also means the corpus as it stands has a portability problem one level below the one
`sweep fingerprint` was built for.** It is written under `trig = libm`, and any cell whose
optimum leans on pitch +90 is leaning on the sign of a rounding error that vanilla does not have.

## The fix: say what the control can do, then converge

Two additions to `PolishOpts`, both in `Rough` in `src/opt.rs`, both recorded in the profile
header so `sweep verify --optimality` re-certifies against the same objective the writer used:

```
--mu <blocks per deg/tick^2>   price on sum |p[t+1] - 2p[t] + p[t-1]|
--limit <deg>                  largest |pitch| the schedule may use
--presmooth <k>                replace the seed by its local mean over k ticks, before polishing
--cap / --slew-cap             hard caps on the second / first difference (available, not used below)
```

`--mu` is a term in the utility function and says so; `--limit` is a restriction of the
admissible set. Neither is a stopping rule, and with either of them on, `--lag1-floor` is off,
`--tol` is 0.002, and the polish runs until it stops moving. Residuals below are 1e-5 or better.

**These replace the velocity jitter; they do not sit alongside it.** Every cell of the rebuilt
corpus carries `# jitter 0`, meaning it was optimized from the exact swept `v0` -- against 403
cells of the corpus we have, every one of them written under `# jitter 0.1 8 1592598191`. Three
reasons to drop it:

1. **It did not work.** The corpus we have *is* the jittered corpus, and its median `curv_l1` is
   4457, its median `lag1` -0.78, and 91% of its cells park a pitch at the gate.
2. **It answers a different question.** Jitter maximizes a mean over eight random starting
   velocities; the sweep's stated problem is one schedule from one exact `v0`. The price and the
   limit leave that problem alone and restrict the *control*.
3. **It splits what is optimized from what is certified.** With jitter on, the header's `dJ` is
   an exact-`v0` replay while the schedule maximized the jittered mean, so the file's headline
   number is not the quantity that was maximized. With `# jitter 0` they are the same number,
   which is what makes the certificate mean anything.

Velocity jitter is still used, as a *test*: `examples/fragility.rs` perturbs `v0` after the fact
to measure how sharp an optimum is. Testing against a perturbation and training against one are
different things, and only the first survived.

`--presmooth` is the piece that makes it work, and it is not cosmetic. A chattering control is
the discrete stand-in for a *relaxed* control: at each tick the optimizer is really choosing a
distribution over pitches and realizing the mixture by alternating. The ordinary control that
means the same thing is the local mean. Skip it and coordinate ascent can stall, because an l1
penalty on second differences couples three neighboring coordinates -- it is a fused-l1 term,
and coordinate descent has no guarantee on a nonsmooth objective that couples coordinates.
Measured: seeded from a schedule alternating -90/+90, the same price converged (residual 2.2e-7)
to `curv_l1` 1295, against 79 from the same price seeded from the local mean.

### The recipe

```
# 1. relax: solve the problem with no price on the control at all, to convergence
sweep polish --trig mth_lut --n 300 --vy 0.167467 --vz 0.200887 --lambda 0 \
             --passes 200 --tol 0.005 --init <seed> --out relaxed.pitches

# 2. project and polish: local mean, then converge under a price and a margin
sweep polish --trig mth_lut --n 300 --vy 0.167467 --vz 0.200887 --lambda 0 \
             --passes 200 --tol 0.002 --init relaxed.pitches \
             --presmooth 9 --mu 0.0001 --penalty l1 --limit 85 --out flyable.pitches
```

`mu = 1e-4` is the point on the frontier below that sits at the reference cycle's own curvature;
pick a different one by reading the table, not by tuning.

Step 1 is not optional and not a mistake to be avoided: without a relaxation step the price
lands 0.29 blocks lower, and graduated smoothing does not substitute for it.

| what was projected at `mu = 1e-3` | its own J | J after projection |
|---|---|---|
| coordinate ascent from a chattering seed | 22.346 | **22.213** |
| Adam from the reference | 22.198 | 22.073 |
| coordinate ascent from the reference | 22.180 | 22.061 |
| nothing -- `mu` annealed 0.3 -> 0.001 from the reference, never chattering | - | 21.924 |
| nothing -- `mu = 1e-3` straight from the reference | - | 21.923 |

Annealing `mu` down from 0.3 lands within 0.001 blocks of where the direct run lands. The chatter
is worth something as *search* even though it is worthless as an *answer*.

**But the projection is not monotone in the relaxed J, and that is the sharp edge in this
recipe.** A second relaxed optimum, `B`, scores 22.402 -- the highest raw J anything found -- and
projects *worse* than the 22.346 one under every filter but one:

```
projection          from relaxed B (J 22.402)     from relaxed A (J 22.346)
none                            14.23                          -
box 3 / 5 / 9        15.49 / 15.73 / 15.67       22.165 / 22.161 / 22.156
box 15 / 25                       -              22.130 / 22.047
median 3 / 5 / 7 / 9  15.32 / 15.69 / 15.92 / 16.01   -    / 22.179 / 22.167 / -
median 13                       22.136                         -
```

Every one of those is converged -- residuals 1e-5 or better, several at the 200-pass ceiling. So
`B` is simply a worse *basin* for a flyable schedule despite being a better relaxed point, and a
projection width that is fine for one relaxed optimum is catastrophic for another. There is no
way to tell from the relaxed schedule which you have.

The recipe therefore has to be multi-start, which is cheap: relax from two or three seeds,
project each at `k` in {3, 5, 9, 15} with both filters, polish all of them, and keep the best on
`J - mu * sum|d2p|`. Twenty-four runs of half a minute. What is *not* legitimate is picking one
projection, getting 15.66, and reporting it as the answer.

`tools/flyable.sh <outdir> <mu>` is that, end to end:

```
N=300 tools/flyable.sh runs/flyable 1e-4         # cell from N / LAM / VY / VZ, defaults as above
```

and the spread across its 24 candidates on the standard cell is the argument for doing it:

```
score      dJ    curv_l1  curv_max  lag1   candidate
21.851  21.868     169.3      34.3  +0.65  r2_box15      <- best
21.771  21.792     209.5      47.1  +0.56  r3_med9
21.767  21.782     150.5      25.7  +0.70  r3_med5
21.629  21.643     147.1      25.0  +0.71  r1_box9
20.516  20.547     305.1      47.7  +0.52  r2_box9
19.799  19.838     383.6      49.4  +0.51  r1_box5
17.499  17.519     204.1      42.5  +0.63  r1_box3
15.064  15.084     199.3      39.6  +0.66  r3_box3       <- worst
```

Every one of those is converged. Six and three quarters blocks separate the best from the worst,
and nothing about the schedule -- not `lag1`, not `curv_l1`, not looking at it -- distinguishes
them. Only `J` does.

## What it costs: J against hand movement

Every point below is a converged coordinate optimum (`--tol 0.002`, residuals 1e-5 or better) of
`J - mu * sum|d2p|` at `--limit 85`, projected from the same relaxed solution with
`--presmooth 9`. `curv_l1` is `sum |d2p|` in deg/tick^2 -- what the wrist actually does.

| mu | curv_l1 | max abs d2p | TV | lag1 | dJ |
|---|---|---|---|---|---|
| none (relaxed) | 6819 | 281 | 4169 | -0.66 | **21.975** |
| 3e-5 | 424 | 81 | 462 | +0.18 | 21.859 |
| **1e-4** | **147** | **27** | **336** | **+0.71** | **21.829** |
| 2e-4 | 131 | 28 | 328 | +0.75 | 21.817 |
| 5e-4 | 82 | 24 | 300 | +0.81 | 21.745 |
| 1e-3 | 79 | 16 | 291 | +0.84 | 21.728 |
| 2e-3 | 68 | 16 | 292 | +0.84 | 21.630 |
| 5e-3 | 65 | 10 | 294 | +0.88 | 21.494 |
| 1e-2 | 50 | 9 | 272 | +0.90 | 20.707 |
| 2e-2 and up | 34 | 8 | 246 | +0.90 | 13.23 (collapsed to a glide) |
| *reference cycle, flown by a person* | *147* | *38* | *248* | *+0.48* | *21.494* |

Read the bolded row against the italic one: **at the reference cycle's own curvature budget the
optimizer finds `dJ` 21.829 against a person's 21.494 -- 0.335 blocks, +1.6% -- while asking for
a lower peak angular acceleration, 27 deg/tick^2 against 38.** All the chatter above that is
worth 0.146 blocks, **0.7%**.

In blocks on the ground rather than in `J`, over the same 300 ticks:

```
                     dy       dz     curv_l1
reference cycle   21.333  342.243       147
mu = 1e-4         21.876  342.768       147
jitter+lag1floor  21.393  344.827       308
best of 24        21.958  342.647       169
```

Half a block more climb for the same hand movement -- **once**. The next section is why that
sentence needs the "once", and it is the most important correction in this document.

The frontier is very flat over the range anyone cares about. Dropping `curv_l1` from 424 to 79, a
factor of five less hand movement, costs 0.13 blocks. It only turns down below about 65, and by
`mu = 2e-2` the schedule has stopped pumping and is holding a glide -- the collapse `opt::shape`
already knows how to name.

## Most of the gain is a first-cycle transient

> **Scope note, added 2026-09-09 after josie read this.** This section and the two under it are
> about a question the corpus does not ask, and their conclusions must not be carried back into
> it. **The sweep solves: best schedule of `n` ticks starting from the swept `v0`, terminal
> velocity free.** That is the whole specification, and every cell of the rebuilt shard obeys it
> exactly -- all 363 carry `# v0 0.000000000 0.000000000`, the swept value, unmodified.
>
> What follows instead iterates `v0 := v_end` until a schedule ends where it began, which is a
> *self-consistency* condition I introduced, not a requirement of the problem. I went there
> because the reference schedule is something a person flies over and over, so comparing a
> one-shot optimum against it on one cycle seemed unfair. That reasoning only holds if the
> deliverable is a repeatable cycle. **If the deliverable is the best schedule from a given
> `v0` -- which is what the sweep actually produces -- then the first-cycle number *is* the
> answer**, the frontier table above is the right comparison, and nothing below applies.
>
> Concretely: the "+0.37% sustained" figure, and any number here described as being at a fixed
> point or a limit cycle, is measured under that extra condition. The single-cycle numbers from
> the swept `v0` are the frontier table ("What it costs") and the head-to-head section.
>
> Also note that `# structure cyclic` and `# cycles N` in the profile headers have nothing to do
> with any of this. They are shape classifiers on the *pitch schedule* -- whether it shows the
> dive/snap/flick form or has collapsed, and how many dive-then-flick transitions it contains
> (used to flag the `n >= 550` tiling degeneracy). Neither constrains `v0`, and neither is
> affected by the iteration below.

The objective optimizes one cycle with the terminal velocity free. Nothing in it asks the
schedule to be *repeatable*, and the answers are not: the reference cycle returns to within 0.024
of its own starting velocity (summed over the two components), the priced optima to 0.060-0.082. So they end slower than they started -- `mu = 1e-4` hands on a velocity 0.070 blocks/tick down
on the one it was given -- and book the difference as height.

`examples/repeat.rs` flies a schedule several times in a row, each cycle starting where the last
one ended. `dy` per repetition, vanilla trig:

```
                        #1      #2      #3      #4      #5   sum|dv| after #1
reference cycle     21.494  21.494  21.494  21.494  21.494        0.0001
jitter + lag1-floor 21.562  21.566  21.565  21.565  21.565        0.0017
mu = 1e-4           21.975  21.513  21.526  21.525  21.525        0.0821
best of 24          22.039  21.506  21.522  21.521  21.521        0.1029
relaxed (chatters)  22.114  21.661  21.677  21.676  21.676        0.0782
```

Everything settles into a limit cycle within three repetitions, and every optimized schedule
loses about 0.45 blocks doing it. The reference cycle loses nothing, because it already *is* its
own limit cycle -- that is what makes it a cycle a person flies rather than an opening move.

So the single-cycle numbers above are correct for the objective as stated and **wrong as a
statement about how fast you climb**. In sustained blocks per second, which is what
`OPTIMAL_CYCLE_RATE` in `opt.rs` already tracks:

```
reference cycle (a person)   21.494 blocks/cycle   1.43293 blocks/s    +0.00%
jitter + --lag1-floor 0.2    21.565                1.43767             +0.33%
mu = 1e-4 at the same v0     21.525                1.43500             +0.14%
relaxed, chattering          21.677                1.44513             +0.85%
```

Those are each schedule repeated from the reference cycle's `v0`, which is nobody's fixed point
but its own. The next section does it properly.

### The fix needs no change to the objective

`v0` is already a swept axis, so the self-consistent cell is just a fixed point of `v0 := v_end`.
Starting from the reference's velocity it converges in three iterations:

```
iter  v0                        -> v_end                    |dv|      dy
1     (0.167467, 0.200887)      -> (0.097730, 0.188597)   0.08203  21.975
2     (0.097730, 0.188597)      -> (0.100099, 0.188802)   0.00257  21.565
3     (0.100099, 0.188802)      -> (0.100012, 0.188797)   0.00009  21.574
8     (0.100033, 0.188800)      -> (0.100033, 0.188800)   0.00000  21.574
```

`runs/antichatter/fix/i8.pitches`: `curv_l1` 149 against the reference's 147, `curv_max` 26.5
against 38, `lag1` +0.696, residual **6.1e-9**, worst single-tick nudge 0.0145 blocks, and it
repeats at 21.574 forever, handing on exactly the velocity it was given.

The chatter has to be measured the same way or the comparison flatters somebody. Running the
identical iteration with `--mu 0` and no margin, from the same starting schedule, converges to its
own fixed point in three steps. All four of these are exact limit cycles:

```
                                       v0 it settles on        dy every cycle  sum|dv|
reference cycle (a person)      (0.167467, 0.200887)                   21.494  0.0001
priced, chain from the priced        (0.100033, 0.188800)              21.574  0.0000
priced, chain from the relaxed       (0.150303, 0.198255)              21.573  0.0000
relaxed, chattering                  (0.120937, 0.192570)              21.737  0.0000
```

So, sustained: **the best flyable schedule found beats a person by 0.080 blocks a cycle, +0.37%,
and the chatter beats it by 0.163, +0.76%.** That 0.76% is the same quantity the single-cycle
frontier put at 0.7%, arrived at by a completely different route, which is the best evidence
available that both are measured right.

"Found" is doing work in that sentence. Seven chains now agree, from starting velocities spread
across the plane (next section), which says the iteration is stable -- but they were all seeded
from the same family of schedules, so this bounds the *basin*, not the problem. The reference
cycle, by contrast, has one fixed point because it is one schedule, so that side of the
comparison is not a sample.

### The fixed point is set by the basin, not by where you start

Two chains suggested the fixed point was not unique -- they landed on `v0` = (0.100, 0.189) and
(0.150, 0.198), a long way apart, agreeing on `dy` to 0.001. Five more chains say that reading was
wrong. Seeded from the *same* schedule but started from deliberately scattered velocities:

```
start v0            settles at v0            dy      curv_l1  lag1
(0.050, 0.150)   -> (0.100101, 0.188794)   21.581      164   +0.66
(0.250, 0.250)   -> (0.100234, 0.188788)   21.580      164   +0.66
(0.000, 0.000)   -> (0.099965, 0.188912)   21.584      175   +0.62
(0.150, 0.300)   -> (0.099757, 0.188652)   21.573      149   +0.69
(-0.050, 0.190)  -> (0.099803, 0.188661)   21.581      173   +0.62
```

Every one lands on (0.1000, 0.1888) to within 5e-4, from starting velocities spread over
`vy` -0.05 to 0.25 and `vz` 0.15 to 0.30, and each is an exact limit cycle. So the iteration is
**not** sensitive to where it starts -- what selects the fixed point is the seed *schedule*, the
same basin dependence as everywhere else in this document. The (0.150, 0.198) outlier was seeded
from a different relaxed optimum, which is why it is somewhere else while scoring the same.

The best flyable fixed point found is 21.584, at `curv_l1` 175. At the reference cycle's own
curvature budget it is 21.573 at `curv_l1` 149, which is the number to quote against 21.494:
**+0.080 blocks, +0.37%.** Buying the extra 0.010 costs 26 more of `curv_l1`.

Read against the stopping rule, the honest summary is that on *rate* the two are a tie (1.43827
blocks/s against 1.43767), and the whole case for the price and the margin is robustness --
0.0145 blocks lost from a 0.05 degree nudge against 5.369.

## Is the answer actually robust, or only smooth?

Smooth is not the same as not-overfit, so the schedules were re-scored under noise they will
actually see. `dJ` under uniform pitch noise of amplitude `a` on *every* tick, 200 draws, vanilla
trig, 5th percentile in brackets (`examples/sens.rs`):

| | a = 0.001 | a = 0.01 | a = 0.05 | a = 0.5 |
|---|---|---|---|---|
| reference cycle | 21.494 (21.494) | 21.495 (21.494) | 21.495 (21.494) | 21.426 (21.343) |
| **mu = 1e-4, limit 85** | 21.823 (21.822) | 21.820 (21.817) | 21.796 (21.777) | 21.521 (21.317) |
| mu = 1e-3, limit 85 | 21.723 (21.722) | 21.720 (21.717) | 21.704 (21.686) | 21.519 (21.338) |
| relaxed, no price or limit | 21.969 (21.968) | **-19.98 (-44.43)** | -45.88 (-58.36) | -49.86 (-60.20) |

Worst loss from nudging one single tick by 0.05 degrees (`examples/whichtick.rs`):

```
reference cycle         0.0002 blocks   (0.002 summed over all 300 ticks)
mu = 1e-4, limit 85     0.0145 blocks   (0.126 summed)   at tick 192, pitch 0.00013
relaxed, no limit       5.2 to 16.9     (102 summed)     at pitch -90.00000
```

The remaining 0.0145 is the *other* corner in the physics -- the forward-to-up conversion is
gated on `lean_angle < 0`, so pitch 0 is a kink and the snap sits on it at 1e-4 of a degree. It
costs a hundredth of a block, which is the right size for something to be left alone.

At a seventh of a degree of pitch noise -- past any real input precision -- the priced schedule is
still 0.24 blocks ahead of the reference cycle on the mean. At half a degree they meet: 21.521
against 21.426 on the mean, and 21.317 against 21.343 at the fifth percentile, which is a tie.
That is the honest statement of how much of the +1.6% survives: all of it, until the hand is
worse than about half a degree, and none of it after.

## Head to head against the stopping rule

The same cell, under vanilla trig, comparing what `README-sweep.md` recommends today against
this. The old recipe is `--jitter 0.10 --draws 8 --lag1-floor 0.2` seeded from the reference
cycle; it stops itself at 12 passes and the pass budget above that changes nothing (8, 16 and 60
give bit-identical output).

| | dJ | curv_l1 | max abs d2p | lag1 | residual | worst 1-tick nudge | dJ at +-0.01 deg noise |
|---|---|---|---|---|---|---|---|
| reference cycle (a person) | 21.494 | 147 | 38 | +0.48 | - | 0.0002 | 21.495 (21.494) |
| jitter + `--lag1-floor 0.2` | 21.564 | 308 | 73 | +0.26 | 3.4e-4 | **5.369** | **18.464 (10.961)** |
| `--mu 1e-4 --limit 85` | **21.829** | **147** | **27** | +0.71 | **2.5e-6** | **0.015** | **21.820 (21.817)** |

Better on every axis at once: 0.265 blocks more `J`, half the summed curvature, a third of the
peak angular acceleration, and a residual 140 times smaller -- meaning it is actually converged
rather than interrupted.

The interesting columns are the last two. **The stopping rule catches the chatter and does nothing
at all about the boundary parking**, which is the larger fragility: its answer holds three
consecutive ticks at pitch -89.98902, one table cell from the gate, and each of them costs more
than five blocks if the hand is 0.05 degrees out. Under a hundredth of a degree of pitch noise it
loses 3.1 blocks on average and 10.6 at the fifth percentile. That is what "weird and overfit"
actually looks like once the chatter has been suppressed, and `lag1` cannot see it -- it reads a
perfectly respectable +0.26.

## The corpus as it stands

`python3 tools/audit.py runs/corpus`, medians per shard:

```
shard                    cells   curv_l1  curv_max    lag1  at +-90
vy+0.0000_vz+0.0000        403      4167       180   -0.77      91%
vy+0.2000_vz+0.2000         11       542        90   +0.33      91%
vy+0.2000_vz-0.2000         11       534        90   +0.14      91%
vy+0.4000_vz+0.2000         11       667        94   +0.15      91%
vy-0.2000_vz+0.2000         11       466        63   +0.42      82%
vy-0.2000_vz-0.2000         18       437        96   +0.15      94%
ALL                        509      2384       180   -0.70      91%
reference cycle              1       147        38   +0.48       0%
```

Two things, and the second is the one that was not known.

**The compounding under continuation is real and it is most of the grid.** The `(0, 0)` shard
carries the whole `n` x `lambda` plane, so it has the longest continuation paths, and its median
cell sits at `curv_l1` 4167 with `lag1` -0.77 -- fully degenerate, 28 times the reference cycle's
curvature. The nine small shards are 11 to 18 cells each, so their paths are short, and they land
at 437 to 667 with a positive `lag1`. `README-sweep.md` predicted exactly this ("a per-cell pass
budget accumulates along the continuation path") and the guard for it defaults off; this is the
measurement.

**Ninety-one percent of cells park a pitch against the gate, in every shard including the healthy
ones.** The anchor cell itself -- `n = 300, lambda = 0, v0 = 0`, the least-polished cell in the
corpus, `curv_l1` 187 and `lag1` +0.65, by every existing statistic a good profile -- holds three
ticks beyond 89.9 degrees. And the corpus is written under `trig = libm`, where that is the
25-blocks-across-7.6e-6-degrees cliff rather than the merely expensive one. So it is not only
that the degenerate cells are degenerate: the *good* cells are sitting on a discontinuity too,
and no statistic in the header says so.

Rebuilding wants `--trig mth_lut --limit 85 --mu 1e-4 --penalty l1` and the relax/project/polish recipe per
cell. That has not been done.

## Does it hold up away from one cell?

### The whole (0,0) shard, rebuilt

363 cells, through `sweep run` itself rather than a hand-rolled loop, so this is the production
path: breadth-first continuation over the *two-dimensional* `(n, lambda)` plane from one solved
anchor. That is where the old corpus's compounding actually happened, and the 1D lines below
cannot show whether a long path through the plane drifts. 66 minutes.

```
sweep run --trig mth_lut --out runs/antichatter/shard0 --vys 0 --vzs 0 \
          --ns <the corpus's 11> --lams <the corpus's 33> \
          --mu 0.0001 --penalty l1 --limit 85 --passes 60 --tol 0.002 --anchor <the solved n=300 cell>
```

The same 363 cells, before and after:

```
                     curv_l1   curv_max    lag1   cells parked past |pitch| 89.9
the corpus we have      4457        180   -0.78                              91%
rebuilt                  197         39   +0.64                               0%
reference cycle          147         38   +0.48                               0%
```

Medians. The **worst** rebuilt cell in the grid is `curv_l1` 579, `curv_max` 109, `lag1` +0.25 --
better than the old *median* on every one of them -- and no cell anywhere touches the gate.

What it cost, split by regime:

```
  n <= 500, the single cycle the corpus claims to sweep   297 cells   median dJ  +0.014
  n >= 550, where the old cells fly the cycle twice        66 cells   median dJ -27.561
```

So inside the regime the corpus is *about*, the rebuild is **free** -- very slightly positive on
the median, better in 168 cells and worse in 121 -- while cutting hand movement twenty-fold and
eliminating gate parking entirely. All 363 re-certify from their own headers: `sweep verify
--optimality` (at the time plain `sweep verify`, with the search before 2026-10-02) reports 363 ok, no mismatches and no bad headers, worst claimed residual 1.96e-3. The `n >= 550` band is the tiling degeneracy `README-sweep.md`
already names: those old cells climb twice as far because they fly the cycle twice, and the
rebuilt ones stay in the single-cycle family. `fig/10_shard.png` draws all of it; the third row is
the one to look at, because the rebuilt panel is empty.

### One axis of the corpus, rebuilt

`n` at `lambda = 0`, `v0 = 0` -- the spine of the `(0,0)` shard -- solved by `tools/flyable.sh`
at `n = 300` and continued outward, 11 cells, about seven minutes:

```
    n |   old dJ  curv_l1   lag1  @90 |   new dJ  curv_l1   lag1  @90
100 |   -8.050        8  +0.83    0 |   -8.051        3  +1.00    0
150 |   -3.704      221  +0.37    1 |   -2.931      227  +0.43    0
200 |    7.706      259  +0.56    2 |    8.384      192  +0.69    0
250 |   15.479      260  +0.46    3 |   15.815      123  +0.82    0
300 |   19.474      187  +0.65    3 |   19.614      104  +0.77    0
350 |   19.640      985  -0.11    9 |   19.818      133  +0.68    0
400 |   16.538     5315  -0.81   21 |   17.472      213  +0.61    0
450 |   12.922     6201  -0.83   23 |   13.906      210  +0.64    0
500 |    9.392     6869  -0.83   25 |   10.200      170  +0.72    0
550 |   27.011      774  +0.36    7 |    6.604      132  +0.71    0
600 |   41.221     6478  -0.83   25 |    3.051      130  +0.71    0
```

Better `J`, an order of magnitude less curvature, and no cell against the gate, at every `n`
from 100 to 500 at once. At `n = 400` it is +0.93 blocks with 25 times less hand movement.

`n = 550` and `n = 600` look like losses and are not: the old cells there are **flying the
cycle twice**. The `n = 600` cell climbs `dy` 40.98 over `dz` 673.6, which is two of the
`n = 300` cell's 19.5 and 337, and `README-sweep.md` already names that regime degenerate --
"a different regime that wins on J while not being the object under study". The rebuilt cells
stay in the single-cycle family (`# cycles 1`) and fall away from `n = 300` the way the
document says they should. So the corpus contains, unlabeled, exactly the tiling degeneracy it
warns about.

### The other axis of the same shard, rebuilt

`lambda` at `n = 300`, `v0 = 0`, continued outward from the same anchor, 21 cells, about eight
minutes. This is the axis the corpus sweeps most finely, so it is where compounding had the most
room to accumulate, and continuation along `lambda` under a price had not been tested at all:

```
 lambda |    old dJ  curv_l1   lag1  @90 |    new dJ  curv_l1   lag1  @90
  -4.00 |   -50.566     3553  -0.73   13 |   -50.494      192  +0.62    0
  -3.00 |   -36.485     3466  -0.76   11 |   -36.403      131  +0.66    0
  -2.00 |   -20.065     2971  -0.70   12 |   -19.925      117  +0.70    0
  -1.00 |    -1.192      817  -0.35    6 |    -1.041       98  +0.77    0
  -0.50 |     8.934      382  +0.15    5 |     9.063       97  +0.78    0
   0.00 |    19.474      187  +0.65    3 |    19.614      104  +0.77    0
   0.50 |    30.713      487  +0.11    5 |    30.826      114  +0.84    0
   1.00 |    42.617      949  -0.26    6 |    42.617      145  +0.82    0
   2.00 |    67.601     3775  -0.72   19 |    67.448      257  +0.57    0
   3.00 |    93.902     5373  -0.79   22 |    94.044      271  +0.60    0
   4.00 |   121.720     6671  -0.80   29 |   122.122      290  +0.49    0
```

Twenty-one of twenty-one cells: better `dJ` at seventeen of them, within 0.2% at the other four
(`lambda` 1.5 and 2 are 0.1 to 0.15 blocks down), an order of magnitude less curvature
everywhere, `lag1` positive everywhere against -0.80 to +0.65, and **no cell touching the gate**
against 3 to 29 ticks per cell. The old grid degrades symmetrically away from `lambda = 0` --
`curv_l1` goes 187 at the anchor to 3553 and 6671 at the ends -- which is the continuation
compounding, visible along both axes.

### Continuation holds under a price

Walking the winner along `n` at 310 through 400 with a single 40-pass polish per cell
(`--mu 1e-4 --limit 85`, no relaxation, no multi-start):

```
n        310   320   330   340   350   360   380   400
curv_l1  162   183   169   165   192   226   242   257
curv_max  26    30    32    35    37    38    42    39
lag1    +.69  +.67  +.69  +.65  +.61  +.62  +.62  +.61
```

`curv_l1` drifts up by 1.6x over nine hops and `lag1` sags from +0.69 to +0.61, so it is not
free -- but nothing collapses, `curv_max` stays under the reference cycle's 38 until `n = 380`,
and every residual is 1e-5. The unpriced control chain, the same hops seeded from the
chattering optimum, does not survive at all: it *collapses*, `J` going -20.7, -21.4, -22.1 with
total variation 32 -- the schedule stops pumping and holds a glide, which `opt::shape` names.

That matters for cost. Relax plus multi-start is minutes per cell, 30 hours for the grid;
continuation is 15 to 30 seconds, about three hours. A rebuild solves one anchor properly and
continues from it, re-anchoring when `lag1` sags.

### The recipe is not specific to one cell

Four more cells, `mu = 1e-4`, `--limit 85`:

```
cell        relaxed dJ  its curv_l1  best flyable dJ  its curv_l1  chatter worth
n = 150          0.725     458-1150            0.760           95   nothing (-0.035)
n = 450         15.878    7108-7394           15.697          202   0.181  (1.1%)
lambda = -1      0.787     584-4197            0.799          103   nothing (-0.012)
lambda = +1     45.606    4361-5053           45.460          184   0.146  (0.3%)
```

The relaxation chatters every time and the recipe returns something flyable every time. Twice
the *priced* answer beat the relaxed one outright, because the unpriced run converged into a
worse basin -- so the chatter is not reliably worth even the 0.7% it buys on the standard cell.

## What is still open

* **Eight of the ten shards have not been rebuilt.** The `(0,0)` shard has been, in full (below);
  the nine `v0 != 0` shards, 137 cells between them, are as they were.
* **The corpus is written under `trig = libm`.** Any cell leaning on pitch +90 is leaning on the
  sign of an f32 rounding error that vanilla does not have. The whole grid should be rebuilt
  under `--trig mth_lut`, or at minimum `--limit` should be on so it cannot matter. This is the
  same class of problem `sweep fingerprint` exists for, one level down.
* **The projection width is not predictable, only searchable.** `tools/flyable.sh` handles it by
  brute force, and that is honest but it is not understanding.

  One hypothesis was tested and **refuted**, which is worth recording so it is not retried. Looking
  at the two relaxed optima side by side (`runs/antichatter/fig/09_projection.png`) suggests a
  clean mechanism: `A`'s chatter swings between 0 and +90, so its local mean is still nose-down
  and continues the dive, while `B`'s swings between -90 and +90, so its local mean noses *up* for
  forty ticks in the middle of a dive. Both endpoints of `B`'s bang-bang straddle the
  `lean_angle < 0` corner, where the physics is discontinuous, so the mean is not a control that
  does anything like either. That predicts: the further nose-up the projection's mean goes, the
  worse it does.

  It does not hold. Across the fifteen projections already measured, the most nose-up degree the
  projection reaches does not separate the ones that worked from the ones that collapsed --
  `A` box-3 reaches -60 and lands at 22.165, `B` median-13 reaches -90 and lands at 22.136, while
  `B` box-9 reaches only -47 and lands at 15.667. It is the *seed* that decides, almost entirely,
  and `B` median-13 is the one exception any theory has to explain. "B is a worse basin" remains
  the only description that fits.
* **Relax/project does not iterate.** Feeding the projection back in as a fresh relaxation seed
  went 22.402 -> 15.66 -> 18.68 and kept falling. One round, then stop.
* **Every cell of the corpus is a first cycle, and it moves by a median of 2.2 blocks when you
  fly it twice.** `examples/repeat.rs` over all 509 cells, `dy` on the first cycle against `dy`
  at the limit cycle:

```
  shard                    cells  median d1-dss  |d1-dss| med  moves >1 block
  vy+0.0000_vz+0.0000        403         -2.149         2.204            84%
  vy+0.2000_vz+0.4000         11         +1.099         1.375            64%
  vy+0.4000_vz+0.4000         11         +4.837         5.047            82%
  vy-0.2000_vz-0.2000         18         -3.392         3.392           100%
  ALL                        509         -2.003         2.246            83%
```

  The sign follows `v0`: cells that start from rest spend their first cycle building speed and
  do *better* on every cycle after, cells that start fast do worse. Concretely, the `(0,0)` shard
  at `lambda = 0`:

```
      n     dy#1     dy#2     dy#5    rate#1   rate#5   (blocks/s)
    200    5.438   10.650   10.493    0.5438   1.0493
    300   19.064   21.456   21.488    1.2709   1.4325
    500    7.918   11.262   11.262    0.3167   0.4505
```

  The `n = 300` cell's header says `dJ 19.474`; repeated, it settles at 21.49, which is the
  reference cycle's own rate to three decimals -- it converges to the same limit cycle. Neither
  number is wrong and they answer different questions ("launch from a standstill" against "cruise"),
  but the header does not say which one it is, and 83% of the grid moves by more than a block
  between them. The fixed-point iteration above resolves one cell; what a rebuild should sweep,
  and how a fixed-point grid should relate to the current `v0` axis, is not settled here.
* `--cap` and `--slew-cap`, the hard-constraint versions of the same idea, are implemented and
  were not swept. A cap is arguably the more honest instrument than a price -- it says a
  pitch profile is unavailable rather than expensive -- and `cap = 45` would admit every move
  the reference cycle makes with 20% to spare.

## The code has been adversarially reviewed

Nine correctness bugs, found by handing the diff to a second model with the instruction "assume
there is a fourth" (I had already found three myself). None of them changed a published number --
every certificate re-verifies at its claimed residual, and the fixed-point chain moves by at most
8e-6 blocks. They are listed here because the *class* is worth knowing:

| | |
|---|---|
| the coordinate search kept an out-of-range incumbent | a warm start outside `limit` was never brought in |
| the ternary refine re-expanded its bracket to +-90 | |
| the coarse sweep stepped past `hi` | `round((hi-lo)/step)` overshoots for half of all intervals |
| f32 rounding crossed an exact constraint | limit 0.1 wrote 0.10000000149011612 |
| `feasible` returning nothing made the polish a no-op | infeasible seed, zero passes, no error |
| **an infeasible schedule certified at residual 0.0** | the check passed hardest where it should have failed |
| a skipped tick left its prefix stale | the sweep then optimized a trajectory nobody was flying |
| `project_cap` could exit still infeasible | "wide enough is constant" is false for a clamped box filter |
| `project_cap` ignored `slew_cap` entirely | |
| `--block` with `--mu` accepted a loss as a gain | 0.188 -> -4.168 reported as +0.19 |
| `jacobi_step` compared jittered against unjittered | pre-existing |
| resume matched the objective but not the price | |
| `--trig` leaked between files in the examples | a headerless file inherited the last profile's physics |
| a malformed `# rough` header parsed as a default | `typo` read as "no cap" |

Eight of those are the same shape: **an emptiness that reads as success.** An interval with
nothing in it, a filter with nothing left to smooth, a constraint nobody checked -- each returns
the value that means "converged, nothing to do here." That is the failure mode to design against
in this codebase, and `Rough::violation` exists so that the certificate can no longer be one of
them.

## Figures

Under `runs/antichatter/fig` (regenerate with the `tools/plot_*.py` and `tools/look.py` scripts;
`runs/` is gitignored, so they are not in the tree):

```
00_baseline.png    the schedules as they were: reference, hard-polished, jittered
01_hamiltonian.png the per-tick objective curve J(p) -- one peak, not two, so not a singular arc
02_mu.png          the first mu sweep, from both a smooth and a chattering seed: the same price
                   reaches the same place from either side, which is what made it worth pursuing
03_razor.png       one tick scanned across the gate it parks against: libm at +90 (a
                   25-block cliff across 7.6e-6 of a degree), mth_lut at +90 (no cliff),
                   mth_lut at -90 (a 5-block step one table cell wide)
04_frontier.png    J against summed |second difference|, every point converged
05_final.png       the priced schedules at mu = 1e-4, 1e-3, 5e-3
06_robust.png      dJ against pitch noise: the picture of what overfitting costs
07_best.png        reference / today's recipe / this recipe / the relaxed optimum
08_fixpoint.png    the answer: reference, today's recipe, and the fixed-point schedule. The
                   last has 3% of its per-tick delta power above 0.35 cycles/tick against the
                   reference cycle's 14%, so it is the smoothest of the three
10_shard.png       the whole (0,0) shard, old against rebuilt, over the (n, lambda) plane.
                   The third row is the one to look at: the rebuilt panel is empty, because
                   no cell parks at the gate. Drawn by tools/compare.py
09_projection.png  two relaxed optima under three box widths -- the picture behind a
                   hypothesis about why one projects and the other does not, which the
                   numbers then refuted; see "What is still open"
11_profiles.png    twelve representative cells of the (0,0) shard, old schedule under new,
                   spanning n 150/300/450 and lambda -2/0/+1/+3. Drawn by
                   tools/plot_pitch_grid.py
12_zoom.png        one cell (n 450, lambda 0) up close: the whole schedule, then the chatter
                   window, then the same window differenced. Drawn by tools/plot_pitch_zoom.py
13_family.png      the rebuilt schedules alone, as families along lambda and along n, with the
                   reference cycle over the top. Drawn by tools/plot_pitch_family.py
```

### What the profiles look like

`11_profiles.png` is the visual form of the shard table. The rebuilt schedule tracks the old one
almost everywhere -- the two lines are indistinguishable over most of every panel -- and departs
in exactly two places: a burst of tick-rate alternation the old cell spends somewhere in the
glide, and the timing of the flick. Nothing else moved. That is the argument that the price is
removing a pathology rather than reshaping the answer.

`12_zoom.png` is the same thing at one cell. From tick 195 to 236 the old `n 450, lambda 0` cell
slams between exactly 0 and exactly +90 degrees *every tick*, and the differenced panel shows
the change reversing sign on 34 of the 40 steps: a hand cannot do this and a wrist would not survive trying.
The rebuilt cell crosses the same window as a straight ramp from +35.5 to +42.2 degrees, and scores
**a block higher** (13.904 against 12.922) while doing it. The next section is about why that is
not the good news it looks like.

### The priced cell scores higher on the unpriced objective. That is not a win

Several panels of `11_profiles.png` show the restricted, priced solution beating the unrestricted
one on `dJ`, which is the raw `TE(s_n) + w*z_n` and *not* what the rebuild maximized. Read as
"the price improved `J`" that would be nonsense -- a smaller admissible set cannot contain a
better point. Three candidate explanations, checked in order.

**It is not the trig backend.** The old corpus is written under `trig = libm` and the rebuild
under `mth_lut`, so the two headers are not literally comparable. Re-evaluating every schedule
under both (`sweep polish --passes 0`): at this cell the old schedule scores 12.9219 under libm
and 12.9152 under `mth_lut`, the new one 13.8994 and 13.9037. Across all 363 cells the median
shift is -0.006 blocks and the largest is 0.014. The gap is a hundred times that.

**It is not the jitter, though it is worth noticing.** The old cell's header reads
`jitter 0.1 8 1592598191`: **the corpus we have was already regularized**, against a spread of
initial velocities, which was the anti-chatter measure of the day. It carries `curv_l1` 6201 and
23 ticks against the gate anyway. Its `dJ` is a *report* at the exact `v0` -- replay reproduces
12.921859 to the digit -- not the quantity it maximized, but it is the same quantity the rebuilt
header reports, so the comparison is fair.

**It is the basin.** Release each schedule from every restriction -- no price, no limit, no
jitter, `mth_lut`, polished to convergence -- and ask where the raw problem takes it:

| polished raw, from | J | curv_l1 | lag1 | passes | residual |
|---|---|---|---|---|---|
| the old cell | **12.924** | 5968 | -0.56 | 13 | 3.6e-5 |
| the rebuilt cell | **14.081** | 4078 | -0.23 | 84 | 2.2e-4 |

The old cell climbs **0.002 blocks and stops**. It is already a converged local optimum of the
raw problem; it is simply a *worse* one. The rebuilt cell, released, climbs to 14.081 -- and
re-chatters at once, `curv_l1` 209 -> 4078 and `lag1` +0.65 -> -0.23, which is this whole
document in one line.

So the price costs exactly what a price should cost: **0.177 blocks** measured within one basin,
14.081 down to 13.904. What the figure shows is not a constrained problem outscoring an
unconstrained one; it is two pipelines landing in basins **1.16 blocks apart**, and the basin gap
being six times the price. The regularizer makes the answer flyable. The multi-start relaxation
is what makes it good. Those are two different jobs and they are separable.

Shard-wide, with the trig confound removed, this is a wash rather than a gain: over the 297
single-cycle cells the median `dJ(new) - dJ(old)` is **+0.020**, higher in 181 and lower in 116.
Not a systematic improvement -- a basin lottery whose spread dwarfs the price. **Nothing here
should be read as evidence that the regularizer buys objective value.** It buys flyability at a
stated, small, measurable cost, and the cost is invisible in the shard table only because the
search noise is larger.

`13_family.png` is the check the grid cannot make. Along lambda the schedules share one
structure -- an early phase that varies with the price on distance, then a common glide, then a
flick to the same peak -- and along n they are the same schedule with the flick slid later, all
five landing on the reference cycle's own flick-and-glide shape (black dashed). Neighbouring
cells lie on top of each other; the family does not fray from one cell to the next. **That is
what "the continuation does not drift" looks like**, and it is the property the old pass budget
was standing in for.

`07_best.png` is worth a specific look. Today's recipe (row 2) is visually clean -- it has no
chatter, and nothing in a pitch plot tells you it holds three ticks a hundredth of a degree from
a cliff. **Boundary parking is invisible to the eye and to `lag1`.** It shows up only in
`examples/whichtick.rs`, which is why that exists.
