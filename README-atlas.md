# The atlas: many optima per cell, and which of them a person could fly

`README-control.md` produces one schedule per cell and argues it is flyable. This document is the
follow-up question: **what else is there?** The corpus reports an argmax, which says what to fly
and nothing about what the alternatives are, how much the choice costs, or which of the
alternatives survives being flown by a human hand.

So: at one cell, polish 264 different seeds under the *unchanged* objective and keep every
converged result. Then ask what the population looks like, and measure each member against a
battery of human error models.

For the later `mapfine` sweep, [implied lambda](docs/implied-lambda.md) uses the population's upper
utility envelope to infer the exchange rate at which each fixed profile comes closest to the
sampled optimum. This separates a profile's achieved timing from the lambda of the cell whose seed
happened to produce it.

Two decisions up front, both of which are load-bearing.

**Diversity goes in the seed, never in the objective.** The obvious way to collect different
answers is a novelty term that pushes each solve away from the ones already found. That destroys
the product. A profile's entire claim is that its pitches are a coordinate optimum of the
objective named in its own header, and `sweep verify --optimality` re-checks exactly that from
the file alone.
Under a repulsion term every file would be an optimum of something that depends on which other
files happened to exist when it ran -- not reproducible, not checkable, and not stateable in a
header. A seed, by contrast, is where a *strategy* is stated; polishing it under the unchanged
objective answers whether that strategy is a local optimum and what it is worth, and every
catalog entry stays certifiable.

**Random restarts do not work here, and that is a measurement.** At the reference operating point
all 17 unstructured seeds tried -- 9 constant-pitch and 8 white-to-heavily-smoothed noise --
converged to the *same* optimum, the steady glide. Nothing unstructured contains a dive and a
flick, and the glide's basin swallows everything that does not. (At `v0 = 0` the picture is
softer: 1 of 16 noise seeds and 22 of 72 piecewise-constant seeds do reach the cyclic basin. The
basin structure depends on `v0`, so do not generalize either result from one cell -- an earlier
draft of this document did, and was wrong.)

## How seeds work in this repo

Two unrelated things are called "seed" and only one of them matters here.

* **An initial pitch schedule**: `--init <file>`, `seed_from_policy` (fly the tuned Leak policy),
  `seed_from_reference` (tile the 300-tick reference), `rescale`/`stretch`, and the BFS parent in
  `run_shard`. All deterministic. This is the only thing that decides which optimum you land on.
* **`Jitter::seed`** (`--seed`, default `0x5eed1eaf`): the RNG for velocity-jitter draws. The
  post-antichatter pipeline writes `# jitter 0`, so it is inert. Nothing inside `polish` is
  stochastic.

`tools/atlas_seeds.py` writes the families; `tools/atlas_run.sh` polishes each and keeps every
result; `tools/atlas_queue.sh` walks the cells. The families are named strategies, so a converged
optimum can be traced to the entrance it was reached through:

```
const 35   diveflick 64   pwc 72   noise 16   lerp 20   bang 20
bangzero 8   fourlines 3   randwalk 12   kcycle 5   refvariant 8   policy 1
```

`lerp`, `bang`, `bangzero`, `fourlines` and `randwalk` are the crude initializations from the
`cycle-optimizer` worktree (`PitchesUtil::new_4040`, `new_40zero40`, `new_lerp`, `new_rand_walk`,
and the four-line `default_pitches`). They earn their place: a **single straight ramp from +80 to
-80 degrees** is the second-best optimum at `v0 = 0, n = 300` (dJ 19.558 against the best
19.710), and a smoothed white-noise seed is third at 19.557. The top basin has a wide mouth; it
does not need a good seed, only a dive and a flick in roughly the right place.

## The landscape at one cell

`v0 = 0, n = 300, lambda = 0`, `--trig mth_lut --mu 1e-4 --limit 85` (l1; `--penalty l1` since the default became l2), 264 seeds:

```
 123 cyclic     dJ 0.01 .. 19.71
   1 multicycle dJ 0.006
 140 glide      dJ -22.21 (median), spread under 0.03
```

Two strategies, 42 blocks apart, and nothing in between. The glide is a **turnpike solution**: a
short entry transient, then a near-constant hold for 50 to 175 ticks, then a terminal departure.
It is a genuine, certifiable local optimum -- `curv_l1` 4.0 to 12.4, residuals 1.4e-6 to 1.5e-5
with median 6.9e-6 -- and it is what the shape classifier calls `COLLAPSED` and the corpus
excludes as degenerate.

The held pitch is **not** pinned to a single value, which is itself the interesting part. Across
the 16 glides holding for 50 ticks or more it runs -13.61 to -12.52, median **-12.82**; at
`v0 = (0.2, 0.2)` the median is -12.80. That brackets `myopic crit`'s min-sink glide (-13.052) and
`myopic eqrate`'s turnpike for this objective at `w = 0` (-13.233) without matching either. A
turnpike is by definition a place where the objective is flat, so the exact held value being
weakly determined is what you should expect -- but do not quote the glide as "the min-sink pitch";
an earlier draft did, from a handful of profiles at one cell, and it does not survive the
population.

The cyclic optima are **not** a set of clusters. Pairwise mean `|dpitch|` between converged
schedules has three scales, not two. Over all 34716 pairs: **31% under 2 degrees** (the same
optimum), **15% between 2 and 25** (a continuum), **54% above 25** (cycle against glide).
Gap-based clustering fails on this population and should not be used; the interpretable
coordinates are flick tick and value.

### The flick-time ridge

`runs/atlas/cells/flick_v00_n300` sweeps the seed's flick tick from 36 to 288 in steps of 2, 127
converged points. Four regimes:

```
 seed flick    converged flick    dJ         what happens
   36 -  64      274 - 277      -22.20     collapses to the glide
   66 - 216      164 -> 216      13.8 -> 19.7   the ridge
  218 - 284      215 - 216       ~18.9     slides back -- see below, this is not a wall
  286 - 288      263 - 265      -22.21     collapses to the glide
```

Inside the middle band the converged flick genuinely tracks the seed -- **38 distinct converged
flick ticks** over 164 to 215 -- so this is a one-parameter continuum, not a handful of
attractors. But the map is compressed (seeds at 66 land at 164). Below flick 164 there is no
cyclic optimum at all.

(Flick tick here means the *earliest* tick attaining the minimum pitch. Ties are common -- the
schedule often sits at the -85 limit for two or three ticks -- and the convention matters: taking
the latest instead moves the upper end of the range to 217. The distinct-tick count is 38 either
way.)

Both halves of the obvious dichotomy are true at once, which is the interesting part. Every point
on the ridge is a certified coordinate optimum -- coordinate ascent cannot translate a manoeuvre,
because that needs many ticks to move together, so it genuinely cannot slide -- yet the points
are not isolated. "Converged" is accurate; "separate basins" is not.

**The upper wall is not a wall.** Seeded past tick 216 the polish walks the flick back to 215-216
and stops there, and for a while I read that as the objective refusing late flicks. It is the
optimizer, not the objective. Stop the polish early and the late flick simply stays:

```
                  cyclic optima   flick ticks held
  3 passes             63            144 - 265
  10 passes            77            136 - 272
  30 passes            79            134 - 262
  400 passes (full)   111            164 - 216
```

More optimization does not find better late flicks, it *destroys* them -- the manoeuvre slides
back toward 205 and the schedule stops being a late one. So the instrument for pricing a late
snap is a **stopping time**, not a constraint: seed the flick late, run a few passes, and read
off what a decent-but-unconverged late schedule is worth. Nothing is forbidden, so nothing can be
gamed.

`runs/atlas/cells/flicksoft{3,10,30}_v00_n300` are that sweep. Against a peak of 19.716:

```
  ticks late   30 passes            10 passes
     +0        19.716  ( 0.00)      19.696  (-0.01)
     +5        19.482  (-0.23)      19.608  (-0.10)
    +10        18.887  (-0.83)      19.275  (-0.43)
    +15        17.852  (-1.86)      18.404  (-1.30)
    +20        16.590  (-3.13)      17.133  (-2.57)
```

Read the two columns as the width of the answer, not as two results: the pass budget is a free
knob and it moves the number by about half a block at +20 ticks. What is robust across budgets is
the shape -- the cost is negligible for the first five ticks, about 0.1 blocks/tick out to +10,
and roughly 0.25 blocks/tick by +20. **A snap half a second late costs around three blocks, one
sixth of the run**, and the penalty accelerates rather than falling off a cliff.

Beyond +20 the numbers keep going (-7 at +30, -12 at +40) but they are not worth much: a person
who is 40 ticks late has not mistimed a cue, they are flying a different flight.

Two other instruments were tried for this and abandoned, both recorded here because the failure
is more informative than the result:

- **A hard constraint** (`--flick-at t`, which pins the tick at which the schedule first crosses a
  chosen pitch) is gameable, and the gamed solutions are invisible in every scalar -- smooth `dJ`,
  `structure` still `cyclic`, pin verified as held, residual fine. Two modes appeared. A one-tick
  needle to -80 at the pin with the real snap at ~213; and, subtler, riding at **-79.99** from
  tick 216 -- just above the threshold, so no crossing is recorded -- then dipping past -80 at the
  pin, which keeps the below-30 excursion contiguous and so defeats a width test too. A codex
  audit found the second one after I had discarded the first and believed the rest were clean.
  `curv_l1` was the tell all along: 102 at the peak, 308 at the far end.
- **Freezing the manoeuvre and translating it** (`cue()` in `examples/human.rs`) prices the
  opposite extreme, a person who mistimes and adapts nothing. It is a much larger number (-8.8
  blocks at +20 against -3.1) but the schedules it produces look wrong -- the splice leaves a
  discontinuity no hand would fly.

The pinned cells remain on disk under `runs/atlas/cells/flickpin*` and are excluded from every
figure.

### Across cells

Every column is taken from that cell's best **single-cycle** profile, classified by the
`# structure` header, so the flick and hold-0 describe the same schedule as the value beside them.

```
cell                      n  cyc  mlt  gld  best 1-cycle  flick  hold0  best multi  best glide
v00_n150_lam0           150   68    0  196        -2.879     98     12           -     -11.581
v00_n300_lam0           300  123    1  140        19.710    205     13       0.006     -22.202
v00_n450_lam0           450  138    8  118        14.219    335     12      29.235     -32.822
v00_n600_lam0 *         600   36    1   41         3.572    482     13      41.374     -43.441
vy00vz02_n300_lam0      300  127    1  136        21.706    199     12       3.945     -21.391
vy02vz00_n300_lam0      300  119    1  144        19.857    210      9      -0.500     -21.009
vy02vz02_n300_lam0      300  124    1  139        21.837    209     13       3.239     -20.183
v00_n300_lamP2          300  221    7   36        67.577    213     11      36.626       9.383
v00_n300_lamM2          300   92    1  171       -19.305    196     12     -33.015     -36.927
```

`*` incomplete (78 of 264 seeds), so its counts are a lower bound and its best may move.

Four things. **The optimal hold-0 sits at 11 to 13 ticks in eight of the nine cells** -- the
exception is 9 -- across horizons 150 to 600, four starting velocities, and `lambda` from -2 to
+2. Whatever sets it is not the horizon, not the entry speed, and not the exchange rate between
height and distance. That is the strongest invariant in this document and it has no explanation.

**Forward speed at the start is worth much more than vertical speed**: `vz0 = 0.2` buys 2.0 blocks
over `v0 = 0`, `vy0 = 0.2` buys 0.15.

**Lambda changes how easy the cyclic basin is to find, not just what it is worth.** At
`lambda = +2`, where distance is priced, 221 of 264 seeds land cyclic and only 36 glide; at
`lambda = -2` it is 92 against 171. Pricing distance widens the mouth of the basin, which is worth
knowing before choosing seeds for a cell.

**The tiling degeneracy arrives earlier than advertised and then dominates.** Best multi-cycle
against best single cycle: 0.0 vs 19.7 at `n = 300`, **29.2 vs 14.2** at 450, **41.4 vs 3.6** at
600. `README-control.md` places that regime at `n >= 550`. The corpus excludes multi-cycle by
construction, which is a statement about which question is being asked, not about which schedules
score well.

### The catalog certifies

Every atlas profile carries the same header a corpus cell does, so it is checkable from the file
alone. A 40-profile random sample across the six cells: **40 ok, 0 mismatches, 0 bad headers, 0
schedules outside their own stated control limits**, worst claimed residual 1.05e-2 (a cell that
hit the pass ceiling, which is what it claims and re-derives to). Keeping the losers costs nothing
in rigor -- they are optima of the same stated objective as the winner, and say so.

## What a person can actually fly

`examples/human.rs` is the battery. It reports **losses** relative to each profile's own dJ, so
profiles forty blocks apart stay comparable, and prints `-` rather than a number where a model is
undefined (a steady glide has no flick to mistime).

```
shift k   the whole manoeuvre executed k ticks late or early
cue k     only the manoeuvre is mistimed, the dive absorbing the difference. This one is
          suspect: the splice leaves a discontinuity at the join that no hand would fly, so
          read it as an upper bound on the cost of mistiming, not as a model of a person
tremor s  correlated low-frequency noise (4-8 random cosine modes) at sd s
lag tau   first-order lag, a[t] = a[t-1] + (p[t]-a[t-1])(1-exp(-1/tau))
bias b    a constant offset on every pitch
```

Existing coverage was already good on the things it covered and misleading on one: `sens.rs` does
i.i.d. pitch noise, and these optima barely feel it (0.34 blocks at 0.5 degrees of noise on *every*
tick, rounding to a 0.5-degree grid free). **Correlated noise is about four times worse at the
same sigma** -- at 0.5 degrees, i.i.d. mean/p05 is -0.34/-0.58 against correlated -0.62/-2.15 --
because these schedules tolerate error that averages out and are exposed to error that persists.
Human aim error is not white, so `sens.rs` flatters them.

Across the 124 cyclic optima at `v0 = 0, n = 300`, median losses:

```
 constant bias, worst of +-1 / +-2 deg    -5.63
 correlated tremor sigma 1 deg, p05       -3.65
 cue timing, worst at 10 ticks            -2.92
 cue timing, worst at 20 ticks            -9.38
 mouse lag tau = 4                        -1.06
```

The glide is immune to all of it: worst bias -0.20, tremor p05 -0.04, lag -0.01, 20-tick shift
-0.27. So the landscape's real axis is not value but fragility -- the cycle is worth 42 blocks
more and hands back a quarter of that for half a second of mistiming, while the glide hands back
nothing.

The -9.38 at 20 ticks is the no-adaptation extreme, and it is nearly three times the -3.13 that
the stopping-time sweep gives for the same lateness. The gap between those two numbers is the
value of adapting the rest of the flight to a snap you know is late; the truth for a real pilot
is somewhere between, and nothing here pins down where.

### The one lever that is nearly free: hold-0 length

Vary only the length of the hold-0 run (the ticks with `|pitch| <= 1` before the flick), stride 1:

```
 hold-0      dJ    vs best   tremor p05     bias
      1    1.165   -18.545       -1.85    -3.06
      5   12.906    -6.804       -1.16    -1.77     <- most robust
      9   18.272    -1.438       -2.35    -3.55
     11   19.369    -0.341       -3.31    -4.78
     13   19.710     0.000       -4.26    -6.00     <- most valuable
     17   18.739    -0.971       -6.12    -8.29
     25   13.084    -6.626       -8.93   -12.12
     30    8.355   -11.355       -9.89   -14.01
```

Value peaks at 13; robustness peaks at 5; they are eight ticks apart. Robustness is **not**
monotone -- it turns over below 5 -- which a coarser sweep hides. Near the value optimum the
exchange is favorable:

```
 hold-0 12: costs 0.09 dJ, buys 0.47 tremor p05   5.5 : 1
 hold-0 11: costs 0.34 dJ, buys 0.95 tremor p05   2.8 : 1
 hold-0 10: costs 0.78 dJ, buys 1.43 tremor p05   1.8 : 1
 hold-0  8: costs 2.33 dJ, buys 2.32 tremor p05   1.0 : 1
 hold-0  5: costs 6.80 dJ, buys 3.10 tremor p05   0.5 : 1
```

**This is not a scaling artifact**, and there is a matched pair that settles it: hold-0 = 3 and
hold-0 = 30 score almost the same (dJ 8.028 against 8.355) and differ in tremor by seven times
(-1.37 against -9.89) and in bias by six. Loss is not tracking value; it is a property of the
shape. Independently: `|loss|/dJ` at sigma 1 runs 2.9% at L05, 5.4% at L11, 6.7% at L13, 21.2% at
L25, nowhere near constant, and at sigma 2 the L11 schedule *outscores* L13 after perturbation
(17.08 against 16.90) despite its lower nominal value.

**It replicates.** At `v0 = (0.2, 0.2)`, n = 300, from that cell's own best profile: value peaks
at hold-0 = 13, robustness at hold-0 = 5, and hold-0 = 12 costs 0.005 blocks while buying 0.37 of
tremor p05. Same two optima, same eight-tick separation, a different starting velocity.

Caveat the audit insisted on: the gain depends on which statistic you read. L13 -> L11 improves
the tremor *5th percentile* by 1.00 and the worst bias by 1.22, but the tremor *mean* by only
0.27. The bad-day number moves more than the average day.

### Where the fragility does not come from

Recorded so they are not re-derived: the **pre-flick spike** does not matter (capping it from 66.9
to 50 degrees moves tremor p05 by 0.02 blocks), and **dive level** matters only weakly between
comparable profiles -- a controlled shift of the dive on one schedule moves tremor p05 monotonically
(-3.47 at mean 30.2 through -5.18 at 46.2), but across profiles at matched value the partial
correlation is only about -0.2. The between-profile spread is **not explained**: `lerp_+80_-80`
has essentially the same dive mean (37.1 against 38.2) and spike (52 against 50) as a capped
variant of the best profile and is 2.2 blocks more tremor-robust at a cost of 0.15 blocks of
value. That mechanism is still open.

## The exact dynamic program

`src/bin/dp.rs`. The reduction that makes it cheap:

`update_fall_flying_movement(vel, rot)` reads **only** velocity and rotation -- position enters the
simulation solely through `pos += vel` -- and with yaw pinned, `v_x` decays to zero. And the
objective decomposes:

```
J = TE(s_n) + w*z_n = KE(v_n) + y_n + w*z_n,   y_n = sum_t v_y[t],  z_n = sum_t v_z[t]
```

verified numerically against a solved profile to 8e-7. So this is an additive-reward finite-horizon
problem with a **two-dimensional** state `(v_y, v_z)`, per-step reward `v_y' + w*v_z'` at the
post-update velocity, and terminal reward `KE(v)`. Position is a readout, never a state variable.
The earlier attempt in the `main` worktree (`DPKey { y_pos, y_vel, z_vel }`, trilinear
interpolation through an `AHashMap`) carries position in the state, which makes it a 3D hashmap
problem where a 2D dense array suffices.

At `n = 300, v0 = 0, lambda = 0, limit 85`, grid 1025^2, pitch step 1 degree, 30 seconds:

```
 DP value estimate                 19.977
 DP policy rolled out, replayed    19.830   (independently re-scored with a correct header)
 best of 264 multi-start optima    19.710
```

**The velocity box is the correctness risk, and it is subtle.** With `vy` in [-4.5, 1.5] and `vz`
in [-1, 4.5] the DP returns **16.75** instead of 19.98 -- three blocks low -- while its own
boundary check reports only *3* out-of-box transitions on the optimal path. The path really does
stay inside; what is corrupted is the value surface everywhere else, because ~2.85 million
state/control transitions per sweep are being clamped, and that propagates back through the
Bellman recursion. "The optimal path stayed in the box" is not sufficient; the reachable set under
*all* controls must fit. This is the same emptiness-reads-as-success shape catalogued in
`README-control.md`.

**What the DP does and does not establish.** It found a feasible schedule scoring 19.8305, which
is verified by replaying it through the real simulator, so that is a genuine achievable value and
it beats every schedule in the atlas by 0.120 blocks (0.605%). The value estimate 19.977 is
**not a bound in either direction**: bilinear interpolation of `V` can over- or under-estimate the
true continuation value with no curvature guarantee, and maximizing over a discrete pitch grid
restricts the action set. The 0.146 gap between the estimate and its own rollout is direct
evidence that the estimate is not a certificate. So it is wrong to say the atlas is "within 1.3%
of global optimal"; what is established is that the atlas best is 0.605% below a schedule we can
actually fly.

**How the DP's answer differs from the flyable one.** Same flick tick (205 in both), and chatter
dominates ticks 138-188: total variation there is 95 for the flyable profile against 3738 for the
DP. But "the global optimum is the flyable optimum plus chatter" is too strong, and was corrected:
the two differ by more than 3 degrees on **104 ticks, 53 of them outside** that window, 13 by more
than 10 degrees, with outside-window total variation 257 against 771. The flick ramp differs
materially -- the flyable profile descends through ticks 201-204 as (-11.5, -30.2, -53.7, -85)
where the DP uses (0, -9, -26, -47) -- and the DP stays steeper through ticks 208-215. What is
fair to say: the chatter is localized and is the largest single difference, and the price on
curvature buys flyability for about 0.6% of J.

## The entry is nearly free, and its sign is an exact symmetry

At `v0 = 0` the tick-0 pitch enters the dynamics only through `lift_force = cos^2(lean)`, which is
even. `move_hor_length` is zero, so the forward-to-up conversion vanishes; the descent-to-forward
term contributes `look_angle.z / look_hor_length = sign(cos p) = +1` either way; and the turning
term -- which does **not** vanish, since it damps the forward velocity created earlier in the same
function -- is identical because that velocity is identical. Measured: negating tick 0 alone leaves
J at 19.710495501123411, a delta of exactly zero at f64 precision.

So the sign is unidentifiable and the optimizer keeps whatever the seed had; `seed_from_policy`
gives nose-*up*. The DP, which has no such inheritance, starts nose-down (+47.5, +32, +22, +19.5,
+11.5). Nose-down is right, and the whole question is worth **0.003 blocks**:

```
 base (nose-up -50, -31.9, -13.7)   19.710491
 negate tick 0                      19.710491   (+0.000000)
 negate ticks 0-2                   19.712080   (+0.0016)
 DP's nose-down first 5             19.713042   (+0.0026)
 DP's nose-down first 12            19.713654   (+0.0032)
 first 5 ticks level                19.666564   (-0.044)
```

Tick 1 is *not* symmetric -- horizontal speed is no longer zero, so a negative pitch activates the
forward-to-up conversion -- and negating it changes dJ by +0.00057. `README-myopic.md` calls the
entry "the open problem, no rule known"; on this evidence it is unexplained largely because it is
nearly flat. Level is the one clearly wrong choice.

## What is still open

* **Four cells of the queue have not run**: `lambda = +-2` at `v0 = 0`, and `n = 600`.
* **The between-profile robustness spread has no mechanism.** Dive level, pre-flick spike and
  hold-0 length together do not explain why `lerp_+80_-80` is 2.2 blocks more tremor-robust than a
  matched profile at 0.15 blocks of cost. Until something does, "prefer a shorter hold-0" is a
  measured exchange rate, not an explanation.
* **The DP has no way to price curvature**, because curvature is not a function of `(v_y, v_z)`.
  Pricing it needs the previous pitch in the state, which makes the table roughly 170 times
  larger. A slew *constraint* would restrict the control set per state instead and might be
  affordable; that has not been tried.
* **Closed-loop robustness has not been measured.** `src/bin/dp.rs` stores the greedy policy for
  every tick, so a feedback rollout under a perturbed state is available and would be the honest
  model of a pilot who can see they are late. Everything in this document perturbs a *schedule*,
  which assumes open-loop execution.
* **The human reference is not comparable.** A top human reportedly reaches 17-20 blocks
  inconsistently, but not from a standing start and possibly with a lead-in from a sacrificed
  cycle, so it anchors nothing precisely. A replay would fix this and is worth getting.
* **There is no clean `J*(T)` and possibly cannot be one.** The stopping-time sweep prices a late
  snap without forbidding anything, but its answer depends on the pass budget by about half a
  block at +20 ticks, and there is no principled budget. The constraint that would pin `T`
  exactly turned out to be gameable (above). A formulation that makes "the manoeuvre happens at
  T" both exact and unfakeable is still missing.

## Figures

Under `runs/atlas/fig` (regenerate with `python3 tools/plot_atlas.py`; `runs/` is gitignored, so
they are not in the tree). Dark by default -- set `ATLAS_LIGHT=1` for light versions:

```
01_landscape.png  every optimum at v0=0 n=300 by flick time and value, colored by the seed
                  family that reached it, plus the two strategies drawn against each other
02_ridge.png      seed flick tick against converged flick tick -- the connected ridge -- and
                  value along it. Its upper "wall" is an optimizer artifact, see above
04_latesnap.png   what a late snap costs, by stopping time: the same seeds at 3, 10, 30 and 400
                  coordinate passes. More optimization destroys the late flick rather than
                  improving it, so the late part of the curve exists only at small budgets
03_hold0.png      value and tremor robustness against hold-0 length at two starting velocities.
                  The middle panel is the replication: the two curves nearly overlay
profiles/<cell>_overlay.png   every schedule at that cell on one axis, colored by dJ. The
                              flicksoft* cells are stopped early, and their titles say so --
                              those are partial optima, not converged ones
profiles/<cell>_bands.png     the same schedules split evenly over distinct dJ levels, so a
                              mode found 140 times costs one panel instead of six
```

The overlays are the shape view of figure 01, which reduces each profile to a point. Read across
the ten of them and three things show up that the scatter cannot:

- The dive is not a constant. In every `lam=0` cell it drifts monotonically from about 25 deg
  nose-down at the top to about 48 just before the hold-0. That drift is present in all 123
  cyclic profiles at the reference cell, at every value level.
- At `lam=+2` the second mode is a *different shape*, not the same glide at a different value.
  Pricing z turns the flat -13 hold into a shallow profile that starts near 0 and steepens to
  about 25 nose-down. The two-mode structure survives the reweighting; the modes themselves do
  not.
- The multi-cycle mode is rare and it wins by a mile. At both horizons past n=300 that were run,
  the best profile in the cell is a multi-cycle one -- two cells is not "never", but the margin
  is not close:

  | cell | best single-cycle | best multi-cycle | n multi |
  |---|---:|---:|---:|
  | `v00_n300_lam0` | **19.710** | 0.006 | 1 |
  | `v00_n450_lam0` | 14.219 | **29.235** | 8 |
  | `v00_n600_lam0` | 3.572 | **41.374** | 1 |

  Both headline profiles were replay-verified from their own headers rather than trusted:
  29.235446995918 and 41.373970520579, `sweep verify --optimality` ok at 9.29e-6 and 1.10e-5
  (at the time plain `sweep verify`, with the search before 2026-10-02).

  Over these three points the best-in-cell value rises 19.71 / 29.24 / 41.37 while the best
  single cycle falls 19.71 / 14.22 / 3.57. Three horizons do not establish a scaling law, and I
  am not claiming one -- but the sign of the two trends is opposite, and the mechanism is not
  mysterious: a second cycle buys another dive-and-convert, whereas a single cycle at n=600 must
  spend the leftover 300 ticks gliding down. The seeds find multi-cycle rarely (1, 8, 1 of 264)
  because reaching a second cycle from a cold start crosses a long bad region, which is a fact
  about the seeds and the basin, not about the strategy.

`v00_n600_lam0` is 78 of 264 seeds, so its counts are a sample, not a census; the values are
certified profiles either way.

## Corrections made while producing this

Kept because the class is worth knowing, and because every one of them was found by something
other than the check that produced the claim.

| what was claimed | what was wrong |
|---|---|
| the cluster threshold can be read off a bimodal gap | inferred from 18 schedules drawn from exactly two basins; with 112 the widest gap is 1.14x, and the optima form a continuum |
| flicking early costs 9 blocks at 10 ticks | the cue operation spliced at the flick, so "early" deleted the hold-0; that is the -74-block snap-removal result misattributed to timing |
| the dive's shape is worth 30 blocks | the dive segment ran through the transition into the snap, so a linear fit between endpoints pointed the wrong way |
| longer hold-0 is more robust | a population correlation with the sign backwards; the controlled sweep gives the opposite, monotonically |
| shallower dive and longer hold-0 explain fragility | `max(pitch)` is the pre-flick spike, not the dive level; the correlations were driven by profiles pinned at the +-85 limit |
| the DP bound proves the atlas is within 1.3% of optimal | the DP estimate is not a bound in either direction |
| the global optimum is the flyable one plus chatter | they also differ on 53 ticks outside the chatter window, materially in the flick ramp |
| the ridge has 45 distinct converged flick ticks | 38; the count included the six ticks belonging to collapsed profiles |
| the glide holds the min-sink pitch, -13.05 | the held pitch runs -13.61 to -12.52 across profiles, median -12.82, matching neither reference value |
| pair distances 26 / 21 / 54 percent | 31 / 15 / 54, computed on 112 profiles of a cell that had 264 |
| n=450's best single cycle flicks at 147 | 335; 147 belongs to the best multi-cycle, a table-construction error |
| the flick ridge spans +-5 ticks for 0.33 blocks | 0.17 blocks; the wider figure came from an off-center window |

Two patterns. The scientific errors are all **a population correlation standing in for a
controlled test**; the bookkeeping errors are all **a statistic computed on a sample that had
moved** -- a cell still filling, a window off center, a category quietly including its neighbor.

Not one of them was caught by re-running the analysis that produced the claim. They came from a
controlled intervention, a matched pair, a dense sweep, a bound computed by other means, or an
independent auditor given the instruction to falsify and told to assume at least two claims were
wrong. That last one found six errors in this document alone, five of them in numbers I had
transcribed from my own earlier output.
