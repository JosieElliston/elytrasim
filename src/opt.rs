//! The shared core: physics helpers, the myopic rules, the policy that flies them,
//! and the search primitives the optimizer and the sweep both stand on.
//!
//! Yaw is pinned to zero everywhere here, so the whole problem lives in the `(v_y, v_z)`
//! plane and a schedule is just a list of pitches in degrees. `myopic` analyzes one cycle
//! with these; `sweep` solves a grid of them.

use crate::sim::*;
use rayon::prelude::*;

pub const V0: Vec3 = Vec3::new(0.0, 0.167467, 0.200887);
pub const OPTIMAL_CYCLE_RATE: f64 = 1.43335; // blocks/second, REPLAY_PITCHES_300

pub fn rot(p: f64) -> Rot { Rot { x: p as f32, y: 0.0 } }
pub fn gamma(v: Vec3) -> f64 { (-v.y).atan2(v.z).to_degrees() }

pub fn ticked(s: &State, p: f64) -> State { s.ticked(rot(p)) }
#[inline]
fn ticked_cached(s: &State, p: PitchTrig) -> State { s.ticked_cached(p) }
fn cache_pitches(pitches: &[f64]) -> Vec<PitchTrig> {
    pitches.iter().map(|&p| PitchTrig::new(p as f32)).collect()
}
pub fn run_n(s: &State, p: f64, n: usize) -> State {
    let r = rot(p);
    let mut s = s.clone();
    for _ in 0..n { s = s.ticked(r) }
    s
}
/// Replay a schedule from `v0` at the origin. Returns `pitches.len() + 1` states.
pub fn replay_from(v0: Vec3, pitches: &[f64]) -> Vec<State> {
    let trig = cache_pitches(pitches);
    let mut v = vec![State { pos: Vec3::ZERO, vel: v0 }];
    for p in trig { let s = ticked_cached(v.last().unwrap(), p); v.push(s) }
    v
}
pub fn replay(pitches: &[f64]) -> Vec<State> { replay_from(V0, pitches) }
/// A schedule file: whitespace-separated pitches in degrees. Everything from a `#` to the
/// end of its line is a comment, which is where a `sweep` profile keeps its header.
pub fn read_pitches(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace())
        .map(|s| s.parse().unwrap())
        .collect()
}

// The pitch argmax: a coarse global sweep, then a 1-D refinement inside the cells of every
// sampled peak. `examples/argmax_study.rs` measured it against a 0.01-degree sweep over the
// reference cycle's states and a velocity grid (libm): no wrong peak on any objective there, at
// ~110-150 evaluations against the 1562 of the 0.125-degree sweep it replaced.
//
// The objectives are not unimodal -- conversion kinks, the forward-to-up branch at pitch 0, ties
// across +-90 -- so the sweep has to be global. Every sampled peak is refined, not a fixed top
// few: dTE held 10-48 ticks has up to 4-6 peaks of real prominence (`examples/argmax_modes.rs`),
// and at a 2-degree step a sample-level peak count of 4 is the most seen there.
const ARGMAX_STEP: f64 = 2.0;
// Degrees. Finer is precision the physics does not have: its f32 parts make every energy noisy
// at ~1e-7 blocks, and where the peak is flat (one-tick dTE: ~6e-5 blocks/deg^2) that alone
// smears the argmax over ~0.05 degrees; under `mth_lut` the trig staircase makes it ~0.5 at 20
// ticks. Each factor of 10 here costs only ~5 evaluations per peak.
const ARGMAX_TOL: f64 = 1e-3;

/// The pitch maximizing `f` over the whole pitch domain.
pub fn argmax<F: Fn(f64) -> f64>(f: F) -> f64 { argmax_in(f, -90.0, 90.0) }

/// `argmax` over `[lo, hi]` instead of the whole pitch domain. A corpus profile was optimized
/// under a `|pitch| <= limit` control set, so a rule scored against one has to be held to the
/// same set: outside it the pitches are not worse, they are unavailable.
pub fn argmax_in<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64) -> f64 { argmax_in_tol(f, lo, hi, ARGMAX_TOL) }

/// `argmax_in` to a tolerance other than `ARGMAX_TOL`, for a caller that iterates on the argmax
/// and tests convergence below it.
pub fn argmax_in_tol<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, tol: f64) -> f64 {
    let ps = argmax_grid(lo, hi);
    let vs: Vec<f64> = ps.iter().map(|&p| f(p)).collect();
    refine_peaks(&f, &ps, &vs, tol).0
}

/// `pitch_grid` at the argmax's own step. Exposed so a caller that scores many objectives from
/// one rollout per pitch (`dj_argmax_all`) samples where `refine_peaks` expects.
pub fn argmax_grid(lo: f64, hi: f64) -> Vec<f64> { pitch_grid(lo, hi, ARGMAX_STEP) }

/// Both ends of `[lo, hi]` and the multiples of `step` between them. Anchored at 0 rather than at
/// `lo` because pitch 0 is a corner -- the forward-to-up branch switches on below it -- and the
/// snap phase's max sits exactly there; a refinement only approaches a corner, so it has to be a
/// sample.
pub fn pitch_grid(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let (a, b) = ((lo / step).floor() as i64 + 1, (hi / step).ceil() as i64 - 1);
    let inner = (a..=b).map(|k| k as f64 * step).filter(|&p| p > lo && p < hi);
    std::iter::once(lo).chain(inner).chain((hi > lo).then_some(hi)).collect()
}

/// The indices of the local maxima of `vs`, ends included.
pub fn sampled_peaks(vs: &[f64]) -> Vec<usize> {
    let n = vs.len();
    (0..n).filter(|&i| (i == 0 || vs[i] >= vs[i - 1]) && (i + 1 == n || vs[i] >= vs[i + 1])).collect()
}

/// The argmax of `f` and its value, given its values `vs` at the sorted pitches `ps`: a
/// refinement inside the two cells around every local maximum of the samples. A sample beats its
/// refinement when the peak sits on a corner the refinement only approaches.
pub fn refine_peaks(f: &dyn Fn(f64) -> f64, ps: &[f64], vs: &[f64], tol: f64) -> (f64, f64) {
    let n = ps.len();
    let mut best = (ps[0], f64::NEG_INFINITY);
    for i in sampled_peaks(vs) {
        if vs[i] > best.1 { best = (ps[i], vs[i]) }
        let (a, b) = (ps[i.saturating_sub(1)], ps[(i + 1).min(n - 1)]);
        if b - a > tol {
            let r = golden_max(f, a, b, tol);
            if r.1 > best.1 { best = r }
        }
    }
    best
}

/// One coordinate's move in a coordinate ascent: the best pitch on `[lo, hi]` and its score,
/// unrounded. Samples every `step` degrees, then `golden_max`es the two cells around every sampled
/// peak and the cells either side of the incumbent `cur`; under `mth_lut`, then `stair_scan`s
/// around the best of those.
///
/// The incumbent gets its own cells rather than being a sample. Next to a grid point it would
/// shrink that point's cell to the sliver between them, and under `mth_lut` the objective is rough
/// enough that the max can sit in the part cut off: `floor exit` lost up to 3e-3 per tick that
/// way, and ended ~0.01 short on every start height tried. Its own cells are also where a
/// converging ascent's move almost always is.
///
/// The samples and the refinements are independent, so both run in parallel.
pub fn coordinate_argmax(score: &(dyn Fn(f64) -> f64 + Sync), cur: f64, lo: f64, hi: f64, step: f64,
                         tol: f64) -> (f64, f64) {
    let ps = pitch_grid(lo, hi, step);
    let vs: Vec<f64> = ps.par_iter().map(|&p| score(p)).collect();
    let n = ps.len();
    let c = cur.clamp(lo, hi);
    let cells: Vec<(f64, f64)> = sampled_peaks(&vs).into_iter()
        .map(|i| (ps[i.saturating_sub(1)], ps[(i + 1).min(n - 1)]))
        .chain([((c - step).max(lo), c), (c, (c + step).min(hi))])
        .collect();
    let best = (0..n).map(|i| (ps[i], vs[i])).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    let best = cells.into_par_iter()
        .filter(|&(a, b)| b - a > tol)
        .map(|(a, b)| golden_max(score, a, b, tol))
        .reduce(|| best, |a, b| if b.1 > a.1 { b } else { a });
    if trig_mode() == TrigMode::MthLut { stair_scan(score, best, lo, hi, STAIR_SCAN) } else { best }
}

/// Degrees either side of `coordinate_argmax`'s answer that `stair_scan` searches. Moves the
/// polish's certificate still found after golden section were 0.03-0.3 degrees; 0.1 recovered
/// the old dense sweep's J on `floor exit` at every height tried, and 0.3 bought nothing more on
/// the polish benchmarks for half again the ticks (2026-10-02, measured at a 2-degree local step).
const STAIR_SCAN: f64 = 0.1;

/// Every `mth_lut` stair within `w` degrees of `best.0`, one sample each: under the table's trig
/// the objective is constant in pitch's sin and cos across a stair, so a bracketing search
/// settles on *a* local stair, and the better one is often a few stairs away.
fn stair_scan(score: &(dyn Fn(f64) -> f64 + Sync), best: (f64, f64), lo: f64, hi: f64, w: f64) -> (f64, f64) {
    const STAIR: f64 = 360.0 / 65536.0;
    let k = (w / STAIR).ceil() as i64;
    (-k..=k).into_par_iter()
        .map(|i| (best.0 + i as f64 * STAIR).clamp(lo, hi))
        .map(|p| (p, score(p)))
        .reduce(|| best, |a, b| if b.1 > a.1 { b } else { a })
}

/// Golden section for a max of `f` on `[a, b]`, down to `tol`: the best point it scored, and its
/// value.
///
/// Derivative-free on purpose. A finite-difference Newton ascent from the same peaks landed on
/// the wrong pitch on 4-60% of states in `examples/argmax_study.rs`: the energy noise swamps a
/// small difference, and the max often sits on a kink or a corner, where a derivative step
/// oscillates. Brent's method (parabolic steps with a golden fallback) was tried too and taken
/// back out (2026-10-02): ~20% faster per polish pass, but its parabola fits the noise on flat
/// objectives -- a wrong peak and a 0.24-degree miss in the study -- and its trust tests were
/// never exercised against cells clipped by the control limits. Golden's only assumption is the
/// bracket.
pub fn golden_max(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64) -> (f64, f64) {
    const G: f64 = 0.618_033_988_749_894_9;
    let (mut c, mut d) = (b - G * (b - a), a + G * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    while b - a > tol {
        if fc >= fd { (b, d, fd) = (d, c, fc); c = b - G * (b - a); fc = f(c) }
        else { (a, c, fc) = (c, d, fd); d = a + G * (b - a); fd = f(d) }
    }
    if fc >= fd { (c, fc) } else { (d, fd) }
}
// ---------------------------------------------------------------- the rules

/// GAIN. Pitch maximizing the total-energy change over `n` ticks held constant.
/// n = 1 is elytrasim's `argmax_over_pitch_of_delta_energy`; the optimum wants n ~ 20.
pub fn bug_dte_n(s: &State, n: usize) -> f64 {
    let te = s.total_energy();
    argmax(|p| run_n(s, p, n).total_energy() - te)
}

/// GAIN, at a price on distance. Pitch maximizing the *objective's* change over `n` ticks held
/// constant: `dJ = dTE + w*dz`. `w = 0` is exactly `bug_dte_n`.
///
/// Why this is the right generalization, and not merely the obvious one. Expand `dJ` over the
/// rollout: `dTE = (|v_n|^2 - |v_0|^2)/2g + sum v_y`, so `dJ` is
/// `sum_k (v_y,k + w*v_z,k) + (|v_n|^2 - |v_0|^2)/2g`
/// -- the cycle's own running reward `(1, w).v` accumulated over the rollout, plus a terminal
/// value equal to the kinetic energy at the end. So `argmax dJ` is the climb's own optimal-control
/// problem truncated at `n`, with the control frozen and the apex's price vector `mu(apex)`
/// replaced by the energy gradient `v_n/g`. `dTE` gets the running reward wrong as soon as
/// `w != 0`; this gets it exactly right and still gets the terminal price wrong, which is what
/// `n` is left standing in for.
pub fn bug_dj_n(s: &State, n: usize, w: f64, limit: f64) -> f64 {
    let j = |t: &State| t.total_energy() + w * t.pos.z;
    let j0 = j(s);
    argmax_in(|p| j(&run_n(s, p, n)) - j0, -limit, limit)
}

/// `bug_dj_n` for every lookahead `1..=nmax` at once.
///
/// Holding a pitch for `n` ticks is a prefix of holding it for `n + 1`, so one rollout per grid
/// pitch scores every horizon; only the refinement is per-horizon.
pub fn dj_argmax_all(s: &State, nmax: usize, w: f64, limit: f64) -> Vec<f64> {
    let j = |t: &State| t.total_energy() + w * t.pos.z;
    let j0 = j(s);
    let ps = argmax_grid(-limit, limit);
    // vs[n - 1][i]: the score of holding ps[i] for n ticks.
    let mut vs = vec![vec![0.0; ps.len()]; nmax];
    for (i, &p) in ps.iter().enumerate() {
        let trig = PitchTrig::new(p as f32);
        let mut t = s.clone();
        for row in vs.iter_mut() {
            t = t.ticked_cached(trig);
            row[i] = j(&t) - j0;
        }
    }
    vs.iter().enumerate().map(|(k, row)| {
        refine_peaks(&|p| j(&run_n(s, p, k + 1)) - j0, &ps, row, ARGMAX_TOL).0
    }).collect()
}

/// SNAP, stopping. Has the forward speed peaked -- would one more tick of the hold fail to
/// raise `v_z`?
///
/// Parameter-free, and a one-tick lookahead: it asks the tick map rather than the closed form,
/// because the sim's drags are `f32` and the margin at the crossing is only ~5e-4 b/tick. In
/// exact rationals the hold changes `v_z` by `-0.01 v_z - 0.0891 v_y + 0.001782`, so the rule
/// fires on the line `v_z >= 0.1782 - 8.91 v_y`, which passes through the pitch-0 steady glide.
///
/// It reads as "no pitch can raise `v_z` any further", not merely "pitch 0 cannot": for `p >= 0`
/// the map depends on pitch only through `lift = cos^2 p`, and `d(v_z')/d(lift)` is positive
/// whenever `v_y < -0.04`, so pitch 0 is the argmax of `v_z'` over the whole domain there; for
/// `p < 0` the forward-to-up branch switches on and subtracts from `v_z` outright.
///
/// Above `v_y = -0.04` that stops being true -- the argmax lift goes interior, the line bends
/// into a parabola, and above `v_y = 0.08` no pitch raises `v_z` at all. `docs/rising.md` has
/// the whole frontier; `myopic rising` measures it.
pub fn vz_peaked(s: &State) -> bool { ticked(s, 0.0).vel.z <= s.vel.z }

/// DIVE. Pitch whose next tick leaves the flight-path angle at `target`.
///
/// gamma(v') is monotone in pitch at dive speeds, but not at low speed, where two branches
/// reach a given angle and only the nose-down one accelerates. So: scan for the last upward
/// crossing, then bisect. Picking the wrong branch stalls the dive completely.
pub fn bug_gamma_to(s: &State, target: f64) -> f64 {
    let h = |p: f64| gamma(ticked(s, p).vel) - target;
    let (mut lo, mut hi) = (f64::NAN, f64::NAN);
    let (mut prev, mut pp) = (h(-90.0), -90.0);
    for i in 1..=1440 {
        let p = -90.0 + 0.125 * i as f64;
        let c = h(p);
        if prev <= 0.0 && c > 0.0 { lo = pp; hi = p }
        prev = c;
        pp = p;
    }
    if lo.is_nan() { return if h(90.0) < 0.0 { 90.0 } else { -90.0 } }
    for _ in 0..60 { let m = 0.5 * (lo + hi); if h(m) <= 0.0 { lo = m } else { hi = m } }
    0.5 * (lo + hi)
}
/// DIVE, as flown: hold the current angle, leaking toward `g_star` at rate `k` per tick.
/// k = 0 is an exact hold, which keeps whatever angle you entered the dive with and loses.
pub fn bug_dive(s: &State, g_star: f64, k: f64) -> f64 {
    let g0 = gamma(s.vel);
    bug_gamma_to(s, g0 + k * (g_star - g0))
}

/// DIVE, leak-free: hold the current angle, but never shallower than `ceiling()`.
///
/// The floor is derived, not fitted: it is the flight-path angle of the fastest steady glide,
/// which is what the dive's gamma settles onto. Entry overshoots it, and from there the rule is
/// an exact hold -- no rate constant anywhere.
pub fn bug_dive_floor(s: &State) -> f64 {
    bug_gamma_to(s, gamma(s.vel).max(ceiling().1))
}

/// The fastest steady glide, `argmax_p eq_vz(p)`, as (pitch, gamma). Swept rather than refined:
/// eq_vz is flat to 1e-9 across the top, so a ternary search there just wanders. Cached, because
/// each equilibrium is 40k iterations of the velocity map.
pub fn ceiling() -> (f64, f64) {
    static C: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        let (mut bz, mut bp) = (f64::NEG_INFINITY, 0.0);
        for i in 0..=1800 {
            let p = 0.05 * i as f64;
            let z = equilibrium(p).z;
            if z > bz { bz = z; bp = p }
        }
        (bp, gamma(equilibrium(bp)))
    })
}

/// Terminal glide for a constant pitch: iterate the velocity map to its fixed point.
pub fn equilibrium(p: f64) -> Vec3 {
    let mut v = Vec3::new(0.0, -0.5, 1.0);
    for _ in 0..40000 { v = update_fall_flying_movement(v, rot(p)) }
    v
}

/// The velocity a schedule settles onto when flown back to back: the fixed point of
/// `v -> (replay the whole schedule from v).vel`.
///
/// This is what makes a *steady-state* profile possible without a constraint or a penalty
/// anywhere. A cycle that ends at a different velocity than it started is not wrong, it is
/// just not the cycle you would fly twice; iterating the schedule finds the one you would.
/// At the fixed point `v_final == v0` identically, so there is nothing left to enforce.
///
/// Iterated to convergence, not damped and not truncated to a lap. Damping would make `v0` a
/// lagged average of past schedules, and it is exactly that lag that could make the outer
/// alternation ring; converging leaves `v0` a memoryless function of the schedule. The map is
/// a strong contraction with a unique attractor -- measured, from starts as far off as
/// `(-3, 3)` and from schedules as degenerate as all-`+90`, chatter, and uniform random, it
/// lands on the same velocity to 1e-13 in 4 to 29 laps.
///
/// `guess` only buys laps. Warm-starting from the previous schedule's fixed point is worth
/// doing and is never load-bearing: the answer does not depend on it.
pub fn steady_vel(pitches: &[f64], guess: Vec3) -> Vec3 {
    const TOL: f64 = 1e-12;
    const MAX_LAPS: usize = 200;          // 29 is the worst seen; this only guards divergence
    let trig = cache_pitches(pitches);
    let mut v = guess;
    for _ in 0..MAX_LAPS {
        let mut s = State { pos: Vec3::ZERO, vel: v };
        for &t in &trig { s = ticked_cached(&s, t) }
        let d = (s.vel.y - v.y).abs() + (s.vel.z - v.z).abs();
        v = s.vel;
        if !v.y.is_finite() || !v.z.is_finite() { return guess }
        if d < TOL { break }
    }
    v
}

/// L1 distance between two velocities in the `(v_y, v_z)` plane. A closure *test*, not a speed
/// difference -- see the note in `examples/repeat.rs`.
pub fn dv_l1(a: Vec3, b: Vec3) -> f64 { (a.y - b.y).abs() + (a.z - b.z).abs() }

// ---------------------------------------------------------------- cycle segmentation

pub fn segment(ps: &[f64], st: &[State], tag: &str) -> Option<(usize, usize, usize, usize, usize)> {
    let apex: Vec<usize> = (1..ps.len() - 1).filter(|&t| st[t].vel.y > 0.0 && st[t + 1].vel.y <= 0.0).collect();
    if apex.len() < 3 { eprintln!("{tag}: only {} apexes, need 3", apex.len()); return None }
    let (a0, a1) = (apex[1], apex[2]);                       // middle cycle, apex to apex
    // The dive ends where the nose comes down and stays down. Require most of the cycle's
    // speed to be built first: a polished dive often has a level stretch early on, which
    // otherwise reads as the snap and collapses every window downstream.
    let v_top = (a0..a1).map(|t| st[t].vel.length()).fold(0.0, f64::max);
    let t_snap = (a0 + 30..a1 - 3)
        .find(|&t| ps[t] < 5.0 && ps[t + 1] < 5.0 && ps[t + 2] < 5.0 && st[t].vel.length() > 0.8 * v_top)
        .unwrap_or(a1);
    let t_gain = (t_snap..a1).find(|&t| st[t].vel.y > 0.0).unwrap_or(a1);
    let t_gend = (t_gain + 10..a1).find(|&t| ps[t] > 0.0).unwrap_or(a1);
    if t_snap <= a0 + 30 || t_gend <= t_gain + 5 {
        eprintln!("{tag}: could not segment the cycle (snap {t_snap}, gain {t_gain}..{t_gend} in {a0}..{a1})");
        return None;
    }
    Some((a0, t_snap, t_gain, t_gend, a1))
}

// ------------------------------------------------- the price vector, non-circularly

pub type M2 = [f64; 4];                                          // row-major [a b; c d] over (y, z)
pub fn mt_vec(m: &M2, v: (f64, f64)) -> (f64, f64) {             // M^T v
    (m[0] * v.0 + m[2] * v.1, m[1] * v.0 + m[3] * v.1)
}
pub fn mt_mat(a: &M2, m: &M2) -> M2 {                            // A^T M
    [a[0] * m[0] + a[2] * m[2], a[0] * m[1] + a[2] * m[3],
     a[1] * m[0] + a[3] * m[2], a[1] * m[1] + a[3] * m[3]]
}
/// d(next velocity)/d(velocity), central differences in the (v_y, v_z) plane.
pub fn jac(v: Vec3, p: f64) -> M2 {
    let h = 1e-6;
    let r = rot(p);
    let (ya, yb) = (update_fall_flying_movement(Vec3::new(0.0, v.y - h, v.z), r),
                    update_fall_flying_movement(Vec3::new(0.0, v.y + h, v.z), r));
    let (za, zb) = (update_fall_flying_movement(Vec3::new(0.0, v.y, v.z - h), r),
                    update_fall_flying_movement(Vec3::new(0.0, v.y, v.z + h), r));
    [(yb.y - ya.y) / (2.0 * h), (zb.y - za.y) / (2.0 * h),
     (yb.z - ya.z) / (2.0 * h), (zb.z - za.z) / (2.0 * h)]
}

/// The periodic costate of a closed cycle: `mu_t = c + A_t^T mu_{t+1}` with `mu_N = mu_0`.
///
/// `(I - M) mu_0 = b` leaves no freedom, so the tangency conditions it implies are predictions
/// rather than fits. `None` when the monodromy makes `I - M` singular and `mu` is undetermined.
pub fn periodic_mu(st: &[State], ps: &[f64], w: f64) -> Option<Vec<(f64, f64)>> {
    let (c, n) = ((1.0, w), ps.len());
    let (mut b, mut m) = ((0.0, 0.0), [1.0, 0.0, 0.0, 1.0]);
    for t in (0..n).rev() {
        let a = jac(st[t].vel, ps[t]);
        let ab = mt_vec(&a, b);
        b = (c.0 + ab.0, c.1 + ab.1);
        m = mt_mat(&a, &m);
    }
    let d = [1.0 - m[0], -m[1], -m[2], 1.0 - m[3]];
    let det = d[0] * d[3] - d[1] * d[2];
    if det.abs() < 1e-12 { return None }
    let mut mu = vec![(0.0, 0.0); n + 1];
    mu[n] = ((d[3] * b.0 - d[1] * b.1) / det, (-d[2] * b.0 + d[0] * b.1) / det);
    for t in (0..n).rev() {
        let a = jac(st[t].vel, ps[t]);
        let am = mt_vec(&a, mu[t + 1]);
        mu[t] = (c.0 + am.0, c.1 + am.1);
    }
    Some(mu)
}

// ------------------------------------------------- the gain arc, in closed form

// The tick map's pitch coefficients on the climbing arc, where `v_y` is high enough that the
// down-to-forward branch is off and the pitch is nose-up. Everything the gain phase does passes
// through `s = sin|pitch|`, linearly except for the lift.
/// The sim's per-tick drags, as `f32` literals widened -- 0.9800000190734863, 0.9900000095367432.
pub const DRAG_Y: f64 = 0.98_f32 as f64;
pub const DRAG_Z: f64 = 0.99_f32 as f64;
pub const GAIN_UP: f64 = 0.128 * DRAG_Y;            // up bought, per unit s per unit v_z
pub const GAIN_LIFT: f64 = 1.5 * GRAVITY * DRAG_Y;  // lift given up, per unit s
pub const GAIN_FWD: f64 = 0.036 * DRAG_Z;           // forward spent, per unit s per unit v_z
/// `GAIN_UP / GAIN_FWD` -- the elytra's forward-to-up exchange rate, 3.519641. The same number
/// ends the pitch-0 hold; see `vz_peaked` and README-myopic.md.
pub const GAIN_RATE: f64 = GAIN_UP / GAIN_FWD;

/// Is the down-to-forward branch off? True exactly when the gain arc's algebra applies.
pub fn dive_branch_off(v: Vec3, p: f64) -> bool {
    let l = (p.to_radians().cos()).powi(2);
    v.y + GRAVITY * (0.75 * l - 1.0) >= 0.0
}

/// GAIN, exact. The interior stationary pitch, from the price vector and `v_z` alone.
///
/// On the climbing arc `v_y' = DRAG_Y(v_y + g(0.75cos^2 p - 1) + 0.128 s v_z)` and
/// `v_z' = DRAG_Z v_z (1 - 0.036 s)`, so `d(mu.v')/ds = 0` reads
/// `mu_y (GAIN_UP v_z - GAIN_LIFT s) = GAIN_FWD mu_z v_z`: the price of the up you buy equals
/// the price of the forward you spend. Note what is absent -- `v_y` does not appear.
pub fn gain_pitch(vz: f64, mu: (f64, f64)) -> f64 {
    let s = vz * (GAIN_UP * mu.0 - GAIN_FWD * mu.1) / (GAIN_LIFT * mu.0);
    -s.clamp(0.0, 1.0).asin().to_degrees()
}

/// The same condition read backwards: the price ratio `mu_z/mu_y` a nose-up pitch implies.
pub fn gain_ratio(vz: f64, p: f64) -> f64 {
    (GAIN_UP - GAIN_LIFT * (-p.to_radians().sin()) / vz) / GAIN_FWD
}

// ------------------------------------------- the gain arc without a price: one constant

// Far from the entry the costate recursions reduce, in continuous time, to `ds/dt = -(K GAIN_FWD
// / 2) v_z` with `s = sin|pitch|`, and that integrates to `s (s + GAIN_LAW_A) = K v_z`. So the
// climb's body needs no price vector at all, only `K`. See "A gain marker without search" in
// README-myopic.md.

/// `2 (1 - DRAG_Z) / GAIN_FWD`, 0.5612. Derived. The data only bracket it: free fits give 0.53 to
/// 0.61 depending on where the body of the climb is cut.
pub const GAIN_LAW_A: f64 = 2.0 * (1.0 - DRAG_Z) / GAIN_FWD;
/// `-(1 - DRAG_Y) / (0.75 g DRAG_Y)`, -0.3401: the far field's `dK/dw`, used here as the line
/// through the two objectives' measured `K` (0.771 at w = 0, 0.670 at w = 0.261). Not confirmed as
/// a slope: horizon-free cycles at fixed length give -0.30 to -0.36 with the body cut at v_z 0.6,
/// -0.24 to -0.33 on the clean plateau.
pub const GAIN_LAW_DKDW: f64 = -(1.0 - DRAG_Y) / (0.75 * GRAVITY * DRAG_Y);
/// `K` at `w = 0`, measured: 0.749..0.771 over horizon-free cycles of 256..450 ticks. The far-field
/// theory alone gives 0.599; the difference is the clock to the entry, and is not derived.
pub const GAIN_LAW_K0: f64 = 0.76;

/// The law's constant at a price on distance.
pub fn gain_law_k(w: f64) -> f64 { GAIN_LAW_K0 + GAIN_LAW_DKDW * w }

/// GAIN, search-free. The nose-up pitch with `s (s + GAIN_LAW_A) = k v_z`: no price, no horizon,
/// no rollout, and `v_y` absent. Holds through the body of the climb, not its end -- the pitch
/// comes back to 0 on a clock to the entry that this does not see.
pub fn gain_law_pitch(vz: f64, k: f64) -> f64 {
    let h = 0.5 * GAIN_LAW_A;
    let s = (h * h + k * vz.max(0.0)).sqrt() - h;
    -s.clamp(0.0, 1.0).asin().to_degrees()
}

/// The law read backwards: the `k` a nose-up pitch implies at `v_z`. Flat along the body of an
/// optimal climb; that flatness is the claim.
pub fn gain_law_phi(vz: f64, p: f64) -> f64 {
    let s = (-p.to_radians().sin()).max(0.0);
    s * (s + GAIN_LAW_A) / vz
}

// ---------------------------------------------------------------- the policy

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dive { Leak, Floor, Hold, Target }

#[derive(Clone, Copy, Debug)]
pub struct P {
    pub g_star: f64, pub k: f64, pub s_switch: f64, pub vy_flick: f64, pub s_exit: f64,
    pub slew: f64, pub p_push: f64, pub p_flick: f64, pub n_gain: usize, pub dive: Dive,
}

/// Has the hold finished? A NaN `vy_flick` asks `vz_peaked` instead of the tuned threshold, so
/// the two stopping rules can be compared with everything else about the policy held fixed.
fn snap_done(s: &State, par: P) -> bool {
    if par.vy_flick.is_nan() { vz_peaked(s) } else { s.vel.y >= par.vy_flick }
}

/// Fly the four bugs, switching on state rather than on the clock, with a pitch rate limit.
pub fn fly(par: P, ticks: usize) -> (Vec<f64>, Vec<u8>, Vec<State>) { fly_pre(par, ticks, &[]) }

/// As `fly`, but the first `pre.len()` ticks of *every* dive replay `pre` open-loop instead of
/// asking the rule. The prefix is part of the control law, not a one-off initial condition, so
/// the limit cycle it settles into is the honest measure of "what does the rule cost once the
/// early dive is handled for it".
pub fn fly_pre(par: P, ticks: usize, pre: &[f64]) -> (Vec<f64>, Vec<u8>, Vec<State>) {
    fly_from(V0, par, ticks, pre)
}

/// As `fly_pre`, from an arbitrary start velocity. The sweep needs this: the policy is the
/// seed for every cell, and `v0` is one of the swept axes.
pub fn fly_from(v0: Vec3, par: P, ticks: usize, pre: &[f64]) -> (Vec<f64>, Vec<u8>, Vec<State>) {
    let mut s = State { pos: Vec3::ZERO, vel: v0 };
    let (mut ps, mut ph, mut st) = (vec![], vec![], vec![s.clone()]);
    let (mut phase, mut last) = (0u8, 0.0f64);
    let mut dive_t = 0usize;
    for _ in 0..ticks {
        let was = phase;
        phase = match phase {
            0 if s.vel.length() >= par.s_switch => 1,     // dive  -> snap
            1 if snap_done(&s, par)             => 2,     // snap  -> flick
            2 if last <= par.p_flick + 1e-9     => 3,     // flick -> gain, once the ramp lands
            3 if s.vel.length() <= par.s_exit   => 0,     // gain  -> dive
            p => p,
        };
        if phase == 0 { if was != 0 { dive_t = 0 } } else { dive_t = 0 }
        let want = match phase {
            0 if dive_t < pre.len() => pre[dive_t],
            0 => if gamma(s.vel) < 0.0 { par.p_push } else {
                match par.dive {
                    Dive::Leak => bug_dive(&s, par.g_star, par.k),
                    Dive::Floor => bug_dive_floor(&s),
                    Dive::Hold => bug_gamma_to(&s, gamma(s.vel)),
                    Dive::Target => bug_gamma_to(&s, ceiling().1),
                }
            },
            1 => 0.0,
            2 => par.p_flick,
            _ => bug_dte_n(&s, par.n_gain),
        };
        // the prefix is the optimum's own schedule, so do not slew-limit it
        let p = if phase == 0 && dive_t < pre.len() { want }
                else { want.clamp(last - par.slew, last + par.slew) };
        if phase == 0 { dive_t += 1 }
        s = ticked(&s, p);
        last = p;
        ps.push(p); ph.push(phase); st.push(s.clone());
    }
    (ps, ph, st)
}

pub fn rate_of(par: P, t: usize) -> f64 { fly(par, t).2[t].pos.y / t as f64 * 20.0 }

// ---------------------------------------------------------------- the objective

/// Reference scales for the normalized weight. `REPLAY_PITCHES_300` climbs 21.5 blocks while
/// covering 330 blocks of z, so `lambda = 1` is the weight at which a cycle's height gain and
/// its distance are worth the same.
pub const Y_REF: f64 = 21.5;
pub const Z_REF: f64 = 330.0;

/// `w` from the normalized weight `lambda`. `w` is a pure exchange rate in block units:
/// blocks of height per block of distance.
pub fn w_of_lambda(lambda: f64) -> f64 { lambda * Y_REF / Z_REF }

/// What a profile is optimal *for*. This is the whole content of a profile's header: given
/// these four numbers and the physics, the optimum is determined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Objective {
    pub v0: Vec3,
    pub n: usize,
    pub lambda: f64,
}

impl Objective {
    pub fn w(&self) -> f64 { w_of_lambda(self.lambda) }

    /// `J = TE(s) + w*z`, both terms in blocks. Terminal velocity is free.
    pub fn j(&self, s: &State) -> f64 { s.total_energy() + self.w() * s.pos.z }

    pub fn replay(&self, pitches: &[f64]) -> Vec<State> { replay_from(self.v0, pitches) }

    /// `J` at the end of the schedule.
    pub fn eval(&self, pitches: &[f64]) -> f64 {
        let trig = cache_pitches(pitches);
        let mut s = State { pos: Vec3::ZERO, vel: self.v0 };
        for p in trig { s = ticked_cached(&s, p) }
        self.j(&s)
    }

    /// `J` averaged over the jitter draws: the value of flying this schedule from a starting
    /// velocity you do not know exactly.
    pub fn eval_jittered(&self, pitches: &[f64], dv: &[Vec3]) -> f64 {
        let trig = cache_pitches(pitches);
        dv.iter().map(|d| {
            let mut s = State { pos: Vec3::ZERO, vel: self.v0 + *d };
            for &p in &trig { s = ticked_cached(&s, p) }
            self.j(&s)
        }).sum::<f64>() / dv.len() as f64
    }
}

// ---------------------------------------------------------------- robustness

/// Optimize against a *distribution of starting states* rather than one exact state.
///
/// Polished hard against a single exact trajectory, the optimizer finds a singular arc and
/// chatters between 0 and 90 degrees every tick through the late dive. That is not a numerical
/// artifact: at 90 degrees `cos(lean_angle)` is zero, so the lift force vanishes *and*
/// `look_hor_length` vanishes, which gates off both the descent-to-forward conversion and the
/// turning term. Bang-bang between "full aero" and "no aero" really does beat any fixed pitch
/// there -- and no human can fly it.
///
/// A slew-rate limit does not fix this, because the cycle contains genuine flicks: the snap
/// drops to 0 and the flick covers ~88 degrees in about six ticks. Any limit loose enough to
/// keep those is loose enough to keep the chatter.
///
/// Perturbing the *pitches* does not fix it either, and was measured: held-fixed draws made the
/// chatter worse (total variation 2406 against 3493), and per-pass resampling cut it only 29%
/// while costing 2.3 blocks of climb. The reason is that the schedule is not especially
/// sensitive to pitch at the chattering end -- `lift_force = cos^2(lean_angle)` is flat at 90
/// degrees, so a degree of pitch error there barely moves anything.
///
/// What works is perturbing the **initial velocity**. A knife-edge gain is knife-edge in the
/// *state*: it exists at one point in `(v_y, v_z)` and the tick that reaches it. Start from a
/// spread of velocities and no single schedule can sit on that point, because the draws arrive
/// at different states at every tick -- while a real flick, which is robust and merely has to
/// happen, keeps its value across the whole spread. This also states the honest problem: you do
/// not know your velocity to three decimals when you start a cycle.
///
/// **Superseded for chatter control, and off by default.** What is written above held when
/// stopping time was the only other lever: jitter suppresses chatter *only while the polish is
/// stopped early*, and polished hard it does nothing (`README-sweep.md` has the numbers -- total
/// variation is flat across an 8x range of `sigma` while `dz` falls monotonically). The l1
/// curvature price in `Rough` replaced it, because that one survives running to convergence.
/// Current runs set `sigma = 0` and carry `mu`; only `runs/corpus` and `runs/veljit`, the older
/// generation, were optimized at `sigma = 0.1`. Reach for this when the question really is
/// "does this schedule work from a starting velocity I do not know exactly", which is what it
/// measures and what it is still good for -- not when the question is chatter.
///
/// `sigma` 0.1 on each component is what this was settled on with, against a reference start of
/// about `(0.17, 0.20)` -- a perturbation of the same order as the velocity itself, so the
/// schedule is being asked to work over a genuinely wide basin rather than to be locally smooth.
/// The draws are *common random numbers* within a pass: one set, reused for every candidate at
/// every tick, so the smoothed objective is a deterministic function of the schedule and the
/// line search is not chasing sampling noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jitter {
    /// Standard deviation of the initial-velocity error, in blocks/tick, applied independently
    /// to `v_y` and `v_z`. 0 disables jitter.
    pub sigma: f64,
    /// How many draws to average over. Cost is linear in this.
    pub draws: usize,
    /// Redraw the perturbations every pass instead of holding one set fixed.
    ///
    /// This is the difference between smoothing and regularizing, and it matters. Held fixed,
    /// the draws define a deterministic surrogate objective with knife edges of its own, and
    /// the optimizer finds those instead. Redrawn each pass, no single realization can be
    /// exploited, so a gain has to survive fresh draws to be kept.
    pub resample: bool,
    pub seed: u64,
}

impl Default for Jitter {
    fn default() -> Self { Jitter { sigma: 0.0, draws: 8, resample: true, seed: 0x5eed_1eaf } }
}

impl Jitter {
    pub fn is_on(&self) -> bool { self.sigma > 0.0 && self.draws > 0 }
    pub fn k(&self) -> usize { if self.is_on() { self.draws.max(1) } else { 1 } }

    /// The velocity offsets, one per draw. Deterministic in `seed`, so a profile can be
    /// re-certified against the same smoothed objective it was optimized under.
    pub fn draws_at_0(&self) -> Vec<Vec3> { self.draws_at(0) }

    /// The offsets for pass `pass`. Identical for every pass unless `resample` is set.
    pub fn draws_at(&self, pass: u64) -> Vec<Vec3> {
        if !self.is_on() { return vec![Vec3::ZERO] }
        let salt = if self.resample { pass.wrapping_mul(0x9e37_79b9_7f4a_7c15) } else { 0 };
        let mut st = self.seed.wrapping_add(salt).wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut next = || -> f64 {                       // splitmix64 -> uniform in (0, 1)
            st = st.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = st;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut normal = || {                            // Box-Muller
            let (u1, u2) = (next(), next());
            self.sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        (0..self.draws).map(|_| {
            let (y, z) = (normal(), normal());
            Vec3 { x: 0.0, y, z }
        }).collect()
    }
}

// ---------------------------------------------------------------- what a hand can do

/// A cost on how much *hand movement* a schedule asks for, and optional hard caps on it.
///
/// Why the second difference and not the first. Total variation cannot separate a flick from
/// chatter, and the numbers say why: the reference cycle's sharpest move is 38.3 deg/tick while
/// chatter runs at 90, a factor of 2.3, so any slew budget wide enough for the snap is wide
/// enough for the alternation. The *rate of change* of pitch separates them cleanly. Measured
/// over the whole 300-tick schedule:
///
/// ```text
///                       max|dp|  max|d2p|  p95|d2p|  sum|d2p|
/// reference cycle          38.3      37.9       0.4     147.1
/// hard-polished (chatter)  90.0     180.0     153.2    4211.1
/// ```
///
/// A factor of 4.7 at the max and 380 at the 95th percentile. The reference is a straight line
/// almost everywhere with a handful of corners -- its curvature is *sparse* -- which is why the
/// norm here is l1 and not l2. `sum |d2p|` with an l1 penalty is l1 trend filtering, whose
/// solutions are piecewise linear with adaptively placed breakpoints: it buys a corner wherever
/// one earns its keep and charges nothing for the straight runs between them. That is the shape
/// a hand produces, and it is the shape the reference cycle already has.
///
/// This is a *stated* term in the utility function, not a stopping rule. The point is that the
/// polish can then run to convergence: `mu` names the exchange rate between blocks of `J` and
/// degrees per tick squared of wrist, and the answer at that rate is a real optimum rather than
/// wherever the optimizer happened to be when it was interrupted.
///
/// **`Default` is not current practice.** It is `mu = 0, limit = 90` -- the *unpriced* objective,
/// because `certify` and `residuals` need exactly that to check a profile against plain `J`.
/// Runs that are actually flown set the price: `runs/atlas` uses `mu = 1e-4, limit = 85` and
/// `runs/antichatter` uses `mu = 1e-3, limit = 85`, both l1 (as of commit bf5b9e3). Seeding new work from
/// a profile means inheriting *its* header, not this default -- and the two older generations,
/// `runs/corpus` and `runs/veljit`, carry no `rough` line at all and were regularized by jitter
/// and stopping time instead. See the generations table in `README.md`.
/// The shape of `Rough`'s price on each second difference `x` (deg/tick^2), before `mu`.
/// `Huber(d)` is `L1` with the corner rounded off: `x^2/2d` inside `d`, the same slope as `L1`
/// outside. `L2(d)` is that quadratic everywhere. Spelled `l1`, `huber:<d>`, `l2:<d>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PriceShape { L1, L2(f64), Huber(f64) }

impl PriceShape {
    pub fn parse(s: &str) -> Result<PriceShape, String> {
        let (k, d) = match s.split_once(':') {
            None => (s, None),
            Some((k, d)) => (k, Some(d.parse::<f64>().map_err(|e| format!("bad width in {s:?}: {e}"))?)),
        };
        if let Some(d) = d { if !(d.is_finite() && d > 0.0) { return Err(format!("{s:?}: the width must be finite and > 0")) } }
        match (k, d) {
            ("l1", None) => Ok(PriceShape::L1),
            ("l2", Some(d)) => Ok(PriceShape::L2(d)),
            ("huber", Some(d)) => Ok(PriceShape::Huber(d)),
            _ => Err(format!("bad price shape {s:?}: l1, l2:<d> or huber:<d>")),
        }
    }
    pub fn f(self, x: f64) -> f64 {
        let a = x.abs();
        match self {
            PriceShape::L1 => a,
            PriceShape::L2(d) => a * a / (2.0 * d),
            PriceShape::Huber(d) => if a <= d { a * a / (2.0 * d) } else { a - 0.5 * d },
        }
    }
    /// `df/dx`. `L1` has a corner at 0, where this returns the subgradient 0; everywhere else
    /// it is the derivative.
    pub fn df(self, x: f64) -> f64 {
        match self {
            PriceShape::L1 => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
            PriceShape::L2(d) => x / d,
            PriceShape::Huber(d) => x.clamp(-d, d) / d,
        }
    }
}

/// The `--penalty` default for new solves. why? on the floor study a quadratic price removed the
/// chatter at no cost (README-floor 6b), and on the `v0 = 0, n = 300` atlas cell it moved the
/// flick by one tick at a cost of 0.007 blocks of `J` (runs/pencycle). Headers written before
/// 2026-09-24 carry no `penalty` line and are l1, which is what a missing line still reads as.
pub const DEFAULT_PENALTY: &str = "l2:2";

impl std::fmt::Display for PriceShape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self { PriceShape::L1 => write!(f, "l1"), PriceShape::L2(d) => write!(f, "l2:{d}"),
                     PriceShape::Huber(d) => write!(f, "huber:{d}") }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rough {
    /// Blocks of `J` charged per degree/tick^2 of summed |second difference|, or per unit of
    /// `shape` of it.
    pub mu: f64,
    /// What `mu` multiplies, per second difference. `L1` unless asked.
    pub shape: PriceShape,
    /// Blocks of `J` charged per degree/tick of summed |first difference| -- plain total
    /// variation. Off by default; kept so the two norms can be compared on one footing.
    pub mu_tv: f64,
    /// Hard cap on |p[t+1] - 2p[t] + p[t-1]|, deg/tick^2. A constraint rather than a price, so
    /// inside the feasible set it does not distort `J` at all. `INFINITY` disables it.
    pub cap: f64,
    /// Hard cap on |p[t+1] - p[t]|, deg/tick. `INFINITY` disables it.
    pub slew_cap: f64,
    /// Largest |pitch| the schedule may use, in degrees. 90 -- the game's own clamp -- allows
    /// the optimizer right up to a discontinuity, and it goes there.
    ///
    /// Both ends of the range are gates, not limits. `look_hor_length` is |cos(pitch)| and every
    /// conversion term is guarded by `look_hor_length > 0`, so where cos underflows to zero the
    /// aerodynamics switch off entirely, and where it changes sign "forward" reverses. Which of
    /// those you hit depends on the trig:
    ///
    /// * `mth_lut` (vanilla). `Mth::cos(-90 deg)` indexes `SIN[0]`, which is exactly 0.0, so the
    ///   gate fails and pitch -90 is a dead tick -- no lift, no conversion, no turning, just
    ///   gravity. At +90 the same table gives +9.6e-5 and the gate passes. The flick wants the
    ///   conversion at its maximum, which is at -90, so the optimizer parks at -89.989: one
    ///   table cell short of the cliff. Overshoot by 0.05 degrees and the schedule loses 5.2
    ///   blocks.
    /// * `libm`. `cos(90 deg)` in f32 is -4.4e-8, *negative*, so `look_angle.z/look_hor_length`
    ///   flips to -1 and the turning term becomes `-0.2 * v_z` -- twenty percent of your forward
    ///   speed per tick. The optimizer parks where cos is +2e-7, within 1e-5 degrees of that.
    ///
    /// Neither is a knob anyone can hold. Minecraft delivers rotation in steps of about
    /// 0.15 * sensitivity degrees, so a margin of a degree or so costs the schedule almost
    /// nothing and removes both cliffs. This is a *control-space* statement, like `cap`: the
    /// pitches outside it are not worse, they are unavailable.
    pub limit: f64,
    /// The tick at which the schedule must first reach `flick_pitch`. This is a restriction of
    /// the control set, not a price: before this tick the pitch is strictly above the threshold,
    /// at this tick it is at or below it, and afterward the ordinary constraints take over.
    pub flick_at: Option<usize>,
    /// What counts as reaching the flick, in degrees.
    pub flick_pitch: f64,
}

impl Default for Rough {
    fn default() -> Self {
        Rough { mu: 0.0, shape: PriceShape::L1, mu_tv: 0.0, cap: f64::INFINITY, slew_cap: f64::INFINITY, limit: 90.0,
                flick_at: None, flick_pitch: -80.0 }
    }
}

/// The five pitches `Rough` needs around tick `t`, lifted out of the schedule so a coordinate
/// search can hold them while the schedule itself is mid-sweep. `None` off the ends.
pub type Win5 = [Option<f64>; 5];

/// The window at `t`: indices `t-2 ..= t+2`, `None` where that runs off the schedule.
pub fn win5(p: &[f64], t: usize) -> Win5 {
    let n = p.len() as isize;
    let t = t as isize;
    std::array::from_fn(|k| {
        let i = t + k as isize - 2;
        if i >= 0 && i < n { Some(p[i as usize]) } else { None }
    })
}

impl Rough {
    pub fn is_on(&self) -> bool {
        self.mu != 0.0 || self.mu_tv != 0.0 || self.cap.is_finite() || self.slew_cap.is_finite()
            || self.limit < 90.0
    }
    pub fn prices(&self) -> bool { self.mu != 0.0 || self.mu_tv != 0.0 }

    /// The whole schedule's roughness cost, in blocks of `J`.
    pub fn cost(&self, p: &[f64]) -> f64 {
        let mut c = 0.0;
        if self.mu != 0.0 {
            for j in 0..p.len().saturating_sub(2) { c += self.mu * self.shape.f(p[j] - 2.0 * p[j + 1] + p[j + 2]) }
        }
        if self.mu_tv != 0.0 {
            for j in 0..p.len().saturating_sub(1) { c += self.mu_tv * (p[j + 1] - p[j]).abs() }
        }
        c
    }

    /// Adds `d cost / dp` to `g`: the gradient of `cost`, with `PriceShape::df`'s subgradient 0 at
    /// an `L1` corner, and the same for the total-variation term.
    pub fn grad(&self, p: &[f64], g: &mut [f64]) {
        if self.mu != 0.0 {
            for j in 0..p.len().saturating_sub(2) {
                let a = self.mu * self.shape.df(p[j] - 2.0 * p[j + 1] + p[j + 2]);
                g[j] += a;
                g[j + 1] -= 2.0 * a;
                g[j + 2] += a;
            }
        }
        if self.mu_tv != 0.0 {
            for j in 0..p.len().saturating_sub(1) {
                let a = self.mu_tv * PriceShape::L1.df(p[j + 1] - p[j]);
                g[j + 1] += a;
                g[j] -= a;
            }
        }
    }

    /// Just the terms the pitch at the window's center takes part in, evaluated at `x`. This is
    /// what a coordinate search has to add to its score: everything else is a constant.
    pub fn local(&self, w: &Win5, x: f64) -> f64 {
        if !self.prices() { return 0.0 }
        let g = |i: usize| if i == 2 { Some(x) } else { w[i] };
        let mut c = 0.0;
        if self.mu != 0.0 {
            for j in 0..3 {
                if let (Some(a), Some(b), Some(d)) = (g(j), g(j + 1), g(j + 2)) {
                    c += self.mu * self.shape.f(a - 2.0 * b + d)
                }
            }
        }
        if self.mu_tv != 0.0 {
            for j in 1..3 {
                if let (Some(a), Some(b)) = (g(j), g(j + 1)) { c += self.mu_tv * (b - a).abs() }
            }
        }
        c
    }

/// How far outside the admissible set a schedule is, in degrees; 0 when it is inside.
    ///
    /// This exists because `certify` was quietly lying without it. `residuals_reg` asks
    /// `feasible` what each pitch may be, and when the *neighbors* already break a cap the
    /// answer is empty, so the search collapses to the single point it started from and reports
    /// a gain of zero. A schedule with 240 deg/tick^2 of curvature under a cap of 0 certified at
    /// residual 0.0 -- a perfect score for a profile that does not satisfy its own header.
    ///
    /// A residual only means anything for a schedule that is *in* the set it claims to be in, so
    /// `certify_reg` checks this first and returns infinity when it is not.
    pub fn violation(&self, p: &[f64]) -> f64 {
        let mut v: f64 = 0.0;
        for (t, &x) in p.iter().enumerate() {
            v = v.max(x.abs() - self.limit);
            if let Some(flick) = self.flick_at {
                if t < flick && x <= self.flick_pitch {
                    // Equality is outside a strict boundary too. Measure to the first pitch the
                    // simulator can actually store above it, so equality cannot look feasible.
                    v = v.max(f32_strictly_above(self.flick_pitch)
                        .map_or(f64::INFINITY, |lo| lo - x));
                } else if t == flick {
                    v = v.max(x - self.flick_pitch);
                }
            }
        }
        if let Some(flick) = self.flick_at {
            if !self.flick_pitch.is_finite() || flick >= p.len() { v = f64::INFINITY }
        }
        if self.cap.is_finite() {
            for j in 0..p.len().saturating_sub(2) {
                v = v.max((p[j] - 2.0 * p[j + 1] + p[j + 2]).abs() - self.cap)
            }
        }
        if self.slew_cap.is_finite() {
            for j in 0..p.len().saturating_sub(1) {
                v = v.max((p[j + 1] - p[j]).abs() - self.slew_cap)
            }
        }
        v.max(0.0)
    }

    /// The interval the center pitch may take without breaking a control restriction,
    /// intersected with `[lo, hi]`. Returns `None` when the restrictions and the neighbors are
    /// already inconsistent, which a warm start can arrive in; the caller then leaves the tick
    /// alone.
    pub fn feasible(&self, w: &Win5, t: usize, lo: f64, hi: f64) -> Option<(f64, f64)> {
        let (mut a, mut b) = (lo.max(-self.limit), hi.min(self.limit));
        let mut clip = |l: f64, h: f64| { a = a.max(l); b = b.min(h) };
        if let Some(flick) = self.flick_at {
            if !self.flick_pitch.is_finite() { return None }
            if t < flick {
                clip(f32_strictly_above(self.flick_pitch)?, f64::INFINITY);
            } else if t == flick {
                clip(f64::NEG_INFINITY, self.flick_pitch);
            }
        }
        if self.cap.is_finite() {
            let c = self.cap;
            // |p[t-2] - 2p[t-1] + x| <= c
            if let (Some(p0), Some(p1)) = (w[0], w[1]) { let m = 2.0 * p1 - p0; clip(m - c, m + c) }
            // |p[t-1] - 2x + p[t+1]| <= c
            if let (Some(p1), Some(p3)) = (w[1], w[3]) { let s = p1 + p3; clip(0.5 * (s - c), 0.5 * (s + c)) }
            // |x - 2p[t+1] + p[t+2]| <= c
            if let (Some(p3), Some(p4)) = (w[3], w[4]) { let m = 2.0 * p3 - p4; clip(m - c, m + c) }
        }
        if self.slew_cap.is_finite() {
            let c = self.slew_cap;
            if let Some(p1) = w[1] { clip(p1 - c, p1 + c) }
            if let Some(p3) = w[3] { clip(p3 - c, p3 + c) }
        }
        if a <= b { Some((a, b)) } else { None }
    }
}

/// Summed |second difference| of the schedule, deg/tick^2. The statistic `Rough::mu` prices,
/// and the one that separates a flick from chatter -- see `Rough`.
pub fn curvature_l1(p: &[f64]) -> f64 {
    (0..p.len().saturating_sub(2)).map(|j| (p[j] - 2.0 * p[j + 1] + p[j + 2]).abs()).sum()
}

/// The largest |second difference|, deg/tick^2: the peak angular acceleration the schedule asks
/// the hand for. 37.9 on the reference cycle, 180 -- the geometric maximum -- on a chattering one.
pub fn curvature_max(p: &[f64]) -> f64 {
    (0..p.len().saturating_sub(2)).map(|j| (p[j] - 2.0 * p[j + 1] + p[j + 2]).abs()).fold(0.0, f64::max)
}

// ---------------------------------------------------------------- the floor

/// A floor `depth` blocks below the start, and a price for coming near it.
///
/// The schedule starts at `y = 0` like every other in this crate, so the floor sits at
/// `y = -depth`; `depth` is the initial height above it. The price is paid once per tick on the
/// state *after* that tick, `states[1..]`, never on the start:
///
/// ```text
/// h   = y + depth                                    clearance, blocks
/// pen = weight * (max(0, margin - h) / margin)^2      the bubble: 0 above `margin`, `weight` at h = 0
///     + wall * max(0, -h)                             under the floor, linear in depth
/// ```
///
/// Why a bubble and a wall, and not a barrier. The coordinate search samples the whole pitch
/// range at every tick, and most candidates crash; a log barrier scores every one of them
/// `-inf` and the sweep cannot tell a schedule that crashes at tick 80 from one that crashes at
/// tick 3. The wall keeps them ranked -- a trajectory that stays under the floor longer, or
/// deeper, pays more -- and the bubble makes first contact continuous instead of a cliff.
///
/// The bubble is conservative on purpose: at a finite `margin` the optimum stands off the floor,
/// so feasibility is judged by `clearance`, never by the penalty being small.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor {
    /// Initial height above the floor, blocks. `INFINITY` turns the floor off.
    pub depth: f64,
    /// Thickness of the bubble, blocks.
    pub margin: f64,
    /// Blocks of `J` charged per tick spent *at* the floor; falls quadratically to 0 at `margin`.
    pub weight: f64,
    /// Blocks of `J` charged per tick per block under the floor.
    pub wall: f64,
}

impl Default for Floor {
    fn default() -> Self { Floor { depth: f64::INFINITY, margin: 0.5, weight: 1.0, wall: 100.0 } }
}

impl Floor {
    pub fn is_on(&self) -> bool { self.depth.is_finite() }

    /// The price of one post-tick state.
    #[inline]
    pub fn tick(&self, s: &State) -> f64 {
        let h = s.pos.y + self.depth;
        if h >= self.margin { return 0.0 }
        let b = (self.margin - h) / self.margin;
        self.weight * b * b + self.wall * (-h).max(0.0)
    }

    /// The whole replay's price, `states[1..]`. 0 when off.
    pub fn cost(&self, states: &[State]) -> f64 {
        if !self.is_on() { return 0.0 }
        states[1..].iter().map(|s| self.tick(s)).sum()
    }

    /// The least clearance over `states[1..]`, blocks. Negative means the replay went through
    /// the floor. This, not `cost`, is what feasibility is judged on.
    pub fn clearance(&self, states: &[State]) -> f64 {
        states[1..].iter().map(|s| s.pos.y + self.depth).fold(f64::INFINITY, f64::min)
    }

    /// Ticks flown before the first state under the floor: the schedule's endurance, capped at
    /// its own length.
    pub fn survived(&self, states: &[State]) -> usize {
        states[1..].iter().position(|s| s.pos.y + self.depth < 0.0).unwrap_or(states.len() - 1)
    }
}

/// `J` averaged over the draws, less the floor's price on each draw's replay. With the floor off
/// this is exactly `eval_jittered`.
fn eval_floored(obj: &Objective, pitches: &[f64], dv: &[Vec3], floor: &Floor) -> f64 {
    if !floor.is_on() { return obj.eval_jittered(pitches, dv) }
    let trig = cache_pitches(pitches);
    dv.iter().map(|d| {
        let mut s = State { pos: Vec3::ZERO, vel: obj.v0 + *d };
        let mut c = 0.0;
        for &p in &trig { s = ticked_cached(&s, p); c += floor.tick(&s) }
        obj.j(&s) - c
    }).sum::<f64>() / dv.len() as f64
}

// ---------------------------------------------------------------- the optimizer

/// Degrees: the sample spacing of `certify`'s global pass. Finer than the polish's own global
/// passes on purpose: the certificate is the profile's claim, re-derived by `verify
/// --optimality` from the file alone, and costs one pass where the polish costs a hundred.
pub const CERTIFY_STEP: f64 = 0.25;

/// Degrees: the polish's and the certificate's per-tick pitch tolerance. A pitch is flown as an
/// `f32`, ~4e-6 degrees apart at 50.
pub const PITCH_TOL: f64 = 1e-5;

/// How hard to polish. The defaults are `cmd_polish`'s, which is the schedule every number in
/// `README-myopic.md` was produced with.
#[derive(Clone, Copy, Debug)]
pub struct PolishOpts {
    pub max_passes: usize,
    /// Sweep the whole pitch range every this many passes; otherwise search near the current
    /// pitch. The global sweep is what handles the objective not being unimodal in pitch.
    pub global_every: usize,
    /// Degrees between the samples of a global pass, and of a local one. Each sampled peak is
    /// then refined to `pitch_tol`, so these set how narrow a peak can be and still be found, not
    /// the precision. The global step is coarse because a global pass is looking for another
    /// basin; the local one is finer because the moves it has to find are smaller.
    ///
    /// Under `mth_lut` the per-tick objective is a ragged staircase of 0.0055-degree stairs, and
    /// the better stair is often 0.03-0.3 degrees from where golden section settles; with only
    /// samples and golden section the polish stalled at a residual of 1e-5 to 5e-5 under every
    /// control constraint tried. `coordinate_argmax`'s `stair_scan` is what finds those stairs,
    /// not the sample spacing.
    ///
    /// Cost is counted in simulated ticks, because sweeps run `RAYON_NUM_THREADS=1` and
    /// parallelize over cells. Against the old 0.25/0.05 sweep plus 70 ternary steps (2026-10-02,
    /// `mth_lut`, n = 300): 2/0.5 with the stair scan spends 1.7-4.2x fewer ticks, and its J was
    /// better or equal on 11 of 12 benchmark cells and on 3 of 5 atlas-setting cells (mean
    /// +3.9e-4 blocks there). A 0.05 local step without the scan cost twice the ticks for a
    /// slightly worse J; narrowing the local span to 4 degrees lost 3e-2 on the unconstrained cell.
    pub global_step: f64,
    pub local_span: f64,
    pub local_step: f64,
    /// Degrees: how finely each tick's best pitch is refined. See `PITCH_TOL`.
    pub pitch_tol: f64,
    /// Stop once the schedule has gained less than this many blocks of `J` over the last
    /// `stall_window` passes. Default 0.1: past that the optimizer is tuning below the
    /// precision anyone could fly, and the extra precision is the part most likely to be
    /// brittle.
    ///
    /// Note what this is *not*. The per-tick residual that `certify` reports bounds only
    /// single-coordinate moves, and a schedule can sit blocks below a better optimum with no
    /// single tick improving by more than a hundredth -- because reaching it needs many ticks
    /// to move together. Measured: a 130-tick cell stopped at residual 0.066 came out 4.7
    /// blocks short of the same cell polished properly. So convergence is judged on progress
    /// in `J`, and the residual is reported as a certificate rather than used as a target.
    pub tol: f64,
    /// Block size for `block_step`, run before each global pass. 0 disables it, which is the
    /// default: it is a wash on wall-clock. See `block_step`.
    pub block: usize,
    /// Optimize against a jittered schedule. See `Jitter`; costs `draws` times as much.
    pub jitter: Jitter,
    /// How many passes of near-no-progress to require before stopping. Coordinate ascent on
    /// this problem stalls and then jumps -- a global pass finds a new basin every so often --
    /// so a single quiet pass means nothing.
    pub stall_window: usize,
    /// Stop as soon as the schedule turns degenerate: the lag-1 correlation of its per-tick
    /// changes falling below this ends the polish and *discards* the pass that crossed it.
    ///
    /// This is the regularizer. Polished to convergence the exact objective genuinely prefers a
    /// chattering singular arc -- residual 9.4e-6 at 200 passes, so it is a real coordinate
    /// optimum, not a failure to converge -- and no perturbation of the objective that was
    /// measured avoids it: initial-velocity jitter is flat in schedule smoothness across an 8x
    /// range of sigma (total variation 925, 871, 838, 826, 884 at sigma 0, 0.05, 0.1, 0.2, 0.4
    /// over 60 passes) while costing up to 8 blocks of distance. Stopping time is what works.
    ///
    /// It matters more under continuation than it looks: a warm-started cell inherits its
    /// neighbor's polish, so a per-cell pass budget accumulates along the path and a cell far
    /// from the anchor has been polished many times over. A budget cannot bound that; a
    /// property of the schedule can.
    ///
    /// `f64::NEG_INFINITY` disables the guard.
    pub lag1_floor: f64,
    /// What the schedule is charged for hand movement, and any hard caps on it. See `Rough`.
    ///
    /// This is the alternative to a stopping rule. `lag1_floor` bounds chatter by ending the
    /// polish before it appears, which leaves the answer depending on when the optimizer was
    /// interrupted; a price on curvature bounds it by making it cost something, which leaves an
    /// answer that can be polished to convergence and certified.
    pub rough: Rough,
    /// Re-solve `v0` to the schedule's own steady state after each pass, instead of holding the
    /// objective's stated `v0` fixed. See `steady_vel`.
    ///
    /// Off by default, and deliberately: the single-cycle problem is the simpler object and is
    /// still the one most questions are about. This is the same opt-in the `cycle-optimizer`
    /// made with its `Optimizer<const STEADY_STATE: bool>`.
    ///
    /// The update lands at the *end* of a pass, after `last_gain` and the stall test, for the
    /// reason the gradient-descent version froze it through the finite differences: `before`
    /// and `last_gain` have to be evaluated against the same objective or their difference
    /// mixes the schedule step with the `v0` move, and the stall test is then reading a number
    /// that is not a gain. Convergence also needs `v0` to have stopped moving, not just `J` --
    /// otherwise the schedule is written out with a header `v0` that is not its own fixed
    /// point, which is the one thing this mode exists to prevent.
    pub steady: bool,
    /// A floor under the trajectory, priced per tick. Off by default. See `Floor`.
    pub floor: Floor,
}

impl Default for PolishOpts {
    fn default() -> Self {
        PolishOpts {
            max_passes: 200, global_every: 4, global_step: 2.0,
            local_span: 8.0, local_step: 0.5, pitch_tol: PITCH_TOL, tol: 0.1, block: 0, stall_window: 12,
            // Off by default. It reads well but interacts badly with continuation: a warm
            // start arrives with a seam where `stretch` joined, that reads as chatter, and the
            // guard then stops the polish that would have repaired it -- freezing a chain of
            // cells at an unimproved copy of their parent. Stopping time is applied as a pass
            // budget instead. Set it explicitly for a single cold polish, where it works.
            lag1_floor: f64::NEG_INFINITY,
            jitter: Jitter::default(),
            rough: Rough::default(),
            steady: false,
            floor: Floor::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Polished {
    pub pitches: Vec<f64>,
    pub j: f64,
    pub passes: usize,
    /// What the last pass gained. Small means the schedule stopped moving under *this* search.
    pub last_gain: f64,
    /// What a fresh full global pass can still find, from `certify`. This is the number that
    /// makes the file's claim checkable; it does not depend on how the schedule was reached.
    pub residual: f64,
    /// The degeneracy statistic at the schedule that was kept. See `lag1`.
    pub lag1: f64,
    /// Whether the polish ended because the schedule was turning degenerate rather than
    /// because it stopped gaining. Recorded because it changes what the residual means: the
    /// profile is deliberately short of the coordinate optimum, not converged to it.
    pub stopped_degenerate: bool,
    /// What the roughness price took off `j`. Reported separately so the raw `J` of a
    /// regularized answer is still readable: the schedule is worth `j` blocks and cost
    /// `rough_cost` blocks of wrist to fly.
    pub rough_cost: f64,
    /// The `v0` the schedule is optimal for. Equal to the objective's `v0` unless
    /// `PolishOpts::steady` was set, in which case it is the schedule's own fixed point and the
    /// caller must write *this* into the header, not the `v0` it asked for.
    pub v0: Vec3,
    /// How far `v0` moved on the last pass, L1 in `(v_y, v_z)`. 0 when `steady` is off.
    pub dv0: f64,
}

/// The nearest `f32` to `x` that lies inside `[lo, hi]`, or `None` if the interval holds none.
///
/// Pitch is stored as `f32`, so every candidate is rounded before it is flown -- and rounding is
/// nearest-even, which can step back *out* of an interval the caller just clamped into. Small:
/// a zero-pass polish at `limit = 0.1` wrote 0.10000000149011612, over by 1.5e-9 degrees. But
/// `Rough::violation` gates certification now, so "over by 1.5e-9" is the difference between a
/// profile satisfying its own header and failing its own `verify`.
fn f32_inside(x: f64, lo: f64, hi: f64) -> Option<f64> {
    let nudge = |v: f32, up: bool| -> f32 {
        if v == 0.0 { return if up { f32::from_bits(1) } else { -f32::from_bits(1) } }
        let b = v.to_bits();
        f32::from_bits(if (v > 0.0) == up { b + 1 } else { b - 1 })
    };
    let mut v = x.clamp(lo, hi) as f32;
    for _ in 0..3 { if (v as f64) < lo { v = nudge(v, true) } else { break } }
    for _ in 0..3 { if (v as f64) > hi { v = nudge(v, false) } else { break } }
    ((v as f64) >= lo && (v as f64) <= hi).then_some(v as f64)
}

/// The least `f32` strictly greater than `x`. The pre-flick half of the admissible set is open
/// in real arithmetic, but schedules are flown and written as `f32`; turning it into this closed
/// endpoint lets every later clamp and refinement preserve the strict condition.
fn f32_strictly_above(x: f64) -> Option<f64> {
    if !x.is_finite() || x >= f32::MAX as f64 { return None }
    let mut v = x as f32;
    if v as f64 <= x {
        let bits = v.to_bits();
        v = if v == 0.0 { f32::from_bits(1) }
            else { f32::from_bits(if v > 0.0 { bits + 1 } else { bits - 1 }) };
    }
    Some(v as f64)
}

/// Best pitch for tick `t`, holding every other tick fixed, scored on the exact tail: a
/// `coordinate_argmax` over `[lo, hi]`, sampled every `step` degrees and refined to `tol`.
/// Returns `(pitch, J, J at cur)` with the pitch already rounded to `f32`, because the sim casts
/// pitch to `f32` anyway -- so the value returned is the one that will actually be flown, and its
/// score is the score of flying it.
///
/// `pen` is the roughness price of putting a given pitch here -- the terms of `Rough::cost`
/// that this tick takes part in. It is subtracted from the score, so the coordinate search
/// optimizes the same regularized objective the polish reports, not `J` alone.
fn best_pitch_at(obj: &Objective, s: &[State], tail: &[PitchTrig],
                 cur: f64, lo: f64, hi: f64, step: f64, tol: f64,
                 pen: &(dyn Fn(f64) -> f64 + Sync), floor: &Floor) -> (f64, f64, f64) {
    // One prefix state per jitter draw. The draws differ only in where they started, so the
    // tail is the exact schedule flown from each of them -- this is E[J | v0 + dv] under common
    // random numbers, with no per-tick noise to average away.
    //
    // The floor's price is charged on the ticks this pitch can move, `t+1..`; the prefix's share
    // is the same for every candidate, so leaving it out changes no comparison.
    let score = |p: f64| -> f64 {
        let trig = PitchTrig::new(p as f32);
        s.iter().map(|s0| {
            let mut st = ticked_cached(s0, trig);
            if floor.is_on() {
                let mut c = floor.tick(&st);
                for &q in tail { st = ticked_cached(&st, q); c += floor.tick(&st) }
                obj.j(&st) - c
            } else {
                for &q in tail { st = ticked_cached(&st, q) }
                obj.j(&st)
            }
        }).sum::<f64>() / s.len() as f64 - pen(p)
    };
    let j_cur = score(cur);
    // The incumbent has to be *inside* the interval, or a warm start that arrives out of bounds
    // is never brought in: it would simply out-score every candidate and be kept. `j_cur` still
    // reports the score of where we actually are, because that is what a residual is measured
    // against.
    // No representable pitch inside the interval: leave the tick alone rather than step out of it.
    let Some(start) = f32_inside(cur, lo, hi) else { return (cur, j_cur, j_cur) };
    let j_start = if start == cur { j_cur } else { score(start) };
    let (p, _) = coordinate_argmax(&score, start, lo, hi, step, tol);
    // Round first, then score: the schedule that gets written must be the schedule that was
    // measured, or a cell can be certified on a pitch it does not contain. Rounded *into* the
    // interval, so the search never returns a pitch outside the admissible set the caller just
    // computed.
    let refined = f32_inside(p, lo, hi).map(|r| (r, score(r)));
    // Strictly better only, starting from the incumbent: a tie never moves the tick.
    let (mut bp, mut bs) = (start, j_start);
    if let Some((r, v)) = refined { if v > bs { (bp, bs) = (r, v) } }
    (bp, bs, j_cur)
}

/// What a full global pass would do at every tick, without doing it: the best pitch, the move
/// it implies, and what that move is worth. This is the raw material for both `certify` (the
/// largest gain) and for asking whether the moves point the same way.
pub fn residuals(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter) -> Vec<(f64, f64)> {
    residuals_reg(obj, pitches, step, jit, Rough::default())
}

/// `residuals` against the regularized objective. A profile written under a roughness price is
/// a coordinate optimum of *that* objective, so its certificate has to be re-derived under the
/// same price or it will report a gain the writer deliberately did not take.
pub fn residuals_reg(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter, rough: Rough)
    -> Vec<(f64, f64)> {
    residuals_floored(obj, pitches, step, jit, rough, Floor::default())
}

/// `residuals_reg` with the floor's price as well: the certificate of a floored polish.
pub fn residuals_floored(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter, rough: Rough,
                         floor: Floor) -> Vec<(f64, f64)> {
    let dv = jit.draws_at_0();
    let trig = cache_pitches(pitches);
    let states = jittered_replays(obj, &trig, &dv);
    (0..pitches.len())
        .map(|t| {
            let row: Vec<State> = states.iter().map(|st| st[t].clone()).collect();
            let w = win5(pitches, t);
            let pen = |x: f64| rough.local(&w, x);
            let (lo, hi) = rough.feasible(&w, t, -90.0, 90.0).unwrap_or((pitches[t], pitches[t]));
            let (p, j, j_cur) = best_pitch_at(obj, &row, &trig[t + 1..], pitches[t],
                                              lo, hi, step, PITCH_TOL, &pen, &floor);
            (p - pitches[t], j - j_cur)
        })
        .collect()
}

/// One replay per jitter draw. `out[k][t]` is the state at tick `t` having started from
/// `v0 + dv[k]`.
fn jittered_replays(obj: &Objective, pitches: &[PitchTrig], dv: &[Vec3]) -> Vec<Vec<State>> {
    dv.iter().map(|d| {
        let mut v = vec![State { pos: Vec3::ZERO, vel: obj.v0 + *d }];
        for &p in pitches {
            let s = ticked_cached(v.last().unwrap(), p);
            v.push(s);
        }
        v
    }).collect()
}

/// One full global pass that changes nothing, reporting the largest gain it could have made.
///
/// This is the profile's certificate. It re-derives the claim from `(v0, n, lambda, pitches)`
/// alone, so it can be run by anyone holding the file and says nothing about how the schedule
/// was found. A residual at or below the writer's tolerance means the schedule is a coordinate
/// optimum to that tolerance.
///
/// **Under resampled jitter this is a pass-zero audit, not the objective the last pass saw.**
/// It always draws `jit.draws_at_0()`, while `polish` with `resample` set redraws every pass, so
/// the two are different samples of the same smoothed objective. Measured on a small fixture
/// (n = 5, sigma = 0.3, three draws): the returned schedule scores 3.668 under the pass-zero
/// draws and 3.005 under the final pass's, a difference of 0.66 blocks. It is reproducible from
/// the header, which is what a certificate needs to be, but it is neither a statement about the
/// population nor an independent holdout -- the first pass optimized on exactly those draws.
/// With jitter off or `--fixed-draws` the distinction disappears, and every profile in
/// `README-control.md` has jitter off.
pub fn certify(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter) -> f64 {
    certify_reg(obj, pitches, step, jit, Rough::default())
}

/// `certify` under a roughness price. See `residuals_reg`.
///
/// Infinite for a schedule that is outside the admissible set its own header names. A residual
/// is "how much could one coordinate move gain", and the coordinate search cannot move at all
/// when the neighbors are already illegal -- so without this check an infeasible profile scores
/// a perfect 0.0. See `Rough::violation`.
pub fn certify_reg(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter, rough: Rough) -> f64 {
    certify_floored(obj, pitches, step, jit, rough, Floor::default())
}

/// `certify_reg` against the floored objective.
pub fn certify_floored(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter, rough: Rough,
                       floor: Floor) -> f64 {
    if rough.violation(pitches) > 0.0 { return f64::INFINITY }
    residuals_floored(obj, pitches, step, jit, rough, floor).iter().map(|x| x.1).fold(0.0, f64::max)
}

/// Is the per-tick correction *smooth in t*?
///
/// A different question from whether the corrections share a sign overall. A delta that is
/// smoothly positive over the first half and negative over the second sums to nothing, so it
/// looks like balanced noise in bulk, while being highly structured and worth exploiting.
/// What matters for a big-step method is whether neighboring ticks want the same correction,
/// because then the whole error lives in a handful of smooth modes.
///
/// Returns the lag-1 correlation of the deltas, the mean length of a same-sign run, and the
/// fraction of the deltas' energy captured by the first `k` cosine modes for a few `k`.
pub fn delta_structure(d: &[f64]) -> (f64, f64, Vec<(usize, f64)>) {
    let n = d.len();
    // Ticks whose pitch is already best are exactly 0, and a run of zeros correlates with
    // itself perfectly -- which would report a smooth correction where there is no correction
    // at all. Measure on the ticks that actually want to move.
    let live: Vec<f64> = d.iter().cloned().filter(|x| x.abs() > 1e-9).collect();
    let m = live.len();
    let lag1 = if m > 2 {
        let mean = live.iter().sum::<f64>() / m as f64;
        let var: f64 = live.iter().map(|x| (x - mean).powi(2)).sum();
        // adjacent *in the schedule*, both live
        let pairs: Vec<(f64, f64)> = (0..n - 1)
            .filter(|&t| d[t].abs() > 1e-9 && d[t + 1].abs() > 1e-9)
            .map(|t| (d[t] - mean, d[t + 1] - mean)).collect();
        if var > 0.0 && !pairs.is_empty() {
            pairs.iter().map(|(a, b)| a * b).sum::<f64>() / var * (m as f64 / pairs.len() as f64)
        } else { 0.0 }
    } else { 0.0 };

    let mut runs = 1usize;
    for t in 1..m { if (live[t] >= 0.0) != (live[t - 1] >= 0.0) { runs += 1 } }
    let run_len = m as f64 / runs as f64;

    // DCT-II. O(n^2), and n is at most a few hundred.
    let total: f64 = d.iter().map(|x| x * x).sum();
    let coef: Vec<f64> = (0..n).map(|k| {
        (0..n).map(|t| d[t] * (std::f64::consts::PI / n as f64
                               * (t as f64 + 0.5) * k as f64).cos()).sum::<f64>()
    }).collect();
    // Parseval for DCT-II: sum_k c_k^2 * (2 - [k==0]) / (2n) = sum_t d_t^2
    let energy = |k: usize| -> f64 {
        let e: f64 = (0..k.min(n)).map(|j| coef[j] * coef[j] * if j == 0 { 1.0 } else { 2.0 }).sum();
        if total > 0.0 { (e / (2.0 * n as f64) / total).min(1.0) } else { 0.0 }
    };
    let ks = [1, 2, 4, 8, 16, 32].iter().map(|&k| (k, energy(k))).collect();
    (lag1, run_len, ks)
}

/// Move every tick toward its own best pitch at once, and line-search how far to go.
///
/// Coordinate ascent changes one tick at a time, so when many ticks want to move the same way
/// it makes the move n times over, each one partly undone by its neighbors. Measured: a
/// schedule 4.7 blocks short of its optimum had 0.79 coherence -- the sum of its per-tick moves
/// was 709 degrees against 901 degrees of absolute movement, nearly all one-signed -- while a
/// well-polished one sat at -0.09, balanced noise. So the one-sidedness is a symptom of being
/// stuck, and this step is the cure for exactly that case and a no-op otherwise.
///
/// Measured, and it does not pay, which is why it is off by default. The best step is only
/// `alpha` 0.23 to 0.43 and captures 23-43% of the summed per-tick gains: the ticks want to
/// move together but they *interact*, so moving them all at once invalidates each one's target.
/// Coordinate ascent already handles that by updating the state as it sweeps, and matches or
/// beats this at every pass count. Kept because the diagnostic is the evidence.
///
/// Costs one global pass plus a handful of replays. Returns `None` when no step helps.
pub fn jacobi_step(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter)
    -> Option<(Vec<f64>, f64, f64)> {
    jacobi_step_reg(obj, pitches, step, jit, Rough::default())
}

/// `jacobi_step` against the regularized objective, and against a consistent one.
///
/// The original compared a *jittered* baseline with *unjittered* candidates -- `base` came from
/// `eval_jittered` and the line search from `eval` -- so under jitter the reported gain was the
/// difference of two different objectives. It is a diagnostic and off by default, which is why
/// nothing depended on it, but a diagnostic that reports a number has to report the right one.
pub fn jacobi_step_reg(obj: &Objective, pitches: &[f64], step: f64, jit: Jitter, rough: Rough)
    -> Option<(Vec<f64>, f64, f64)> {
    let d = residuals_reg(obj, pitches, step, jit, rough);
    let dv = jit.draws_at_0();
    let score = |p: &[f64]| obj.eval_jittered(p, &dv) - rough.cost(p);
    let base = score(pitches);
    let at = |alpha: f64| -> Vec<f64> {
        pitches.iter().zip(&d).enumerate()
            .map(|(t, (p, (delta, _)))| {
                let (lo, hi) = rough.feasible(&win5(pitches, t), t, -90.0, 90.0)
                    .unwrap_or((*p, *p));
                f32_inside(p + alpha * delta, lo, hi).unwrap_or(*p)
            })
            .collect()
    };
    // Coarse sweep then bisection-free refine: the gain along alpha is not unimodal either,
    // since the schedule crosses the pitch-0 corner at different alphas for different ticks.
    let (mut best_a, mut best_j) = (0.0, base);
    let mut alpha = 0.02;
    while alpha <= 1.6 {
        let j = score(&at(alpha));
        if j > best_j { best_j = j; best_a = alpha }
        alpha += 0.02;
    }
    if best_a == 0.0 { return None }
    // refine around the winner
    let (lo, hi) = (best_a - 0.02, best_a + 0.02);
    for i in 1..20 {
        let a = lo + (hi - lo) * i as f64 / 20.0;
        let j = score(&at(a));
        if j > best_j { best_j = j; best_a = a }
    }
    let out = at(best_a);
    if rough.violation(&out) > 0.0 { return None }
    Some((out, best_a, best_j - base))
}

/// Line-search a separate step size for each block of ticks.
///
/// The corrections are locally correlated but not globally so: lag-1 around +0.46 to +0.50 on
/// an under-converged cell, with same-sign runs of 15 to 24 ticks, while only a quarter of
/// their energy sits in the first 32 cosine modes. That is the shape of an error that is
/// roughly constant over a phase and unrelated between phases -- so one step size for the
/// whole schedule averages independent corrections against each other and finds nothing, and a
/// low-mode projection misses them too. A step size per block matches the measured structure.
///
/// Blocks are taken greedily, so each one's line search already sees the blocks before it.
///
/// Measured, and it is a wash, which is why it is off by default. It does beat a single
/// whole-schedule step, confirming the diagnosis -- but the block size barely matters (5
/// through 80 all land within 0.01 blocks), so the gain is not really about matching the block
/// structure, and on wall-clock it buys at 16 passes what plain coordinate ascent reaches at
/// 19, while costing 22% more per pass.
///
/// The underlying reason no big step pays: each tick's target is computed holding the others
/// fixed, and they interact strongly. The best step along the aggregate direction is only
/// `alpha` 0.23 to 0.43 and captures 23-43% of the summed per-tick gains. Coordinate ascent
/// already accounts for the interaction by updating the state as it sweeps, which is exactly
/// what a simultaneous step throws away.
pub fn block_step(obj: &Objective, pitches: &[f64], step: f64, block: usize, jit: Jitter)
    -> Option<(Vec<f64>, f64)> {
    block_step_reg(obj, pitches, step, block, jit, Rough::default())
}

/// `block_step` against the regularized objective.
///
/// The unregularized one is *wrong* to use under a price and not subtly: it line-searches on
/// `J` alone, so it accepts a step that raises `J` while raising the curvature far more, and
/// reports the size of the `J` gain as though that were the improvement. Measured at `mu = 0.1`
/// it took a schedule from 0.188 to -4.168 on the regularized objective and called it a gain of
/// 0.19; at `mu = 1`, from 0.188 to -48.2. `polish` calls this one.
pub fn block_step_reg(obj: &Objective, pitches: &[f64], step: f64, block: usize, jit: Jitter,
                      rough: Rough) -> Option<(Vec<f64>, f64)> {
    let d = residuals_reg(obj, pitches, step, jit, rough);
    let dv = jit.draws_at_0();
    let score = |p: &[f64]| obj.eval_jittered(p, &dv) - rough.cost(p);
    let base = score(pitches);
    let mut cur = pitches.to_vec();
    let mut best_j = base;
    for start in (0..pitches.len()).step_by(block) {
        let end = (start + block).min(pitches.len());
        let at = |alpha: f64| -> Vec<f64> {
            let mut v = cur.clone();
            for t in start..end {
                let (lo, hi) = rough.feasible(&win5(&cur, t), t, -90.0, 90.0)
                    .unwrap_or((cur[t], cur[t]));
                v[t] = f32_inside(cur[t] + alpha * (pitches[t] + d[t].0 - cur[t]), lo, hi)
                    .unwrap_or(cur[t]);
            }
            v
        };
        let (mut ba, mut bj) = (0.0, best_j);
        let mut alpha = 0.05;
        while alpha <= 1.5 {
            let j = score(&at(alpha));
            if j > bj { bj = j; ba = alpha }
            alpha += 0.05;
        }
        if ba > 0.0 { cur = at(ba); best_j = bj }
    }
    // A block step moves whole spans at once, so it can leave the admissible set even though
    // every endpoint is clamped -- reject rather than repair, since the caller has a perfectly
    // good schedule already.
    if rough.violation(&cur) > 0.0 { return None }
    (best_j > base).then(|| (cur, best_j - base))
}

/// How still `v0` has to be, summed over a `stall_window` of passes, before a steady-state
/// polish may call itself converged. L1 in `(v_y, v_z)`.
///
/// Summed rather than per-pass because a drift and a jitter of the same per-pass size are the
/// same number and completely different situations: measured on n=150, 9.5e-3 a pass read as
/// converged while it carried `v0y` from 0.568 to 0.430 over 40 passes.
///
/// Absolute rather than priced against `tol`. Pricing it was tried and was a mistake: `v0` error
/// costs about 5.5 blocks of `TE` per unit (measured 2.0 at n=100, 4.0 at n=600, 5.3 at n=150,
/// 5.5 at n=300, linear over 1e-3 to 1e-2), so `tol = 0.1` admits a windowed drift of 0.018 --
/// and the polish then stopped at 18 passes with `dv0` at 3e-4, six orders short of converged,
/// writing a header `v0` the schedule does not reproduce. `tol` is a coarse threshold on the
/// *schedule's* gain and does not transfer.
///
/// The value is set by what the iteration actually reaches, not by what is tolerable: the inner
/// solve converges to 1e-12 and the outer drift falls to 1e-10 on its own, so this is slack by
/// two orders and still worth under 1e-6 blocks.
const STEADY_DRIFT_TOL: f64 = 1e-7;

/// Coordinate ascent over the schedule: sweep `t = 0..n`, replacing each pitch with the best
/// one given the rest. Non-unimodality in pitch is why the sweep has to be global, and the
/// corner at pitch 0 -- the forward-to-up conversion is gated on `lean_angle < 0` -- is why it
/// cannot be replaced by a derivative method.
pub fn polish(obj: &Objective, init: &[f64], opts: PolishOpts) -> Polished {
    assert_eq!(init.len(), obj.n, "schedule length must match the objective's horizon");
    // Neither line search prices the floor, and a steady cycle has no floor to speak of.
    assert!(!opts.floor.is_on() || (opts.block == 0 && !opts.steady),
            "a floored polish supports neither --block nor --steady");
    let floor = opts.floor;
    // Project the seed into the admissible set first. A warm start routinely arrives outside it
    // -- every relaxed solution parks at +-90 -- and the sweep is a local move, not a repair:
    // where the neighbors are out of bounds `Rough::feasible` has nothing to offer and every
    // tick is skipped.
    let mut pitches = project_cap(init, opts.rough);
    // `v0` is an iterate in steady mode, so the objective cannot stay borrowed. Seeded from the
    // schedule we were handed: under continuation that is the previous cell's answer, which is
    // already near its own fixed point, so the first solve is a few laps rather than a cold one.
    let mut obj = *obj;
    if opts.steady { obj.v0 = steady_vel(&pitches, obj.v0) }
    let mut trig = cache_pitches(&pitches);
    let mut dv = opts.jitter.draws_at(0);
    let mut states = jittered_replays(&obj, &trig, &dv);
    let (mut passes, mut last_gain) = (0, f64::INFINITY);
    let (mut dv0, mut stopped_degenerate) = (0.0f64, false);
    let mut recent: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut drift: std::collections::VecDeque<f64> = std::collections::VecDeque::new();

    for pass in 0..opts.max_passes {
        if opts.jitter.is_on() && opts.jitter.resample && pass > 0 {
            dv = opts.jitter.draws_at(pass as u64);
            states = jittered_replays(&obj, &trig, &dv);
        }
        let before = eval_floored(&obj, &pitches, &dv, &floor) - opts.rough.cost(&pitches);
        let prev = pitches.clone();
        let global = pass % opts.global_every == 0;
        // Before each global pass, try moving the whole schedule at once. When the per-tick
        // moves point the same way this leaps; when they do not it finds no step and costs
        // one pass.
        if global && opts.block > 0 {
            if let Some((next, gain)) = block_step_reg(&obj, &pitches, opts.global_step, opts.block,
                                                       opts.jitter, opts.rough) {
                if gain > 0.0 {
                    pitches = next;
                    trig = cache_pitches(&pitches);
                    states = jittered_replays(&obj, &trig, &dv);
                }
            }
        }
        let mut worst_tick = 0.0f64;
        for t in 0..obj.n {
            let cur = pitches[t];
            let (lo, hi, step) = if global {
                (-90.0, 90.0, opts.global_step)
            } else {
                ((cur - opts.local_span).max(-90.0), (cur + opts.local_span).min(90.0), opts.local_step)
            };
            let row: Vec<State> = states.iter().map(|st| st[t].clone()).collect();
            // The neighbors as they stand right now, so the price this tick pays reflects the
            // ticks already moved in this sweep -- Gauss-Seidel on the regularized objective,
            // not on `J` with a correction bolted on afterwards.
            let w = win5(&pitches, t);
            let pen = |x: f64| opts.rough.local(&w, x);
            // A tick with no admissible move still has to advance its own prefix: the tick
            // before it may have moved this pass, which makes every state downstream stale, and
            // the rest of the sweep would then optimize against a trajectory nobody is flying.
            let Some((lo, hi)) = opts.rough.feasible(&w, t, lo, hi) else {
                for state in &mut states {
                    state[t + 1] = ticked_cached(&state[t], trig[t])
                }
                continue
            };
            let (np, j, j_cur) = best_pitch_at(&obj, &row, &trig[t + 1..], cur,
                                               lo, hi, step, opts.pitch_tol, &pen, &floor);
            if global { worst_tick = worst_tick.max(j - j_cur) }
            pitches[t] = np;
            trig[t] = PitchTrig::new(np as f32);
            // every draw's prefix must stay consistent with the pitch just changed, or the
            // next tick's line search optimizes against a stale state and the schedule diverges
            for state in &mut states {
                state[t + 1] = ticked_cached(&state[t], trig[t]);
            }
        }
        // The regularizer: if this pass pushed the schedule into chatter, throw the pass away
        // and stop. Discarding rather than merely stopping matters -- the crossing pass is the
        // one that did the damage, and keeping it would write out exactly what we are avoiding.
        // Only a pass that makes it *worse* stops the polish. A warm start can arrive already
        // below the floor -- `stretch` leaves a seam at the join, which reads as chatter -- and
        // an absolute test would then revert every pass and freeze the schedule, propagating an
        // unimproved copy down the whole continuation chain. Measured: three consecutive cells
        // with identical lag-1 and falling dy, each writing out its stretched parent untouched.
        let l1_now = lag1(&pitches);
        if l1_now < opts.lag1_floor && l1_now < lag1(&prev) {
            pitches = prev;
            stopped_degenerate = true;
            break;
        }
        passes = pass + 1;
        last_gain = eval_floored(&obj, &pitches, &dv, &floor) - opts.rough.cost(&pitches) - before;
        let _ = worst_tick;
        recent.push_back(last_gain);
        if recent.len() > opts.stall_window { recent.pop_front(); }
        // Stop when the last `stall_window` passes together earned less than `tol`. Judging on
        // one pass would stop in the lulls between the jumps that the global passes find.
        // In steady mode `v0` must also have settled: `J` can go quiet for a window while the
        // fixed point is still walking, and stopping there writes a header `v0` the schedule
        // does not actually reproduce.
        let stalled = recent.len() == opts.stall_window && recent.iter().sum::<f64>() < opts.tol;
        // `v0` has to have stopped drifting as well as `J` having stopped gaining. See
        // `STEADY_DRIFT_TOL` for why this is summed over the window and why it is absolute.
        let v0_settled = !opts.steady
            || (drift.len() == opts.stall_window
                && drift.iter().sum::<f64>() < STEADY_DRIFT_TOL);
        if stalled && v0_settled { break }

        // Only now, with the pass scored and the stall test read, does the objective move. See
        // `PolishOpts::steady`.
        if opts.steady {
            let next = steady_vel(&pitches, obj.v0);
            dv0 = dv_l1(next, obj.v0);
            if dv0 > 0.0 {
                obj.v0 = next;
                states = jittered_replays(&obj, &trig, &dv);   // every prefix starts at v0
            }
            drift.push_back(dv0);
            if drift.len() > opts.stall_window { drift.pop_front(); }
        }
    }

    let j = obj.eval(&pitches);
    let residual = certify_floored(&obj, &pitches, CERTIFY_STEP, opts.jitter, opts.rough, floor);
    let l1 = lag1(&pitches);
    let rough_cost = opts.rough.cost(&pitches);
    Polished { pitches, j, passes, last_gain, residual, lag1: l1, stopped_degenerate, rough_cost,
               v0: obj.v0, dv0 }
}

// ---------------------------------------------------------------- the profile file

/// A solved cell, as it lives on disk.
///
/// The invariant the format is built around: the header states everything that determines the
/// optimum -- objective, initial conditions, horizon, physics -- so the claim "these pitches
/// are optimal" is checkable by `certify` from the file alone. How the schedule was reached is
/// deliberately absent, because with warm-start continuation the initialization is a path
/// through the grid rather than a compact set of hyperparameters, and it does not matter.
#[derive(Clone, Debug)]
pub struct Profile {
    pub obj: Objective,
    pub trig: TrigMode,
    /// The movement implementation used to optimize and certify this schedule.
    pub flight: FlightMode,
    /// The jitter the schedule was optimized against. Part of the utility function, so a
    /// profile optimized at sigma = 1 is a different object from one optimized at sigma = 0,
    /// and `verify --optimality` re-certifies against whatever the header says.
    pub jitter: Jitter,
    /// The roughness price the schedule was optimized under. Part of the utility function in
    /// exactly the way `jitter` is: a profile written at `mu = 0.002` is a coordinate optimum
    /// of a different objective than one written at `mu = 0`, and `verify --optimality` has to know which.
    pub rough: Rough,
    pub commit: String,
    pub pitches: Vec<f64>,
    /// The residual a full global pass could still find, in blocks of `J`.
    pub residual: f64,
    pub passes: usize,
}

impl Profile {
    pub fn to_string(&self) -> String {
        let o = &self.obj;
        let st = o.replay(&self.pitches);
        let (s0, sn) = (&st[0], st.last().unwrap());
        let mut out = String::new();
        let mut w = |l: &str| { out.push_str(l); out.push('\n') };
        w("# elytrasim optimal pitch schedule");
        w("# objective   maximize J = TE(s_n) + w*z_n; TE in blocks (KE = |v|^2/(2g), PE = y); v_n free");
        w(&format!("# lambda      {}", self.obj.lambda));
        w(&format!("# w           {:.10}          # lambda * Y_REF/Z_REF, Y_REF = {Y_REF}, Z_REF = {Z_REF}", o.w()));
        w(&format!("# v0          {:.9} {:.9}    # vy vz", o.v0.y, o.v0.z));
        w(&format!("# n           {}", o.n));
        w(&format!("# trig        {}", self.trig));
        w(&format!("# flight      {}", self.flight));
        if self.jitter.is_on() {
            w(&format!("# jitter      {} {} {}    # v0 sigma in blocks/tick, draws, seed",
                       self.jitter.sigma, self.jitter.draws, self.jitter.seed));
        } else {
            w("# jitter      0                     # optimized from the exact starting velocity");
        }
        if self.rough.is_on() {
            w(&format!("# rough       {} {} {} {} {}    # mu (per deg/tick^2), mu_tv (per deg/tick), \
cap, slew_cap, |pitch| limit",
                       self.rough.mu, self.rough.mu_tv, self.rough.cap, self.rough.slew_cap,
                       self.rough.limit));
        } else {
            w("# rough       0 0 inf inf 90         # no price on hand movement, no pitch margin");
        }
        if self.rough.shape != PriceShape::L1 {
            w(&format!("# penalty     {}                # shape of the mu price per second difference; l1 if absent", self.rough.shape));
        }
        if let Some(t) = self.rough.flick_at {
            w(&format!("# flick       {t} {}              # first tick at or below this pitch",
                       self.rough.flick_pitch));
        }
        w(&format!("# commit      {}", self.commit));
        w(&format!("# dJ          {:.6}              # J(s_n) - J(s_0)", o.j(sn) - o.j(s0)));
        w(&format!("# dte         {:.6}              # TE(s_n) - TE(s_0), blocks", sn.total_energy() - s0.total_energy()));
        w(&format!("# dy          {:.6}", sn.pos.y - s0.pos.y));
        w(&format!("# dz          {:.6}", sn.pos.z - s0.pos.z));
        w(&format!("# v_end       {:.9} {:.9}", sn.vel.y, sn.vel.z));
        // Derived from the pitches, like dy and dz, and recorded for the same reason: the first
        // thing any analysis does is separate the pump cycles from the glides, and needing a
        // forward pass to do it is friction. See opt::shape.
        let sh = shape(&self.pitches);
        w(&format!("# structure   {}", sh.structure));
        w(&format!("# cycles      {}                     # dive-then-flick transitions; the \
corpus sweeps one cycle, so more than one is degenerate", sh.cycles));
        // How much hand movement the schedule asks for. Chatter shows up here and nowhere else
        // in the header: the reference cycle sits near 248 degrees, a hard-polished one at 3000+.
        w(&format!("# variation   {:.1}                 # summed |pitch change|, deg",
                   total_variation(&self.pitches)));
        w(&format!("# lag1        {:+.3}                # lag-1 correlation of the per-tick changes; below ~0.2 is chatter", lag1(&self.pitches)));
        // The curvature statistics: what the schedule asks of the wrist. `curv_l1` is what a
        // roughness price is levied on, `curv_max` is the peak angular acceleration -- 37.9 on
        // the reference cycle and 180, the geometric maximum, on a chattering one.
        w(&format!("# curv_l1     {:.1}                 # summed |second difference|, deg/tick^2",
                   curvature_l1(&self.pitches)));
        w(&format!("# curv_max    {:.1}                 # peak |second difference|, deg/tick^2",
                   curvature_max(&self.pitches)));
        w(&format!("# certified   full global pass at {:.2}deg, exact tail eval, improves J by {:.2e} ({} passes)",
                   CERTIFY_STEP, self.residual, self.passes));
        for p in &self.pitches { w(&format!("{p}")) }
        out
    }

    /// Read a profile back. The header is authoritative: a file replayed under the physics or
    /// weight it was *not* optimized for is not the same object, so `verify` reads the mode
    /// from the file rather than the command line.
    pub fn parse(text: &str) -> Result<Profile, String> {
        let field = |k: &str| -> Option<String> {
            text.lines()
                .find_map(|l| l.strip_prefix("# ")?.strip_prefix(k)
                    .map(|v| v.split('#').next().unwrap_or("").trim().to_string()))
        };
        let need = |k: &str| field(k).ok_or_else(|| format!("missing header field '{k}'"));
        let num = |k: &str| -> Result<f64, String> {
            need(k)?.parse().map_err(|e| format!("bad '{k}': {e}"))
        };
        let v0 = need("v0")?;
        let mut vs = v0.split_whitespace();
        let mut nextf = |what: &str| -> Result<f64, String> {
            vs.next().ok_or_else(|| format!("v0 needs two numbers ({what})"))?
              .parse().map_err(|e| format!("bad v0: {e}"))
        };
        let (vy, vz) = (nextf("vy")?, nextf("vz")?);
        let pitches = text.lines()
            .flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace())
            .map(|s| s.parse::<f64>().map_err(|e| format!("bad pitch {s:?}: {e}")))
            .collect::<Result<Vec<f64>, _>>()?;
        let n = num("n")? as usize;
        if pitches.len() != n {
            return Err(format!("header says n = {n} but the file holds {} pitches", pitches.len()));
        }
        Ok(Profile {
            obj: Objective { v0: Vec3::new(0.0, vy, vz), n, lambda: num("lambda")? },
            trig: need("trig")?.parse()?,
            // Profiles predating the algebraic route used the reference implementation.
            flight: field("flight").map(|v| v.parse()).transpose()?.unwrap_or_default(),
            jitter: match field("jitter") {
                None => Jitter::default(),
                Some(v) => {
                    let f: Vec<&str> = v.split_whitespace().collect();
                    let g = |i: usize| f.get(i).and_then(|x| x.parse().ok());
                    Jitter {
                        sigma: g(0).unwrap_or(0.0),
                        draws: g(1).unwrap_or(1.0) as usize,
                        resample: Jitter::default().resample,
                        seed: f.get(2).and_then(|x| x.parse().ok())
                                .unwrap_or(Jitter::default().seed),
                    }
                }
            },
            // A header field is the whole claim, so a typo in it has to be an error and not a
            // silent default: `# rough 0 0 typo inf 85` used to parse as "no cap".
            rough: {
                let mut rough = match field("rough") {
                    None => Rough::default(),
                    Some(v) => {
                    let f: Vec<&str> = v.split_whitespace().collect();
                    let g = |i: usize, d: f64| -> Result<f64, String> {
                        match f.get(i) {
                            None => Ok(d),
                            Some(x) => x.parse().map_err(|e| format!("bad 'rough' field {i} {x:?}: {e}"))
                                        .and_then(|y: f64| if y.is_nan() {
                                            Err(format!("'rough' field {i} is NaN")) } else { Ok(y) }),
                        }
                    };
                    if f.len() > 5 { return Err(format!("'rough' takes at most 5 numbers, got {}", f.len())) }
                    Rough { mu: g(0, 0.0)?, mu_tv: g(1, 0.0)?,
                            cap: g(2, f64::INFINITY)?, slew_cap: g(3, f64::INFINITY)?,
                            limit: g(4, 90.0)?, ..Rough::default() }
                    }
                };
                if let Some(v) = field("penalty") {
                    rough.shape = PriceShape::parse(v.split('#').next().unwrap_or("").trim())?;
                }
                let flick: Vec<String> = text.lines().filter_map(|l| {
                    l.strip_prefix("# ")?.strip_prefix("flick")
                        .map(|v| v.split('#').next().unwrap_or("").trim().to_string())
                }).collect();
                if flick.len() > 1 {
                    return Err(format!("'flick' must appear at most once, got {} lines", flick.len()))
                }
                if let Some(v) = flick.first() {
                    let f: Vec<&str> = v.split_whitespace().collect();
                    if f.len() != 2 {
                        return Err(format!("'flick' needs exactly 2 fields, got {}", f.len()))
                    }
                    rough.flick_at = Some(f[0].parse()
                        .map_err(|e| format!("bad 'flick' tick {:?}: {e}", f[0]))?);
                    rough.flick_pitch = f[1].parse()
                        .map_err(|e| format!("bad 'flick' pitch {:?}: {e}", f[1]))?;
                    if !rough.flick_pitch.is_finite() {
                        return Err("'flick' pitch must be finite".into())
                    }
                }
                rough
            },
            commit: field("commit").unwrap_or_default(),
            pitches,
            residual: field("certified").and_then(|s| s.split("improves J by ").nth(1)
                .and_then(|r| r.split_whitespace().next()?.parse().ok())).unwrap_or(f64::NAN),
            passes: 0,
        })
    }
}

/// The git commit the binary was built from, for the header. Resolved at build time by
/// build.rs; "unknown" if the build had no git.
pub fn commit_hash() -> &'static str { env!("ELYTRASIM_COMMIT") }

// ---------------------------------------------------------------- seeding a cell

/// The best-scoring constants for each dive rule, as tuned by `myopic policy <rule> --opt`
/// against the 300-tick limit cycle. `Leak` is the strongest of the four (95.9% of the optimal
/// cycle) and so is the default seed.
///
/// Note what these are and are not. They fly a *limit cycle*: the phase switches are on state,
/// not on the clock, so the policy is horizon-independent -- but that is not the same as being
/// horizon-*appropriate*. At `n = 100` the horizon is shorter than one cycle, so the optimum is
/// not a truncated cycle at all and there is no reason these thresholds should seed it well.
/// Continuation down the `n` axis is the answer; `sweep pilot` measures whether it is needed.
pub fn tuned_policy(dive: Dive, n_gain: usize) -> P {
    match dive {
        Dive::Leak => P { g_star: 17.73, k: 0.055, s_switch: 2.40, vy_flick: -0.260,
                          s_exit: 0.21, slew: 12.7, p_push: 23.0, p_flick: -88.5, n_gain, dive },
        Dive::Floor => P { g_star: f64::NAN, k: f64::NAN, s_switch: 2.177, vy_flick: -0.2685,
                           s_exit: 0.45, slew: 12.92, p_push: 24.01, p_flick: -79.44, n_gain, dive },
        Dive::Hold => P { g_star: f64::NAN, k: f64::NAN, s_switch: 2.127, vy_flick: -0.260,
                          s_exit: 0.29, slew: 8.34, p_push: 47.0, p_flick: -41.51, n_gain, dive },
        Dive::Target => P { g_star: f64::NAN, k: f64::NAN, s_switch: 2.177, vy_flick: -0.2685,
                            s_exit: 0.45, slew: 12.92, p_push: 24.01, p_flick: -79.44, n_gain, dive },
    }
}

/// A cold start for a cell: fly the tuned policy from this cell's `v0` for `n` ticks.
pub fn seed_from_policy(obj: &Objective) -> Vec<f64> {
    fly_from(obj.v0, tuned_policy(Dive::Leak, 20), obj.n, &[]).0
}

/// A cold start from the reference cycle, tiled and cut to length. The cycle is ~300 ticks, so
/// for `n > 300` this repeats it; phases have absolute durations, not fractional ones, so the
/// schedule is extended rather than rescaled.
pub fn seed_from_reference(n: usize) -> Vec<f64> {
    let r = crate::replay_pitches::REPLAY_PITCHES_300;
    (0..n).map(|i| r[i % r.len()] as f64).collect()
}

/// Resample a schedule to length `n`, linearly: a cycle of period 300 becomes a cycle of
/// period `n`.
///
/// The third seed for continuation, and the one that covers the gap tiling leaves. Tiling only
/// lands well at integer multiples of the cycle -- at n = 400 a tiled 300-tick cycle ends 100
/// ticks into the next one, mid-dive, which is where the energy goes. Rescaling instead asks
/// for a single cycle that happens to take n ticks.
///
/// Phases do have absolute durations, so this is not physically exact for large ratios; it is a
/// seed, and it competes with the others on measured `J` rather than on principle.
pub fn rescale(pitches: &[f64], n: usize) -> Vec<f64> {
    if pitches.is_empty() { return vec![0.0; n] }
    if pitches.len() == 1 { return vec![pitches[0]; n] }
    let m = pitches.len();
    (0..n).map(|i| {
        let x = i as f64 * (m - 1) as f64 / (n.max(2) - 1) as f64;
        let (lo, f) = (x.floor() as usize, x - x.floor());
        let hi = (lo + 1).min(m - 1);
        ((1.0 - f) * pitches[lo] + f * pitches[hi]) as f32 as f64
    }).collect()
}

/// Adapt a solved neighbor's schedule to a different horizon, for continuation along `n`.
///
/// This is `rescale`: a cycle of period `m` becomes a cycle of period `n`. The corpus sweeps
/// **one** cycle across `num_ticks`, so a 500-tick cell must be a single cycle that takes 500
/// ticks -- not the 300-tick cycle flown once and two-thirds, and not two 250s, which is
/// something you can tile post hoc from two cells of this corpus if you want it.
///
/// Tiling was tried and is wrong, however good it looks. Seeding n = 600 with the 300-tick
/// cycle twice scores dJ 41.08 against 19.47 at n = 300 -- but that is a two-cycle schedule,
/// degenerate here in the same way the constant-pitch-up glide is: a different regime that
/// wins on J while not being the object under study. Total energy *should* fall away from
/// n = 300, because 300 is the optimal length for a single cycle.
///
/// Phases have absolute durations, so a uniform time rescale is not physically exact for large
/// ratios. It is a seed, and the polish moves it; what matters is that it stays in the
/// single-cycle family.
pub fn stretch(pitches: &[f64], n: usize) -> Vec<f64> {
    if pitches.len() == n { return pitches.to_vec() }
    rescale(pitches, n)
}

// ---------------------------------------------------------------- determinism

/// Local averaging over `k` ticks: the projection a chattering schedule needs before it can be
/// polished under a roughness price.
///
/// A chattering control is the discrete stand-in for a *relaxed* control -- at each tick the
/// optimizer is really choosing a distribution over pitches, and it realizes the mixture by
/// alternating. The ordinary control that means the same thing is the local mean, which is what
/// this computes. Without it, coordinate ascent under an l1 curvature price can stall: the price
/// couples three neighboring coordinates, so it is a fused-l1 term, and coordinate descent is
/// not guaranteed to reach a stationary point of a nonsmooth objective that couples coordinates.
/// Measured: seeded from a schedule alternating -90/+90 it converged (residual 2.2e-7) to
/// `curv_l1` 1295, against 162 from the same price seeded from the local mean.
///
/// `k <= 1` is the identity. The window is clamped at the ends rather than wrapped, which means
/// the ends are *not* faithful: a ramp comes out flattened there (0,1,2,3,... at width 5 becomes
/// 0.4, 0.8, 2, 3, ...). That matters twice -- it is why the entry and the tail of a projected
/// schedule need the polish to repair them, and it is why widening this filter does not converge
/// to a constant, so `project_cap` cannot use "wide enough is constant" as a termination
/// argument.
///
/// The width matters and is not free: a box filter smears the snap and the flick along with the
/// chatter, and from a destroyed cycle the price-constrained polish cannot rebuild one. Sweep it.
/// Projecting one relaxed optimum (J 22.346) at k = 3, 5, 9, 15, 25 gives 22.165, 22.161, 22.156,
/// 22.130, 22.047 -- all fine; projecting a *more violently* chattering one (J 22.402) at k = 9
/// lands at 15.66, converged (residual 8.9e-6) and 6.5 blocks worse. So the projection width is a
/// hyperparameter to sweep and score on the regularized objective, not a constant to pick once.
pub fn smooth_box(p: &[f64], k: usize) -> Vec<f64> {
    if k <= 1 { return p.to_vec() }
    let n = p.len() as isize;
    let h = (k / 2) as isize;
    (0..n).map(|i| {
        let (mut s, mut c) = (0.0, 0.0);
        for d in -h..=h { s += p[(i + d).clamp(0, n - 1) as usize]; c += 1.0 }
        (s / c) as f32 as f64
    }).collect()
}

/// Median over a window of `k` ticks. Available, and **not** the right projection here -- kept
/// because the reason is worth writing down.
///
/// A median filter looks like the obvious choice: it passes a step through untouched while
/// removing short runs, which is exactly the shape of "a piecewise-smooth control with a few
/// genuine corners plus noise on top". But it removes runs shorter than *half the window*, and a
/// tick-rate alternation has runs of length one with fifty percent duty -- so every window holds
/// a majority of whichever value sits at its center, and the median reproduces the alternation
/// instead of removing it. Chatter is not impulsive noise; it is a square wave at Nyquist.
///
/// Measured, projecting the same relaxed optimum and then polishing at `mu = 1e-3`: median 3, 5,
/// 7, 9 give J 15.32, 15.69, 15.92, 16.01, against 22.16 for the local mean. Only at width 13,
/// where the chatter is irregular enough for the majority to break, does it recover (22.14).
///
/// `k` is forced odd and `k <= 1` is the identity.
pub fn smooth_median(p: &[f64], k: usize) -> Vec<f64> {
    if k <= 1 { return p.to_vec() }
    let k = k | 1;
    let n = p.len() as isize;
    let h = (k / 2) as isize;
    (0..n).map(|i| {
        let mut w: Vec<f64> = (-h..=h).map(|d| p[(i + d).clamp(0, n - 1) as usize]).collect();
        w.sort_by(f64::total_cmp);
        w[w.len() / 2]
    }).collect()
}

/// Bring a schedule inside the admissible set -- the per-tick pitch interval, `slew_cap` and
/// `cap` -- by integrating a rate-limited tracker through it from the left.
///
/// A coordinate sweep cannot do this on its own. `Rough::feasible` asks what one pitch may be
/// given its neighbors, and when the neighbors are themselves out of bounds the answer is
/// nothing at all -- every tick returns `None` and the polish is a no-op. So a seed that breaks
/// the cap has to be projected before the sweep starts, exactly as one that breaks `limit` is.
///
/// The core of it is the causal projection: walk forward, and wherever the schedule asks for
/// more than `cap` of angular acceleration, give it `cap`. That is what a hand with a bounded
/// acceleration would produce trying to track the target.
///
/// On its own it is not enough, and the failure is immediate rather than exotic. A tracker
/// following a violent seed accumulates rate, runs into `+-limit`, and the clamp *is* a large
/// second difference -- so the pass ends with a violation where it hit the wall. Alternating
/// -60/+60 with `cap = 25` and `limit = 90` comes out with a worst of 90, unchanged in kind from
/// the 240 it started with.
///
/// So: pre-smooth first, widening the window until the causal pass comes out clean. This
/// terminates, because at a window as wide as the schedule the mean is a constant and every
/// second difference is zero. The width it settles on is a fair measure of how far outside the
/// admissible set the seed was.
pub fn project_cap(p: &[f64], rough: Rough) -> Vec<f64> {
    // Quantize as we go. The projection's output is a control that will be flown, so it has to be
    // representable *and* inside the set -- clamping in f64 and rounding afterwards can land back
    // outside, which is the whole point of `f32_inside`.
    let empty = [None; 5];
    if let Some(t) = rough.flick_at {
        assert!(t < p.len(), "cannot project seed: flick tick {t} is outside a {}-tick schedule", p.len());
    }
    let bounds: Vec<(f64, f64)> = (0..p.len()).map(|t| {
        let (lo, hi) = rough.feasible(&empty, t, -90.0, 90.0)
            .unwrap_or_else(|| panic!("cannot project seed: admissible interval is empty at tick {t}"));
        if f32_inside(p[t], lo, hi).is_none() {
            panic!("cannot project seed: admissible interval contains no representable pitch at tick {t}")
        }
        (lo, hi)
    }).collect();
    let q32 = |t: usize, x: f64| {
        let (lo, hi) = bounds[t];
        f32_inside(x, lo, hi)
            .unwrap_or_else(|| panic!("cannot project seed: no representable pitch at tick {t}"))
    };
    let causal = |src: &[f64]| -> Vec<f64> {
        let mut q: Vec<f64> = src.iter().enumerate().map(|(t, &x)| q32(t, x)).collect();
        // Rate first, then acceleration: a schedule inside the slew limit is a milder input to
        // the curvature pass, and the curvature pass cannot make the rate worse than 2*cap.
        if rough.slew_cap.is_finite() {
            for j in 0..q.len().saturating_sub(1) {
                let d = q[j + 1] - q[j];
                if d.abs() > rough.slew_cap {
                    q[j + 1] = q32(j + 1, q[j] + d.clamp(-rough.slew_cap, rough.slew_cap));
                }
            }
        }
        if rough.cap.is_finite() {
            for j in 0..q.len().saturating_sub(2) {
                let d = q[j + 2] - 2.0 * q[j + 1] + q[j];
                if d.abs() > rough.cap {
                    q[j + 2] = q32(j + 2, 2.0 * q[j + 1] - q[j]
                                                   + d.clamp(-rough.cap, rough.cap));
                }
            }
        }
        q
    };
    if !rough.cap.is_finite() && !rough.slew_cap.is_finite() { return causal(p) }   // already q32
    // `violation` is the termination test, so this cannot claim success on a constraint it
    // forgot to enforce -- which is exactly how the first version of this function was wrong.
    let mut k = 1;
    while k <= p.len() + 1 {
        let q = causal(&smooth_box(p, k));
        if rough.violation(&q) == 0.0 { return q }
        k += 2;
    }
    // A wide `smooth_box` is *not* a constant: it clamps at the ends rather than wrapping, so it
    // keeps a slope, and the widening loop can run out while still failing the cap. Fall back to
    // the flattest thing there is, whose differences are all exactly zero. A timed flick makes a
    // constant unavailable; clamp that constant into each tick's box and reject it loudly if the
    // resulting required crossing cannot satisfy a coupled cap. It throws the schedule away,
    // which is why it is a fallback and not the method.
    let m = p.iter().sum::<f64>() / p.len().max(1) as f64;
    let q: Vec<f64> = (0..p.len()).map(|t| q32(t, m)).collect();
    if rough.violation(&q) == 0.0 { return q }
    let t = first_violation_tick(&q, rough).unwrap_or(rough.flick_at.unwrap_or(0));
    panic!("cannot project seed into the admissible set at tick {t}")
}

fn first_violation_tick(p: &[f64], rough: Rough) -> Option<usize> {
    if let Some(flick) = rough.flick_at {
        if flick >= p.len() || !rough.flick_pitch.is_finite() { return Some(flick) }
    }
    for (t, &x) in p.iter().enumerate() {
        if x.abs() > rough.limit { return Some(t) }
        if let Some(flick) = rough.flick_at {
            if (t < flick && x <= rough.flick_pitch) || (t == flick && x > rough.flick_pitch) {
                return Some(t)
            }
        }
    }
    if rough.slew_cap.is_finite() {
        for j in 0..p.len().saturating_sub(1) {
            if (p[j + 1] - p[j]).abs() > rough.slew_cap { return Some(j + 1) }
        }
    }
    if rough.cap.is_finite() {
        for j in 0..p.len().saturating_sub(2) {
            if (p[j] - 2.0 * p[j + 1] + p[j + 2]).abs() > rough.cap { return Some(j + 2) }
        }
    }
    None
}

/// FNV-1a over the raw bits of a canonical replay, for checking that two machines agree.
///
/// The `Mth` table removes the f32 trig divergence between platforms, but `lift_force` still
/// goes through f64 `cos` in `entity.rs` -- faithfully, because vanilla's `liftForce` really is
/// `Math.cos` on a double -- so bit-identity across libms is *not* guaranteed. This makes any
/// drift a number that shows up in a diff rather than a silent change in the corpus.
pub fn physics_fingerprint() -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut feed = |b: u64| { h ^= b; h = h.wrapping_mul(0x100000001b3) };
    let mut v = Vec3::new(0.0, 0.167467, 0.200887);
    for i in 0..20_000i32 {
        // sweep the whole pitch range, and restart when the state runs away so the walk stays
        // in the region the sweep actually visits
        let p = ((i % 361) - 180) as f64 * 0.5;
        v = update_fall_flying_movement(v, rot(p));
        if !v.y.is_finite() || v.y.abs() > 1e6 { v = Vec3::new(0.0, 0.167467, 0.200887) }
        feed(v.y.to_bits());
        feed(v.z.to_bits());
    }
    h
}

// ---------------------------------------------------------------- structure

/// Whether a schedule still flies the pump cycle, or has fallen into the glide.
///
/// This is *not* the same question as whether it gains energy, and conflating the two is a
/// mistake: at short horizons a genuine cycle can lose height and still be a cycle. A 120-tick
/// optimum loses 10.5 blocks while showing the full dive / snap / flick / gain structure, and
/// is not collapsed. What collapse means is that the schedule stops pumping and just holds a
/// glide -- pitch stays in a narrow band a few degrees below level, never noses down to build
/// speed and never flicks up to cash it in.
///
/// The two regimes are separated by a mile, so the thresholds are not delicate: measured
/// against known-collapsed and known-uncollapsed 110-, 115- and 120-tick optima, the cyclic
/// ones span roughly [-55, +90] degrees and the collapsed ones [-15.3, +0.10].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    /// Dives to build speed, snaps flat, flicks up, and eases down through the gain. One cycle.
    Cyclic,
    /// Holds a glide. No nose-down dive and no flick.
    Collapsed,
    /// Two or more cycles inside the horizon.
    ///
    /// Degenerate for this corpus in the same way the collapsed glide is: a different regime
    /// that happens to score better, not the object being swept. 300 ticks is the optimal
    /// length for *one* cycle, so total energy falling away from 300 is the answer, not a
    /// defect -- and a 500-tick cell must be a single cycle that takes 500 ticks. Two 250s
    /// tiled is something you can build post hoc from two cells of this corpus.
    MultiCycle,
}

/// A schedule noses down past this to build speed. The collapsed glide never exceeds +0.11.
const DIVE_PITCH: f64 = 20.0;
/// And flicks up past this to convert it. The collapsed glide bottoms out near -15.3.
const FLICK_PITCH: f64 = -30.0;

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub structure: Structure,
    pub pitch_min: f64,
    pub pitch_max: f64,
    /// Ticks spent within 1 degree of level after the dive -- the snap.
    pub flat_ticks: usize,
    /// Dive-then-flick transitions. One is the object being swept; more is degenerate.
    pub cycles: usize,
}

/// Total variation of the schedule: summed absolute pitch change, in degrees. How much hand
/// movement the whole flight asks for, and the direct measure of chatter -- a hard-polished
/// 300-tick schedule reaches 3243 degrees against the reference cycle's 35.
pub fn total_variation(pitches: &[f64]) -> f64 {
    pitches.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// Lag-1 correlation of the per-tick pitch changes: the degeneracy statistic.
///
/// Total variation cannot tell a flick from chatter -- both are large moves. What separates
/// them is whether neighboring ticks move *together*. A real manoeuvre is a run of same-signed
/// steps, so the deltas correlate positively; chatter alternates, so they correlate negatively.
/// Measured on the reference cycle this reads +0.48, on the same schedule polished 200 passes
/// -0.74, and the sign flip is the whole diagnosis.
pub fn lag1(pitches: &[f64]) -> f64 {
    if pitches.len() < 3 { return 0.0 }
    let d: Vec<f64> = pitches.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    let den: f64 = d.iter().map(|x| (x - mean).powi(2)).sum();
    if den <= 0.0 { return 1.0 }        // constant or perfectly linear: maximally smooth
    d.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum::<f64>() / den
}

/// How many dive-then-flick transitions the schedule makes: the number of cycles.
///
/// Scanned with hysteresis, so the chatter in a degenerate schedule -- which crosses both
/// thresholds repeatedly within a few ticks -- does not read as extra cycles. A cycle is
/// counted only when the schedule has been committed to the dive (above `DIVE_PITCH`) and then
/// commits to the climb (below `FLICK_PITCH`); it must return to the dive side before another
/// can be counted.
pub fn cycles(pitches: &[f64]) -> usize {
    let (mut n, mut armed) = (0usize, false);
    for &p in pitches {
        if p > DIVE_PITCH { armed = true }
        else if armed && p < FLICK_PITCH { n += 1; armed = false }
    }
    n
}

pub fn shape(pitches: &[f64]) -> Shape {
    let pitch_max = pitches.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pitch_min = pitches.iter().cloned().fold(f64::INFINITY, f64::min);
    let flat_ticks = pitches.iter().filter(|p| p.abs() <= 1.0).count();
    let n_cycles = cycles(pitches);
    let structure = if !(pitch_max > DIVE_PITCH && pitch_min < FLICK_PITCH) {
        Structure::Collapsed
    } else if n_cycles >= 2 {
        Structure::MultiCycle
    } else {
        Structure::Cyclic
    };
    Shape { structure, pitch_min, pitch_max, flat_ticks, cycles: n_cycles }
}

impl std::fmt::Display for Structure {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Structure::Cyclic => "cyclic",
            Structure::Collapsed => "COLLAPSED",
            Structure::MultiCycle => "MULTICYCLE",
        })
    }
}
