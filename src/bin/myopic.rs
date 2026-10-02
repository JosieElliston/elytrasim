//! Which *myopic* metrics does the globally optimal climb cycle agree with, phase by phase?
//!
//! The optimal cycle splits into five phases. Three of them follow a rule with no fitted constant
//! in it and the gain phase's has one measured constant, `K` at w = 0; the entry has no rule, and
//! is the open problem:
//!
//! | phase | ticks | rule                                                  |
//! |-------|-------|-------------------------------------------------------|
//! | entry | ~40   | **no rule known** -- pitches hard down, unexplained    |
//! | dive  | ~150  | hold the flight-path angle exactly; no constants       |
//! | snap  | ~14   | pitch 0, held until one more tick stops raising v_z   |
//! | flick | ~6    | ramp to about -88 deg; the values do not matter       |
//! | gain  | ~86   | s(s + A) = K v_z, s = sin|pitch|; one constant per w   |
//!
//! The dive was long described as leaking toward ~16.6 deg at a rate `k`. That was an artifact
//! of asking one rule to cover the entry as well: hand the first ~60 ticks to the optimum and
//! the exact hold beats every leaking variant (97.9% of the cycle). `k`, `g_star` and the floor
//! clamp are all entry corrections, kept here only because they still fly the whole cycle
//! open-loop. See "The leak was an entry correction" in README-myopic.md.
//!
//! The gain phase's lookahead is gone too. On the climbing arc the price of upward velocity is a
//! pure clock at `DRAG_Y` and the stationary pitch has a closed form in `v_z` and the price ratio
//! alone -- `v_y` does not enter -- so the phase is its own optimal-control problem, from here to
//! the apex, with no horizon constant. What it cannot derive is `kappa = mu_z(apex)`, what the
//! dive will pay for the forward speed it is handed; `gain` measures all of it and
//! `tools/gain_phase.py` runs it over a corpus. That climb-to-apex rule is still a search, though.
//! What you can fly without one is `gain_law_pitch`: through the body of the climb
//! `sin|p| (sin|p| + A) = K v_z`, `A` read off the tick map and `K` measured
//! per objective: 0.771 for the climb, 0.670 for the level cycle. The climb's end is a clock to the entry and is not in it.
//! `gainlaw` measures it; `tools/gainlaw_refs.sh` builds the horizon-free cycles it needs.
//!
//! The snap's *pitch* was never in doubt; when to start and stop holding it was. `vz_peaked` is
//! the myopic answer for the stop and is exact at lambda = 0. The exact answer at both ends and
//! every price is a pair of corner conditions on the costate, with every constant read off the
//! tick map: hold 0 rather than pitch up while `0.036*DRAG_Z*mu_z >= 0.128*DRAG_Y*mu_y` (the
//! elytra's own forward-to-up exchange rate, 3.519641), and rather than pitch down while
//! `DRAG_Y*(0.056 - 0.1*v_y)*mu_y >= 0.09*DRAG_Z*(v_y + 0.04)*mu_z`. Parameter-free, but `mu`
//! needs the whole future. `adjoint` solves it and `tools/glide_phase.py --switch` checks both:
//! inside one tick on every one of the 1233 steady cycles.
//!
//! Everything here is measured against `sim`'s physics with yaw pinned to zero, so the whole
//! problem lives in the (v_y, v_z) plane. See README-myopic.md for the numbers.
//!
//! `myopic --help` lists the subcommands and `myopic <subcommand> --help` their arguments.
//!
//! `--trig libm|mth_lut` picks the trig implementation for any subcommand. `mth_lut` is
//! Minecraft's own 65536-entry sine table; `libm` (the default) is the platform's, which is
//! what every number in README-myopic.md was measured with.
//! `--flight reference|algebraic` selects the direct port or the faster yaw-zero equations.
//!
//! A schedule file is whitespace-separated pitches in degrees. `<off>` is the tick offset of
//! the cycle to read, so a 3x-tiled 900-tick flight is read horizon-free at offset 300.

use clap::{Parser, Subcommand};
use elytrasim::opt::*;
use elytrasim::sim::*;
use rayon::prelude::*;

/// Which myopic metrics the globally optimal climb cycle agrees with, phase by phase.
///
/// A schedule file is whitespace-separated pitches in degrees. `<off>` is the tick offset of the
/// cycle to read, so a 3x-tiled 900-tick flight is read horizon-free at offset 300.
#[derive(Parser)]
struct Cli {
    /// Trig route, anywhere on the line: `libm` (the default) or `mth_lut`.
    #[arg(long, global = true)]
    trig: Option<TrigMode>,
    /// Flight kernel, anywhere on the line: `reference` or `algebraic`.
    #[arg(long, global = true)]
    flight: Option<FlightMode>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Cycle closure and climb rate of the built-in profiles.
    Profiles,
    /// Terminal-glide table, and the fastest steady glide.
    Eq,
    /// Steady glide maximizing the objective rate, per w.
    Eqrate {
        /// Also print the equilibrium locus as CSV, and stop there.
        #[arg(long)]
        locus: bool,
    },
    /// Steady-glide curves vs pitch, as CSV.
    #[command(allow_negative_numbers = true)]
    Glide {
        #[arg(default_value_t = 0.05)]
        step: f64,
        #[arg(default_value_t = -90.0)]
        lo: f64,
        #[arg(default_value_t = 90.0)]
        hi: f64,
    },
    /// The three critical points of the steady glide, both sides.
    Crit,
    /// When forward speed can rise at all, and what a rise costs; see docs/rising.md.
    Rising {
        #[arg(default_value = "all", value_parser = ["frontier", "pitch", "clock", "tradeoff", "down", "all"])]
        part: String,
    },
    /// Coordinate-ascent polish of a schedule; maximizes TE + w*z.
    #[command(allow_negative_numbers = true)]
    Polish {
        path: String,
        #[arg(default_value_t = 40)]
        passes: usize,
        #[arg(default_value_t = 0.0)]
        w: f64,
    },
    /// Per-tick dump: pitch, gamma, and each rule's answer.
    Cycle { path: String, off: usize },
    /// RMS pitch error of a menu of rules, per phase.
    Score { path: String, off: usize, lo: usize, hi: usize },
    /// Implied lookahead n*(t) through the gain phase.
    Probe { path: String, off: usize, lo: usize, hi: usize },
    /// Auto-detect the phases and summarize the fit.
    Family { path: String, tag: String },
    /// Fit the dive's first-order gamma decay and its asymptote.
    Floor { path: String, tag: String },
    /// Shadow prices from the optimum, and the glide they pick.
    Prices { path: String, tag: String },
    /// Flight-path angle at ten points through the dive.
    Gprofile { path: String, tag: String },
    /// The one-tick sensitivity of `sens`, one side at a time.
    #[command(allow_negative_numbers = true)]
    Sides { path: String, delta: f64 },
    /// Sweep the dive's leak rate `k`, with the other constants fixed and then re-tuned.
    Ksweep,
    /// Hand the first T ticks of every dive to the optimum, then let each rule take over.
    Prefix { path: String },
    /// What a pitch error at one tick costs the cycle, re-converged, in blocks per second.
    #[command(allow_negative_numbers = true)]
    Sens {
        path: String,
        delta: f64,
        /// Per-tick CSV.
        #[arg(long)]
        dump: bool,
    },
    /// One tick, spelled out: how next tick's velocity swings with pitch, and where the price sits.
    Swing { path: String, t: usize },
    /// Print one cycle of a schedule: the middle of a 3x tiling, or the policy's limit cycle.
    Cyclecut { path: String },
    /// Is the price vector's consistency special to the optimum? `amp` perturbs a control.
    #[command(allow_negative_numbers = true)]
    Consist { path: String, amp: f64 },
    /// How sharply the one-tick score picks out the optimum's pitch.
    Singular { path: String },
    /// The climbing arc: the two costate clocks, the closed-form stationary pitch, the countdown
    /// two pitches read out, and with `--bvp` the climb-to-apex rule's own pitches.
    #[command(allow_negative_numbers = true)]
    Gain {
        path: String,
        w: Option<f64>,
        #[arg(long)]
        bvp: bool,
        #[arg(long)]
        dump: bool,
    },
    /// The search-free gain law on a closed cycle: the constant read off the optimum's climb, and
    /// the law flown closed-loop against argmax dTE over 20 ticks.
    #[command(allow_negative_numbers = true)]
    Gainlaw {
        path: String,
        w: Option<f64>,
        /// Fly this K instead of the one read off the optimum.
        #[arg(long)]
        k: Option<f64>,
        /// Pitch limit, degrees.
        #[arg(long)]
        limit: Option<f64>,
        #[arg(long)]
        dump: bool,
    },
    /// Solve the periodic price vector and test the one-tick rule; a sweep profile supplies its
    /// own v0 and w.
    #[command(allow_negative_numbers = true)]
    Adjoint {
        path: String,
        w: Option<f64>,
        #[arg(long)]
        dump: bool,
    },
    /// Argmax pitch for every lookahead 1..N, per tick.
    Sweepn { path: String, off: usize, lo: usize, hi: usize, nmax: usize },
    /// The gain phase's lookahead rule at a price on distance: argmax `dJ = dTE + w*dz` against
    /// argmax `dTE`, both held `n` ticks, scored on the profile's own climb.
    #[command(allow_negative_numbers = true)]
    Djn {
        path: String,
        #[arg(default_value_t = 48)]
        nmax: usize,
        /// Override the profile's own w. A flag, not a positional: a bare number there reads as
        /// the lookahead.
        #[arg(long)]
        w: Option<f64>,
        #[arg(long)]
        limit: Option<f64>,
        #[arg(long)]
        dump: bool,
        #[arg(long)]
        wsweep: bool,
        #[arg(long)]
        equiv: bool,
    },
    /// Pointwise rules for an arbitrary initial velocity.
    #[command(allow_negative_numbers = true)]
    Rules { path: String, vy0: f64, vz0: f64, nmax: usize },
    /// Fly the four bugs.
    Policy {
        #[arg(default_value = "leak", value_parser = ["leak", "floor", "hold", "target"])]
        dive: String,
        /// Tune the shared constants first.
        #[arg(long)]
        opt: bool,
        /// Swap the tuned snap->flick threshold for the parameter-free "forward speed has
        /// peaked" rule.
        #[arg(long)]
        vzpeak: bool,
        /// The gain lookahead.
        #[arg(long, default_value_t = 20)]
        ngain: usize,
    },
}

// ---------------------------------------------------------------- subcommands

fn cmd_profiles() {
    use elytrasim::replay_pitches as rp;
    let named: [(&str, &[f32]); 5] = [
        ("FORTY_FORTY", rp::FORTY_FORTY), ("FOUR_LINES_300", rp::FOUR_LINES_300),
        ("REPLAY_PITCHES_200", rp::REPLAY_PITCHES_200), ("REPLAY_PITCHES_300", rp::REPLAY_PITCHES_300),
        ("REPLAY_PITCHES_400", rp::REPLAY_PITCHES_400),
    ];
    println!("{:<20} {:>6} {:>10} {:>9} {:>9} {:>12}", "profile", "ticks", "|v_N - v_0|", "climb b/s", "dist b/s", "closes?");
    for (name, ps) in named {
        let ps: Vec<f64> = ps.iter().map(|&p| p as f64).collect();
        if ps.is_empty() { continue }
        let st = replay(&ps);
        let end = st.last().unwrap();
        let dv = (end.vel - V0).length();
        let n = ps.len() as f64;
        println!("{name:<20} {:>6} {dv:>10.2e} {:>9.4} {:>9.3} {:>12}",
                 ps.len(), end.pos.y / n * 20.0, end.pos.z / n * 20.0,
                 if dv < 1e-5 { "yes" } else { "no" });
    }
}

fn cmd_eq() {
    let (bzp, _) = ceiling();
    let e = equilibrium(bzp);
    println!("fastest steady glide: pitch {bzp:.3}  v_z {:.5}  v_y {:.5}  gamma {:.4}", e.z, e.y, gamma(e));
    println!("  -- this angle is the floor the dive's flight-path angle settles onto.\n");
    println!("{:>6} {:>10} {:>10} {:>9} {:>8}", "pitch", "eq v_y", "eq v_z", "|eq|", "gamma");
    for p in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 53.35, 60.0, 75.0, 90.0] {
        let e = equilibrium(p);
        println!("{p:>6.2} {:>10.5} {:>10.5} {:>9.5} {:>8.4}", e.y, e.z, e.length(), gamma(e));
    }
}

fn cmd_polish(path: &str, passes: usize, w: f64) {
    let mut pitches = read_pitches(path);
    let n = pitches.len();
    let obj = |s: &State| s.total_energy() + w * s.pos.z;
    let tail = |s: &State, ps: &[f64], i: usize| {
        let mut s = s.clone();
        for k in i..ps.len() { s = ticked(&s, ps[k]) }
        obj(&s)
    };
    eprintln!("start: J = {:.6}  y = {:.4}  (w = {w})", tail(&replay(&pitches)[n], &pitches, n), replay(&pitches)[n].pos.y);
    for pass in 0..passes {
        let mut states = replay(&pitches);
        let mut improved = 0.0;
        for t in 0..n {
            let s = states[t].clone();
            let f = |p: f64| tail(&ticked(&s, p), &pitches, t + 1);
            let cur = pitches[t];
            // every fourth pass sweeps the whole range, in case a tick is in the wrong basin
            let (lo, hi, step) = if pass % 4 == 0 { (-90.0, 90.0, 0.25) }
                                 else { ((cur - 8.0).max(-90.0), (cur + 8.0).min(90.0), 0.05) };
            let (mut bp, mut bs) = (cur, f(cur));
            let steps = ((hi - lo) / step).round() as i64;
            for i in 0..=steps { let p = lo + step * i as f64; let v = f(p); if v > bs { bs = v; bp = p } }
            let (mut a, mut b) = ((bp - step).max(-90.0), (bp + step).min(90.0));
            for _ in 0..70 { let (m1, m2) = (a+(b-a)/3.0, b-(b-a)/3.0); if f(m1) < f(m2) { a = m1 } else { b = m2 } }
            let pr = 0.5 * (a + b);
            let (np, ns) = if f(pr) > bs { (pr, f(pr)) } else { (bp, bs) };
            let np = np as f32 as f64;                       // the sim casts pitch to f32 anyway
            let base = f(cur);
            if ns > base { improved += ns - base; pitches[t] = np }
            // the prefix must stay consistent with the pitches just changed, or the next
            // tick's line search optimizes against a stale state and the schedule diverges
            states[t + 1] = ticked(&states[t], pitches[t]);
        }
        let st = replay(&pitches);
        if pass % 5 == 4 || improved < 1e-12 {
            eprintln!("pass {pass:>3}: J = {:.6}  y = {:.4}  z = {:.2}  |v_end| = {:.5}  (+{improved:.2e})",
                      tail(&st[n], &pitches, n), st[n].pos.y, st[n].pos.z, st[n].vel.length());
        }
        if improved < 1e-12 { break }
    }
    for p in &pitches { println!("{p}") }
}

fn cmd_cycle(path: &str, off: usize) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    println!("rel,pitch,vy,vz,speed,gamma,te,dte,hold_gamma,dte1,dte8,dte20");
    for rel in 0..300.min(ps.len() - off) {
        let t = off + rel;
        let s = &st[t];
        println!("{rel},{:.4},{:.6},{:.6},{:.6},{:.4},{:.5},{:.6},{:.4},{:.4},{:.4},{:.4}",
                 ps[t], s.vel.y, s.vel.z, s.vel.length(), gamma(s.vel), s.total_energy(),
                 st[t + 1].total_energy() - s.total_energy(),
                 bug_gamma_to(s, gamma(s.vel)), bug_dte_n(s, 1), bug_dte_n(s, 8), bug_dte_n(s, 20));
    }
}

fn cmd_score(path: &str, off: usize, lo: usize, hi: usize) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let rms = |f: &dyn Fn(&State) -> f64| {
        let (mut acc, mut n) = (0.0, 0);
        for rel in lo..hi { let d = f(&st[off + rel]) - ps[off + rel]; acc += d * d; n += 1 }
        (acc / n as f64).sqrt()
    };
    let med = |f: &dyn Fn(&State) -> f64| {
        let mut v: Vec<f64> = (lo..hi).map(|rel| (f(&st[off + rel]) - ps[off + rel]).abs()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("ticks {lo}..{hi} of the cycle at offset {off}");
    println!("{:<22} {:>8} {:>8}", "rule", "RMS", "median");
    let row = |name: String, f: &dyn Fn(&State) -> f64| println!("{name:<22} {:>8.2} {:>8.2}", rms(f), med(f));
    for n in [1usize, 2, 4, 8, 12, 16, 20, 24, 32, 48] {
        row(format!("dTE / {n} ticks"), &|s| bug_dte_n(s, n));
    }
    row("hold gamma".into(), &|s| bug_gamma_to(s, gamma(s.vel)));
    row("gamma -> 16.577".into(), &|s| bug_gamma_to(s, 16.577));
    row("max next speed".into(), &|s| argmax(|p| ticked(s, p).vel.length(), 0.125));
    row("min next |v_y|".into(), &|s| argmax(|p| -ticked(s, p).vel.y.abs(), 0.125));
    row("hold gamma, floored".into(), &bug_dive_floor);
}

fn cmd_probe(path: &str, off: usize, lo: usize, hi: usize) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    println!("rel,pitch,n_star,err_dte1,err_dte20,err_dte48,gamma");
    for rel in lo..hi {
        let s = &st[off + rel];
        let opt = ps[off + rel];
        let (mut bn, mut bd) = (0usize, f64::INFINITY);
        for n in 1..=90 { let d = (bug_dte_n(s, n) - opt).abs(); if d < bd { bd = d; bn = n } }
        println!("{rel},{opt:.3},{bn},{:.3},{:.3},{:.3},{:.2}",
                 bug_dte_n(s, 1) - opt, bug_dte_n(s, 20) - opt, bug_dte_n(s, 48) - opt, gamma(s.vel));
    }
}

/// Dump the whole tick x lookahead matrix of argmax pitches, so the spread across horizons
/// can be read directly. A narrow spread means the implied lookahead is barely determined.
fn cmd_sweepn(path: &str, off: usize, lo: usize, hi: usize, nmax: usize) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    print!("rel,pitch");
    for n in 1..=nmax { print!(",n{n}") }
    println!();
    for rel in lo..hi {
        let s = &st[off + rel];
        print!("{rel},{:.4}", ps[off + rel]);
        for n in 1..=nmax { print!(",{:.4}", bug_dte_n(s, n)) }
        println!();
    }
}

fn hold_gamma_branches(s: &State) -> Vec<f64> {
    let h = |p: f64| gamma(ticked(s, p).vel) - gamma(s.vel);
    let step = 0.125;
    let mut roots = vec![];
    let (mut pp, mut prev) = (-90.0, h(-90.0));
    for i in 1..=1440 {
        let p = -90.0 + step * i as f64;
        let cur = h(p);
        if prev == 0.0 {
            roots.push(pp);
        } else if cur == 0.0 {
            roots.push(p);
        } else if prev.is_sign_positive() != cur.is_sign_positive() {
            let (mut lo, mut hi, increasing) = (pp, p, cur > prev);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let value = h(mid);
                if (value <= 0.0) == increasing { lo = mid } else { hi = mid }
            }
            roots.push(0.5 * (lo + hi));
        }
        pp = p;
        prev = cur;
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots.dedup_by(|a, b| (*a - *b).abs() < 0.02);
    // Mth's lookup table can jump across zero at a cell boundary without attaining it.
    // Those are not solutions to gamma(v') = gamma(v), even though a sign-only bisection
    // returns the boundary.  Keep only roots that actually satisfy the equation.
    roots.retain(|&p| h(p).abs() < 1e-4);
    roots
}

fn dte_argmax_branches(s: &State, n: usize) -> (Vec<f64>, Option<f64>) {
    let f = |p: f64| run_n(s, p, n).total_energy() - s.total_energy();
    // The mapfine profile was optimized with an 85-degree pitch cap.  The Mth lookup table
    // also makes the exact poles singular, so they are not meaningful competing controls.
    let (lo, hi) = (-85.0, 85.0);
    let step = 0.125;
    let steps = ((hi - lo) / step) as usize;
    let values: Vec<f64> = (0..=steps).map(|i| f(lo + step * i as f64)).collect();
    let mut peaks = vec![];
    for i in 0..=steps {
        let left = if i == 0 { f64::NEG_INFINITY } else { values[i - 1] };
        let right = if i == steps { f64::NEG_INFINITY } else { values[i + 1] };
        if values[i] < left || values[i] < right { continue }
        let p = lo + step * i as f64;
        if peaks.last().is_some_and(|&(q, _): &(f64, f64)| p - q < step * 1.5) { continue }
        let (mut bp, mut bs) = (p, values[i]);
        if i > 0 && i < steps {
            let (mut a, mut b) = (p - step, p + step);
            for _ in 0..60 {
                let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
                if f(m1) < f(m2) { a = m1 } else { b = m2 }
            }
            let q = 0.5 * (a + b);
            let score = f(q);
            if score > bs { bp = q; bs = score }
        }
        peaks.push((bp, bs));
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let best = peaks[0].1;
    let tolerance = 1e-8_f64.max(best.abs() * 1e-10);
    let runner_up_gap = peaks.iter().skip(1).map(|&(_, score)| best - score)
        .find(|&gap| gap > tolerance);
    let mut branches: Vec<f64> = peaks.iter().filter(|&&(_, score)| best - score <= tolerance)
        .map(|&(p, _)| p).collect();
    branches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (branches, runner_up_gap)
}

/// Dump the optimum and pointwise rules used by the mapfine comparison plot.
/// Atlas profiles supply their initial velocity explicitly instead of using `V0`.
fn cmd_rules(path: &str, vy0: f64, vz0: f64, nmax: usize) {
    let ps = read_pitches(path);
    let st = replay_from(Vec3::new(0.0, vy0, vz0), &ps);
    print!("tick,optimum,hold_gamma");
    for n in 1..=nmax { print!(",n{n}") }
    println!();
    let (mut max_hold, mut max_argmax, mut multi_argmax, mut closest_gap, mut hold_residual) =
        (0usize, 0usize, 0usize, f64::INFINITY, 0.0_f64);
    for t in 0..ps.len() {
        let s = &st[t];
        let holds = hold_gamma_branches(s);
        max_hold = max_hold.max(holds.len());
        for &p in &holds {
            hold_residual = hold_residual.max((gamma(ticked(s, p).vel) - gamma(s.vel)).abs());
        }
        let hold = bug_gamma_to(s, gamma(s.vel));
        print!("{t},{:.5},{hold:.5}", ps[t]);
        for n in 1..=nmax {
            let (branches, gap) = dte_argmax_branches(s, n);
            max_argmax = max_argmax.max(branches.len());
            if branches.len() > 1 { multi_argmax += 1 }
            if let Some(gap) = gap { closest_gap = closest_gap.min(gap) }
            print!(",{:.5}", branches[0]);
        }
        println!();
    }
    eprintln!("branch check: at most {max_hold} hold-gamma roots (max residual {hold_residual:.3e} deg) and {max_argmax} tied global argmax modes; {multi_argmax} tick/lookahead cells have multiple argmax modes; closest non-tied local mode is {closest_gap:.3e} blocks below the maximum");
}

/// Find the cycle in a schedule without being told where it is, then report how well the
/// dive and gain rules fit it. Used to check the rules across a family of optimal cycles.
/// Middle cycle of a 3x-tiled schedule, split into its four phases: `(a0, t_snap, t_gain,
/// t_gend, a1)`. Apex to apex, so the dive is `a0..t_snap` and the gain is `t_gain..t_gend`.
fn cmd_family(path: &str, tag: &str) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let Some((a0, t_snap, t_gain, t_gend, a1)) = segment(&ps, &st, tag) else { return };
    let period = a1 - a0;
    let (dy, dz) = (st[a1].pos.y - st[a0].pos.y, st[a1].pos.z - st[a0].pos.z);

    let (mut acc, mut n, mut gmin, mut gmax) = (0.0, 0, f64::MAX, f64::MIN);
    let mut habs = vec![];
    for t in a0 + period / 8..t_snap {
        let d = bug_gamma_to(&st[t], gamma(st[t].vel)) - ps[t];
        acc += d * d; n += 1; habs.push(d.abs());
        let g = gamma(st[t].vel); gmin = gmin.min(g); gmax = gmax.max(g);
    }
    habs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (hold_rms, hold_med) = ((acc / n as f64).sqrt(), habs[habs.len() / 2]);

    let (mut best, mut r1, mut r20) = ((0usize, f64::MAX), 0.0, 0.0);
    for k in 1..=60usize {
        let (mut acc, mut m) = (0.0, 0);
        for t in t_gain..t_gend { let d = bug_dte_n(&st[t], k) - ps[t]; acc += d * d; m += 1 }
        let rms = (acc / m as f64).sqrt();
        if k == 1 { r1 = rms } else if k == 20 { r20 = rms }
        if rms < best.1 { best = (k, rms) }
    }
    let mut ns: Vec<usize> = (t_gain..t_gend).map(|t| {
        let (mut bn, mut bd) = (0usize, f64::INFINITY);
        for k in 1..=90 { let d = (bug_dte_n(&st[t], k) - ps[t]).abs(); if d < bd { bd = d; bn = k } }
        bn
    }).collect();
    ns.sort();
    println!("{tag:>8} | period {period:>4} | climb {:>6.4} | dist {:>6.3} | dive {:>3}t gamma {:>5.2}..{:<5.2} \
              hold RMS {:>5.2} med {:>4.2} | gain {:>3}t best n={:<3} rms {:>5.2} | n=1 {:>6.2} | n=20 {:>5.2} | n* med {:>2}",
             dy / period as f64 * 20.0, dz / period as f64 * 20.0, t_snap - a0, gmin, gmax,
             hold_rms, hold_med, t_gend - t_gain, best.0, best.1, r1, r20, ns[ns.len() / 2]);
}

/// The steady glide that maximizes the *objective rate* `v_y + w*v_z`.
///
/// Turnpike candidate for the dive's flight-path-angle floor. Maximizing `TE + w*z` over a
/// fixed number of ticks is maximizing the time-average of `d/dt (TE + w*z) = v_y + w*v_z`,
/// so if the dive were asymptoting to the best available *steady* state for the objective,
/// the floor would track this angle. Note what it reduces to at w = 0: the minimum-sink
/// glide, not the fastest one.
/// Ticks per second: `sim` works in blocks/tick, but these two commands report blocks/second.
const TPS: f64 = 20.0;

/// Steady-glide curves against pitch, as CSV, for `tools/plot_glide.py`.
///
/// The horizontal axis is `sim`'s `z` (yaw is pinned to zero); it is reported as `vx` because
/// that is the convention everywhere outside the sim. Velocities are blocks/second. The glide
/// ratio is a ratio of two velocities, so it is dimensionless and the same in either unit.
fn cmd_glide(step: f64, lo: f64, hi: f64) {
    println!("# steady glide vs pitch, trig={}, velocities in blocks/second", trig_mode());
    println!("pitch,vy,vx,speed,glide,gamma");
    let n = ((hi - lo) / step).round() as i64;
    let rows: Vec<String> = (0..=n).into_par_iter().map(|i| {
        let p = lo + step * i as f64;
        let e = equilibrium(p);
        format!("{p:.6},{:.10},{:.10},{:.10},{:.9},{:.6}",
                e.y * TPS, e.z * TPS, e.length() * TPS, e.z / -e.y, gamma(e))
    }).collect();
    for r in rows { println!("{r}") }
}

/// The three critical points of the steady glide, each read from both sides.
///
/// The interesting one is pitch 0. For `p >= 0` the tick map depends on pitch *only* through
/// `lift_force = cos^2(p)`: yaw is zero, so `look_angle.x` is zero and every use of
/// `look_angle.z / look_hor_length` collapses to 1, cancelling `cos(p)` out of the direction
/// terms, and the `lean_angle < 0` branch is off. `cos^2` is even, so every curve here is
/// even-and-flat to second order on the right of 0. For `p < 0` that branch switches on with a
/// term linear in `sin(p)`, which is a nonzero one-sided slope. So 0 is a *corner*, not a jump.
fn cmd_crit() {
    let crit = |name: &str, p0: f64, key: &dyn Fn(Vec3) -> f64| {
        // refine on a fine sweep; the tops are flat to ~1e-9, so a ternary search wanders
        let (mut bp, mut bs) = (p0, f64::NEG_INFINITY);
        for i in -2000..=2000 {
            let p = p0 + 0.001 * i as f64;
            if !(-90.0..=90.0).contains(&p) { continue }
            let v = key(equilibrium(p));
            if v > bs { bs = v; bp = p }
        }
        println!("\n{name}: argmax at pitch {bp:.4}");
        println!("{:>12} {:>16} {:>16} {:>14} {:>12}", "pitch", "vy", "vx", "glide", "gamma");
        let mut ps = vec![bp];
        for e in [1e-1, 1e-2, 1e-3, 1e-5, 1e-7] { ps.push(bp - e); ps.push(bp + e) }
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for p in ps {
            let e = equilibrium(p);
            let mark = if p == bp { " <-" } else { "" };
            println!("{p:>12.7} {:>16.10} {:>16.10} {:>14.9} {:>12.6}{mark}",
                     e.y * TPS, e.z * TPS, e.z / -e.y, gamma(e));
        }
    };
    println!("steady glide critical points, trig={}, velocities in blocks/second", trig_mode());
    crit("max glide ratio (blocks forward per block fallen)", 0.0, &|v| v.z / -v.y);
    crit("min sink (max vy)", -13.0, &|v| v.y);
    crit("max forward speed (max vx)", 53.0, &|v| v.z);
}

// ---------------------------------------------------------------- the acceleration frontier

/// Largest horizontal speed at which *some* pitch still raises it, in closed form.
///
/// Over all pitches and yaws `h' <= 0.99 h + 0.891 D` with `D = -0.1 L (v_y - 0.08 + 0.06 L)`
/// and `L = cos^2 p`: the nose-up branch only subtracts, and a yaw off the velocity only
/// subtracts. Maximizing `D` over `L` in `[0,1]` and solving `h' = h` gives three branches --
/// pitch 0 saturating the lift below `v_y = -0.04`, an interior lift above it, and no lift at
/// all once gravity can no longer be beaten. See `docs/rising.md`.
fn frontier_h(vy: f64) -> f64 {
    if vy >= 0.08 { 0.0 }
    else if vy >= -0.04 { 3712.5 * (0.008 - 0.1 * vy).powi(2) }
    else { 8.91 * (0.02 - vy) }        // = the vz_peaked line, 0.1782 - 8.91 v_y
}

/// The pitch that maximizes `h'`, in closed form. Never nose-up: every nose-up pitch is beaten
/// by the nose-down pitch with the same lift, which pays no pull-up.
fn frontier_pitch(vy: f64) -> f64 {
    (((0.008 - 0.1 * vy) / 0.012).clamp(0.0, 1.0) as f64).sqrt().acos().to_degrees()
}

fn dh(vy: f64, vz: f64, p: f64) -> f64 {
    ticked(&State { pos: Vec3::ZERO, vel: Vec3::new(0.0, vy, vz) }, p).vel.z - vz
}
fn best_dh(vy: f64, vz: f64) -> (f64, f64) {
    let p = argmax(|p| dh(vy, vz, p), 0.05);
    (dh(vy, vz, p), p)
}

fn rising_frontier() {
    println!("# largest v_z at which some pitch still raises v_z, bisected against the sim");
    println!("{:>9} {:>12} {:>12} {:>10} {:>9} {:>9}",
             "v_y", "sim", "closed", "rel", "p* sim", "p* cls");
    for vy in [-1.2, -1.0, -0.6, -0.4, -0.36040, -0.2, -0.14949, -0.1, -0.04,
               -0.02, 0.0, 0.02, 0.04, 0.06, 0.079, 0.08, 0.2] {
        if best_dh(vy, 0.0).0 <= 0.0 {
            println!("{vy:>9.5} {:>12.6} {:>12.6} {:>10} {:>9} {:>9.3}",
                     0.0, frontier_h(vy), "-", "-", frontier_pitch(vy));
            continue
        }
        let (mut lo, mut hi) = (0.0, 20.0);
        for _ in 0..80 { let m = 0.5 * (lo + hi); if best_dh(vy, m).0 > 0.0 { lo = m } else { hi = m } }
        let f = 0.5 * (lo + hi);
        println!("{vy:>9.5} {f:>12.6} {:>12.6} {:>10.1e} {:>9.3} {:>9.3}",
                 frontier_h(vy), f / frontier_h(vy) - 1.0, best_dh(vy, f).1, frontier_pitch(vy));
    }
}

fn rising_pitch() {
    println!("# v_y >= 0.08: no pitch can touch h. h'/h over all nose-down pitches, and the");
    println!("# best nose-up one, at 0.01 deg resolution.");
    println!("{:>7} {:>7}  {:>15} {:>15} {:>15}", "v_y", "v_z", "max h'/h, p>=0", "min h'/h, p>=0", "max h'/h, p<0");
    for vy in [0.08, 0.2, 1.0] {
        for vz in [0.2, 1.0, 3.4] {
            let r = |p: f64| (dh(vy, vz, p) + vz) / vz;
            let (mut mx, mut mn, mut up) = (f64::MIN, f64::MAX, f64::MIN);
            for i in 0..=9000 {
                mx = mx.max(r(i as f64 * 0.01));
                mn = mn.min(r(i as f64 * 0.01));
                up = up.max(r(-(i as f64) * 0.01 - 0.01));
            }
            println!("{vy:>7.2} {vz:>7.2}  {mx:>15.12} {mn:>15.12} {up:>15.12}");
        }
    }
    println!("\n# same claim in three dimensions: worst h'/h over 4e6 random (v, pitch, yaw)");
    println!("# draws with v_y >= 0.08, v_x and v_z in [-4, 4], yaw in [-360, 360]");
    let (mut worst, mut arg) = (f64::MIN, (0f64, 0f64, 0f64, 0f64, 0f64));
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
    for _ in 0..4_000_000 {
        let (vx, vy, vz) = (rnd() * 8.0 - 4.0, 0.08 + rnd() * 4.0, rnd() * 8.0 - 4.0);
        let (p, y) = (rnd() * 180.0 - 90.0, rnd() * 720.0 - 360.0);
        let h = (vx * vx + vz * vz).sqrt();
        if h < 1e-9 { continue }
        let v = update_fall_flying_movement(Vec3::new(vx, vy, vz), Rot { x: p as f32, y: y as f32 });
        let r = (v.x * v.x + v.z * v.z).sqrt() / h;
        if r > worst { worst = r; arg = (vx, vy, vz, p, y) }
    }
    println!("worst h'/h {worst:.15} at v ({:.3}, {:.3}, {:.3}) pitch {:.2} yaw {:.2}; \
              0.99f32 = {:.15}", arg.0, arg.1, arg.2, arg.3, arg.4, 0.99_f32 as f64);

    println!("\n# steepest nose-up pitch that still raises h (blank: none does)");
    let hs = [0.5f64, 1.0, 1.5, 2.0, 2.5, 3.0, 3.389];
    println!("{:>7} | {}", "v_y", hs.map(|h| format!("{h:>8.3}")).join(""));
    for vy in [-0.1, -0.2, -0.3, -0.4, -0.6, -0.8, -1.0, -1.5, -2.0] {
        let row: Vec<String> = hs.iter().map(|&h| {
            if dh(vy, h, 0.0) <= 0.0 { return format!("{:>8}", "-") }
            let (mut lo, mut hi) = (-90.0, 0.0);
            for _ in 0..60 { let m = 0.5 * (lo + hi); if dh(vy, h, m) > 0.0 { hi = m } else { lo = m } }
            format!("{:>8.2}", 0.5 * (lo + hi))
        }).collect();
        println!("{vy:>7.2} | {}", row.join(""));
    }
}

/// Hold `p` until `v_y < 0.08`; the ticks it takes, and the height it buys.
fn rise_leg(vy0: f64, h0: f64, p: f64) -> (usize, f64, f64) {
    let mut s = State { pos: Vec3::ZERO, vel: Vec3::new(0.0, vy0, h0) };
    let te0 = s.total_energy();
    let mut t = 0;
    while s.vel.y >= 0.08 && t < 500 { s = ticked(&s, p); t += 1 }
    (t, s.total_energy() - te0, s.pos.y)
}

fn rising_clock() {
    println!("# the rise clock: how long `v_y >= 0.08` lasts, per held pitch. h' = 0.99 h every");
    println!("# tick of it whatever the pitch, so `0.99^T` is the whole horizontal cost.");
    println!("{:>8} | {:>23} | {:>23} | {:>23}", "",
             "pitch 90, no lift", "pitch 45, half lift", "pitch 0, full lift");
    println!("{:>8} | {:>4} {:>7} {:>6} {:>4} | {:>4} {:>7} {:>6} {:>4} | {:>4} {:>7} {:>6} {:>4}",
             "v_y0", "T", "0.99^T", "dy", "T<0", "T", "0.99^T", "dy", "T<0", "T", "0.99^T", "dy", "T<0");
    for vy0 in [0.08, 0.10, 0.15, 0.167467, 0.20, 0.30, 0.40, 0.50, 0.80] {
        let cells: Vec<String> = [89.9, 45.0, 0.0].iter().map(|&p| {
            let (t1, _, dy) = rise_leg(vy0, 1.0, p);
            let mut s = State { pos: Vec3::ZERO, vel: Vec3::new(0.0, vy0, 1.0) };
            let mut t2 = 0;
            while s.vel.y >= 0.0 && t2 < 500 {
                s = ticked(&s, if s.vel.y >= 0.08 { p } else { frontier_pitch(s.vel.y) });
                t2 += 1;
            }
            format!("{t1:>4} {:>7.4} {dy:>6.3} {t2:>4}", 0.99_f64.powi(t1 as i32))
        }).collect();
        println!("{vy0:>8.4} | {} | {} | {}", cells[0], cells[1], cells[2]);
    }
    println!("\n# zero lift is exactly solvable: v_y(k) = -3.92 + (v_y0 + 3.92) 0.98^k, so");
    println!("# T = ln((v_y0 + 3.92)/4) / 0.020203, rounded up");
    for vy0 in [0.10, 0.20, 0.30, 0.50, 0.80] {
        println!("  v_y0 {vy0:>5.2}  T = {:>6.3} -> {} (sim: {})",
                 ((vy0 + 3.92) / 4.0f64).ln() / 0.98f64.ln().abs(),
                 (((vy0 + 3.92) / 4.0f64).ln() / 0.98f64.ln().abs()).ceil(),
                 rise_leg(vy0, 1.0, 89.9).0);
    }
}

fn rising_tradeoff() {
    println!("# lift while rising buys height and costs ticks. Compared at the end of the rise,");
    println!("# where both are back at v_y < 0.08: above h*, dumping the lift wins on TE.");
    println!("{:>7} {:>8} {:>8} {:>10} {:>10} {:>10}",
             "v_y0", "T(p=90)", "T(p=0)", "extra dy", "h* sim", "h* closed");
    for vy0 in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.80, 1.20] {
        let (t90, _, y90) = rise_leg(vy0, 1.0, 89.9);
        let (t0, _, y0) = rise_leg(vy0, 1.0, 0.0);
        if t90 == t0 {
            println!("{vy0:>7.2} {t90:>8} {t0:>8} {:>10.3} {:>10} {:>10}", y0 - y90, "-", "-");
            continue
        }
        let (mut lo, mut hi) = (0.05, 12.0);
        for _ in 0..60 {
            let m = 0.5 * (lo + hi);
            if rise_leg(vy0, m, 0.0).1 - rise_leg(vy0, m, 89.9).1 > 0.0 { lo = m } else { hi = m }
        }
        let pred = (0.16 * (y0 - y90)
                    / (0.99f64.powi(2 * t90 as i32) - 0.99f64.powi(2 * t0 as i32))).sqrt();
        println!("{vy0:>7.2} {t90:>8} {t0:>8} {:>10.3} {:>10.4} {pred:>10.4}",
                 y0 - y90, 0.5 * (lo + hi));
    }
    println!("\n# for scale: pitch-0 steady glide h = {:.5}, fastest steady glide h = {:.5}",
             equilibrium(0.0).z, equilibrium(ceiling().0).z);
}

fn rising_down() {
    println!("# one tick from v = (0, 0.30, 2.00) at pitches approaching straight down, trig {}",
             trig_mode());
    println!("{:>10} {:>14} {:>12} {:>14} {:>12}", "pitch", "Mth::cos(p)", "lift", "v_z'/v_z", "v_y'");
    for p in [0.0f64, 45.0, 80.0, 89.0, 89.9, 89.99, 89.999, 90.0, -89.999, -90.0] {
        let lean = (p as f32) * (std::f64::consts::PI / 180.0) as f32;
        let s = State { pos: Vec3::ZERO, vel: Vec3::new(0.0, 0.30, 2.00) };
        let v = ticked(&s, p).vel;
        println!("{p:>10.4} {:>14.4e} {:>12.4e} {:>14.9} {:>12.7}",
                 Mth::cos(lean), (lean as f64).cos().powi(2), v.z / 2.00, v.y);
    }
}

fn cmd_rising(part: &str) {
    println!("# trig = {}, flight = {}\n", trig_mode(), flight_mode());
    match part {
        "frontier" => rising_frontier(),
        "pitch" => rising_pitch(),
        "clock" => rising_clock(),
        "tradeoff" => rising_tradeoff(),
        "down" => rising_down(),
        "all" => { rising_frontier(); println!(); rising_pitch(); println!();
                   rising_clock(); println!(); rising_tradeoff(); println!(); rising_down() }
        _ => eprintln!("rising: want frontier|pitch|clock|tradeoff|down|all"),
    }
}

fn cmd_eqrate(locus: bool) {
    // TE is in blocks, so potential energy is exactly height and v_y carries weight 1.
    const G: f64 = 1.0;
    // The locus is a curve in the (v_z, v_y) plane parameterized by pitch. Build it once;
    // each point is 40k iterations of the velocity map.
    let step = 0.05;
    let tab: Vec<(f64, Vec3)> = (0..=(180.0 / step) as i64)
        .map(|i| { let p = -90.0 + step * i as f64; (p, equilibrium(p)) })
        .collect();
    let pick = |f: &dyn Fn(Vec3) -> f64| -> (f64, Vec3) {
        let (mut bp, mut bs) = (0.0, f64::NEG_INFINITY);
        for &(p, e) in &tab { let v = f(e); if v > bs { bs = v; bp = p } }
        let (mut a, mut b) = ((bp - step).max(-90.0), (bp + step).min(90.0));
        for _ in 0..80 {
            let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
            if f(equilibrium(m1)) < f(equilibrium(m2)) { a = m1 } else { b = m2 }
        }
        let p = 0.5 * (a + b);
        if f(equilibrium(p)) > bs { (p, equilibrium(p)) } else { (bp, equilibrium(bp)) }
    };

    println!("reference points on the equilibrium locus");
    println!("{:<28} {:>8} {:>10} {:>10} {:>9} {:>9}", "", "pitch", "eq v_y", "eq v_z", "|eq|", "gamma");
    let refs: [(&str, &dyn Fn(Vec3) -> f64); 3] = [
        ("fastest glide  (max v_z)", &|e: Vec3| e.z),
        ("min-sink glide (max v_y)", &|e: Vec3| e.y),
        ("best glide ratio (min y)", &|e: Vec3| -gamma(e)),
    ];
    for (name, f) in refs {
        let (p, e) = pick(f);
        println!("{name:<28} {p:>8.3} {:>10.5} {:>10.5} {:>9.5} {:>9.4}", e.y, e.z, e.length(), gamma(e));
    }

    // The observed floors, from README-myopic.md's family table.
    // w is in the block convention; README-myopic.md's family table predates it, so its
    // w column is these divided by GRAVITY. The gammas are unchanged.
    let obs: &[(f64, f64)] = &[(-0.125, 12.81), (-0.0625, 14.62), (-0.025, 16.80), (0.0, 16.72),
                               (0.025, 16.32), (0.0625, 15.94), (0.125, 15.36), (0.250, 14.45)];
    println!("\nturnpike glide for the objective TE + w*z, vs the observed dive floor");
    println!("{:>7} {:>8} {:>10} {:>10} {:>9} {:>9} {:>12} {:>9}",
             "w", "pitch", "eq v_y", "eq v_z", "|eq|", "gamma_eq", "rate", "observed");
    for &(w, g_obs) in obs {
        let (p, e) = pick(&|e: Vec3| G * e.y + w * e.z);
        println!("{w:>7.4} {p:>8.3} {:>10.5} {:>10.5} {:>9.5} {:>9.4} {:>12.6} {g_obs:>9.2}",
                 e.y, e.z, e.length(), gamma(e), G * e.y + w * e.z);
    }
    if locus {
        println!("\npitch,eq_vy,eq_vz,speed,gamma");
        for i in 0..=3600 { let p = -90.0 + 0.05 * i as f64; let e = equilibrium(p);
            println!("{p:.2},{:.6},{:.6},{:.6},{:.4}", e.y, e.z, e.length(), gamma(e)) }
        return;
    }
    println!("\nthe same sweep, wider, to show which way the turnpike actually moves");
    println!("{:>9} {:>8} {:>10} {:>10} {:>9}", "w", "pitch", "eq v_y", "eq v_z", "gamma_eq");
    for w in [-12.5, -2.5, -1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0, 2.5, 12.5, 125.0] {
        let (p, e) = pick(&|e: Vec3| G * e.y + w * e.z);
        println!("{w:>9.3} {p:>8.3} {:>10.5} {:>10.5} {:>9.4}", e.y, e.z, gamma(e));
    }
}

/// What angle is the dive's flight-path angle actually heading for?
///
/// Near w = 0 gamma dips to a plateau and comes back up as the snap approaches, so the trough
/// is a real floor. On the min-distance side it climbs monotonically and there is no floor at
/// all. `min gamma over the whole dive` cannot tell those apart -- on the climbing shapes it
/// returns the early transient instead, which is what produced the spurious non-monotonicity
/// in the family table. So take the minimum over the middle of the dive only, and say whether
/// it is interior; if it is not, there is no floor to report.
///
/// (A first-order decay `gamma_{t+1} - gamma_t = k (c - gamma_t)` fits badly -- the shape is
/// dip-then-rise, not decay -- and on a polished schedule the tick-scale jitter enters
/// regressor and response with opposite signs and inflates k. Not worth reporting.)
fn cmd_floor(path: &str, tag: &str) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let Some((a0, t_snap, _, _, a1)) = segment(&ps, &st, tag) else { return };
    let (lo, hi) = (a0 + (a1 - a0) / 8, t_snap);             // same window family scores the hold on
    let g: Vec<f64> = (lo..hi).map(|t| gamma(st[t].vel)).collect();
    let m = g.len() - 1;
    let (i, &trough) = g[m / 5..=4 * m / 5].iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    let at = (m / 5 + i) as f64 / m as f64;
    let interior = at > 0.22 && at < 0.78;
    // How much forward speed does that angle buy? Invert it onto the nose-down branch of the
    // equilibrium locus and read the steady speed there against the ceiling.
    let (mut bp, mut bd) = (0.0, f64::MAX);
    for i in 400..=1800 {
        let q = 0.05 * i as f64;
        let d = (gamma(equilibrium(q)) - trough).abs();
        if d < bd { bd = d; bp = q }
    }
    let vz = equilibrium(bp).z;
    const VZ_MAX: f64 = 3.3887937;                           // fastest steady glide, from `eq`
    println!("{tag:>9} | dive {:>3}t | gamma {:>5.2} -> {:>5.2} | trough {:>5.2} at {:>3.0}% {} \
              | that glide: pitch {bp:>5.2} v_z {vz:>7.5} = {:>6.3}% of the ceiling",
             m + 1, g[0], g[m], trough, 100.0 * at,
             if interior { "        " } else { "(no floor)" }, 100.0 * vz / VZ_MAX);
}

/// The costate direction, read off the optimum itself.
///
/// The optimal pitch satisfies the stationary condition `lambda_{t+1} . df/dp = 0`. With yaw
/// pinned the state is two-dimensional, so that one equation pins `lambda` up to sign and
/// scale: it is the normal to the reachable curve's tangent. Sign is fixed by requiring height
/// to be worth something. What comes back is the cycle's *actual* price of distance in units of
/// height, `lambda_z / lambda_y`, which is the number the objective weight `w` was
/// standing in for.
fn costate_dir(v: Vec3, p: f64) -> (f64, f64) {
    let h = 1e-3;
    let (a, m, b) = (update_fall_flying_movement(v, rot(p - h)), update_fall_flying_movement(v, rot(p)),
                     update_fall_flying_movement(v, rot(p + h)));
    let (dy, dz) = ((b.y - a.y) / (2.0 * h), (b.z - a.z) / (2.0 * h));
    let (mut ny, mut nz) = (-dz, dy);
    let n = ny.hypot(nz);
    ny /= n; nz /= n;
    // Sign from the second-order condition: at a maximum of lambda.f the curvature must be
    // non-positive. Do NOT assume lambda_y > 0 -- in a dive, upward velocity is worth less
    // than nothing, and forcing the other branch picks out a nose-up glide that makes no sense.
    let (fyy, fzz) = ((b.y - 2.0 * m.y + a.y) / (h * h), (b.z - 2.0 * m.z + a.z) / (h * h));
    if ny * fyy + nz * fzz > 0.0 { ny = -ny; nz = -nz }
    (ny, nz)
}

/// Does the dive's floor sit where the *measured* prices say a steady glide should?
///
/// `eqrate` asks which equilibrium maximizes `v_y + w*v_z` and gets the wrong answer.
/// This asks the same question with the shadow prices the optimum is actually using, recovered
/// from its own stationary condition, which is the only version of the turnpike claim that
/// has a chance of being true.
fn cmd_prices(path: &str, tag: &str) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let Some((a0, t_snap, _, _, a1)) = segment(&ps, &st, tag) else { return };
    let step = 0.05;
    let tab: Vec<(f64, Vec3)> = (0..=(180.0 / step) as i64)
        .map(|i| { let p = -90.0 + step * i as f64; (p, equilibrium(p)) }).collect();
    let best_eq = |ly: f64, lz: f64| -> (f64, Vec3) {
        let (mut bp, mut be, mut bs) = (0.0, Vec3::ZERO, f64::NEG_INFINITY);
        for &(p, e) in &tab { let v = ly * e.y + lz * e.z; if v > bs { bs = v; bp = p; be = e } }
        (bp, be)
    };
    println!("{tag}: costate through the dive (w is what eqrate assumed)");
    println!("{:>5} {:>8} {:>8} {:>8} {:>8} {:>9} {:>10} {:>9} {:>9} {:>9}",
             "rel", "pitch", "gamma", "l_y", "l_z", "lz/ly", "eq pitch", "eq gamma", "eq v_z", "argmax-p");
    println!("{:>88}   (last column: global argmax of lambda.f minus the optimum's pitch)", "");
    let (lo, hi) = (a0 + (a1 - a0) / 8, t_snap);
    for t in (lo..hi).step_by(((hi - lo) / 12).max(1)) {
        let (ly, lz) = costate_dir(st[t].vel, ps[t]);
        let (ep, ee) = best_eq(ly, lz);
        // not circular: the tangency fixes lambda locally, so recovering p_t as the *global*
        // argmax of lambda.f is a real check that the extracted lambda is the optimum's own
        let chk = argmax(|q| { let f = update_fall_flying_movement(st[t].vel, rot(q)); ly * f.y + lz * f.z }, 0.125);
        println!("{:>5} {:>8.3} {:>8.3} {ly:>8.4} {lz:>8.4} {:>9.4} {ep:>10.3} {:>9.4} {:>9.5} {:>9.3}",
                 t - a0, ps[t], gamma(st[t].vel), lz / ly, gamma(ee), ee.z, chk - ps[t]);
    }
    // the price the cycle puts on distance, averaged over the settled half of the dive
    let mid = (lo + hi) / 2;
    let (mut sy, mut sz) = (0.0, 0.0);
    for t in mid..hi { let (ly, lz) = costate_dir(st[t].vel, ps[t]); sy += ly; sz += lz }
    let (n, ep_pair) = ((hi - mid) as f64, ());
    let _ = ep_pair;
    let (my, mz) = (sy / n, sz / n);
    let (ep, ee) = best_eq(my, mz);
    println!("settled dive: mean lambda ({my:.4}, {mz:.4}), lz/ly {:.4}  ->  equilibrium pitch {ep:.3}, \
              gamma {:.4}, v_z {:.5}", mz / my, gamma(ee), ee.z);
}

/// Flight-path angle at ten points through the dive, so the *shape* is visible.
///
/// This is what settles the question `min gamma` was getting wrong: near w = 0 the dive dips
/// and comes back up, so the minimum is a real plateau, but on the min-distance side gamma
/// climbs monotonically and there is no floor at all -- there the minimum is just the start of
/// the dive, and reading it as an asymptote invents a trend that is not there.
fn cmd_gprofile(path: &str, tag: &str) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let Some((a0, t_snap, _, _, a1)) = segment(&ps, &st, tag) else { return };
    let (lo, hi) = (a0 + (a1 - a0) / 8, t_snap);
    let g: Vec<f64> = (lo..hi).map(|t| gamma(st[t].vel)).collect();
    let m = g.len() - 1;
    print!("{tag:>9} |");
    for i in 0..=10 { print!(" {:>5.2}", g[m * i / 10]) }
    println!(" | min {:>5.2} at {:>3.0}%",
             g.iter().cloned().fold(f64::MAX, f64::min),
             100.0 * g.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f64 / m as f64);
}


/// Is the optimum the one-tick argmax of a linear score on next tick's velocity?
///
/// Over a closed cycle the objective is `sum_t c . v_{t+1}` with `c = (1, w)`, because the
/// kinetic terms cancel when the cycle closes. Pontryagin then says the optimum maximizes
/// `mu_{t+1} . f(v_t, p)` at every tick -- a genuinely myopic, one-tick, horizon-free score --
/// where the price vector obeys `mu_t = c + A_t^T mu_{t+1}`, `A_t = df/dv`.
///
/// That recursion plus periodicity (`mu_N = mu_0`) pins `mu` completely: solve `(I - M) mu_0 = b`
/// where `M` and `b` accumulate the recursion around the loop. **No free parameters.** So the
/// tangency at each of the N ticks is a separate falsifiable prediction, unlike reading `mu` off
/// the optimum's own pitch, which is stationary by construction. Reported as the gap in degrees
/// between the optimum's pitch and the argmax of the score.
fn cmd_adjoint(path: &str, w: Option<f64>, dump: bool) {
    // A sweep profile states its own starting velocity and price. Replaying it from the
    // built-in `V0` instead is simply a different flight -- the cycle does not close and every
    // number below is of something else -- so read the header where there is one, and fall back
    // to `V0` only for a bare list of pitches.
    let prof = Profile::parse(&std::fs::read_to_string(path)
                              .unwrap_or_else(|e| panic!("{path}: {e}"))).ok();
    let ps = read_pitches(path);
    let (v0, src) = match &prof { Some(p) => (p.obj.v0, "header"), None => (V0, "built-in V0") };
    let w = w.or_else(|| prof.as_ref().map(|p| p.obj.w())).unwrap_or(0.0);
    let st = replay_from(v0, &ps);
    let n = ps.len();
    let close = (st[n].vel - st[0].vel).length();
    println!("{path}: {n} ticks, v0 ({:.6}, {:.6}) from the {src}, |v_N - v_0| = {close:.3e}, w = {w}",
             v0.y, v0.z);
    if close > 1e-3 { println!("  !! not a closed cycle; the periodic adjoint does not apply") }
    let c = (1.0, w);

    let (mut b, mut m) = ((0.0, 0.0), [1.0, 0.0, 0.0, 1.0]);
    for t in (0..n).rev() {
        let a = jac(st[t].vel, ps[t]);
        let ab = mt_vec(&a, b);
        b = (c.0 + ab.0, c.1 + ab.1);
        m = mt_mat(&a, &m);
    }
    // (I - M) mu_0 = b
    let d = [1.0 - m[0], -m[1], -m[2], 1.0 - m[3]];
    let det = d[0] * d[3] - d[1] * d[2];
    println!("  monodromy M = [{:.4} {:.4}; {:.4} {:.4}], det(I-M) = {det:.4e}", m[0], m[1], m[2], m[3]);
    if det.abs() < 1e-12 { println!("  !! singular; mu is not determined"); return }
    let mut mu = vec![(0.0, 0.0); n + 1];
    mu[n] = ((d[3] * b.0 - d[1] * b.1) / det, (-d[2] * b.0 + d[0] * b.1) / det);
    for t in (0..n).rev() {
        let a = jac(st[t].vel, ps[t]);
        let am = mt_vec(&a, mu[t + 1]);
        mu[t] = (c.0 + am.0, c.1 + am.1);
    }
    let drift = ((mu[0].0 - mu[n].0).hypot(mu[0].1 - mu[n].1)) / mu[0].0.hypot(mu[0].1);
    println!("  mu_0 = ({:.4}, {:.4}), periodicity residual {drift:.2e}", mu[0].0, mu[0].1);

    // the prediction: p_t is the global argmax over pitch of mu_{t+1} . f(v_t, p)
    let mut err: Vec<f64> = vec![];
    for t in 0..n {
        let (my, mz) = mu[t + 1];
        let q = argmax(|p| { let f = update_fall_flying_movement(st[t].vel, rot(p)); my * f.y + mz * f.z }, 0.125);
        err.push(q - ps[t]);
    }
    let rms = |a: &[f64]| (a.iter().map(|x| x * x).sum::<f64>() / a.len() as f64).sqrt();
    let mut srt: Vec<f64> = err.iter().map(|x| x.abs()).collect();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  argmax gap over all {n} ticks: RMS {:.4}deg  median {:.4}  p90 {:.4}  max {:.4}",
             rms(&err), srt[n / 2], srt[n * 9 / 10], srt[n - 1]);
    if let Some((a0, t_snap, t_gain, t_gend, a1)) = segment(&ps, &st, path) {
        for (name, lo, hi) in [("dive", a0, t_snap), ("snap", t_snap, t_gain),
                               ("flick", t_gain, t_gend), ("gain", t_gend, a1)] {
            if hi > lo { println!("    {name:<6} {:>3}t  RMS {:>8.4}  max {:>8.4}",
                                  hi - lo, rms(&err[lo..hi]), err[lo..hi].iter().fold(0.0f64, |m, x| m.max(x.abs()))) }
        }
    }
    if dump {
        println!("t,pitch,vy,vz,gamma,mu_y,mu_z,ratio,gap");
        for t in 0..n {
            println!("{t},{:.4},{:.6},{:.6},{:.4},{:.6},{:.6},{:.6},{:.4}", ps[t], st[t].vel.y, st[t].vel.z,
                     gamma(st[t].vel), mu[t + 1].0, mu[t + 1].1, mu[t + 1].1 / mu[t + 1].0, err[t]);
        }
    }
}

/// The climbing arc, as `(start, apex)` on a *doubled* tick index so an arc that straddles the
/// schedule's cut is still one run. The apex is where the down-to-forward branch switches on;
/// of the ticks that end such a run, take the one ending the longest, so a one-tick dropout
/// mid-climb cannot cut the arc short. `None` when the branch never switches at all.
fn climbing_arc(st: &[State], ps: &[f64]) -> Option<(usize, usize)> {
    let n = ps.len();
    let off = |t: usize| dive_branch_off(st[t % n].vel, ps[t % n]);
    let run = |t: usize| (1..n).map(|k| t - k).take_while(|&u| off(u - 1)).count();
    let t_apex = *(n + 1..=2 * n).filter(|&t| off(t - 1) && !off(t))
        .collect::<Vec<_>>().iter().max_by_key(|&&t| run(t))?;
    Some((t_apex - run(t_apex), t_apex))
}

/// The gain phase's lookahead rule, generalized from `dTE` to the cycle's own objective.
///
/// `argmax dTE over n held ticks` is elytrasim's own rule with the horizon stretched, and it
/// knows nothing about the price on distance: at `w != 0` it is maximizing the wrong thing. The
/// generalization holds the same pitch for the same `n` ticks and maximizes `dJ = dTE + w*dz`
/// instead, with the profile's own `w` -- see `bug_dj_n`, which is where the claim that this is
/// the *right* generalization is argued. Both are scored against the optimum's own pitches over
/// its climbing arc, at every lookahead `1..=nmax`, so the table says which `n` each rule wants
/// as well as how well it does there.
///
/// Three scoring sets, because which ticks count is a real choice and the answer moves with it:
/// `arc` is the whole climb, `up` drops the ticks where the optimum is not nose-up (the closed
/// form does not apply there and neither rule is being asked about them), and `free` further
/// drops the ticks where the optimum sits on its own `|pitch| <= limit` bound, where any
/// sufficiently nose-up rule agrees for free. `free` is the honest column.
fn cmd_djn(path: &str, w_over: Option<f64>, lim_over: Option<f64>, nmax: usize,
           dump: bool, wsweep: bool, equiv: bool) {
    let prof = Profile::parse(&std::fs::read_to_string(path)
                              .unwrap_or_else(|e| panic!("{path}: {e}"))).ok();
    let ps = read_pitches(path);
    let v0 = prof.as_ref().map_or(V0, |p| p.obj.v0);
    let w = w_over.or_else(|| prof.as_ref().map(|p| p.obj.w())).unwrap_or(0.0);
    let lam = prof.as_ref().map_or(f64::NAN, |p| p.obj.lambda);
    // A `cyclecut` is a bare list of pitches: it carries neither the price it was polished at
    // nor the control set, so both can be supplied on the command line.
    let limit = lim_over.or_else(|| prof.as_ref().map(|p| p.rough.limit)).unwrap_or(90.0);
    let st = replay_from(v0, &ps);
    let n = ps.len();
    let close = (st[n].vel - st[0].vel).length();
    let Some((t_gain, t_apex)) = climbing_arc(&st, &ps) else {
        println!("# {path}: no apex -- the down-to-forward branch never switches"); return };

    // which ticks of the arc each set keeps
    let up = |t: usize| ps[t % n] < -1e-9 && st[t % n].vel.z > 0.0;
    let free = |t: usize| up(t) && ps[t % n] > -(limit - 0.1);
    let sets: [(&str, Vec<usize>); 3] = [
        ("arc", (t_gain..t_apex).collect()),
        ("up", (t_gain..t_apex).filter(|&t| up(t)).collect()),
        ("free", (t_gain..t_apex).filter(|&t| free(t)).collect()),
    ];

    println!("# {path}");
    println!("# lambda {lam}  w {w:.10}  limit {limit}  ticks {n}  |v_N - v_0| {close:.2e}");
    println!("# arc {t_gain}..{t_apex} ({} ticks)  up {}  free {}",
             t_apex - t_gain, sets[1].1.len(), sets[2].1.len());
    if sets[2].1.is_empty() { println!("# no interior nose-up ticks; nothing to score"); return }

    // one `dj_argmax_all` per tick per rule serves every lookahead
    let arc: &[usize] = &sets[0].1;
    let rows: Vec<(usize, f64, Vec<f64>, Vec<f64>)> = arc.par_iter().map(|&t| {
        let s = &st[t % n];
        (t, ps[t % n], dj_argmax_all(s, nmax, w, limit), dj_argmax_all(s, nmax, 0.0, limit))
    }).collect();

    let stat = |e: &mut Vec<f64>| -> (f64, f64, f64) {
        let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
        let bias = e.iter().sum::<f64>() / e.len() as f64;
        let mut a: Vec<f64> = e.iter().map(|x| x.abs()).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (rms, a[a.len() / 2], bias)
    };
    if equiv {
        // Is the price a second horizon knob? The gain phase's pitch depends on the state only
        // through `v_z` and the price *ratio* (`gain_pitch`), so two rules that trace the same
        // ratio along the arc give the same pitches whatever their parameters say. For each
        // lookahead at the profile's own `w`, find the `dTE` lookahead whose pitches match it
        // best and report what is left over -- if that residual is far below the RMS either
        // rule scores against the optimum, the two knobs are the same knob.
        let free_t: Vec<usize> = sets[2].1.clone();
        let at_w = |wr: f64| -> Vec<Vec<f64>> {
            free_t.par_iter().map(|&t| dj_argmax_all(&st[t % n], nmax, wr, limit)).collect() };
        let (dj, dte) = (at_w(w), at_w(0.0));
        let rms = |a: &[Vec<f64>], i: usize, b: &[Vec<f64>], j: usize| -> f64 {
            (a.iter().zip(b).map(|(x, y)| (x[i - 1] - y[j - 1]).powi(2)).sum::<f64>()
             / a.len() as f64).sqrt() };
        println!("n_dj,best_n_dte,residual_deg,dj_vs_opt_deg");
        for k in 1..=nmax {
            let m = (1..=nmax).min_by(|&x, &y|
                rms(&dj, k, &dte, x).partial_cmp(&rms(&dj, k, &dte, y)).unwrap()).unwrap();
            let vs_opt = (dj.iter().zip(&free_t)
                .map(|(v, &t)| (v[k - 1] - ps[t % n]).powi(2)).sum::<f64>()
                / dj.len() as f64).sqrt();
            println!("{k},{m},{:.4},{vs_opt:.4}", rms(&dj, k, &dte, m));
        }
        return
    }
    if wsweep {
        // Is the profile's own `w` the one the rule wants? Sweep the rule's price over a grid
        // in lambda units and report, for each, the best lookahead and what it scores. If the
        // hypothesis is right the minimum sits at the cell's own lambda; if the curve is flat
        // the rule is not reading the price at all.
        let free_t: Vec<usize> = sets[2].1.clone();
        println!("lam_rule,w_rule,best_n,rms,med,bias,rms_at_20");
        for i in -24..=24 {
            let lr = 0.5 * i as f64;
            let wr = w_of_lambda(lr);
            let picks: Vec<Vec<f64>> = free_t.par_iter()
                .map(|&t| dj_argmax_all(&st[t % n], nmax, wr, limit)).collect();
            let score = |k: usize| {
                let e: Vec<f64> = picks.iter().zip(&free_t)
                    .map(|(v, &t)| v[k - 1] - ps[t % n]).collect();
                let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
                let bias = e.iter().sum::<f64>() / e.len() as f64;
                let mut a: Vec<f64> = e.iter().map(|x| x.abs()).collect();
                a.sort_by(|x, y| x.partial_cmp(y).unwrap());
                (rms, a[a.len() / 2], bias)
            };
            let bk = (1..=nmax).min_by(|&x, &y|
                score(x).0.partial_cmp(&score(y).0).unwrap()).unwrap();
            let (rms, med, bias) = score(bk);
            println!("{lr},{wr:.6},{bk},{rms:.4},{med:.4},{bias:+.4},{:.4}",
                     score(20.min(nmax)).0);
        }
        return
    }
    // The climb's last ten-odd ticks, where the pitch comes back to 0, are a separate story from
    // its body: the README's own bvp rule breaks down there too. Split the free set three
    // quarters / one quarter along the arc so the two cannot be confused.
    let cut = t_gain + 3 * (t_apex - t_gain) / 4;
    let early = |t: usize| free(t) && t < cut;
    let late = |t: usize| free(t) && t >= cut;
    println!("n,dj_rms,dj_med,dj_bias,dte_rms,dte_med,dte_bias,dj_rms_up,dte_rms_up,\
              dj_rms_arc,dte_rms_arc,dj_rms_early,dte_rms_early,dj_rms_late,dte_rms_late");
    for k in 1..=nmax {
        let err = |pick: &dyn Fn(&(usize, f64, Vec<f64>, Vec<f64>)) -> f64, keep: &dyn Fn(usize) -> bool|
            rows.iter().filter(|r| keep(r.0)).map(|r| pick(r) - r.1).collect::<Vec<f64>>();
        let dj = |r: &(usize, f64, Vec<f64>, Vec<f64>)| r.2[k - 1];
        let dte = |r: &(usize, f64, Vec<f64>, Vec<f64>)| r.3[k - 1];
        let (a, b, c) = stat(&mut err(&dj, &free));
        let (d, e, f) = stat(&mut err(&dte, &free));
        let (g, ..) = stat(&mut err(&dj, &up));
        let (h, ..) = stat(&mut err(&dte, &up));
        let (i, ..) = stat(&mut err(&dj, &|_| true));
        let (j, ..) = stat(&mut err(&dte, &|_| true));
        let nan = (f64::NAN, f64::NAN, f64::NAN);
        let (p, ..) = if rows.iter().any(|r| early(r.0)) { stat(&mut err(&dj, &early)) } else { nan };
        let (q, ..) = if rows.iter().any(|r| early(r.0)) { stat(&mut err(&dte, &early)) } else { nan };
        let (u, ..) = if rows.iter().any(|r| late(r.0)) { stat(&mut err(&dj, &late)) } else { nan };
        let (v, ..) = if rows.iter().any(|r| late(r.0)) { stat(&mut err(&dte, &late)) } else { nan };
        println!("{k},{a:.4},{b:.4},{c:+.4},{d:.4},{e:.4},{f:+.4},{g:.4},{h:.4},{i:.4},{j:.4},\
                  {p:.4},{q:.4},{u:.4},{v:.4}");
    }
    if dump {
        // per tick: what the optimum flew, and the lookahead at which each rule would have said it
        println!("# t,rel,pitch,vy,vz,dj_nstar,dte_nstar,dj_err20,dte_err20,free");
        for (t, p, dj, dte) in &rows {
            let s = &st[t % n];
            let nstar = |v: &Vec<f64>| (1..=nmax).min_by(|&a, &b|
                (v[a - 1] - p).abs().partial_cmp(&(v[b - 1] - p).abs()).unwrap()).unwrap();
            println!("{t},{},{p:.4},{:.6},{:.6},{},{},{:+.4},{:+.4},{}",
                     t - t_gain, s.vel.y, s.vel.z, nstar(dj), nstar(dte),
                     dj[19.min(nmax - 1)] - p, dte[19.min(nmax - 1)] - p, free(*t) as u8);
        }
    }
}

/// The gain phase, in closed form: two clocks, one corner, and one number handed over.
///
/// The Jacobian is triangular on each arc of the cycle, in complementary directions:
///
///   * where the down-to-forward branch is off (`v_y` high enough), `d v_z'/d v_y = 0`, so
///     `mu_y` obeys `mu_y,t = 1 + DRAG_Y mu_y,t+1` -- autonomous, state- and control-free. It
///     is a clock: `mu_y = 50 - (50 - mu_y(apex)) DRAG_Y^(T-t)`, with `50 = 1/(1 - DRAG_Y)`.
///     The price of upward velocity is the discounted count of ticks left before you stop
///     climbing.
///   * where the pitch is not nose-up, `d v_y'/d v_z = 0`, so `mu_z,t = w + DRAG_Z mu_z,t+1` --
///     the same thing one axis over, running at `DRAG_Z` through the whole dive and hold.
///
/// Only the flick has neither. So the gain phase's pitch, which is `gain_pitch(v_z, mu)`, needs
/// exactly two non-local numbers: the apex (which its own trajectory sets) and `kappa =
/// mu_z(apex)`, the price the dive will pay for the forward speed it is handed. `gain --bvp`
/// scores the rule that follows -- maximize `sum (v_y + w v_z)` to the apex plus `kappa v_z`
/// there -- against the cycle's own pitches.
fn cmd_gain(path: &str, w: Option<f64>, bvp: bool, dump: bool) {
    let prof = Profile::parse(&std::fs::read_to_string(path)
                              .unwrap_or_else(|e| panic!("{path}: {e}"))).ok();
    let ps = read_pitches(path);
    let v0 = prof.as_ref().map_or(V0, |p| p.obj.v0);
    let w = w.or_else(|| prof.as_ref().map(|p| p.obj.w())).unwrap_or(0.0);
    let st = replay_from(v0, &ps);
    let n = ps.len();
    let close = (st[n].vel - st[0].vel).length();
    let Some(mu) = periodic_mu(&st, &ps, w) else {
        println!("{path}: monodromy singular, mu undetermined"); return };
    // Work on a doubled index so an arc that straddles the cut is still one run.
    let at = |t: usize| (t % n, &st[t % n], ps[t % n], mu[t % n + 1]);
    let off = |t: usize| { let (_, s, p, _) = at(t); dive_branch_off(s.vel, p) };
    let Some((t_gain, t_apex)) = climbing_arc(&st, &ps) else {
        println!("{path}: no apex -- the down-to-forward branch never switches"); return };
    let (mu_y_t, kappa) = (mu[t_apex % n].0, mu[t_apex % n].1);
    println!("{path}: {n} ticks, |v_N - v_0| = {close:.2e}, w = {w}");
    if close > 1e-3 { println!("  !! not a closed cycle; the periodic costate does not apply") }
    println!("  climbing arc {t_gain}..{t_apex} ({} ticks), apex prices mu = ({mu_y_t:.4}, {kappa:.4})",
             t_apex - t_gain);

    // the two clocks, as residuals against the recursion they are supposed to satisfy
    let (mut cy, mut cz) = (vec![], vec![]);
    for t in 0..n {
        let (_, s, p, m1) = at(t);
        let m0 = mu[t];
        if dive_branch_off(s.vel, p) { cy.push(((m0.0 - (1.0 + DRAG_Y * m1.0)) / m0.0.abs().max(1e-9)).abs()) }
        if p >= 0.0 { cz.push(((m0.1 - (w + DRAG_Z * m1.1)) / m0.1.abs().max(1e-9)).abs()) }
    }
    let med = |mut v: Vec<f64>| -> f64 { if v.is_empty() { return f64::NAN }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    println!("  clock residuals: mu_y over {:>3} branch-off ticks {:.1e}   mu_z over {:>3} \
              nose-down ticks {:.1e}", cy.len(), med(cy), cz.len(), med(cz));

    // the closed-form stationary pitch, and the countdown two consecutive pitches read out
    let interior = |t: usize| { let (_, s, p, _) = at(t);
        off(t) && p < -1e-9 && p > -89.9 && s.vel.z > 0.0 };
    let (mut gap, mut miss) = (vec![], vec![]);
    let mut rows = vec![];
    for t in t_gain..t_apex {
        if !interior(t) { continue }
        let (_, s, p, m1) = at(t);
        gap.push(gain_pitch(s.vel.z, m1) - p);
        let (r0, tau_true) = (gain_ratio(s.vel.z, p), (t_apex - t - 1) as f64);
        let tau = if interior(t + 1) {
            let (_, _, p1, _) = at(t + 1);
            let s1 = -p1.to_radians().sin();
            let k = (GAIN_UP * s1 + DRAG_Z * (1.0 - 0.036 * s1) * gain_ratio(at(t + 1).1.vel.z, p1)) / DRAG_Y;
            let m = (w - k) / (r0 - k);
            // mu_y = mu_inf - (mu_inf - mu_y(apex)) DRAG_Y^tau, inverted for tau
            let mu_inf = 1.0 / (1.0 - DRAG_Y);
            let tau = ((mu_inf - m) / (mu_inf - mu_y_t)).ln() / DRAG_Y.ln();
            if tau.is_finite() { miss.push(tau - tau_true) }
            tau
        } else { f64::NAN };
        rows.push((t, p, s.vel.y, s.vel.z, m1.0, m1.1, r0, tau, tau_true));
    }
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    println!("  stationary pitch over {} interior ticks: RMS {:.4}deg  median {:.4}",
             gap.len(), rms(&gap), med(gap.iter().map(|x| x.abs()).collect()));
    if !miss.is_empty() {
        let inside: f64 = miss.iter().filter(|x| x.abs() < 1.0).count() as f64 / miss.len() as f64;
        let mut v = miss.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // the level of the miss is mu_y(apex) and is fed back in above; what the clock claims
        // is that the miss does not move along the arc, so the spread is the falsifiable part
        println!("  two-tick countdown to the apex over {} pairs: median miss {:+.2} ticks, \
                  {:.0}% inside one tick, p10..p90 spread {:.2}",
                 miss.len(), med(miss.clone()), 100.0 * inside,
                 v[v.len() * 9 / 10] - v[v.len() / 10]);
    }
    // the gain phase ends where the hold ends, on the same corner: mu_z/mu_y = GAIN_RATE
    let t_corner = (t_gain..t_apex).find(|&t| { let (_, _, _, m) = at(t);
        m.0 > 0.0 && m.1 / m.0 >= GAIN_RATE });
    let t_flat = (t_gain..t_apex).find(|&t| at(t).2 > -1e-9);   // the optimizer's zero is not exactly 0
    match (t_corner, t_flat) {
        (Some(a), Some(b)) => println!("  nose-up ends at {b}, the corner mu_z/mu_y = {GAIN_RATE:.6} \
                                        fires at {a} ({:+} ticks)", a as i64 - b as i64),
        _ => println!("  the gain phase does not return to pitch 0 inside the cycle"),
    }

    if bvp {
        for (lab, mt) in [("measured apex prices", (mu_y_t, kappa)),
                          ("mu_y(apex) forced to 0", (0.0, kappa)),
                          ("both forced to 0", (0.0, 0.0))] {
            let err: Vec<f64> = rows.iter().map(|&(t, p, ..)| {
                gain_bvp(&st[t % n], w, mt, p, 400).0 - p }).collect();
            if err.is_empty() { continue }
            println!("  climb-to-apex rule, {lab:<22}: RMS {:.4}deg  median {:.4}  max {:.4}",
                     rms(&err), med(err.iter().map(|x| x.abs()).collect()),
                     err.iter().fold(0.0f64, |m, x| m.max(x.abs())));
        }
        {
            let err: Vec<f64> = rows.iter().map(|&(_t, p, ..)| p).collect();
            if !err.is_empty() {
            let base: Vec<f64> = rows.iter().map(|&(t, p, ..)| bug_dte_n(&st[t % n], 20) - p).collect();
            println!("  for comparison, argmax dTE over 20 ticks  : RMS {:.4}deg  median {:.4}",
                     rms(&base), med(base.iter().map(|x| x.abs()).collect()));
            }
        }
    }
    if dump {
        println!("t,pitch,vy,vz,mu_y,mu_z,r,tau_2tick,tau_true,bvp_err,bvp_n,dte20_err");
        for (t, p, vy, vz, my, mz, r, tau, tt) in rows {
            let (q, hz) = if bvp { gain_bvp(&st[t % n], w, (mu_y_t, kappa), p, 400) } else { (p, 0) };
            println!("{t},{p:.4},{vy:.6},{vz:.6},{my:.6},{mz:.6},{r:.6},{tau:.4},{tt},{:.4},{hz},{:.4}",
                     q - p, if bvp { bug_dte_n(&st[t % n], 20) - p } else { 0.0 });
        }
    }
}

/// The climb's own two-point problem: from `s`, choose pitches until the apex to maximize
/// `sum (v_y + w v_z)` plus `kappa v_z` at the apex. Forward-backward sweep; returns the first
/// pitch and the apex it settled on. There is no lookahead constant -- the horizon is the apex,
/// which the candidate trajectory locates itself. `mu_t` is the apex price vector: the climb
/// cannot derive it, because it is what the dive will pay for what the climb hands over.
pub fn gain_bvp(s: &State, w: f64, mu_t: (f64, f64), p_init: f64, cap: usize) -> (f64, usize) {
    let c = (1.0, w);
    let mut ps = vec![p_init; cap];
    let mut n = cap;
    for _ in 0..80 {
        let mut st = vec![s.clone()];
        n = cap;
        for k in 0..cap {
            // the arc ends where the down-to-forward branch switches on -- the same event that
            // bounds `climbing arc`, not the slightly later v_y <= 0
            if k > 0 && !dive_branch_off(st[k].vel, ps[k]) { n = k; break }
            st.push(ticked(&st[k], ps[k]));
        }
        let mut mu = vec![(0.0, 0.0); n + 1];
        mu[n] = mu_t;
        for k in (0..n).rev() {
            let j = jac(st[k].vel, ps[k]);
            let am = mt_vec(&j, mu[k + 1]);
            mu[k] = (c.0 + am.0, c.1 + am.1);
        }
        let mut worst: f64 = 0.0;
        for k in 0..n {
            let q = argmax(|p| { let f = update_fall_flying_movement(st[k].vel, rot(p));
                                 mu[k + 1].0 * f.y + mu[k + 1].1 * f.z }, 0.03125);
            worst = worst.max((q - ps[k]).abs());
            ps[k] = 0.5 * ps[k] + 0.5 * q;      // damped; the undamped sweep oscillates
        }
        if worst < 1e-4 { break }
    }
    (ps[0], n)
}

/// The climb flown by `gain_law_pitch` instead of by search, against the cycle's own climb.
///
/// The window runs from the tick the optimum comes off its nose-up stop to the tick it starts
/// pitching down for the entry, or the end of the climbing arc if that comes first. Every rule
/// flies the window closed-loop from the optimum's own state there, and what it hands the entry
/// is priced with the periodic costate at the window's end, so the cost is first-order in blocks
/// of the cycle's own `J` and the accounting is the same for every rule. `phi` is the law read
/// backwards off the optimum's own pitches; its being flat through the body is the claim.
///
/// `--k <x>` flies a `k` of your choosing alongside `gain_law_k(w)`, which is how a marker built for
/// one objective is scored on another's cycle. `v_c`, where the optimum's pitch first returns to
/// 0, is the cycle's own number: the row that uses it is an oracle for the end, not a rule.
fn cmd_gainlaw(path: &str, w: Option<f64>, k_over: Option<f64>, lim_over: Option<f64>, dump: bool) {
    let prof = Profile::parse(&std::fs::read_to_string(path)
                              .unwrap_or_else(|e| panic!("{path}: {e}"))).ok();
    let ps = read_pitches(path);
    let v0 = prof.as_ref().map_or(V0, |p| p.obj.v0);
    let w = w.or_else(|| prof.as_ref().map(|p| p.obj.w())).unwrap_or(0.0);
    let limit = lim_over.or_else(|| prof.as_ref().map(|p| p.rough.limit)).unwrap_or(90.0);
    let st = replay_from(v0, &ps);
    let n = ps.len();
    let Some(mu) = periodic_mu(&st, &ps, w) else {
        println!("{path}: monodromy singular, mu undetermined"); return };
    let Some((t_gain, t_apex)) = climbing_arc(&st, &ps) else {
        println!("{path}: no apex -- the down-to-forward branch never switches"); return };
    let (p_at, v_at) = (|t: usize| ps[t % n], |t: usize| st[t % n].vel);
    let t0 = (t_gain..t_apex).rev().find(|&t| p_at(t) <= -limit + 0.01).map_or_else(
        || (t_gain..t_apex).min_by(|&a, &b| p_at(a).total_cmp(&p_at(b))).unwrap(), |t| t + 1);
    let te = (t0..t_apex).find(|&t| p_at(t) > 1.0).unwrap_or(t_apex);
    let v_c = (t0..te).find(|&t| p_at(t) > -0.5).map(|t| v_at(t).z);
    let k = gain_law_k(w);
    let c = (1.0, w);
    let j_opt: f64 = (0..n).map(|t| c.0 * st[t + 1].vel.y + c.1 * st[t + 1].vel.z).sum();

    let body: Vec<f64> = (t0..te).filter(|&t| v_at(t).z > 0.6 && p_at(t) < -0.5)
        .map(|t| gain_law_phi(v_at(t).z, p_at(t))).collect();
    let pct = |v: &[f64], q: f64| { let mut v = v.to_vec(); v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q).round() as usize] };
    println!("{path}: {n} ticks, w = {w}, k = {k:.4} ({GAIN_LAW_K0} {GAIN_LAW_DKDW:+.4} w)");
    println!("  climb window {t0}..{te} ({} ticks): off the {limit} stop at v_z {:.3}, corner {}",
             te - t0, v_at(t0).z, v_c.map_or("never".into(), |v| format!("at v_z {v:.3}")));
    if !body.is_empty() {
        println!("  phi = s(s + {GAIN_LAW_A:.4})/v_z over the body (v_z > 0.6, {} ticks): median {:.4}  \
                  p10..p90 {:.4}..{:.4}", body.len(), pct(&body, 0.5), pct(&body, 0.1), pct(&body, 0.9));
    }

    // fly `rule` over the window from the optimum's own state; the entry prices what it hands over
    let fly = |rule: &dyn Fn(&State) -> f64| -> (f64, f64) {
        let mut s = st[t0 % n].clone();
        let (mut dj, mut e2) = (0.0, 0.0);
        for t in t0..te {
            let p = rule(&s).clamp(-limit, limit);
            e2 += (p - p_at(t)).powi(2);
            s = ticked(&s, p);
            if t + 1 < te {
                let o = v_at(t + 1);
                dj += c.0 * (s.vel.y - o.y) + c.1 * (s.vel.z - o.z);
            }
        }
        let (m, o) = (mu[te % n], v_at(te));
        dj += m.0 * (s.vel.y - o.y) + m.1 * (s.vel.z - o.z);
        (dj, (e2 / (te - t0) as f64).sqrt())
    };
    let law = move |kk: f64| move |s: &State| gain_law_pitch(s.vel.z, kk);
    let mut rows: Vec<(String, Box<dyn Fn(&State) -> f64>)> = vec![
        (format!("law, k(w) = {k:.4}"), Box::new(law(k))),
    ];
    if let Some(vc) = v_c {
        rows.push((format!("law, k(w), 0 below v_c"),
                   Box::new(move |s: &State| if s.vel.z < vc { 0.0 } else { gain_law_pitch(s.vel.z, k) })));
    }
    if !body.is_empty() {
        let kb = pct(&body, 0.5);
        rows.push((format!("law, the body's own k = {kb:.4}"), Box::new(law(kb))));
    }
    if let Some(ko) = k_over { rows.push((format!("law, k = {ko:.4}"), Box::new(law(ko)))) }
    rows.push(("argmax dTE, 20 ticks held".into(), Box::new(move |s: &State| bug_dj_n(s, 20, 0.0, limit))));
    println!("  flown closed-loop over the window, what it hands the entry priced at mu(entry):");
    println!("    {:<34} {:>9} {:>13} {:>11}", "rule", "pitch RMS", "cost, blocks", "% of J");
    for (lab, rule) in &rows {
        let (dj, rms) = fly(rule.as_ref());
        println!("    {lab:<34} {rms:>8.2}d {dj:>+13.4} {:>+10.3}%", 100.0 * dj / j_opt);
    }
    if dump {
        println!("t,pitch,vy,vz,phi,law,dte20");
        for t in t0..te {
            let s = &st[t % n];
            println!("{t},{:.4},{:.6},{:.6},{:.5},{:.4},{:.4}", p_at(t), s.vel.y, s.vel.z,
                     gain_law_phi(s.vel.z, p_at(t)), gain_law_pitch(s.vel.z, k), bug_dj_n(s, 20, 0.0, limit));
        }
    }
}

/// How sharply does the one-tick score pick out the optimum's pitch?
///
/// Pontryagin says the optimum maximizes `mu . f(v, p)` over p every tick. That is exact, but it
/// only *determines* the pitch if the score has curvature at its maximum. Where the score is flat
/// the condition is satisfied by a whole range of pitches and the maximum principle says nothing
/// -- a singular arc -- and the control has to come from somewhere else, typically a feedback law
/// in the state. Which is exactly the shape of "hold the flight-path angle".
///
/// Uses the per-tick tangency direction, so `p_t` is stationary by construction and the curvature
/// there is clean. Reports `d^2 S/dp^2` in degrees^-2 and the half-width over which the score
/// stays within 1e-6 of its maximum.
fn cmd_singular(path: &str) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let n = ps.len();
    let mut curv = vec![0.0; n];
    let mut half = vec![0.0; n];
    for t in 0..n {
        let (uy, uz) = costate_dir(st[t].vel, ps[t]);
        let sc = |p: f64| { let f = update_fall_flying_movement(st[t].vel, rot(p)); uy * f.y + uz * f.z };
        let (h, s0) = (0.5, sc(ps[t]));
        curv[t] = (sc(ps[t] + h) - 2.0 * s0 + sc(ps[t] - h)) / (h * h);
        let (mut lo, mut hi) = (0.0, 90.0);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if s0 - sc(ps[t] + m).max(sc(ps[t] - m)) < 1e-6 { lo = m } else { hi = m }
        }
        half[t] = 0.5 * (lo + hi);
    }
    println!("{path}: curvature of the one-tick score at the optimum's own pitch");
    println!("{:>7} {:>5} {:>12} {:>12}", "phase", "t", "d2S/dp2", "half-width");
    let show = |name: &str, lo: usize, hi: usize| {
        let c: Vec<f64> = curv[lo..hi].iter().map(|x| x.abs()).collect();
        let w: Vec<f64> = half[lo..hi].to_vec();
        let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        println!("{name:>7} {:>5} {:>12.2e} {:>10.2}deg", hi - lo, med(c), med(w));
    };
    if let Some((a0, t_snap, t_gain, t_gend, a1)) = segment(&ps, &st, path) {
        show("dive", a0, t_snap); show("snap", t_snap, t_gain);
        show("flick", t_gain, t_gend); show("gain", t_gend, a1);
    } else {
        show("all", 0, n);
        for (name, lo, hi) in [("entry", 1, 20), ("entry30", 1, 30), ("dive", 20, 190),
                               ("snap", 190, 205), ("flick", 205, 213), ("gain", 213, 295)] {
            show(name, lo, hi);
        }
    }
}

/// What does a pitch error at tick t actually cost the cycle?
///
/// The curvature of `mu . f` is a statement about a linear score in abstract units. This is the
/// operational version: nudge one tick's pitch, let the schedule re-converge to its own limit
/// cycle, and read the change in climb rate. Answers "does the pitch at this tick matter", in
/// blocks per second, with no theory in between.
fn cmd_sens(path: &str, delta: f64, dump: bool) {
    let base = read_pitches(path);
    let n = base.len();
    let rate = |ps: &[f64]| -> f64 {
        let mut v = V0;
        for _ in 0..40 { for &p in ps { v = update_fall_flying_movement(v, rot(p)) } }
        let mut s = State { pos: Vec3::ZERO, vel: v };
        for &p in ps { s = ticked(&s, p) }
        s.pos.y / n as f64 * 20.0
    };
    let base_rate = rate(&base);
    println!("{path}: climb {base_rate:.5} b/s; cost of a {delta}deg pitch error at one tick");
    let mut cost = vec![0.0; n];
    for t in 0..n {
        let mut up = base.clone(); up[t] += delta;
        let mut dn = base.clone(); dn[t] -= delta;
        // symmetric part: the second-order cost, which is what "flat" is really asking about
        cost[t] = base_rate - 0.5 * (rate(&up) + rate(&dn));
    }
    let show = |name: &str, lo: usize, hi: usize| {
        let mut v: Vec<f64> = cost[lo..hi].iter().cloned().collect();
        let sum: f64 = v.iter().sum();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let _ = sum;
        // a correlated shift of the whole phase is NOT the sum of the one-tick costs -- the cross
        // terms dominate -- so shift it for real and re-converge
        let corr = |d: f64| { let mut q = base.clone(); for t in lo..hi { q[t] += d } rate(&q) };
        println!("{name:>11} {:>4}t | one tick: median {:>9.2e} max {:>9.2e} | whole phase off by \
                  {delta}deg: {:>+8.4} / {:>+8.4} b/s",
                 hi - lo, v[v.len() / 2], v[v.len() - 1], corr(delta) - base_rate, corr(-delta) - base_rate);
    };
    let st = replay(&base);
    let v_top = (0..n).map(|t| st[t].vel.length()).fold(0.0, f64::max);
    let t_snap = (30..n - 3).find(|&t| base[t] < 5.0 && base[t + 1] < 5.0 && base[t + 2] < 5.0
                                    && st[t].vel.length() > 0.8 * v_top).unwrap_or(n * 2 / 3);
    let t_gain = (t_snap..n).find(|&t| st[t].vel.y > 0.0).unwrap_or(t_snap);
    let t_gend = (t_gain + 10..n).find(|&t| base[t] > 0.0).unwrap_or(n);
    show("dive", 20, t_snap); show("snap+flick", t_snap, t_gain); show("gain", t_gain, t_gend);
    if dump {
        println!("t,pitch,gamma,cost");
        for t in 0..n { println!("{t},{:.4},{:.3},{:.6e}", base[t], gamma(st[t].vel), cost[t]) }
    }
}

/// One tick, spelled out: how next tick's velocity swings with pitch, and where the price sits.
fn cmd_swing(path: &str, t: usize) {
    let ps = read_pitches(path);
    let st = replay(&ps);
    let (v, p) = (st[t].vel, ps[t]);
    println!("tick {t}: v = ({:.5}, {:.5}), gamma {:.2}, pitch {p:.3}", v.y, v.z, gamma(v));
    println!("{:>8} {:>11} {:>11}", "pitch", "next v_y", "next v_z");
    for d in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let f = update_fall_flying_movement(v, rot(p + d));
        println!("{:>8.3} {:>11.6} {:>11.6}", p + d, f.y, f.z);
    }
    let h = 1e-3;
    let (a, b) = (update_fall_flying_movement(v, rot(p - h)), update_fall_flying_movement(v, rot(p + h)));
    let (dy, dz) = ((b.y - a.y) / (2.0 * h), (b.z - a.z) / (2.0 * h));
    println!("df/dp = ({dy:.6}, {dz:.6}) per degree  -- the direction next velocity slides as you pitch");
    let (uy, uz) = costate_dir(v, p);
    println!("price = ({uy:.6}, {uz:.6})  -- perpendicular to it: dot = {:.3e}", uy * dy + uz * dz);
}

/// Print one cycle of a schedule: the middle cycle of a 3x tiling, or the policy's limit cycle.
/// Both are needed as controls for `consist`, which requires a genuinely closed cycle.
fn cmd_cyclecut(what: &str) {
    let (ps, st, tag): (Vec<f64>, Vec<State>, &str) = if what == "policy" {
        let par = P { g_star: 17.73, k: 0.055, s_switch: 2.40, vy_flick: -0.260,
                      s_exit: 0.21, slew: 12.7, p_push: 23.0, p_flick: -88.5, n_gain: 20, dive: Dive::Leak };
        let (ps, _, st) = fly(par, 2000);
        (ps, st, "policy")
    } else {
        let ps = read_pitches(what);
        let st = replay(&ps);
        (ps, st, "file")
    };
    // last two apexes, so the policy has settled onto its limit cycle
    let apex: Vec<usize> = (1..ps.len() - 1).filter(|&t| st[t].vel.y > 0.0 && st[t + 1].vel.y <= 0.0).collect();
    let k = apex.len();
    if k < 3 { eprintln!("{tag}: {k} apexes, need 3"); return }
    let (a0, a1) = (apex[k - 2], apex[k - 1]);
    eprintln!("{tag}: cycle {a0}..{a1} ({} ticks)", a1 - a0);
    for t in a0..a1 { println!("{}", ps[t]) }
}

/// Is the price vector's consistency special to the optimum, or does any profile have it?
///
/// Every schedule has, at each tick, *some* direction making its pitch stationary -- just take
/// the normal to `df/dp`. That part is vacuous. What is not vacuous is whether the directions
/// found tick by tick are mutually consistent: the true costate must satisfy
/// `mu_t = c + A_t^T mu_{t+1}` and close on itself around the cycle, which leaves no freedom at
/// all. So solve for `mu` from the recursion alone and then measure, at every tick, the angle
/// between it and the perpendicular to `df/dp`. Zero for an optimum; the control is a
/// deliberately perturbed schedule, re-closed into its own limit cycle.
///
/// The angle is the right metric rather than the argmax gap, because the one-tick score is very
/// flat in the dive (see `singular`) and a fraction of a percent in `mu` throws the argmax by ten
/// degrees there without meaning anything.
fn cmd_consist(path: &str, amp: f64) {
    let tag = path.rsplit('/').next().unwrap().trim_end_matches(".txt");
    let base = read_pitches(path);
    let n = base.len();
    let ps: Vec<f64> = base.iter().enumerate()
        .map(|(t, &p)| p + amp * (std::f64::consts::TAU * t as f64 / n as f64).sin()).collect();
    // the cycle map is strongly contracting, so iterating the schedule lands on its limit cycle
    let mut v = V0;
    for _ in 0..60 { for &p in &ps { v = update_fall_flying_movement(v, rot(p)) } }
    let mut st = vec![State { pos: Vec3::ZERO, vel: v }];
    for &p in &ps { let s = ticked(st.last().unwrap(), p); st.push(s) }
    let close = (st[n].vel - st[0].vel).length();
    let c = (1.0, 0.0);

    let (mut b, mut m) = ((0.0, 0.0), [1.0, 0.0, 0.0, 1.0]);
    for t in (0..n).rev() {
        let a = jac(st[t].vel, ps[t]);
        let ab = mt_vec(&a, b);
        b = (c.0 + ab.0, c.1 + ab.1);
        m = mt_mat(&a, &m);
    }
    let d = [1.0 - m[0], -m[1], -m[2], 1.0 - m[3]];
    let det = d[0] * d[3] - d[1] * d[2];
    let mut mu = vec![(0.0, 0.0); n + 1];
    mu[n] = ((d[3] * b.0 - d[1] * b.1) / det, (-d[2] * b.0 + d[0] * b.1) / det);
    for t in (0..n).rev() {
        let a = jac(st[t].vel, ps[t]);
        let am = mt_vec(&a, mu[t + 1]);
        mu[t] = (c.0 + am.0, c.1 + am.1);
    }

    // angle between mu_{t+1} and the perpendicular to df/dp: the stationary residual
    let mut res = vec![0.0; n];
    for t in 0..n {
        let h = 1e-3;
        let (a, bb) = (update_fall_flying_movement(st[t].vel, rot(ps[t] - h)),
                       update_fall_flying_movement(st[t].vel, rot(ps[t] + h)));
        let (dy, dz) = ((bb.y - a.y) / (2.0 * h), (bb.z - a.z) / (2.0 * h));
        let (my, mz) = mu[t + 1];
        let cosang = (my * dy + mz * dz).abs() / (my.hypot(mz) * dy.hypot(dz));
        res[t] = cosang.min(1.0).asin().to_degrees();
    }
    let med = |lo: usize, hi: usize| {
        if hi <= lo { return f64::NAN }
        let mut v: Vec<f64> = res[lo..hi].to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // the cycle starts at an apex, so segment it in place rather than needing a 3x tiling
    let v_top = (0..n).map(|t| st[t].vel.length()).fold(0.0, f64::max);
    let t_snap = (30..n - 3).find(|&t| ps[t] < 5.0 && ps[t + 1] < 5.0 && ps[t + 2] < 5.0
                                    && st[t].vel.length() > 0.8 * v_top).unwrap_or(n * 2 / 3);
    let t_gain = (t_snap..n).find(|&t| st[t].vel.y > 0.0).unwrap_or(t_snap);
    let t_gend = (t_gain + 10..n).find(|&t| ps[t] > 0.0).unwrap_or(n);
    println!("{tag:>11} amp {amp:>4.1} | climb {:>7.4} b/s | closure {close:>8.1e} | median degrees off perpendicular: \
              all {:>6.3} | dive({:>3}) {:>6.3} | snap+flick({:>2}) {:>6.3} | gain({:>3}) {:>6.3}",
             st[n].pos.y / n as f64 * 20.0, med(0, n),
             t_snap - 20, med(20, t_snap), t_gain - t_snap, med(t_snap, t_gain),
             t_gend - t_gain, med(t_gain, t_gend));
}

/// Two things the flatness result raises but does not answer.
///
/// `sides`: the one-tick sensitivity in `sens` is a symmetric second difference, which hides
/// whether the two directions cost the same. At the snap the pitch sits a tenth of a degree above
/// zero, and zero is a corner of the physics -- the forward-to-up conversion is gated on
/// `lean_angle < 0.0` -- so there is every reason to expect one side to be free and the other a
/// cliff.
///
/// `ksweep`: independent per-tick jitter in the dive costs nothing, but `k` is not per-tick
/// jitter, it sets the dive's whole trend. Sweeps it with the other constants fixed, and again
/// with the two that could absorb it re-tuned, which is the test of whether it is a real degree
/// of freedom or a redundant one.
fn cmd_sides(path: &str, delta: f64) {
    let base = read_pitches(path);
    let n = base.len();
    let rate = |ps: &[f64]| -> f64 {
        let mut v = V0;
        for _ in 0..40 { for &p in ps { v = update_fall_flying_movement(v, rot(p)) } }
        let mut s = State { pos: Vec3::ZERO, vel: v };
        for &p in ps { s = ticked(&s, p) }
        s.pos.y / n as f64 * 20.0
    };
    let r0 = rate(&base);
    let st = replay(&base);
    println!("{path}: climb {r0:.5} b/s; cost of {delta}deg at one tick, each side separately");
    println!("{:>5} {:>8} {:>8} {:>13} {:>13}", "t", "pitch", "gamma", "nose down", "nose up");
    for t in 0..n {
        let interesting = (190..215).contains(&t) || t % 30 == 0;
        if !interesting { continue }
        let (mut up, mut dn) = (base.clone(), base.clone());
        up[t] += delta; dn[t] -= delta;
        println!("{t:>5} {:>8.3} {:>8.2} {:>13.2e} {:>13.2e}",
                 base[t], gamma(st[t].vel), rate(&up) - r0, rate(&dn) - r0);
    }
}

/// Where in the dive does the leak actually earn its keep?
///
/// Josie's hypothesis: most of the difference between leak rates is spent early, around the
/// stretch where the flight-path angle is still being dragged down and the delta-TE family is
/// bimodal. If so, the leak is an *entry* correction rather than a rule for the whole dive, and
/// an exact hold -- no constant at all -- should be nearly free once something else handles the
/// first stretch.
///
/// Hands the first T ticks of every dive to the optimum's own pitches, then lets the rule take
/// over, and retunes the switch speed for each cell so no rule is judged on constants tuned for
/// another. `g_star` stays pinned to the derived ceiling throughout: it is the fastest steady
/// glide's flight-path angle, which the physics fixes, and letting a tuner move it would turn a
/// derived constant into a fitted one.
fn cmd_prefix(path: &str) {
    const TICKS: usize = 1500;
    let opt = read_pitches(path);
    let g_star = ceiling().1;
    let ts = [0usize, 5, 10, 20, 30, 40, 60, 80, 120];
    let rules: [(&str, Dive, f64); 4] = [("hold current", Dive::Hold, f64::NAN),
                                         ("target g*", Dive::Target, f64::NAN),
                                         ("floor clamp", Dive::Floor, f64::NAN),
                                         ("leak k=.04", Dive::Leak, 0.040)];
    let rate = |par: P, pre: &[f64]| fly_pre(par, TICKS, pre).2[TICKS].pos.y / TICKS as f64 * 20.0;
    // best climb over the switch speed: a coarse global grid, because a single seed gets stuck,
    // then a shrinking-step refine
    let best = |par: P, pre: &[f64]| -> (f64, f64) {
        let (mut bv, mut bs) = (f64::NEG_INFINITY, par.s_switch);
        for i in 0..=32 {
            let mut q = par; q.s_switch = 1.4 + 0.05 * i as f64;
            let v = rate(q, pre);
            if v > bv { bv = v; bs = q.s_switch }
        }
        let mut step = 0.05;
        for _ in 0..25 {
            let mut improved = false;
            for d in [-1.0f64, 1.0] {
                let mut q = par; q.s_switch = (bs + d * step).clamp(1.2, 3.2);
                let v = rate(q, pre);
                if v > bv { bv = v; bs = q.s_switch; improved = true }
            }
            if !improved { step *= 0.6 }
        }
        (bv, bs)
    };
    // the (prefix, rule) matrix is embarrassingly parallel
    let cells: Vec<(usize, usize)> = (0..ts.len()).flat_map(|a| (0..rules.len()).map(move |b| (a, b))).collect();
    let out: Vec<f64> = cells.par_iter()
        .map(|&(a, b)| {
            let (_, dive, k) = rules[b];
            let par = P { g_star, k, s_switch: 2.40, vy_flick: -0.260, s_exit: 0.21,
                          slew: 12.7, p_push: 23.0, p_flick: -88.5, n_gain: 20, dive };
            best(par, &opt[..ts[a]]).0
        })
        .collect();

    println!("{path}: dive prefix flown open-loop from the optimum, rule takes over after");
    println!("g_star pinned at {g_star:.4}; s_switch retuned for every cell");
    println!("hold / target / floor are all parameter-free; the leak is the reference");
    print!("{:>7}", "prefix");
    for (n, _, _) in rules { print!(" {n:>13}") }
    println!(" {:>12}", "leak - hold");
    for (a, t) in ts.iter().enumerate() {
        print!("{t:>7}");
        for b in 0..rules.len() { print!(" {:>13.4}", out[a * rules.len() + b]) }
        println!(" {:>12.4}", out[a * rules.len() + 3] - out[a * rules.len()]);
    }
}

fn cmd_ksweep() {
    let ticks = 1500;
    // g_star is NOT free: it is the flight-path angle of the fastest steady glide, which the
    // physics fixes. Letting a tuner move it turns a derived constant into a fitted one and
    // makes k look redundant when it is only being absorbed.
    let g_star = ceiling().1;
    let base = P { g_star, k: 0.055, s_switch: 2.40, vy_flick: -0.260,
                   s_exit: 0.21, slew: 12.7, p_push: 23.0, p_flick: -88.5, n_gain: 20, dive: Dive::Leak };
    println!("g_star pinned at the derived ceiling, {g_star:.4} deg; only s_switch is retuned");
    println!("{:>7} {:>13} {:>13} {:>9}", "k", "as tuned", "retuned", "s_switch");
    for k in [0.005, 0.01, 0.02, 0.03, 0.04, 0.055, 0.07, 0.10, 0.15, 0.25, 0.50, 1.00] {
        let mut p = base; p.k = k;
        let asis = rate_of(p, ticks);
        // Let the two constants that could stand in for the leak try to absorb the change.
        // Coordinate descent from a single seed is not enough: from the k = 0.055 constants it
        // gets stuck against the s_switch clamp for k = 0.15 and 0.25 while k = 0.5 finds 1.28,
        // which is an optimizer artifact and not the leak rate failing. Seed on a grid instead.
        let mut q = p;
        let mut best = f64::NEG_INFINITY;
        for si in 0..=32 {
            let mut r = p;
            r.s_switch = 1.4 + 0.05 * si as f64;
            let v = rate_of(r, ticks);
            if v > best { best = v; q = r }
        }
        let mut step = 0.05;
        for _ in 0..25 {
            let cur = rate_of(q, ticks);
            let (mut bv, mut bq) = (cur, q);
            for d in [-1.0f64, 1.0] {
                let mut r = q;
                r.s_switch = (r.s_switch + d * step).clamp(1.2, 3.2);
                let v = rate_of(r, ticks);
                if v > bv { bv = v; bq = r }
            }
            if bv > cur { q = bq } else { step *= 0.6 }
        }
        println!("{k:>7.3} {asis:>13.4} {:>13.4} {:>9.3}", rate_of(q, ticks), q.s_switch);
    }
}


fn cmd_policy(optimize: bool, dive: Dive, vzpeak: bool, ng: usize) {
    let ticks = 1500;
    // the shared constants, as tuned by `policy <rule> --opt` for each dive rule in turn
    let mut par = match dive {
        Dive::Leak => P { g_star: 17.73, k: 0.055, s_switch: 2.40, vy_flick: -0.260,
                          s_exit: 0.21, slew: 12.7, p_push: 23.0, p_flick: -88.5, n_gain: ng, dive },
        Dive::Floor => P { g_star: f64::NAN, k: f64::NAN, s_switch: 2.177, vy_flick: -0.2685,
                           s_exit: 0.45, slew: 12.92, p_push: 24.01, p_flick: -79.44, n_gain: ng, dive },
        Dive::Hold => P { g_star: f64::NAN, k: f64::NAN, s_switch: 2.127, vy_flick: -0.260,
                          s_exit: 0.29, slew: 8.34, p_push: 47.0, p_flick: -41.51, n_gain: ng, dive },
        // steer straight at the derived floor rather than holding what you have; wants its own
        // constants, so `policy target --opt` before reading anything into its score
        Dive::Target => P { g_star: f64::NAN, k: f64::NAN, s_switch: 2.177, vy_flick: -0.2685,
                            s_exit: 0.45, slew: 12.92, p_push: 24.01, p_flick: -79.44, n_gain: ng, dive },
    };
    // the stopping rule under test: `--vzpeak` drops the tuned threshold for `opt::vz_peaked`
    if vzpeak { par.vy_flick = f64::NAN }
    if optimize {
        // g_star and k exist only for the leaking dive; the others are shared
        let mut js: Vec<usize> = if dive == Dive::Leak { (0..8).collect() } else { (2..8).collect() };
        // there is no threshold to tune when the stopping rule is the parameter-free one
        if par.vy_flick.is_nan() { js.retain(|&j| j != 3) }
        let js = js;
        let mut step = [3.0, 0.03, 0.30, 0.10, 0.08, 8.0, 6.0, 12.0];
        for _ in 0..26 {
            for &j in &js {
                let cur = rate_of(par, ticks);
                let (mut bv, mut bp) = (cur, par);
                for d in [-1.0f64, 1.0] {
                    let mut q = par;
                    let s = d * step[j];
                    match j {
                        0 => q.g_star += s, 1 => q.k = (q.k + s).clamp(0.002, 1.0),
                        2 => q.s_switch += s, 3 => q.vy_flick += s,
                        4 => q.s_exit = (q.s_exit + s).max(0.05), 5 => q.slew = (q.slew + s).clamp(2.0, 90.0),
                        6 => q.p_push += s, _ => q.p_flick = (q.p_flick + s).clamp(-90.0, -10.0),
                    }
                    let v = rate_of(q, ticks);
                    if v > bv { bv = v; bp = q }
                }
                if bv > cur { par = bp } else { step[j] *= 0.55 }
            }
        }
    }
    let (ps, ph, st) = fly(par, ticks);
    let r = rate_of(par, ticks);
    eprintln!("n_gain {:>2}  {r:.5} b/s  {:>5.1}% of the optimal cycle   {par:?}", par.n_gain, r / OPTIMAL_CYCLE_RATE * 100.0);
    if dive == Dive::Floor { eprintln!("dive floor: gamma {:.4}, the glide at pitch {:.3}", ceiling().1, ceiling().0) }
    println!("tick,phase,pitch,vy,vz,speed,gamma,te");
    for t in 0..ticks {
        println!("{t},{},{:.5},{:.6},{:.6},{:.6},{:.4},{:.6}", ph[t], ps[t],
                 st[t].vel.y, st[t].vel.z, st[t].vel.length(), gamma(st[t].vel), st[t].total_energy());
    }
}

fn main() {
    let cli = Cli::parse();
    if let Some(t) = cli.trig { set_trig_mode(t) }
    if let Some(f) = cli.flight { set_flight_mode(f) }
    match cli.cmd {
        Cmd::Profiles => cmd_profiles(),
        Cmd::Eq => cmd_eq(),
        Cmd::Eqrate { locus } => cmd_eqrate(locus),
        Cmd::Glide { step, lo, hi } => cmd_glide(step, lo, hi),
        Cmd::Crit => cmd_crit(),
        Cmd::Rising { part } => cmd_rising(&part),
        Cmd::Polish { path, passes, w } => cmd_polish(&path, passes, w),
        Cmd::Cycle { path, off } => cmd_cycle(&path, off),
        Cmd::Score { path, off, lo, hi } => cmd_score(&path, off, lo, hi),
        Cmd::Probe { path, off, lo, hi } => cmd_probe(&path, off, lo, hi),
        Cmd::Family { path, tag } => cmd_family(&path, &tag),
        Cmd::Floor { path, tag } => cmd_floor(&path, &tag),
        Cmd::Prices { path, tag } => cmd_prices(&path, &tag),
        Cmd::Gprofile { path, tag } => cmd_gprofile(&path, &tag),
        Cmd::Sides { path, delta } => cmd_sides(&path, delta),
        Cmd::Ksweep => cmd_ksweep(),
        Cmd::Prefix { path } => cmd_prefix(&path),
        Cmd::Sens { path, delta, dump } => cmd_sens(&path, delta, dump),
        Cmd::Swing { path, t } => cmd_swing(&path, t),
        Cmd::Cyclecut { path } => cmd_cyclecut(&path),
        Cmd::Consist { path, amp } => cmd_consist(&path, amp),
        Cmd::Singular { path } => cmd_singular(&path),
        Cmd::Gain { path, w, bvp, dump } => cmd_gain(&path, w, bvp, dump),
        Cmd::Gainlaw { path, w, k, limit, dump } => cmd_gainlaw(&path, w, k, limit, dump),
        Cmd::Adjoint { path, w, dump } => cmd_adjoint(&path, w, dump),
        Cmd::Sweepn { path, off, lo, hi, nmax } => cmd_sweepn(&path, off, lo, hi, nmax),
        Cmd::Djn { path, nmax, w, limit, dump, wsweep, equiv } =>
            cmd_djn(&path, w, limit, nmax, dump, wsweep, equiv),
        Cmd::Rules { path, vy0, vz0, nmax } => cmd_rules(&path, vy0, vz0, nmax),
        Cmd::Policy { dive, opt, vzpeak, ngain } => {
            let dive = match dive.as_str() { "floor" => Dive::Floor, "hold" => Dive::Hold,
                                             "target" => Dive::Target, _ => Dive::Leak };
            cmd_policy(opt, dive, vzpeak, ngain)
        }
    }
}
