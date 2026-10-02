//! Flight over a floor: how long, and how far, before the trajectory first goes under `y = 0`.
//!
//! The polish only knows a fixed horizon, so endurance is a search over it: a horizon `n` is
//! *feasible* when some schedule keeps every one of its `n` post-tick states at or above the
//! floor, and the endurance is the largest feasible `n`. Inside one horizon the objective is the
//! usual `J = TE(s_n) + w*z_n` less the floor's price (see `Floor` in `opt.rs`); `J` only picks
//! which feasible schedule to hand the next horizon, and feasibility is read off the replay.
//!
//! Subcommands: `probe` (constant-pitch baselines), `solve` (one horizon), `safety` (is a horizon
//! feasible?), `endure` (the longest feasible horizon), `depth` and `laps` (read a schedule), and
//! `exit` (first-exit ascent: the interpolated exit time or distance). `floor <cmd> --help` has the
//! options.
//!
//! An init spec is `hold:<p>`, `pump:<p_down>,<k>,<p_up>` (p_down for k ticks, then p_up),
//! `tile:<file>` (a cycle repeated), or a pitch file (last pitch repeated). `exit` also takes
//! `minipump[:<d>[,<k>]]`, see `minipump`, and `pumps:<K>[,<key>=<v>..]`, a K-climb
//! floor-grazing pump flown by events (keys `d lvl pull a a2 g end`; any not given is fitted to
//! the seed's own exit score), see `Pumps`.
//!
//! Physics is `mth_lut` trig and `reference` flight, fixed: vanilla's table, and the kernel
//! that is fastest on arm64 (README-sweep.md).

use elytrasim::opt::*;
use elytrasim::sim::*;
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Constant-pitch baselines: glides, and ticks to the floor.
    Probe,
    /// Polish one horizon.
    #[command(allow_negative_numbers = true)]
    Solve {
        #[command(flatten)]
        c: Common,
        #[arg(long, default_value_t = 40)]
        n: usize,
        #[arg(long, default_value = "hold:-13")]
        init: String,
        #[arg(long, default_value_t = 200)]
        passes: usize,
        #[arg(long)]
        out: Option<String>,
    },
    /// Max soft-min clearance: is this horizon feasible?
    #[command(allow_negative_numbers = true)]
    Safety {
        #[command(flatten)]
        c: Common,
        #[arg(long, default_value_t = 40)]
        n: usize,
        #[arg(long, default_value = "hold:-13")]
        init: String,
        #[arg(long, default_value_t = 200)]
        passes: usize,
    },
    /// The longest feasible horizon.
    #[command(allow_negative_numbers = true)]
    Endure {
        #[command(flatten)]
        c: Common,
        #[arg(long, default_value_t = 200)]
        passes: usize,
        #[arg(long, default_value_t = 150)]
        nmax: usize,
        /// Stop after this many infeasible horizons in a row.
        #[arg(long, default_value_t = 3)]
        misses: usize,
        /// Bubble shrink stages after the first polish; see `polish_annealed`.
        #[arg(long, default_value_t = 0)]
        anneal: usize,
        /// Factor on margin and weight per anneal stage.
        #[arg(long, default_value_t = 0.3)]
        shrink: f64,
        #[arg(long)]
        out: Option<String>,
    },
    /// How far a schedule dips below its start, from `--vy/--vz`.
    #[command(allow_negative_numbers = true)]
    Depth {
        #[command(flatten)]
        c: Common,
        #[arg(long)]
        file: String,
        /// Also print every this many ticks.
        #[arg(long)]
        every: Option<usize>,
    },
    /// Every dip and peak of the clearance before the exit, and the energy above the floor
    /// there: does a pump gain per lap?
    #[command(allow_negative_numbers = true)]
    Laps {
        #[command(flatten)]
        c: Common,
        #[arg(long)]
        file: String,
    },
    /// First-exit ascent on the interpolated exit; the cap doubles while the answer survives it.
    #[command(allow_negative_numbers = true)]
    Exit(ExitArgs),
}

/// The objective, the floor's price and the polish's settings, shared by the subcommands.
#[derive(clap::Args)]
struct Common {
    /// Initial v_y, blocks/tick.
    #[arg(long, default_value_t = 0.0)]
    vy: f64,
    /// Initial v_z, blocks/tick.
    #[arg(long, default_value_t = 0.0)]
    vz: f64,
    #[arg(long, default_value_t = 0.0)]
    lambda: f64,
    /// Start height above the floor, blocks.
    #[arg(long, default_value_t = 4.0)]
    y0: f64,
    #[arg(long, default_value_t = Floor::default().margin)]
    margin: f64,
    #[arg(long, default_value_t = Floor::default().weight)]
    weight: f64,
    #[arg(long, default_value_t = Floor::default().wall)]
    wall: f64,
    /// Roughness price.
    #[arg(long, default_value_t = 1e-4)]
    mu: f64,
    /// Largest |pitch|, degrees.
    #[arg(long, default_value_t = 85.0)]
    limit: f64,
    #[arg(long, default_value_t = 1e-3)]
    tol: f64,
}

impl Common {
    fn obj(&self, n: usize) -> Objective {
        Objective { v0: Vec3::new(0.0, self.vy, self.vz), n, lambda: self.lambda }
    }
    fn floor(&self) -> Floor {
        Floor { depth: self.y0, margin: self.margin, weight: self.weight, wall: self.wall }
    }
    fn opts(&self, passes: usize) -> PolishOpts {
        PolishOpts {
            max_passes: passes,
            tol: self.tol,
            rough: Rough { mu: self.mu, limit: self.limit, ..Default::default() },
            floor: self.floor(),
            ..Default::default()
        }
    }
}

#[derive(clap::Args)]
struct ExitArgs {
    #[command(flatten)]
    c: Common,
    /// The first cap, ticks.
    #[arg(long, default_value_t = 1200)]
    n: usize,
    /// The largest cap, ticks.
    #[arg(long, default_value_t = 4800)]
    nmax: usize,
    #[arg(long, value_enum, default_value_t = Exit::Time)]
    mode: Exit,
    /// An init spec (module docs), or `minipump[:<d>[,<k>]]` or `pumps:<K>[,<key>=<v>..]`.
    #[arg(long, default_value = "hold:-13")]
    init: String,
    /// With `pumps`: keep this schedule to its last apex, then fly the K laps.
    #[arg(long)]
    after: Option<String>,
    #[arg(long, default_value_t = 100)]
    passes: usize,
    /// Weight on kinetic energy at the exit.
    #[arg(long, default_value_t = 0.0)]
    ke: f64,
    /// margin,weight: blocks, and blocks of energy per tick at contact.
    #[arg(long, value_delimiter = ',', num_args = 1, default_value = "1,0")]
    bubble: Vec<f64>,
    /// Multiplies both bubble terms: under `tick`, after every pass, across cap doublings;
    /// under `grad`, once per stall stage, restarting at every cap.
    #[arg(long, default_value_t = 1.0)]
    shrink: f64,
    /// l1|l2:<d>|huber:<d>, the shape `--mu` prices.
    #[arg(long, default_value = DEFAULT_PENALTY)]
    penalty: String,
    /// Which sweeps a pass makes, in order: `tick,box:3:9,ramp,shift`.
    #[arg(long)]
    moves: Option<String>,
    /// Shorthand for `--moves tick,shift`.
    #[arg(long, conflicts_with = "moves")]
    shift: bool,
    /// `tick`: `ascend`. `grad`: `ascend_grad`. `grad+tick`: `ascend_grad`, then `ascend` from its
    /// answer at every cap.
    #[arg(long, default_value = "tick", value_parser = ["tick", "grad", "grad+tick"])]
    method: String,
    #[arg(long, default_value_t = 2000)]
    iters: usize,
    #[arg(long, default_value_t = 10)]
    mem: usize,
    #[arg(long, default_value_t = 5.0)]
    max_step: f64,
    #[arg(long, default_value_t = 1e-4)]
    c1: f64,
    /// Dip-projection margins, or `off` for no dip projection and no rejection rule.
    #[arg(long, default_value = "1e-2,3e-3,1e-3")]
    graze: String,
    /// Bubble shrink stages for `grad`.
    #[arg(long, default_value_t = 0)]
    anneal: usize,
    #[arg(long)]
    out: Option<String>,
}

fn init(spec: &str, n: usize) -> Vec<f64> {
    if let Some(p) = spec.strip_prefix("hold:") { return vec![p.parse().unwrap(); n] }
    if let Some(r) = spec.strip_prefix("pump:") {
        let v: Vec<f64> = r.split(',').map(|x| x.parse().unwrap()).collect();
        let k = v[1] as usize;
        return (0..n).map(|t| if t < k { v[0] } else { v[2] }).collect();
    }
    if let Some(f) = spec.strip_prefix("tile:") {
        let c = read_pitches(f);
        return (0..n).map(|t| c[t % c.len()]).collect();
    }
    init_from(&read_pitches(spec), n)
}

/// What a schedule does over the floor, from its replay alone.
struct Flown { survived: usize, clearance: f64, z: f64, te: f64, cost: f64 }

fn flown(obj: &Objective, floor: &Floor, p: &[f64]) -> Flown {
    let st = obj.replay(p);
    let s = st.last().unwrap();
    Flown { survived: floor.survived(&st), clearance: floor.clearance(&st), z: s.pos.z,
            te: s.total_energy() + floor.depth, cost: floor.cost(&st) }
}

/// Each pitch as the shortest decimal that round-trips its `f32`. Not a display rounding: an
/// optimum here skims the floor at zero margin, and writing pitches to four places moved one
/// replay 0.004 blocks through it.
fn fmt_pitches(p: &[f64]) -> String {
    p.chunks(10).map(|c| c.iter().map(|&x| format!("{}", x as f32)).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>().join("\n")
}

/// Soft minimum of the clearance over `states[1..]`: `-tau * ln sum exp(-h/tau)`, shifted by the
/// hard minimum so it cannot overflow. Never above the hard minimum, and within `tau * ln n` of it.
fn softmin(hs: impl Iterator<Item = f64> + Clone, tau: f64) -> f64 {
    let m = hs.clone().fold(f64::INFINITY, f64::min);
    m - tau * hs.map(|h| (-(h - m) / tau).exp()).sum::<f64>().ln()
}

/// The feasibility oracle: coordinate ascent on the soft-minimum clearance, the discrete HJ
/// safety value `max_u min_t h_t`. A horizon is feasible exactly when this can be pushed to 0 or
/// above, which the energy objective does not ask for -- it only prices the floor -- so a horizon
/// the energy polish reports infeasible is re-tried here before it is written off.
///
/// Returns the schedule and its hard clearance. Stops early once the hard clearance is
/// non-negative: past that point the oracle has answered its question.
fn safety(obj: &Objective, floor: &Floor, init: &[f64], limit: f64, passes: usize) -> (Vec<f64>, f64) {
    let tau = 0.01;
    let mut p = init.to_vec();
    let hard = |p: &[f64]| floor.clearance(&obj.replay(p));
    let mut best = hard(&p);
    for _ in 0..passes {
        if best >= 0.0 { break }
        for t in 0..p.len() {
            let st = obj.replay(&p);
            let pre: Vec<f64> = st[1..=t].iter().map(|s| s.pos.y + floor.depth).collect();
            let tail: Vec<PitchTrig> = p[t + 1..].iter().map(|&q| PitchTrig::new(q as f32)).collect();
            let score = |x: f64| -> f64 {
                let mut s = st[t].clone().ticked_cached(PitchTrig::new(x as f32));
                let mut hs = pre.clone();
                hs.push(s.pos.y + floor.depth);
                for &q in &tail { s = s.ticked_cached(q); hs.push(s.pos.y + floor.depth) }
                softmin(hs.iter().copied(), tau)
            };
            let mut bp = p[t];
            let (r, _) = coordinate_argmax(&score, p[t], -limit, limit, 2.0, PITCH_TOL);
            let r = r as f32 as f64;
            if score(r) > score(bp) { bp = r }
            p[t] = bp as f32 as f64;
        }
        let now = hard(&p);
        if now <= best + 1e-9 { best = best.max(now); break }
        best = now;
    }
    (p, best)
}

/// Ticks a constant pitch survives, and the best such pitch on a 0.1-degree grid.
fn best_hold(obj: &Objective, floor: &Floor, limit: f64) -> (usize, f64) {
    let mut best = (0usize, 0.0);
    for i in (-(limit * 10.0) as i64)..=((limit * 10.0) as i64) {
        let q = 0.1 * i as f64;
        let t = floor.survived(&obj.replay(&vec![q; 400]));
        if t > best.0 { best = (t, q) }
    }
    best
}

const COLD: &[&str] = &["hold:-13", "hold:0", "pump:30,6,-20", "pump:60,4,-30"];

/// Endurance: the largest horizon with a feasible schedule, by walking `n` upward from the best
/// constant pitch's survival.
///
/// Each horizon is polished from the previous horizon's answer (last pitch repeated) and from
/// every `COLD` start; the feasible result with the best priced objective is kept and seeds the
/// next horizon. A horizon nobody made feasible goes to `safety` before it counts as a miss, and
/// the walk stops after `--misses` consecutive misses. A miss is evidence, not proof: the inner
/// search is local.
fn endure(a: &Common, passes: usize, nmax: usize, misses_max: usize, anneal: usize, shrink: f64,
          out: Option<&str>) {
    let floor = a.floor();
    let opts = a.opts(passes);
    let (n0, q0) = best_hold(&a.obj(1), &floor, opts.rough.limit);
    let mut best: Option<(usize, Vec<f64>)> = None;
    let mut far: Option<(usize, Vec<f64>, f64)> = None;         // the feasible horizon reaching furthest
    let mut prev = vec![q0; n0];
    let mut misses = 0;
    let mut n = n0;
    while n <= nmax && misses < misses_max {
        let obj = a.obj(n);
        let mut starts: Vec<(String, Vec<f64>)> = vec![("warm".into(), init_from(&prev, n))];
        for c in COLD { starts.push((c.to_string(), init(c, n))) }
        let mut found: Option<(String, Vec<f64>, f64)> = None;
        let mut near: Vec<(String, Vec<f64>)> = Vec::new();
        for (tag, s) in starts {
            let p = polish_annealed(opts, anneal, shrink, &obj, &s);
            let st = obj.replay(&p);
            // Compared on `J` alone once feasible: the stages ran at different bubbles, so their
            // priced objectives are not on one scale, and feasibility is already settled.
            let score = obj.eval(&p);
            if floor.clearance(&st) >= 0.0 {
                if found.as_ref().map_or(true, |f| score > f.2) { found = Some((tag, p, score)) }
            } else { near.push((tag, p)) }
        }
        if found.is_none() {
            for (tag, s) in near {
                let (p, c) = safety(&obj, &floor, &s, opts.rough.limit, 200);
                if c >= 0.0 {
                    // Feasible but not energy-optimal: polish it once more, and keep whichever
                    // of the two is still feasible.
                    let r = polish(&obj, &p, opts);
                    let keep = if floor.clearance(&obj.replay(&r.pitches)) >= 0.0 { r.pitches } else { p };
                    found = Some((format!("{tag}+safety"), keep, f64::NAN));
                    break;
                }
            }
        }
        match found {
            Some((tag, p, _)) => {
                let f = flown(&obj, &floor, &p);
                eprintln!("  n {n:>3} feasible via {tag:<16} clearance {:+.4} z {:7.3}", f.clearance, f.z);
                misses = 0;
                if far.as_ref().map_or(true, |b| f.z > b.2) { far = Some((n, p.clone(), f.z)) }
                prev = p.clone();
                best = Some((n, p));
            }
            None => { eprintln!("  n {n:>3} infeasible"); misses += 1 }
        }
        n += 1;
    }
    let (n, p) = best.expect("the best constant pitch is feasible at its own survival");
    let obj = a.obj(n);
    let f = flown(&obj, &floor, &p);
    let capped = if n == nmax { " (hit --nmax)" } else { "" };
    let text = format!(
        "# endure y0 {} v0 ({}, {}) lambda {} margin {} weight {} wall {} mu {} limit {}\n\
         # hold: {n0} ticks at pitch {q0:.1};  optimized: {n} ticks{capped}, clearance {:+.4}, z {:.4}\n{}\n",
        floor.depth, obj.v0.y, obj.v0.z, obj.lambda, floor.margin, floor.weight, floor.wall,
        opts.rough.mu, opts.rough.limit, f.clearance, f.z, fmt_pitches(&p));
    let (fn_, fp, fz) = far.unwrap();
    let text = format!("{text}# furthest: {fn_} ticks, z {fz:.4}\n{}\n", fmt_pitches(&fp));
    match out {
        Some(o) => { std::fs::write(o, &text).unwrap();
                     print!("{}", text.lines().filter(|l| l.starts_with('#')).collect::<Vec<_>>().join("\n") + "\n") }
        None => print!("{text}"),
    }
}

/// Polish, then shrink the bubble and polish again, `--anneal` times: margin and weight each
/// times `--shrink`, the wall untouched. Keeps the last stage that is still feasible, or the first
/// stage's answer if none is.
///
/// The bubble is conservative by design, and how conservative is not a detail: a range optimum
/// skims the floor for many ticks and pays the bubble on every one of them, so at a fixed bubble
/// the distance reached is set by the ratio of `weight` to `lambda`, not by `lambda`. Annealing is
/// the interior-point schedule: each stage starts from a schedule the previous one kept off the
/// floor, so it only has to give up margin, never find it.
fn polish_annealed(mut opts: PolishOpts, stages: usize, shrink: f64, obj: &Objective, init: &[f64]) -> Vec<f64> {
    let floor = opts.floor;
    let mut p = polish(obj, init, opts).pitches;
    for _ in 0..stages {
        opts.floor.margin *= shrink;
        opts.floor.weight *= shrink;
        let q = polish(obj, &p, opts).pitches;
        if floor.clearance(&obj.replay(&q)) >= 0.0 { p = q } else { break }
    }
    p
}

/// A seed of the right length from a schedule of another: truncate, or repeat the last pitch.
fn init_from(p: &[f64], n: usize) -> Vec<f64> {
    let mut v = p.to_vec();
    let last = *v.last().unwrap_or(&0.0);
    v.resize(n, last);
    v
}

/// The two steady-glide constants, from `probe`.
const MIN_SINK: f64 = 0.0708;    // blocks/tick, at pitch -13.0
const BEST_GLIDE: f64 = 10.10;   // blocks of z per block of height, at pitch 0

/// What a first-exit schedule is scored on. Every variant reads only the replay up to the first
/// state under the floor, so the schedule's later ticks are dead and the horizon is just a cap.
#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum Exit {
    /// The fractional tick at which the replay crosses the floor.
    Time,
    /// `z` at that fractional tick.
    Dist,
}

impl Exit {
    /// Score units per block of energy: what one block of `TE` buys at min sink or at the best
    /// glide ratio. `mu` is quoted in blocks of energy, as in `polish`; charged unconverted in
    /// ticks or in blocks of `z` it was 14x or 10x weaker than there, and the schedules showed it
    /// -- one-tick spikes to 75 degrees where the pitch barely matters.
    fn per_block(self) -> f64 { match self { Exit::Time => 1.0 / MIN_SINK, Exit::Dist => BEST_GLIDE } }
}

/// Where the replay exits, interpolated: `(t*, z(t*))`, from the first pair of states that
/// straddles the floor. `h` is affine along the secant between the two, so
/// `t* = (k-1) + h[k-1] / (h[k-1] - h[k])`, and `z` is read off the same secant.
///
/// Why interpolate. The integer exit tick is a step function of every pitch, so a coordinate
/// search sees a plateau with cliffs; the secant makes it continuous as the crossing slides from
/// one tick to the next -- at `h[k] = 0` both readings give `t* = k`. What it does *not* smooth is
/// a touch-and-go: a dip that goes under and comes back makes an earlier pair the first one, and
/// `t*` jumps down.
///
/// A replay that never exits scores its horizon plus a value-to-go for the energy it still holds:
/// that energy spent at min sink (`Time`) or at the best glide ratio (`Dist`), the two steady-glide
/// constants from `probe`. Crude, but it only has to rank schedules that all survive the cap.
fn exit_score(states: &[State], depth: f64, mode: Exit) -> f64 {
    for k in 1..states.len() {
        let (a, b) = (states[k - 1].pos.y + depth, states[k].pos.y + depth);
        if b < 0.0 {
            let f = a / (a - b);
            return match mode {
                Exit::Time => (k - 1) as f64 + f,
                Exit::Dist => states[k - 1].pos.z + f * (states[k].pos.z - states[k - 1].pos.z),
            };
        }
    }
    let s = states.last().unwrap();
    let e = s.total_energy() + depth;
    match mode {
        Exit::Time => (states.len() - 1) as f64 + e / MIN_SINK,
        Exit::Dist => s.pos.z + e * BEST_GLIDE,
    }
}

/// The energy the replay still holds where it crosses the floor, in blocks: `TE + y0` along the
/// same secant as `t*`, which at `y = -y0` is all kinetic. `None` if it never crosses.
fn exit_energy(states: &[State], depth: f64) -> Option<f64> {
    (1..states.len()).find(|&k| states[k].pos.y + depth < 0.0).map(|k| {
        let (a, b) = (states[k - 1].pos.y + depth, states[k].pos.y + depth);
        let f = a / (a - b);
        let (e0, e1) = (states[k - 1].total_energy(), states[k].total_energy());
        e0 + f * (e1 - e0) + depth
    })
}

/// What the ascent maximizes: the exit score plus `ke` times the energy left at the crossing,
/// converted to score units by `Exit::per_block`. `ke < 0` charges for reaching the floor with
/// speed to spare -- a schedule that exits fast has flown badly; `ke > 0` is the PE+KE-style
/// proxy, crediting speed the exit alone does not see. Survivors get nothing extra: their
/// value-to-go already counts all their energy.
fn utility(states: &[State], depth: f64, mode: Exit, ke: f64) -> f64 {
    let base = exit_score(states, depth, mode);
    if ke == 0.0 { return base }
    base + exit_energy(states, depth).map_or(0.0, |e| ke * mode.per_block() * e)
}

/// The bubble's price on a replay: `weight * (max(0, margin - h) / margin)^2` per tick, in
/// blocks of energy -- the same shape as `Floor`, without its wall, since going under the floor
/// ends the flight here. `weight == 0` is off.
///
/// Continuous in the schedule at both ends. Entering the bubble starts at zero with zero slope.
/// At the floor, the crossing interval is charged the full-contact price times `f`, the same
/// fraction of the tick that `t*` uses: charging only the states at or above the floor would
/// make a state at `h = 0+` pay `weight` and one at `0-` pay nothing, a jump of `weight` every
/// time the crossing slides past a tick.
fn bubble_cost(states: &[State], depth: f64, (margin, weight): (f64, f64)) -> f64 {
    if weight == 0.0 { return 0.0 }
    let pen = |h: f64| weight * ((margin - h).max(0.0) / margin).powi(2);
    let mut c = 0.0;
    for k in 1..states.len() {
        let (a, b) = (states[k - 1].pos.y + depth, states[k].pos.y + depth);
        if b < 0.0 { return c + a / (a - b) * weight }
        c += pen(b);
    }
    c
}

/// A move that changes many pitches at once, sized by one number `d`, searched like a pitch.
/// Why: the curvature price couples neighbors, so a change that needs several pitches to move
/// together -- a ramp that should start a tick later, a lump that should spread -- can be
/// uphill along every single pitch.
///
/// * `Shift`: `d` added to every live pitch from `t` on. Changes two second differences.
/// * `Box(k)`: `d` added to pitches `t .. t+k`. Changes four.
/// * `Ramp`: `d * (s - t)` added to every live pitch `s > t`. Changes the second difference
///   centered on `t`, and the one at the live/dead boundary, since the dead tail is held flat
///   (that term is the last live slope): a sweep of ramps is coordinate search in the second
///   differences, where the price is separable but for that one term. At `t = 0` it is the
///   schedule's initial slope; the initial pitch is a `Shift` at 0, which a ramp sweep also
///   tries. Clamping at the limit and the `f32` round break the ramp's exactness.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Move { Shift, Box(usize), Ramp }

impl Move {
    fn parse(s: &str) -> Vec<Move> {
        match s.split(':').collect::<Vec<_>>().as_slice() {
            ["shift"] => vec![Move::Shift],
            ["ramp"] => vec![Move::Ramp],
            ["box", ks @ ..] => ks.iter().map(|k| Move::Box(k.parse().unwrap())).collect(),
            _ => panic!("bad move {s}"),
        }
    }
    /// `p` with the move applied at `t` over the live pitches `..end`; the dead ones after it are
    /// set to the new last live pitch, as `ascend` keeps them.
    fn apply(self, p: &[f64], t: usize, end: usize, d: f64, lim: f64) -> Vec<f64> {
        let mut q = p.to_vec();
        for s in t..end {
            let dx = match self {
                Move::Shift => d,
                Move::Box(k) => if s < t + k { d } else { 0.0 },
                Move::Ramp => d * (s - t) as f64,
            };
            if dx != 0.0 { q[s] = (p[s] + dx).clamp(-lim, lim) as f32 as f64 }
        }
        if end > 0 { let last = q[end - 1]; q[end..].iter_mut().for_each(|x| *x = last) }
        q
    }
    /// The grid `d` is scanned on. A ramp's grid is in degrees at the tail's end, so a long tail
    /// gets fine slopes and a short one coarse.
    fn grid(self, t: usize, end: usize) -> (Vec<f64>, f64) {
        match self {
            Move::Ramp => { let l = end.saturating_sub(t).max(1) as f64;
                            ((-80..=80).map(|i| 0.5 * i as f64 / l).collect(), 0.5 / l) }
            _ => ((-40..=40).map(|i| 0.25 * i as f64).collect(), 0.25),
        }
    }
}

/// The first dead pitch: the one after the pitch whose state is the first under the floor, or
/// `n` if the replay never goes under.
fn live_end(st: &[State], depth: f64) -> usize {
    (1..st.len()).find(|&k| st[k].pos.y + depth < 0.0).unwrap_or(st.len() - 1)
}

/// Coordinate ascent on a first-exit score over a fixed cap. The same sweep as `polish` -- a
/// `coordinate_argmax` of every tick's pitch on a 2-degree grid, the `f32` round, Gauss-Seidel
/// -- but scored on the whole replay rather than on a terminal state, and every pass
/// global. `tick: false` drops that sweep; each of `moves` then gets a sweep of its own, in order,
/// every pass. The curvature price is quoted in blocks of energy as in `polish`, converted to
/// score units by `Exit::per_block`. Ticks after the exit are dead: they are not searched, and
/// after each pass they are set to the last live pitch, so the price never sees the seed's
/// leftover tail.
fn ascend(obj: &Objective, depth: f64, init: &[f64], mode: Exit, ke: f64, rough: Rough,
          passes: usize, tol: f64, bail: bool, tick: bool, moves: &[Move], bub: &mut (f64, f64), shrink: f64)
          -> (Vec<f64>, f64, usize) {
    let pen = Rough { mu: rough.mu * mode.per_block(), ..rough };
    let lim = rough.limit;
    let mut p = init.to_vec();
    let flatten = |p: &mut Vec<f64>| {
        let st = obj.replay(p);
        if let Some(k) = (1..st.len()).find(|&k| st[k].pos.y + depth < 0.0) {
            let last = p[k - 1];
            p[k..].iter_mut().for_each(|q| *q = last);
        }
    };
    flatten(&mut p);
    let pb = mode.per_block();
    let total = |p: &[f64], b: (f64, f64)| {
        let st = obj.replay(p);
        utility(&st, depth, mode, ke) - pb * bubble_cost(&st, depth, b) - pen.cost(p)
    };
    let mut best = total(&p, *bub);
    let mut quiet = 0;
    let mut done = 0;
    for _ in 0..passes {
        done += 1;
        let before = best;
        let b = *bub;
        for t in 0..if tick { p.len() } else { 0 } {
            let st = obj.replay(&p);
            // Dead tick: the replay is already under the floor by the state this pitch produces.
            if (1..=t).any(|k| st[k].pos.y + depth < 0.0) { break }
            let tail: Vec<PitchTrig> = p[t + 1..].iter().map(|&q| PitchTrig::new(q as f32)).collect();
            let w = win5(&p, t);
            let score = |x: f64| -> f64 {
                let mut v = st[..=t].to_vec();
                let mut s = st[t].ticked_cached(PitchTrig::new(x as f32));
                v.push(s.clone());
                for &q in &tail {
                    if s.pos.y + depth < 0.0 { break }
                    s = s.ticked_cached(q);
                    v.push(s.clone());
                }
                utility(&v, depth, mode, ke) - pb * bubble_cost(&v, depth, b) - pen.local(&w, x)
            };
            let mut bp = p[t];
            let (r, _) = coordinate_argmax(&score, p[t], -lim, lim, 2.0, PITCH_TOL);
            // Round first, then score: the pitch kept is the pitch flown.
            let r = r.clamp(-lim, lim) as f32 as f64;
            if score(r) > score(bp) { bp = r }
            p[t] = bp as f32 as f64;
        }
        // Multi-pitch moves (see `Move`), each scored on the whole price since they are not local.
        for &mv in moves {
            // A ramp sweep also moves the initial pitch, the one coordinate a ramp cannot reach.
            let slots: Vec<(Move, usize)> = if mv == Move::Ramp { std::iter::once((Move::Shift, 0))
                .chain((0..p.len()).map(|t| (Move::Ramp, t))).collect() } else { (0..p.len()).map(|t| (mv, t)).collect() };
            for (mv, t) in slots {
                let st = obj.replay(&p);
                if (1..=t).any(|k| st[k].pos.y + depth < 0.0) { break }
                let end = live_end(&st, depth);
                // The candidate as it would be kept: moved, then flattened from its *own* exit,
                // which can come before the schedule's. Pricing the moved dead tail instead
                // would let the end-of-pass flatten undo an accepted gain.
                let realize = |d: f64| -> (Vec<f64>, f64) {
                    let mut q = mv.apply(&p, t, end, d, lim);
                    let mut v = st[..=t].to_vec();
                    let mut s = st[t].clone();
                    for &x in &q[t..] {
                        if s.pos.y + depth < 0.0 { break }
                        s = s.ticked_cached(PitchTrig::new(x as f32));
                        v.push(s.clone());
                    }
                    let k = v.len() - 1;
                    if k < q.len() { let last = q[k - 1]; q[k..].iter_mut().for_each(|x| *x = last) }
                    let sc = utility(&v, depth, mode, ke) - pb * bubble_cost(&v, depth, b) - pen.cost(&q);
                    (q, sc)
                };
                let score = |d: f64| realize(d).1;
                let (grid, h) = mv.grid(t, end);
                let s0 = score(0.0);
                let (mut bd, mut bs) = grid.par_iter().map(|&d| (d, score(d)))
                    .reduce(|| (0.0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
                // To the precision the old 30 ternary steps reached, (2/3)^30 of the 2h bracket.
                let (r, _) = golden_max(&score, bd - h, bd + h, h * 1e-5);
                let sr = score(r);
                if sr > bs { bd = r; bs = sr }
                if bs > s0 { p = realize(bd).0 }
            }
        }
        flatten(&mut p);
        best = total(&p, b);
        // The shrinking bubble: an interior-point schedule, one step per pass.
        *bub = (b.0 * shrink, b.1 * shrink);
        // `bail`: the caller will grow the cap the moment the flight outlasts it, so every pass
        // after that would only be optimizing the value-to-go guess. Hand it back now.
        if bail && obj.replay(&p).iter().all(|s| s.pos.y + depth >= 0.0) { break }
        if best - before < tol { quiet += 1; if quiet >= 2 { break } } else { quiet = 0 }
    }
    (p, best, done)
}

/// `utility` less the bubble (`bubble_cost`, in score units), and its gradient in every pitch,
/// from one replay and one backward sweep (`elytrasim::adjoint`).
///
/// At an exit the seeds sit on the two states that straddle the floor: `t* = (k-1) + f` with
/// `f = a / (a - b)`, `a = h[k-1]`, `b = h[k]`, so `dt*/da = -b/(a-b)^2` and `dt*/db = a/(a-b)^2`;
/// `z(t*)` adds `1 - f` and `f` on the two `z`s and `(z_k - z_{k-1}) df` on the two heights. The
/// crossing's *index* is a step function and has no derivative: this is the gradient of the
/// score with the crossing held at the pair it is on, which is exact wherever the score is
/// continuous. A touch-and-go (an earlier dip about to go under) is invisible to it; see
/// `ascend_grad` for how that is handled. A survivor is seeded on its last state with the
/// value-to-go's `d/d TE`.
///
/// The bubble is seeded on each priced state's height, `-2 weight (margin - h) / margin^2` per
/// state inside it, and its crossing term `f * weight` through the same `df/da`, `df/db` as `t*`.
fn utility_grad(obj: &Objective, depth: f64, mode: Exit, ke: f64, bub: (f64, f64), p: &[f64]) -> ExitGrad {
    use elytrasim::adjoint::{backprop_costates, te_cot};
    let st = obj.replay(p);
    let pb = mode.per_block();
    let u = utility(&st, depth, mode, ke) - pb * bubble_cost(&st, depth, bub);
    let k = live_end(&st, depth);
    let mut seeds: Vec<(usize, [f64; 4])> = Vec::new();
    let scaled = |c: [f64; 4], w: f64| c.map(|x| x * w);
    let y = [1.0, 0.0, 0.0, 0.0];
    let crossed = st[k].pos.y + depth < 0.0;
    let (margin, weight) = bub;
    if weight != 0.0 {
        // The states `bubble_cost` prices: `1 .. k-1` before a crossing, `1 ..= k` for a survivor.
        for j in 1..if crossed { k } else { k + 1 } {
            let h = st[j].pos.y + depth;
            if h < margin { seeds.push((j, scaled(y, pb * 2.0 * weight * (margin - h) / (margin * margin)))) }
        }
        if crossed {
            let (a, b) = (st[k - 1].pos.y + depth, st[k].pos.y + depth);
            let d = a - b;
            seeds.push((k - 1, scaled(y, -pb * weight * (-b / (d * d)))));
            seeds.push((k, scaled(y, -pb * weight * (a / (d * d)))));
        }
    }
    if crossed {
        let (a, b) = (st[k - 1].pos.y + depth, st[k].pos.y + depth);
        let d = a - b;
        let f = a / d;
        let (dfa, dfb) = (-b / (d * d), a / (d * d));
        match mode {
            Exit::Time => { seeds.push((k - 1, scaled(y, dfa))); seeds.push((k, scaled(y, dfb))) }
            Exit::Dist => {
                let dz = st[k].pos.z - st[k - 1].pos.z;
                seeds.push((k - 1, [dfa * dz, 1.0 - f, 0.0, 0.0]));
                seeds.push((k, [dfb * dz, f, 0.0, 0.0]));
            }
        }
        if ke != 0.0 {
            // ke * per_block * (e0 + f (e1 - e0) + depth)
            let c = ke * mode.per_block();
            let (e0, e1) = (st[k - 1].total_energy(), st[k].total_energy());
            seeds.push((k - 1, scaled(te_cot(&st[k - 1]), c * (1.0 - f))));
            seeds.push((k, scaled(te_cot(&st[k]), c * f)));
            seeds.push((k - 1, scaled(y, c * (e1 - e0) * dfa)));
            seeds.push((k, scaled(y, c * (e1 - e0) * dfb)));
        }
    } else {
        match mode {
            Exit::Time => seeds.push((k, scaled(te_cot(&st[k]), 1.0 / MIN_SINK))),
            Exit::Dist => { seeds.push((k, scaled(te_cot(&st[k]), BEST_GLIDE))); seeds.push((k, [0.0, 1.0, 0.0, 0.0])) }
        }
    }
    let (g, av) = backprop_costates(p, &st, &seeds);
    ExitGrad { u, g, av, k, st }
}

/// A gradient of `utility` with what `ascend_grad` needs around it.
struct ExitGrad {
    u: f64,
    /// `dU/dp`, the derivative of the branches the replay took.
    g: Vec<f64>,
    /// Each tick's velocity costate, from `backprop_costates`.
    av: Vec<[f64; 2]>,
    /// The live length, `live_end`.
    k: usize,
    st: Vec<State>,
}

/// `p` with its dead tail -- every pitch from `k`, the live length -- set to the last live one,
/// as `ascend` keeps it.
fn flat_tail(p: &mut [f64], k: usize) {
    if k > 0 && k < p.len() { let last = p[k - 1]; p[k..].iter_mut().for_each(|x| *x = last) }
}

/// What the gradient ascent maximizes, on a schedule as it would be kept: flattened from its own
/// exit, then `utility - bubble - price`, exactly `ascend`'s `total`. Returns the flattened
/// schedule, its score and its lowest pre-exit dip (`pre_exit_min`).
fn grad_total(obj: &Objective, depth: f64, mode: Exit, ke: f64, bub: (f64, f64), pen: &Rough, q: &[f64])
              -> (Vec<f64>, f64, f64) {
    let st = obj.replay(q);
    let k = live_end(&st, depth);
    let mut q = q.to_vec();
    flat_tail(&mut q, k);
    let sc = utility(&st, depth, mode, ke) - mode.per_block() * bubble_cost(&st, depth, bub) - pen.cost(&q);
    (q, sc, pre_exit_min(&st, k, depth))
}

/// Folds every dead pitch's share of a gradient onto the last live one: the dead tail is a copy
/// of it, so that is the chain rule.
fn fold_tail(g: &mut [f64], k: usize) {
    if k > 0 && k < g.len() {
        let tail: f64 = g[k..].iter().sum();
        g[k - 1] += tail;
        g[k..].iter_mut().for_each(|x| *x = 0.0);
    }
}

/// The ascent direction's raw material at a schedule whose tail is flat from its exit: the score,
/// its one-sided derivatives, and the replay.
///
/// Where a pitch sits on the forward-to-up switch (`climb_switch_partials`), the score has a
/// corner in it: raising the pitch keeps the branch off, lowering it turns it on. `up` is the
/// derivative of the branch taken (for raising), `down` the one for lowering. Elsewhere they are
/// equal. The price's gradient is smooth for `l2`/`huber` and a subgradient for `l1`.
struct Slopes { f: f64, up: Vec<f64>, down: Vec<f64>, eg: ExitGrad }

fn slopes(obj: &Objective, depth: f64, mode: Exit, ke: f64, bub: (f64, f64), pen: &Rough, p: &[f64]) -> Slopes {
    let eg = utility_grad(obj, depth, mode, ke, bub, p);
    let mut gp = vec![0.0; p.len()];
    pen.grad(p, &mut gp);
    let mut up: Vec<f64> = eg.g.iter().zip(&gp).map(|(a, b)| a - b).collect();
    let mut down = up.clone();
    for t in 0..eg.k.min(p.len()) {
        if let Some(d) = climb_switch_partials(eg.st[t].vel, p[t]) {
            down[t] += eg.av[t][0] * d[0] + eg.av[t][1] * d[1];
        }
    }
    fold_tail(&mut up, eg.k);
    fold_tail(&mut down, eg.k);
    Slopes { f: eg.u - pen.cost(p), up, down, eg }
}

/// The steepest-ascent vector from one-sided slopes: each pitch moves the way that climbs, at
/// that side's slope, and stays put at a corner that is a local maximum in it (`up <= 0 <=
/// down`). Where the two agree this is the gradient.
fn steepest(s: &Slopes) -> Vec<f64> {
    s.up.iter().zip(&s.down).map(|(&u, &d)| {
        let (a, b) = (u.max(0.0), d.min(0.0));
        if a >= -b { a } else { b }
    }).collect()
}

/// `d` moved as little as possible (Euclidean) to satisfy `a . d >= b` for every `(a, b)`:
/// Hildreth's dual coordinate ascent, which for a handful of half-spaces converges in a few
/// sweeps.
fn project_halfspaces(d: &mut [f64], cons: &[(Vec<f64>, f64)]) {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norms: Vec<f64> = cons.iter().map(|(a, _)| dot(a, a)).collect();
    let mut mu = vec![0.0; cons.len()];
    for _ in 0..500 {
        let mut worst: f64 = 0.0;
        for (j, (a, b)) in cons.iter().enumerate() {
            if norms[j] == 0.0 { continue }
            let r = b - dot(a, d);
            let delta = (r / norms[j]).max(-mu[j]);
            if delta != 0.0 { mu[j] += delta; for i in 0..d.len() { d[i] += delta * a[i] } }
            worst = worst.max(r);
        }
        if worst <= 1e-12 { break }
    }
}

/// The least clearance over the states strictly before the exit pair (`1 .. k-1`), or over all
/// of them for a survivor: the dips `ascend_grad` keeps off the floor. Infinite if there are none.
fn pre_exit_min(st: &[State], k: usize, depth: f64) -> f64 {
    let last = if st[k].pos.y + depth < 0.0 { k.saturating_sub(1) } else { k + 1 };
    (1..last.min(st.len())).map(|j| st[j].pos.y + depth).fold(f64::INFINITY, f64::min)
}

/// Settings for `ascend_grad`.
#[derive(Clone, Debug)]
struct GradOpts {
    /// Iterations per cap, at most; each takes at most one new gradient. In practice the ascent
    /// stalls first (see `ascend_grad`).
    iters: usize,
    /// L-BFGS memory, pairs.
    mem: usize,
    /// Largest change of any one pitch in one step, degrees: a trust region on the direction,
    /// applied before the line search.
    max_step: f64,
    /// Armijo's sufficient-increase constant.
    c1: f64,
    /// The clearance, blocks, every pre-exit dip is steered to keep (see `ascend_grad`), one
    /// stage per entry: when the ascent stalls at one it moves on to the next. Empty is off
    /// (`--graze off`): no projection and no rejection rule, so only the bubble, if any, keeps
    /// dips off the floor.
    margins: Vec<f64>,
    /// The bubble, `(margin, weight)` as in `bubble_cost`; weight 0 is off.
    bub: (f64, f64),
    /// The bubble's margin and weight are multiplied by this at every stage after the first.
    shrink: f64,
    /// Stages at least (`--anneal` + 1): a stall ends one, and the next starts from its answer
    /// with the graze margin's next entry (or its last) and the bubble shrunk once more.
    stages: usize,
}

impl GradOpts {
    fn stages(&self) -> usize { self.margins.len().max(self.stages).max(1) }
    /// The graze margin at stage `i`, or `None` when the graze is off.
    fn margin(&self, i: usize) -> Option<f64> { self.margins.get(i).or(self.margins.last()).copied() }
    fn bubble(&self, i: usize) -> (f64, f64) {
        let r = self.shrink.powi(i as i32);
        (self.bub.0 * r, self.bub.1 * r)
    }
}

/// Why `ascend_grad` stopped, for the log.
#[derive(Clone, Copy, Debug)]
enum GradStop { Iters, Survives, ZeroGradient, NotAscent, LineSearch }

/// Gradient ascent on the same first-exit score as `ascend`: projected L-BFGS with a
/// backtracking (Armijo) line search on the true objective, `grad_total`, the pitch box
/// `|p| <= limit` enforced by clamping each trial point.
///
/// Why a line search on the true objective rather than a fixed-rate method like Adam. The score
/// is not smooth everywhere: a touch-and-go drops it by whole ticks, the forward-to-up switch at
/// pitch 0 is a corner, and under `mth_lut` it is a staircase at 0.0055 degrees. A trial point is
/// kept only if the replay says it is better, so none of those can make the iterate worse; the
/// gradient only proposes.
///
/// Two things the plain gradient cannot see, handled explicitly:
///
/// * **Pre-exit dips.** A local minimum of the clearance before the exit pair is a constraint
///   `h_j >= 0` that the score only feels by falling off a cliff when it breaks (a
///   touch-and-go). Unhandled, the ascent walks a dip onto the floor within a few dozen steps,
///   then every trial crosses it and the line search stalls there. So every dip under one block
///   contributes its own gradient `a_j = dh_j/dp` (one more backward sweep), and the direction
///   is projected onto `a_j . d >= margin - h_j`: to first order, a full step leaves the dip at
///   `margin` or above. The optimum therefore stands `margin` off the floor at its dips.
/// * **The forward-to-up switch at pitch 0.** The seed holds many pitches at exactly 0, where
///   the branch taken (off) has zero pitch-derivative, so the taken-branch gradient never lowers
///   them. `slopes` reports both one-sided derivatives there and `steepest` picks the side that
///   climbs.
///
/// A smooth alternative to the first item is the bubble (`bubble_cost`, differentiated in
/// `utility_grad`): a price that rises as a dip nears the floor, so the score itself, not a
/// constraint, holds the dip where the bubble's slope balances the utility's. `GradOpts` can run
/// either, both, or neither.
///
/// Returns the schedule, its score (at the last stage's bubble), the number of gradients taken
/// and the last stage's bubble.
fn ascend_grad(obj: &Objective, depth: f64, init: &[f64], mode: Exit, ke: f64, rough: Rough,
               o: &GradOpts, bail: bool) -> (Vec<f64>, f64, usize, (f64, f64)) {
    use elytrasim::adjoint::backprop_costates;
    let pen = Rough { mu: rough.mu * mode.per_block(), ..rough };
    let lim = rough.limit;
    let n = init.len();
    let clamp = |q: &mut [f64]| q.iter_mut().for_each(|x| *x = x.clamp(-lim, lim));
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut p = init.to_vec();
    clamp(&mut p);
    let mut stage = 0;
    let mut bub = o.bubble(0);
    p = grad_total(obj, depth, mode, ke, bub, &pen, &p).0;
    let mut s = slopes(obj, depth, mode, ke, bub, &pen, &p);
    let mut g = steepest(&s);
    let mut hist: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> = Default::default();
    let mut used = 0;
    let mut fails = 0;
    let mut stop = GradStop::Iters;
    // A stall ends the stage; the last stage's stall ends the ascent. A new stage whose bubble
    // differs has a different objective, so its slopes are taken afresh.
    macro_rules! stall { ($why:expr) => {{
        stop = $why;
        if stage + 1 < o.stages() {
            stage += 1; hist.clear(); fails = 0;
            if o.bubble(stage) != bub {
                bub = o.bubble(stage);
                s = slopes(obj, depth, mode, ke, bub, &pen, &p);
                g = steepest(&s);
            }
            continue
        } else { break }
    }} }
    while used < o.iters {
        used += 1;
        // The box: a pitch on its bound whose slope points out of the box is held.
        let mut gb = g.clone();
        for i in 0..n { if (p[i] >= lim && gb[i] > 0.0) || (p[i] <= -lim && gb[i] < 0.0) { gb[i] = 0.0 } }
        // Two-loop recursion, for ascent: d = H gb.
        let mut d = gb.clone();
        let mut alpha = vec![0.0; hist.len()];
        for (i, (sv, y, rho)) in hist.iter().enumerate().rev() { alpha[i] = rho * dot(sv, &d); for j in 0..n { d[j] -= alpha[i] * y[j] } }
        if let Some((sv, y, _)) = hist.back() {
            let gamma = dot(sv, y) / dot(y, y);
            d.iter_mut().for_each(|x| *x *= gamma);
            for (i, (sv, y, rho)) in hist.iter().enumerate() { let b = rho * dot(y, &d); for j in 0..n { d[j] += sv[j] * (alpha[i] - b) } }
        } else {
            let m = gb.iter().fold(0.0f64, |a, x| a.max(x.abs()));
            if m == 0.0 { stall!(GradStop::ZeroGradient) }
            d.iter_mut().for_each(|x| *x /= m);                 // a first step of at most one degree
        }
        for i in 0..n { if (p[i] >= lim && d[i] > 0.0) || (p[i] <= -lim && d[i] < 0.0) { d[i] = 0.0 } }
        let big = d.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        if big > o.max_step { d.iter_mut().for_each(|x| *x *= o.max_step / big) }
        // The dips: every local minimum of the clearance before the exit pair, under one block.
        let (st, k) = (&s.eg.st, s.eg.k);
        let h = |j: usize| st[j].pos.y + depth;
        let last = if st[k].pos.y + depth < 0.0 { k.saturating_sub(1) } else { k + 1 };
        let graze = o.margin(stage);
        let cons: Vec<(Vec<f64>, f64)> = if graze.is_none() { vec![] } else { (1..last.min(st.len() - 1))
            .filter(|&j| h(j) < 1.0 && h(j) <= h(j - 1) && h(j) <= h(j + 1))
            .map(|j| {
                let (mut a, av) = backprop_costates(&p, st, &[(j, [1.0, 0.0, 0.0, 0.0])]);
                // At a pitch on the switch, the side this direction takes.
                for t in 0..j { if d[t] < 0.0 { if let Some(c) = climb_switch_partials(st[t].vel, p[t]) {
                    a[t] += av[t][0] * c[0] + av[t][1] * c[1];
                } } }
                fold_tail(&mut a, k);
                (a, graze.unwrap() - h(j))
            }).collect() };
        project_halfspaces(&mut d, &cons);
        if dot(&d, &g) <= 0.0 {
            // Not an ascent direction once the dips are respected: drop the curvature model, and
            // if even the steepest direction fails, stop.
            if hist.is_empty() { stall!(GradStop::NotAscent) }
            hist.clear();
            continue;
        }
        let lo0 = pre_exit_min(st, k, depth);
        // Backtracking on the true objective. The predicted rise uses the one-sided slopes.
        let mut step = 1.0;
        let mut found = None;
        for _ in 0..40 {
            let mut q: Vec<f64> = (0..n).map(|i| p[i] + step * d[i]).collect();
            clamp(&mut q);
            let (q, sc, lo) = grad_total(obj, depth, mode, ke, bub, &pen, &q);
            let rise: f64 = (0..n).map(|i| { let m = q[i] - p[i]; if m >= 0.0 { s.up[i] * m } else { s.down[i] * m } }).sum();
            // A dip may not sink under half the margin unless it already was lower. why? under
            // `mth_lut` the physics is a staircase in pitch, and a dip parked on the floor is
            // crossed by table noise at every step size, so the line search stalls there. Found
            // on `y0 = 24` endurance, which stalled at t* 298 this way and reaches 411 with it.
            // Off with the graze.
            let dip_ok = graze.map_or(true, |m| lo >= (0.5 * m).min(lo0));
            if sc > s.f && sc >= s.f + o.c1 * rise && dip_ok { found = Some(q); break }
            step *= 0.5;
        }
        let Some(q) = found else {
            fails += 1;
            if hist.is_empty() || fails > 3 { stall!(GradStop::LineSearch) }
            hist.clear();
            continue;
        };
        fails = 0;
        let sq = slopes(obj, depth, mode, ke, bub, &pen, &q);
        let gq = steepest(&sq);
        // A curvature pair for the ascent: s = q - p, y = -(gq - g), kept only when s.y > 0.
        let sv: Vec<f64> = (0..n).map(|i| q[i] - p[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| g[i] - gq[i]).collect();
        let sy = dot(&sv, &y);
        if sy > 1e-12 * dot(&sv, &sv).sqrt() * dot(&y, &y).sqrt() {
            hist.push_back((sv, y, 1.0 / sy));
            if hist.len() > o.mem { hist.pop_front(); }
        }
        p = q;
        s = sq;
        g = gq;
        if bail && s.eg.st.iter().all(|x| x.pos.y + depth >= 0.0) { stop = GradStop::Survives; break }
    }
    let gmax = g.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    eprintln!("grad: cap {n} stop {stop:?} after {used} iters, stage {stage}, score {:.4}, |g|max {gmax:.2e}, \
               pre-exit min clearance {:.3e}", s.f, pre_exit_min(&s.eg.st, s.eg.k, depth));
    (p, s.f, used, bub)
}

/// The minipump seed: `d` degrees nose-down for `k` ticks, pitch 0 for 10, -40 while `v_y` is
/// still rising, then 0. The dive is the part a per-tick search cannot invent from a hold, since
/// no single tick of it pays on its own.
fn minipump(obj: &Objective, d: f64, k: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(obj.n);
    let mut s = State { pos: Vec3::ZERO, vel: obj.v0 };
    let mut climbing = true;
    while p.len() < obj.n {
        let t = p.len();
        let q = if t < k { d } else if t < k + 10 || !climbing { 0.0 } else { -40.0 };
        let next = s.ticked_cached(PitchTrig::new(q as f32));
        if q == -40.0 && next.vel.y < s.vel.y { climbing = false }
        p.push(q);
        s = next;
    }
    p
}

/// A minipump seed with the dive's length `k` fitted -- the `k` whose seed scores best -- and,
/// unless given, the dive angle `d` too, on a 2.5-degree grid.
fn fit_minipump(obj: &Objective, depth: f64, mode: Exit, spec: &str) -> (Vec<f64>, f64, usize, f64) {
    let v: Vec<f64> = spec.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    let ds: Vec<f64> = match v.first() { Some(&d) => vec![d], None => (0..=34).map(|i| 2.5 * i as f64).collect() };
    let ks: Vec<usize> = match v.get(1) { Some(&k) => vec![k as usize], None => (0..obj.n).collect() };
    ds.par_iter().flat_map(|&d| ks.par_iter().map(move |&k| (d, k)))
        .map(|(d, k)| { let p = minipump(obj, d, k); let sc = exit_score(&obj.replay(&p), depth, mode); (p, d, k, sc) })
        .reduce(|| (vec![], 0.0, 0, f64::NEG_INFINITY), |a, b| if b.3 > a.3 { b } else { a })
}

/// The K-climb seed: a floor-grazing pump flown by events, not by tick counts, so one set of
/// parameters means the same flight at every `y0` and on every lap.
///
/// From rest: dive at `d` degrees nose-down until the clearance falls under `lvl`; hold 0 until it
/// falls under `pull`; pull at `a` while `v_y` still rises; then relax the pitch linearly in `v_y`,
/// from `a` at the peak of `v_y` to `a2` as `v_y` reaches 0 (the apex). That is one climb. After each climb but the last, glide at 0 for `g` ticks and dive
/// again; after the last, hold `end` to the exit. `lvl <= pull` skips the level-off.
///
/// Why these phases. They are the laps of the best `y0 = 32` range schedule (v11, 1362 ticks,
/// five dips): a nose-down dive to about 3 blocks, pitch 0 for about ten ticks while the dive's
/// `v_y` turns into forward speed (the `v_y < 0` conversion is largest at pitch 0), a pull that
/// peaks at -47 to -57 as the flight grazes the floor, a climb whose pitch relaxes about linearly
/// to -10 at the apex, then a glide or straight into the next dive. A relax held at a constant
/// pitch instead (the first version) flew laps that each lost about a block at `y0 = 32`, so the
/// fit could not use more than six of them; the same laps optimized gain 3-9 blocks each. Why events: a lap's timing depends on how fast it arrives, so
/// a tick count fitted at one `y0` or on one lap is wrong on the next.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pumps { k: usize, d: f64, lvl: f64, pull: f64, a: f64, a2: f64, g: usize, end: f64 }

impl Pumps {
    fn show(&self) -> String {
        format!("pumps:{},d={},lvl={},pull={},a={},a2={},g={},end={}", self.k, self.d, self.lvl, self.pull, self.a,
                self.a2, self.g, self.end)
    }
}

/// Flies `prefix`, then `Pumps` from where it leaves off, for `n` ticks in all, over a floor
/// `depth` below the start. After a non-empty prefix the controller starts in the glide, as after
/// a climb. Returns the pitches (after the exit, the last pitch repeated) and the exit score in
/// `mode`.
fn pumps(obj: &Objective, depth: f64, mode: Exit, q: &Pumps, prefix: &[f64]) -> (Vec<f64>, f64) {
    #[derive(PartialEq)]
    enum Ph { Dive, Level, Pull, Relax, Glide(usize), End }
    let mut p = Vec::with_capacity(obj.n);
    let mut st = vec![State { pos: Vec3::ZERO, vel: obj.v0 }];
    let mut ph = if q.k == 0 { Ph::End } else if prefix.is_empty() { Ph::Dive } else { Ph::Glide(q.g) };
    let mut climbs = 0;
    let mut vtop = 1.0;
    for &x in &prefix[..prefix.len().min(obj.n)] {
        let s = st.last().unwrap().clone();
        if s.pos.y + depth < 0.0 { break }
        st.push(s.ticked_cached(PitchTrig::new(x as f32)));
        p.push(x);
    }
    while p.len() < obj.n {
        let s = st.last().unwrap().clone();
        let h = s.pos.y + depth;
        if h < 0.0 { break }
        // Phase changes read the current state; `Pull` and `Relax` end on the next state's `v_y`.
        loop {
            let next = match ph {
                Ph::Dive if h < q.lvl => Ph::Level,
                Ph::Level if h < q.pull => Ph::Pull,
                Ph::Glide(0) => Ph::Dive,
                _ => break,
            };
            ph = next;
        }
        let x = match ph { Ph::Dive => q.d, Ph::Level | Ph::Glide(_) => 0.0, Ph::Pull => q.a,
                           Ph::Relax => q.a2 + (q.a - q.a2) * (s.vel.y / vtop).clamp(0.0, 1.0),
                           Ph::End => q.end };
        let nx = s.ticked_cached(PitchTrig::new(x as f32));
        match ph {
            Ph::Pull if nx.vel.y < s.vel.y && nx.vel.y > 0.0 => { ph = Ph::Relax; vtop = s.vel.y }
            Ph::Relax if nx.vel.y < 0.0 => {
                climbs += 1;
                ph = if climbs >= q.k { Ph::End } else { Ph::Glide(q.g) };
            }
            Ph::Glide(r) => ph = Ph::Glide(r.saturating_sub(1)),
            _ => {}
        }
        p.push(x);
        st.push(nx);
    }
    let sc = exit_score(&st, depth, mode);
    let last = p.last().copied().unwrap_or(0.0);
    p.resize(obj.n, last);
    (p, sc)
}

/// `pumps:<K>[,<key>=<v>..]` fitted: every parameter not given is searched, first on a coarse
/// grid, then by coordinate passes on finer ones, all on the seed's own exit score. Keys `d`,
/// `lvl`, `pull`, `a`, `a2`, `g`, `end` as in `Pumps`.
fn fit_pumps(obj: &Objective, depth: f64, mode: Exit, spec: &str, prefix: &[f64]) -> (Pumps, Vec<f64>, f64) {
    let mut it = spec.split(',').filter(|x| !x.is_empty());
    let k: usize = it.next().expect("pumps:<K>").parse().expect("pumps:<K>");
    let mut fixed: Vec<(String, f64)> = Vec::new();
    for kv in it {
        let (key, v) = kv.split_once('=').unwrap_or_else(|| panic!("pumps: {kv} is not key=value"));
        assert!(["d", "lvl", "pull", "a", "a2", "g", "end"].contains(&key), "pumps: unknown key {key}");
        fixed.push((key.into(), v.parse().unwrap_or_else(|e| panic!("pumps: {kv}: {e:?}"))));
    }
    let get = |key: &str| fixed.iter().find(|(x, _)| x == key).map(|(_, v)| *v);
    let set = |mut q: Pumps, i: usize, v: f64| -> Pumps {
        match i { 0 => q.d = v, 1 => q.lvl = v, 2 => q.pull = v, 3 => q.a = v, 4 => q.a2 = v,
                  5 => q.g = v.max(0.0).round() as usize, _ => q.end = v }
        q
    };
    const KEYS: [&str; 7] = ["d", "lvl", "pull", "a", "a2", "g", "end"];
    // Coarse grids, searched jointly, and fine ones, searched a coordinate at a time.
    let coarse: [Vec<f64>; 7] = [
        vec![0.0, 10.0, 20.0, 30.0, 40.0, 55.0], vec![0.0, 1.5, 3.0, 5.0, 8.0],
        vec![0.1, 0.25, 0.5, 1.0, 2.0], vec![-30.0, -45.0, -60.0], vec![0.0, -10.0, -20.0],
        vec![0.0, 15.0, 40.0, 80.0], vec![0.0, -13.0, -22.0]];
    let fine: [Vec<f64>; 7] = [
        (0..=36).map(|i| 2.5 * i as f64).collect(), (0..=40).map(|i| i as f64 / 4.0).collect(),
        (0..=60).map(|i| i as f64 / 20.0).collect(), (0..=40).map(|i| -10.0 - 2.0 * i as f64).collect(),
        (0..=20).map(|i| -2.0 * i as f64).collect(), (0..=40).map(|i| 5.0 * i as f64).collect(),
        (0..=30).map(|i| -1.0 * i as f64).collect()];
    let base = Pumps { k, d: 0.0, lvl: 0.0, pull: 0.0, a: 0.0, a2: 0.0, g: 0, end: 0.0 };
    let base = (0..7).fold(base, |q, i| set(q, i, get(KEYS[i]).unwrap_or(coarse[i][0])));
    let free: Vec<usize> = (0..7).filter(|&i| get(KEYS[i]).is_none()).collect();
    let score = |q: &Pumps| pumps(obj, depth, mode, q, prefix).1;
    let mut grid = vec![base];
    for &i in &free { grid = grid.into_iter().flat_map(|q| coarse[i].iter().map(move |&v| set(q, i, v))).collect() }
    let mut best = grid.par_iter().map(|q| (*q, score(q)))
        .reduce(|| (base, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
    for _ in 0..4 {
        let before = best.1;
        for &i in &free {
            best = fine[i].par_iter().map(|&v| { let q = set(best.0, i, v); (q, score(&q)) })
                .reduce(|| best, |a, b| if b.1 > a.1 { b } else { a });
        }
        if best.1 <= before { break }
    }
    let (p, sc) = pumps(obj, depth, mode, &best.0, prefix);
    (best.0, p, sc)
}

/// First-exit ascent under a cap that doubles, up to `--nmax`, for as long as the answer
/// survives it. Why: a survivor is scored on a value-to-go guess rather than an exit, so a cap
/// that binds decides the answer. The ascent returns after the first pass whose schedule
/// outlasts the cap, and the doubled one starts from that schedule. Doubling is a heuristic, not a bound -- a pump that gains
/// energy survives every cap, and says so in the header.
fn exit_cmd(e: &ExitArgs) {
    let a = &e.c;
    // Dead ticks cost one replay each, not a search, so a cap far past the exit is nearly free;
    // what costs is the live flight, quadratically.
    let mut n = e.n;
    let nmax = e.nmax;
    let depth = a.y0;
    let mode = e.mode;
    let spec = e.init.as_str();
    let mut seed_note = String::new();
    let mut p = if let Some(r) = spec.strip_prefix("pumps:") {
        // Fitted and flown on the largest cap, and the ascent starts at the first doubling of `--n`
        // that the seed does not outlast. why? growing from a cap shorter than the seed's flight
        // keeps only the seed's first `n` pitches and repeats the last one, which erases every
        // lap after the first.
        // `--after <file>`: continue that schedule from its last apex (the last tick whose `v_y`
        // turns negative before its exit) with K more laps -- a continuation in laps, since the
        // seed's own fixed-parameter laps lose energy and stop after a few.
        let prefix = e.after.as_ref().map_or(vec![], |f| {
            let x = read_pitches(f);
            let st = a.obj(x.len()).replay(&x);
            let end = live_end(&st, depth);
            let apex = (1..end).rev().find(|&t| st[t - 1].vel.y > 0.0 && st[t].vel.y <= 0.0).unwrap_or(0);
            x[..apex].to_vec()
        });
        let (q, p, sc) = fit_pumps(&a.obj(nmax), depth, mode, r, &prefix);
        if let Some(f) = &e.after { seed_note = format!("  after {f} ({} ticks)", prefix.len()) }
        let live = Floor { depth, ..Default::default() }.survived(&a.obj(nmax).replay(&p));
        while n < nmax && live >= n { n = (2 * n).min(nmax) }
        seed_note = format!("{seed_note}  seed {} seed score {sc:.4}", q.show());
        p[..n].to_vec()
    } else {
        match spec.strip_prefix("minipump") {
            Some(r) => {
                let (p, d, k, sc) = fit_minipump(&a.obj(n), depth, mode, r.trim_start_matches(':'));
                seed_note = format!("  minipump d {d} k {k} seed score {sc:.4}");
                p
            }
            None => init(spec, n),
        }
    };
    let passes = e.passes;
    let ke = e.ke;
    // `--shrink` is applied to the bubble once per stall stage under `grad` (`GradOpts`).
    assert!(e.bubble.len() == 2, "--bubble wants margin,weight");
    let mut bub = (e.bubble[0], e.bubble[1]);
    let shrink = e.shrink;
    let bub0 = bub;
    let pen_spec = e.penalty.as_str();
    let shape = PriceShape::parse(pen_spec).unwrap_or_else(|e| panic!("--penalty: {e}"));
    let mv_spec = e.moves.clone()
        .unwrap_or(if e.shift { "tick,shift".into() } else { "tick".into() });
    let tick = mv_spec.split(',').any(|m| m == "tick");
    let moves: Vec<Move> = mv_spec.split(',').filter(|m| *m != "tick").flat_map(Move::parse).collect();
    let method = e.method.as_str();
    let (grad, coord) = match method { "tick" => (false, true), "grad" => (true, false), _ => (true, true) };
    if grad {
        // The gradient does not differentiate a multi-pitch move, so refuse them rather than
        // quietly optimize something else.
        if !coord && (e.moves.is_some() || e.shift) {
            panic!("--moves/--shift: --method grad makes no moves; use grad+tick")
        }
    }
    let graze = e.graze.as_str();
    let go = GradOpts { iters: e.iters, mem: e.mem, max_step: e.max_step, c1: e.c1,
                        margins: if graze == "off" { vec![] } else { graze.split(',').map(|x| x.parse().expect("--graze")).collect() },
                        bub, shrink, stages: e.anneal + 1 };
    let clock = std::time::Instant::now();
    let rough = Rough { shape, ..a.opts(passes).rough };
    let mut used = Vec::new();
    // The bubble the gradient's last stage scored with.
    let mut gbub = bub;
    let (obj, sc) = loop {
        let obj = a.obj(n);
        let mut q = init_from(&p, n);
        let mut tag = String::new();
        if grad {
            let (r, _, k, b) = ascend_grad(&obj, depth, &q, mode, ke, rough, &GradOpts { bub, ..go.clone() }, n < nmax);
            q = r.iter().map(|&x| x as f32 as f64).collect();
            tag = format!("g{k}");
            gbub = b;
        }
        let sc = if coord {
            let (r, s, k) = ascend(&obj, depth, &q, mode, ke, rough, passes,
                                   a.tol, n < nmax, tick, &moves, &mut bub, shrink);
            q = r;
            tag = if tag.is_empty() { format!("{k}") } else { format!("{tag}+{k}") };
            s
        } else {
            // Scored as `ascend` scores what it keeps: on the `f32` pitches the replay flies.
            let st = obj.replay(&q);
            utility(&st, depth, mode, ke) - mode.per_block() * bubble_cost(&st, depth, gbub)
                - Rough { mu: rough.mu * mode.per_block(), ..rough }.cost(&q)
        };
        used.push(format!("{tag}@{n}"));
        p = q;
        let fl = Floor { depth, ..Default::default() };
        if fl.survived(&obj.replay(&p)) < n || n >= nmax { break (obj, sc) }
        n = (2 * n).min(nmax);
    };
    let st = obj.replay(&p);
    let fl = Floor { depth, ..Default::default() };
    let k = fl.survived(&st);
    let verdict = if k >= n { "  SURVIVES the cap: score is a value-to-go guess" } else { "" };
    // The default method's header is unchanged; the others name themselves and their settings,
    // and add their wall time.
    let (mnote, wall) = if grad {
        (format!(" method {method} iters {} mem {} max-step {} c1 {} graze {}{}", go.iters, go.mem, go.max_step, go.c1,
                 if go.margins.is_empty() { "off".into() } else { go.margins.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(",") },
                 if go.stages > 1 { format!(" anneal {}", go.stages - 1) } else { String::new() }),
         format!("  wall {:.2}s", clock.elapsed().as_secs_f64()))
    } else { (String::new(), String::new()) };
    let text = format!("# exit {mode:?} y0 {depth} v0 ({}, {}) cap {n} init {spec} mu {} limit {} penalty {pen_spec} moves {mv_spec} ke {ke} bubble {},{} shrink {shrink}{seed_note}{mnote}\n\
                        # score {sc:.4}  t* {:.4}  z(t*) {:.4}  survived {k}  exit KE {:.4}  passes {}{wall}{verdict}\n{}\n",
                       obj.v0.y, obj.v0.z, a.mu, a.limit, bub0.0, bub0.1,
                       exit_score(&st, depth, Exit::Time), exit_score(&st, depth, Exit::Dist),
                       exit_energy(&st, depth).unwrap_or(f64::NAN), used.join(","),
                       fmt_pitches(&p[..(k + 1).min(n)]));
    match &e.out {
        Some(o) => { std::fs::write(o, &text).unwrap(); print!("{}", text.lines().take(2).collect::<Vec<_>>().join("\n") + "\n") }
        None => print!("{text}"),
    }
}

fn probe() {
    let (mut ms, mut msp, mut bg, mut bgp) = (f64::INFINITY, 0.0, 0.0, 0.0);
    for i in -900..=900 {
        let p = 0.1 * i as f64;
        let e = equilibrium(p);
        if e.y < 0.0 && -e.y < ms { ms = -e.y; msp = p }
        if e.y < 0.0 && e.z / -e.y > bg { bg = e.z / -e.y; bgp = p }
    }
    println!("steady glide: min sink {ms:.5} b/tick at pitch {msp:.1}; best glide ratio {bg:.4} at {bgp:.1}");
    println!("constant pitch from v0 = 0, pitch step 0.1:");
    println!("{:>3} {:>6} {:>7} {:>8} {:>7}   {:>12}   {:>12}", "y0", "ticks", "at", "z", "at",
             "hold 0 t, z", "hold -13 t, z");
    for y0 in 1..=8 {
        let fl = Floor { depth: y0 as f64, ..Default::default() };
        let (mut bt, mut btp, mut bz, mut bzp) = (0usize, 0.0, 0.0f64, 0.0);
        for i in -900..=900 {
            let p = 0.1 * i as f64;
            let st = replay_from(Vec3::ZERO, &vec![p; 400]);
            let t = fl.survived(&st);
            if t > bt { bt = t; btp = p }
            if st[t].pos.z > bz { bz = st[t].pos.z; bzp = p }
        }
        let at = |q: f64| { let st = replay_from(Vec3::ZERO, &vec![q; 400]);
                            let t = fl.survived(&st); (t, st[t].pos.z) };
        let ((t0, z0), (t13, z13)) = (at(0.0), at(-13.0));
        println!("{y0:>3} {bt:>6} {btp:>7.1} {bz:>8.3} {bzp:>7.1}   {t0:>4} {z0:>7.3}   {t13:>4} {z13:>7.3}");
    }
}

fn solve(a: &Common, n: usize, spec: &str, passes: usize, out: Option<&str>) {
    let obj = a.obj(n);
    let opts = a.opts(passes);
    let r = polish(&obj, &init(spec, n), opts);
    let f = flown(&obj, &opts.floor, &r.pitches);
    let text = format!(
        "# floor y0 {} margin {} weight {} wall {}\n# n {n} lambda {} v0 ({}, {}) mu {} limit {}\n\
         # init {spec}  passes {}  residual {:.2e}  lag1 {:+.3}\n\
         # survived {}  clearance {:+.4}  z {:.4}  TE {:.4}  floor cost {:.4}  J {:.4}\n{}\n",
        opts.floor.depth, opts.floor.margin, opts.floor.weight, opts.floor.wall,
        obj.lambda, obj.v0.y, obj.v0.z, opts.rough.mu, opts.rough.limit,
        r.passes, r.residual, r.lag1, f.survived, f.clearance, f.z, f.te, f.cost, r.j,
        fmt_pitches(&r.pitches));
    match out {
        Some(o) => { std::fs::write(o, &text).unwrap(); print!("{}", text.lines().take(4).collect::<Vec<_>>().join("\n") + "\n") }
        None => print!("{text}"),
    }
}

fn main() {
    set_trig_mode(TrigMode::MthLut);
    set_flight_mode(FlightMode::Reference);
    match Cli::parse().cmd {
        Cmd::Probe => probe(),
        Cmd::Solve { c, n, init, passes, out } => solve(&c, n, &init, passes, out.as_deref()),
        Cmd::Endure { c, passes, nmax, misses, anneal, shrink, out } =>
            endure(&c, passes, nmax, misses, anneal, shrink, out.as_deref()),
        Cmd::Exit(e) => exit_cmd(&e),
        Cmd::Depth { c: a, file, every } => {
            // How far a schedule's replay goes below its start, from `--vy/--vz`: the floor
            // clearance a repeated cycle needs.
            let p = read_pitches(&file);
            let st = a.obj(p.len()).replay(&p);
            let lo = st.iter().map(|s| s.pos.y).fold(f64::INFINITY, f64::min);
            let hi = st.iter().map(|s| s.pos.y).fold(f64::NEG_INFINITY, f64::max);
            let e = st.last().unwrap();
            println!("min y {lo:.3}  max y {hi:.3}  end y {:.3}  end v ({:.4}, {:.4})", e.pos.y, e.vel.y, e.vel.z);
            if let Some(every) = every {
                for (t, s) in st.iter().enumerate().step_by(every) {
                    println!("  t {t:4}  y {:8.3}  TE {:8.3}  v ({:+.4}, {:.4})", s.pos.y, s.total_energy(), s.vel.y, s.vel.z);
                }
            }
        }
        Cmd::Laps { c: a, file } => {
            // A pump's laps: every local minimum (dip) and maximum (peak) of the clearance
            // before the exit, with the energy above the floor, `TE + y0`, there.
            let p = read_pitches(&file);
            let depth = a.y0;
            let st = a.obj(p.len()).replay(&p);
            let k = live_end(&st, depth);
            let h = |j: usize| st[j].pos.y + depth;
            println!("# exit at state {k} of {}", st.len() - 1);
            for j in 1..k.min(st.len() - 1) {
                let kind = if h(j) <= h(j - 1) && h(j) < h(j + 1) { "dip" }
                           else if h(j) >= h(j - 1) && h(j) > h(j + 1) { "peak" } else { continue };
                let s = &st[j];
                println!("{kind:<4} t {j:5}  h {:9.5}  E {:8.4}  v ({:+.4}, {:.4})  pitch {:.2}",
                         h(j), s.total_energy() + depth, s.vel.y, s.vel.z, p[j - 1]);
            }
        }
        Cmd::Safety { c: a, n, init: spec, passes } => {
            let obj = a.obj(n);
            let (p, c) = safety(&obj, &a.floor(), &init(&spec, n), a.limit, passes);
            println!("# clearance {c:+.5}  survived {}\n{}", a.floor().survived(&obj.replay(&p)), fmt_pitches(&p));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(y: &[f64]) -> Vec<State> {
        y.iter().enumerate().map(|(t, &y)| State { pos: Vec3::new(0.0, y, t as f64), vel: Vec3::ZERO }).collect()
    }

    // The gradient checks run under the default trig mode, `libm`, which no unit test here
    // changes: smooth between branches, so central differences are a fair reference. Under
    // `mth_lut` (a 65536-cell table) they would be differences of a staircase.

    /// A glide, a pull and a glide again, every pitch at least a degree from the forward-to-up
    /// switch at 0, and dyadic, so `p +- 1/16` round-trips through `f32` exactly.
    fn schedule(n: usize) -> Vec<f64> {
        let mut x: u64 = 12345;
        (0..n).map(|t| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (x >> 40) as f64 / (1u64 << 24) as f64 - 0.5;
            let base = if (60..90).contains(&t) { -30.0 } else { 9.0 };
            ((base + 6.0 * r) * 64.0).round() / 64.0
        }).collect()
    }

    /// The worst relative error of `g` against central differences of `f` at `ticks`, and the
    /// finite difference and gradient where it occurred.
    fn fd_worst(f: &dyn Fn(&[f64]) -> f64, g: &[f64], p: &[f64], ticks: &[usize]) -> (f64, f64, f64) {
        let h = 1.0 / 16.0;
        let mut worst = (0.0, 0.0, 0.0);
        for &t in ticks {
            let (mut a, mut b) = (p.to_vec(), p.to_vec());
            a[t] += h;
            b[t] -= h;
            let fd = (f(&a) - f(&b)) / (2.0 * h);
            let err = (fd - g[t]).abs() / g[t].abs().max(1e-2);
            if err > worst.0 { worst = (err, fd, g[t]) }
        }
        worst
    }

    /// What `utility_grad` differentiates: `utility` less the bubble in score units.
    fn priced(obj: &Objective, depth: f64, mode: Exit, ke: f64, bub: (f64, f64), q: &[f64]) -> f64 {
        let st = obj.replay(q);
        utility(&st, depth, mode, ke) - mode.per_block() * bubble_cost(&st, depth, bub)
    }

    /// The first-exit utility's gradient, through the secant at the crossing, for both modes,
    /// with and without the exit-energy term, with and without the bubble, and for a survivor's
    /// value-to-go.
    #[test]
    fn exit_gradient_matches_finite_differences() {
        let n = 200;
        let obj = Objective { v0: Vec3::ZERO, n, lambda: 0.0 };
        let p = schedule(n);
        let st = obj.replay(&p);
        // The floor midway between states 170 and 171, which must be the first to go under it.
        let depth = -0.5 * (st[170].pos.y + st[171].pos.y);
        assert!(st[..=170].iter().all(|s| s.pos.y + depth > 0.05), "the test schedule dips early");
        assert_eq!(live_end(&st, depth), 171);
        // A bubble deep enough to hold the pre-exit dip and the last few states before the
        // crossing, and one too thin to hold anything but the crossing term.
        let lo = pre_exit_min(&st, 171, depth);
        let inside = |m: f64| (1..171).filter(|&j| st[j].pos.y + depth < m).count();
        let (thick, thin) = ((lo + 0.4, 0.3), (0.02, 0.05));
        assert!(inside(thick.0) >= 4 && inside(thin.0) == 0, "bubble holds {} and {}", inside(thick.0), inside(thin.0));
        let ticks = [0, 5, 59, 60, 75, 89, 90, 120, 168, 169, 170];
        for mode in [Exit::Time, Exit::Dist] {
            for ke in [0.0, -0.3] {
                let bare = utility_grad(&obj, depth, mode, ke, (1.0, 0.0), &p).g;
                for bub in [(1.0, 0.0), thick, thin] {
                    let eg = utility_grad(&obj, depth, mode, ke, bub, &p);
                    // The bubble's own share of the gradient, at the ticks checked, relative to the
                    // utility's: a check that it is not lost in the tolerance.
                    let share = ticks.iter().map(|&t| (eg.g[t] - bare[t]).abs() / bare[t].abs().max(1e-2)).fold(0.0, f64::max);
                    if bub.1 != 0.0 { assert!(share > 0.05, "{mode:?} bubble {bub:?}: its gradient is only {share:.1e} of the utility's") }
                    let (err, fd, g) = fd_worst(&|q| priced(&obj, depth, mode, ke, bub, q), &eg.g, &p, &ticks);
                    eprintln!("exit {mode:?} ke {ke} bubble {bub:?}: worst relative error {err:.2e} (fd {fd:.6}, adjoint {g:.6}), bubble share {share:.2}");
                    assert!(err < 1e-4, "{mode:?} ke {ke} bubble {bub:?}: fd {fd} vs adjoint {g}");
                    assert!(eg.g[171..].iter().all(|&x| x == 0.0), "a dead pitch has a gradient");
                }
            }
            // Survivors: the floor far below, and just under the lowest state, inside a bubble.
            let top = -st.iter().map(|s| s.pos.y).fold(f64::INFINITY, f64::min) + 0.05;
            for (d, bub) in [(1000.0, (1.0, 0.0)), (top, (0.5, 0.3))] {
                let eg = utility_grad(&obj, d, mode, 0.0, bub, &p);
                let (err, fd, g) = fd_worst(&|q| priced(&obj, d, mode, 0.0, bub, q), &eg.g, &p, &[0, 60, 150, 199]);
                eprintln!("survivor {mode:?} bubble {bub:?}: worst relative error {err:.2e} (fd {fd:.6}, adjoint {g:.6})");
                assert!(err < 1e-4, "survivor {mode:?} bubble {bub:?}: fd {fd} vs adjoint {g}");
            }
        }
    }

    /// The whole ascent objective -- utility less every curvature price, on the schedule
    /// flattened from its exit -- against `slopes`, which folds the dead tail's price onto the
    /// last live pitch.
    #[test]
    fn ascent_gradient_matches_finite_differences() {
        let n = 200;
        let obj = Objective { v0: Vec3::ZERO, n, lambda: 0.0 };
        let mut p = schedule(n);
        let st = obj.replay(&p);
        let depth = -0.5 * (st[170].pos.y + st[171].pos.y);
        flat_tail(&mut p, 171);
        let bubble = (pre_exit_min(&st, 171, depth) + 0.4, 0.3);
        for (shape, bub) in [(PriceShape::L2(2.0), (1.0, 0.0)), (PriceShape::Huber(0.5), (1.0, 0.0)),
                             (PriceShape::L1, (1.0, 0.0)), (PriceShape::L2(2.0), bubble)] {
            // A price far above the default, so its gradient is not lost under the utility's.
            let pen = Rough { mu: 0.05, shape, ..Default::default() };
            let s = slopes(&obj, depth, Exit::Time, 0.0, bub, &pen, &p);
            assert_eq!(s.up, s.down, "no pitch sits on the switch, so the two slopes agree");
            let f = |q: &[f64]| grad_total(&obj, depth, Exit::Time, 0.0, bub, &pen, q).1;
            let (err, fd, g) = fd_worst(&f, &s.up, &p, &[0, 1, 59, 60, 61, 100, 168, 169, 170]);
            eprintln!("ascent objective, {shape}, bubble {bub:?}: worst relative error {err:.2e} (fd {fd:.6}, adjoint {g:.6})");
            assert!(err < 1e-4, "{shape} bubble {bub:?}: fd {fd} vs adjoint {g}");
        }
    }

    /// At a pitch of exactly 0 the forward-to-up branch is off but one step down turns it on:
    /// `slopes` gives the two one-sided derivatives, and they must match one-sided differences.
    #[test]
    fn switch_slopes_match_one_sided_differences() {
        let n = 200;
        let obj = Objective { v0: Vec3::ZERO, n, lambda: 0.0 };
        let mut p = schedule(n);
        let st = obj.replay(&p);
        let depth = -0.5 * (st[170].pos.y + st[171].pos.y);
        for t in [20, 100, 140] { p[t] = 0.0 }
        let pen = Rough::default();
        let s = slopes(&obj, depth, Exit::Time, 0.0, (1.0, 0.0), &pen, &p);
        let f = |q: &[f64]| grad_total(&obj, depth, Exit::Time, 0.0, (1.0, 0.0), &pen, q).1;
        let h = 1.0 / 1024.0;
        for t in [20, 100, 140] {
            let (mut a, mut b) = (p.clone(), p.clone());
            a[t] += h;
            b[t] -= h;
            let (up, down) = ((f(&a) - f(&p)) / h, (f(&p) - f(&b)) / h);
            eprintln!("switch at {t}: up fd {up:.5} slope {:.5}; down fd {down:.5} slope {:.5}", s.up[t], s.down[t]);
            // One-sided differences carry an O(h) error; the gap between the sides is O(1).
            let near = |fd: f64, g: f64| (fd - g).abs() < 1e-5 + 1e-2 * g.abs();
            assert!(near(up, s.up[t]) && near(down, s.down[t]));
            assert!((s.down[t] - s.up[t]).abs() > 1e-3, "the corner should be visible");
        }
    }

    /// The bubble must not jump as the crossing slides past a tick.
    #[test]
    fn bubble_is_continuous_across_a_tick() {
        let eps = 1e-9;
        let (a, b) = (at(&[0.0, -0.5, -1.0 + eps, -1.5]), at(&[0.0, -0.5, -1.0 - eps, -1.5]));
        let (x, y) = (bubble_cost(&a, 1.0, (0.5, 1.0)), bubble_cost(&b, 1.0, (0.5, 1.0)));
        assert!((x - y).abs() < 1e-6, "{x} {y}");
    }

    /// The point of interpolating: as the crossing slides from one tick to the next, `t*` and
    /// `z(t*)` do not jump.
    #[test]
    fn exit_is_continuous_across_a_tick() {
        let eps = 1e-9;
        let (a, b) = (at(&[0.0, -0.5, -1.0 + eps, -1.5]), at(&[0.0, -0.5, -1.0 - eps, -1.5]));
        for mode in [Exit::Time, Exit::Dist] {
            let (x, y) = (exit_score(&a, 1.0, mode), exit_score(&b, 1.0, mode));
            assert!((x - 2.0).abs() < 1e-6 && (y - 2.0).abs() < 1e-6, "{mode:?}: {x} {y}");
        }
    }
}
