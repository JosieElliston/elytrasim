//! Produce a corpus of optimal pitch schedules over a grid of `(n, lambda, vy0, vz0)`.
//!
//! The product is one file per cell: whitespace-separated pitches in degrees under a `#` header
//! that states the utility function and initial conditions completely. The point of a profile
//! is not where it came from -- it is that the pitches are observably a coordinate optimum of
//! the objective the header names. `sweep verify` re-checks exactly that, from the file alone.
//!
//! Subcommands:
//!   polish  <opts> [--out <file>]   solve one cell and print or write it
//!   verify  <file>...               re-certify profiles against their own headers
//!   bench   [--n <n>]               seconds per pass and per profile, cold and warm
//!   pilot   [--out <dir>]           the coarse grid: axis bounds, strides, seed quality
//!   run     --out <dir> [--shard <i>,..]       the full sweep (shards index --vys x --vzs, vy-major)
//!
//! `sweep <subcommand> --help` lists every option; what follows is what they mean.
//!
//! Cell options: --n, --lambda, --vy, --vz, --passes, --tol, --trig, --flight, --init <file>
//!   --flick-at <tick> restricts the first tick at or below --flick-pitch <deg> (default -80).
//!   A seed given with --init or --anchor does NOT supply the route; --trig/--flight still do,
//!   defaults included, and a seed built under another route is refused. See `require_physics`.
//!
//! Roughness options: --mu <per deg/tick^2>, --penalty l1|l2:<d>|huber:<d> (the shape --mu
//!   prices; `l2:2` by default since 2026-09-24, `l1` before -- see `PriceShape`), --mu-tv, --cap, --slew-cap, --limit <deg>
//!   The l1 curvature price, and the current chatter regularizer. Unlike a pass budget it is a
//!   *price*, so the polish runs to convergence and the answer is a real optimum that `verify`
//!   can re-check. `runs/atlas` was built at `--mu 1e-4 --limit 85`, l1.
//!
//! Steady state: --steady
//!   Off by default. Re-solves `v0` to the schedule's own fixed point after every pass, so the
//!   result is the cycle you would fly back to back: at convergence `v_final == v0` identically.
//!   It moves `v0` and nothing else -- `v_n` stays free and the roughness price stays unwrapped,
//!   both deliberately -- so a steady profile is a repeatable *open-horizon* optimum and its
//!   terminal price at the cut is `dJ/dv_n`, not a periodic costate. See README-sweep.md for the
//!   two places that distinction is visible.
//!   `--vy/--vz` then *seed* that iteration instead of naming the answer, and the written
//!   header states the fixed point actually reached. `v0` stops being a grid axis, so `run`
//!   writes these under `steady/` rather than a `vy.../vz...` shard and takes a single
//!   --vys/--vzs pair. Worth between +0.94 blocks of TE per lap (n=150) and nothing at all
//!   (n~330); see `steady_vel` in `opt.rs`.
//!
//! Jitter options: --jitter <sigma>, --draws <k>, --fixed-draws, --seed <s>
//!   `--jitter` is the standard deviation of the *initial velocity* error, in blocks/tick,
//!   applied independently to v_y and v_z. It was the regularizer before the curvature price
//!   and is superseded for that purpose -- it only suppresses chatter while the polish is
//!   stopped early, and current runs set it to 0. What it is still good for is the question it
//!   actually measures: whether a schedule works from a starting velocity you do not know
//!   exactly. Perturbing the pitches instead does not work and was measured -- see `Jitter`.

use elytrasim::opt::*;
use elytrasim::sim::*;
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------- argument plumbing

#[derive(Parser)]
struct Cli {
    /// Trig route; anywhere on the line. Unset keeps the library default.
    #[arg(long, global = true)]
    trig: Option<TrigMode>,
    /// Flight kernel; anywhere on the line. Unset keeps the library default.
    #[arg(long, global = true)]
    flight: Option<FlightMode>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Solve one cell and print or write it.
    #[command(allow_negative_numbers = true)]
    Polish {
        #[command(flatten)]
        cell: Cell,
        #[command(flatten)]
        polish: Polish,
        #[arg(long, default_value_t = 200)]
        passes: usize,
        /// Seed: a profile or a bare pitch list, stretched to `--n`; default is the policy seed.
        #[arg(long)]
        init: Option<String>,
        /// Median-filter width applied to the seed before polishing.
        #[arg(long, default_value_t = 1)]
        premedian: usize,
        /// Box-filter width applied to the seed after the median.
        #[arg(long, default_value_t = 1)]
        presmooth: usize,
        #[arg(long)]
        out: Option<String>,
    },
    /// Re-certify profiles against their own headers.
    Verify {
        #[arg(required = true)]
        files: Vec<String>,
    },
    /// Seconds per pass and per profile, cold and warm.
    #[command(allow_negative_numbers = true)]
    Bench {
        #[command(flatten)]
        cell: Cell,
        /// Passes per timed block.
        #[arg(long, default_value_t = 10)]
        every: usize,
        #[arg(long, default_value_t = 80)]
        passes: usize,
    },
    /// The coarse grid: axis bounds, strides, seed quality.
    #[command(allow_negative_numbers = true)]
    Pilot {
        #[arg(long, default_value = "sweep-pilot")]
        out: String,
        #[arg(long, default_value_t = 60)]
        passes: usize,
        #[arg(long, value_delimiter = ',', default_value = "100,200,300,400,500")]
        ns: Vec<usize>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true, default_value = "-2,-1,-0.5,0,0.5,1,2")]
        lams: Vec<f64>,
        /// Explicit `vy:vz` points; overrides the --vys x --vzs product.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true, value_parser = parse_vel)]
        vels: Vec<(f64, f64)>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true, default_value = "0,0.1,0.2")]
        vys: Vec<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true, default_value = "0,0.1,0.2")]
        vzs: Vec<f64>,
    },
    /// The full sweep.
    #[command(allow_negative_numbers = true)]
    Run {
        #[command(flatten)]
        grid: GridArgs,
        #[command(flatten)]
        polish: Polish,
        #[arg(long, default_value_t = 200)]
        passes: usize,
        #[arg(long, default_value = "sweep")]
        out: String,
        /// Re-solve cells that already have a file.
        #[arg(long)]
        force: bool,
        /// Seed for the anchor cell of every shard.
        #[arg(long)]
        anchor: Option<String>,
        /// Shard indices to run, into the --vys x --vzs product (vy-major); default all.
        #[arg(long, value_delimiter = ',')]
        shard: Vec<usize>,
    },
    /// The physics fingerprint, route and commit.
    Fingerprint,
    /// Per-tick residuals of one profile, and their structure.
    Residuals { file: String },
    /// The shape of each profile's schedule.
    Structure {
        #[arg(required = true)]
        files: Vec<String>,
    },
}

fn parse_vel(p: &str) -> Result<(f64, f64), String> {
    let (y, z) = p.split_once(':').ok_or_else(|| format!("want vy:vz, got {p:?}"))?;
    let num = |x: &str| x.parse::<f64>().map_err(|e| format!("{x:?}: {e}"));
    Ok((num(y)?, num(z)?))
}

/// One cell's objective.
#[derive(clap::Args)]
struct Cell {
    #[arg(long, default_value_t = 300)]
    n: usize,
    #[arg(long, default_value_t = 0.0)]
    lambda: f64,
    /// Initial v_y. Zero, not the reference cycle's start: (0.167467, 0.200887) is the velocity
    /// REPLAY_PITCHES_300 happens to close on, which makes it special for that one profile and
    /// for nothing else.
    #[arg(long, default_value_t = 0.0)]
    vy: f64,
    #[arg(long, default_value_t = 0.0)]
    vz: f64,
}

impl Cell {
    fn obj(&self) -> Objective {
        Objective { v0: Vec3::new(0.0, self.vy, self.vz), n: self.n, lambda: self.lambda }
    }
}

/// The polish's settings; the module docs say what each means.
#[derive(clap::Args)]
struct Polish {
    #[arg(long, default_value_t = PolishOpts::default().tol)]
    tol: f64,
    #[arg(long, default_value_t = PolishOpts::default().block)]
    block: usize,
    #[arg(long, default_value_t = PolishOpts::default().lag1_floor)]
    lag1_floor: f64,
    /// Re-solve v0 to the schedule's own fixed point after every pass.
    #[arg(long)]
    steady: bool,
    #[arg(long, default_value_t = 0.0)]
    mu: f64,
    /// l1|l2:<d>|huber:<d>, the shape `--mu` prices.
    #[arg(long, default_value = DEFAULT_PENALTY, value_parser = PriceShape::parse)]
    penalty: PriceShape,
    #[arg(long, default_value_t = 0.0)]
    mu_tv: f64,
    #[arg(long, default_value_t = f64::INFINITY)]
    cap: f64,
    #[arg(long, default_value_t = f64::INFINITY)]
    slew_cap: f64,
    #[arg(long, default_value_t = 90.0)]
    limit: f64,
    /// Restrict the first tick at or below --flick-pitch to this tick.
    #[arg(long)]
    flick_at: Option<usize>,
    #[arg(long, default_value_t = -80.0)]
    flick_pitch: f64,
    #[arg(long, default_value_t = 0.0)]
    jitter: f64,
    #[arg(long, default_value_t = 8)]
    draws: usize,
    #[arg(long)]
    fixed_draws: bool,
    #[arg(long, default_value_t = Jitter::default().seed)]
    seed: u64,
}

impl Polish {
    fn opts(&self, passes: usize) -> PolishOpts {
        PolishOpts {
            max_passes: passes,
            tol: self.tol,
            block: self.block,
            lag1_floor: self.lag1_floor,
            // Off unless asked: the single-cycle problem is the simpler object and is still
            // the one most questions are about. `--vy/--vz` then seed the fixed-point
            // iteration rather than naming the answer; see `solve`.
            steady: self.steady,
            rough: Rough {
                mu: self.mu,
                shape: self.penalty,
                mu_tv: self.mu_tv,
                cap: self.cap,
                slew_cap: self.slew_cap,
                limit: self.limit,
                flick_at: self.flick_at,
                flick_pitch: self.flick_pitch,
            },
            jitter: Jitter {
                sigma: self.jitter,
                draws: self.draws,
                resample: !self.fixed_draws,
                seed: self.seed,
            },
            ..Default::default()
        }
    }
}

/// The full sweep's axes; each defaults to the grid in `Grid::from`.
#[derive(clap::Args)]
struct GridArgs {
    /// Default 100..=500 by 10.
    #[arg(long, value_delimiter = ',')]
    ns: Vec<usize>,
    /// Default -1..=1 by 0.05.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    lams: Vec<f64>,
    /// Default 0..=0.2 by 0.02.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    vys: Vec<f64>,
    /// Default 0..=0.2 by 0.02.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    vzs: Vec<f64>,
}

/// Where a cell's file lives. The path names the coordinates that *determine* the answer.
///
/// Without `--steady` that includes `v0`, which is an input: two cells differing only in `v0`
/// are different problems with different answers. With `--steady` `v0` is an output -- the
/// schedule's own fixed point, reached from a seed that barely influences it -- so it is not a
/// coordinate of the grid and naming a directory after it would claim a distinction the files
/// do not have. Steady cells live under `steady/` and state their `v0` in the header.
fn cell_path(dir: &str, o: &Objective, steady: bool) -> String {
    let shard = if steady { "steady".to_string() }
                else { format!("vy{:+.4}_vz{:+.4}", o.v0.y, o.v0.z) };
    format!("{dir}/{shard}/n{:04}_lam{:+.6}.pitches", o.n, o.lambda)
}

fn write_profile(path: &str, p: &Profile) {
    if let Some(d) = std::path::Path::new(path).parent() { std::fs::create_dir_all(d).unwrap() }
    std::fs::write(path, p.to_string()).unwrap_or_else(|e| panic!("{path}: {e}"));
}

// ---------------------------------------------------------------- one cell

/// Returns the profile and the polish record: the record carries *why* the polish stopped,
/// which the file deliberately does not (the file states the objective, not the provenance).
fn solve(obj: &Objective, init: &[f64], opts: PolishOpts) -> (Profile, Polished) {
    let r = polish(obj, init, opts);
    // Under `--steady` the polish re-solved `v0` to the schedule's own fixed point, and *that*
    // is the objective this schedule is a coordinate optimum of. The header has to state it or
    // `verify` replays from a velocity the schedule was never optimized for and the file's
    // claim is simply false. The requested `v0` was a seed and does not survive into the file.
    //
    // Nothing else changes: at convergence the objective is still determined by the four
    // numbers in the header, so no `steady` field is needed. How `v0` was arrived at is
    // provenance, and the file states the objective, not the provenance.
    let obj = Objective { v0: r.v0, ..*obj };
    let profile = Profile {
        obj, trig: trig_mode(), flight: flight_mode(), jitter: opts.jitter, rough: opts.rough,
        commit: commit_hash().to_string(),
        pitches: r.pitches.clone(), residual: r.residual, passes: r.passes,
    };
    (profile, r)
}

/// Refuse a seed profile that was optimized under physics this run is not using.
///
/// A profile states the trig and flight route it was optimized under, and `verify` and the
/// examples replay under the route the file claims. `polish` and `run` do not adopt it -- the
/// route stays whatever `--trig`/`--flight` say, defaults included -- so seeding from an
/// `mth_lut`/`algebraic` profile while the process sits on `libm`/`reference` solves a different
/// problem than the seed solved, and nothing downstream can see it: every header truthfully
/// records the route it was actually run under, so the files come out self-consistent and
/// quietly wrong about which question they answer. The routes differ by about 1 ULP on 2.6% of
/// inputs, which is enough to move where the optimum sits -- re-polishing one `runs/atlas` cell
/// under the defaults took 186 passes to a different fixed point where its own route takes 156.
///
/// Fatal rather than a warning, and with no override flag. Cross-route seeding has no
/// legitimate use -- matching the seed is what `--trig`/`--flight` are for -- and the escape
/// hatch already exists for the rare case that wants it: a headerless pitch list claims no
/// route and is accepted as-is.
fn require_physics(p: &Profile, what: &str) {
    if p.trig != trig_mode() || p.flight != flight_mode() {
        panic!("{what} was optimized under trig {} flight {}, but this run uses trig {} \
                flight {}. Pass --trig/--flight to match it, or seed from a profile built \
                under this route.",
               p.trig, p.flight, trig_mode(), flight_mode());
    }
}

fn cmd_polish_cell(obj: Objective, opts: PolishOpts, init: Option<&str>, premedian: usize, presmooth: usize,
                   out: Option<&str>) {
    let init = match init {
        Some(f) => {
            let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
            let parsed = Profile::parse(&text);
            if let Ok(p) = &parsed { require_physics(p, "--init") }
            stretch(&parsed.map(|p| p.pitches).unwrap_or_else(|_| {
                text.lines().flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace())
                    .map(|s| s.parse().unwrap()).collect()
            }), obj.n)
        }
        None => seed_from_policy(&obj),
    };
    // Project before polishing. A relaxed (chattering) solution has to be locally averaged
    // before a roughness price can improve it -- see `smooth_box`.
    let init = smooth_median(&init, premedian);
    let init = smooth_box(&init, presmooth);
    let t = Instant::now();
    let (p, r) = solve(&obj, &init, opts);
    eprintln!("n {:>4}  lambda {:+.4}  v0 ({:.6}, {:.6})  J {:.6}  residual {:.2e}  \
lag1 {:+.3}  TV {:.0}  curv_l1 {:.0}  curv_max {:.0}  {} passes{}  {:.1}s",
              obj.n, obj.lambda, p.obj.v0.y, p.obj.v0.z, p.obj.eval(&p.pitches), p.residual,
              lag1(&p.pitches), total_variation(&p.pitches), curvature_l1(&p.pitches),
              curvature_max(&p.pitches),
              p.passes, if r.stopped_degenerate { " (stopped: degenerate)" } else { "" },
              t.elapsed().as_secs_f64());
    match out {
        Some(f) => write_profile(f, &p),
        None => print!("{}", p.to_string()),
    }
}

fn cmd_verify(files: &[String]) {
    let mut bad = 0;
    for f in files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        let p = match Profile::parse(&text) { Ok(p) => p, Err(e) => { println!("{f}: BAD HEADER: {e}"); bad += 1; continue } };
        // the header is authoritative: replay under the physics it claims, not the shell's
        set_trig_mode(p.trig);
        set_flight_mode(p.flight);
        let viol = p.rough.violation(&p.pitches);
        let res = certify_reg(&p.obj, &p.pitches, PolishOpts::default().global_step, p.jitter, p.rough);
        let claimed = p.residual;
        let ok = viol == 0.0 && res <= claimed.max(1e-6) * 1.5 + 1e-9;
        println!("{f}: n {:>4} lambda {:+.4} trig {} flight {} | claimed {:.2e}, found {:.2e}  {}{}",
                 p.obj.n, p.obj.lambda, p.trig, p.flight, claimed, res, if ok { "ok" } else { "MISMATCH" },
                 if viol > 0.0 { format!("  (outside its own control limits by {viol:.3} deg)") }
                 else { String::new() });
        if !ok { bad += 1 }
    }
    if bad > 0 { eprintln!("{bad} of {} profiles failed", files.len()); std::process::exit(1) }
}

// ---------------------------------------------------------------- cost

fn cmd_bench(obj: Objective, step: usize, total: usize) {
    eprintln!("trig {}  flight {}  n {}  lambda {:+.4}  v0 ({:.4}, {:.4})",
              trig_mode(), flight_mode(), obj.n, obj.lambda, obj.v0.y, obj.v0.z);

    let seeds: Vec<(&str, Vec<f64>)> = vec![
        ("policy", seed_from_policy(&obj)),
        ("reference", seed_from_reference(obj.n)),
        ("flat 0", vec![0.0; obj.n]),
    ];
    println!("{:>10} {:>7} {:>10} {:>10} {:>9} {:>8}", "seed", "passes", "J", "residual", "dy", "cum s");
    for (name, seed) in seeds {
        let mut cur = seed;
        let (mut done, mut secs) = (0usize, 0.0);
        println!("{name:>10} {:>7} {:>10.5} {:>10} {:>9.3} {:>8}",
                 0, obj.eval(&cur), "-", obj.replay(&cur).last().unwrap().pos.y, "-");
        while done < total {
            let t = Instant::now();
            let r = polish(&obj, &cur, PolishOpts { max_passes: step, ..Default::default() });
            secs += t.elapsed().as_secs_f64();
            done += r.passes;
            cur = r.pitches;
            println!("{name:>10} {done:>7} {:>10.5} {:>10.2e} {:>9.3} {secs:>8.1}",
                     r.j, r.residual, obj.replay(&cur).last().unwrap().pos.y);
            if r.passes < step { break }        // converged inside the block
        }
    }
}

// ---------------------------------------------------------------- the coarse sweep

fn cmd_pilot(dir: &str, passes: usize, ns: &[usize], lams: &[f64], vels: &[(f64, f64)], vys: &[f64],
             vzs: &[f64]) {
    // --vels gives explicit (vy:vz) points; --vys/--vzs give their outer product. The critical
    // points of the velocity box are its four corners, its center, and the reference operating
    // point, which is not a corner and is the only one with a known answer.
    let vels: Vec<(f64, f64)> = if vels.is_empty() {
        vys.iter().flat_map(|&y| vzs.iter().map(move |&z| (y, z))).collect()
    } else { vels.to_vec() };

    let mut cells: Vec<Objective> = vec![];
    for &n in ns { for &lambda in lams { for &(vy, vz) in &vels {
        cells.push(Objective { v0: Vec3::new(0.0, vy, vz), n, lambda });
    }}}
    eprintln!("pilot: {} cells, {passes} passes each, trig {}, flight {}",
              cells.len(), trig_mode(), flight_mode());

    println!("{:>5} {:>8} {:>8} {:>8} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10} {:>8}",
             "n", "lambda", "vy0", "vz0", "J_seed", "J_opt", "dy", "dz", "residual", "close", "passes");
    let t0 = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = cells.len();
    let rows: Vec<String> = cells.par_iter().map(|obj| {
        let seed = seed_from_policy(obj);
        let j_seed = obj.eval(&seed);
        let (p, _) = solve(obj, &seed, PolishOpts { max_passes: passes, ..Default::default() });
        let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if k % 10 == 0 || k == total {
            eprintln!("  {k}/{total} cells, {:.0}s elapsed", t0.elapsed().as_secs_f64());
        }
        let st = obj.replay(&p.pitches);
        let sn = st.last().unwrap();
        write_profile(&cell_path(dir, obj, false), &p);
        format!("{:>5} {:>8.3} {:>8.4} {:>8.4} {j_seed:>10.4} {:>10.4} {:>9.3} {:>9.2} {:>10.2e} {:>10.2e} {:>8}",
                obj.n, obj.lambda, obj.v0.y, obj.v0.z, obj.eval(&p.pitches),
                sn.pos.y, sn.pos.z, p.residual, (sn.vel - obj.v0).length(), p.passes)
    }).collect();
    for r in rows { println!("{r}") }
    eprintln!("pilot: {:.1}s total", t0.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------- the full sweep

/// The grid. Kept in one place so `run`, `manifest` and the resume check cannot disagree.
struct Grid { ns: Vec<usize>, lams: Vec<f64>, vys: Vec<f64>, vzs: Vec<f64> }

impl Grid {
    fn from(a: &GridArgs) -> Grid {
        let or = |v: &[f64], d: Vec<f64>| if v.is_empty() { d } else { v.to_vec() };
        Grid {
            ns: if a.ns.is_empty() { (100..=500).step_by(10).collect() } else { a.ns.clone() },
            lams: or(&a.lams, (-20..=20).map(|i| i as f64 * 0.05).collect()),
            vys: or(&a.vys, (0..=10).map(|i| i as f64 * 0.02).collect()),
            vzs: or(&a.vzs, (0..=10).map(|i| i as f64 * 0.02).collect()),
        }
    }
    /// Index of the value nearest `x`, for placing the anchor.
    fn nearest(v: &[f64], x: f64) -> usize {
        (0..v.len()).min_by(|&i, &j| {
            (v[i] - x).abs().partial_cmp(&(v[j] - x).abs()).unwrap()
        }).unwrap()
    }
}

/// Solve one `(vy0, vz0)` shard: anchor at the cell nearest the reference operating point, then
/// breadth-first outward over the `(n, lambda)` plane, each cell warm-started from the solved
/// neighbor that reached it. Continuation is the whole speedup, and it also keeps the heuristic
/// seed out of the parts of the grid where it has no reason to work -- it is asked once, at the
/// anchor, and never again.
fn run_shard(dir: &str, g: &Grid, vy: f64, vz: f64, opts: PolishOpts, force: bool,
             anchor: Option<&[f64]>) -> (usize, usize) {
    let ns_f: Vec<f64> = g.ns.iter().map(|&n| n as f64).collect();
    let (ni, li) = (Grid::nearest(&ns_f, 300.0), Grid::nearest(&g.lams, 0.0));
    let (nn, nl) = (g.ns.len(), g.lams.len());
    let cell = |i: usize, j: usize| Objective {
        v0: Vec3::new(0.0, vy, vz), n: g.ns[i], lambda: g.lams[j],
    };

    let mut solved: Vec<Option<Vec<f64>>> = vec![None; nn * nl];
    let mut queue = std::collections::VecDeque::from([(ni, li, None::<(usize, usize)>)]);
    let mut seen = vec![false; nn * nl];
    seen[ni * nl + li] = true;
    let (mut done, mut skipped) = (0, 0);

    while let Some((i, j, from)) = queue.pop_front() {
        let obj = cell(i, j);
        let path = cell_path(dir, &obj, opts.steady);
        // Warm start from whichever solved neighbor got here first; the anchor has none.
        // The anchor is the one cell with no solved neighbor. `seed_from_policy` was tuned
        // against the reference cycle's operating point, so away from it -- v0 = 0 especially --
        // an explicit anchor schedule is the difference between seeding the cyclic branch and
        // seeding a collapsed one, and the BFS propagates whichever it gets to the whole shard.
        // Two candidate seeds, and the cell takes whichever actually scores better. Both stay
        // inside the single-cycle family: the corpus sweeps one cycle across num_ticks, so a
        // seed that tiles the cycle is not a cheaper route to the answer, it is a different
        // and degenerate answer.
        //
        // The BFS parent carries the branch -- along lambda, and along n below the anchor where
        // it holds the cyclic regime a cold seed collapses out of. The anchor rescaled to
        // period n is the cold fallback, and wins where the chain has drifted.
        let parent = from.and_then(|(a, b)| solved[a * nl + b].clone())
            .map(|prev| stretch(&prev, obj.n));
        let mut cands: Vec<Vec<f64>> = parent.into_iter().collect();
        if let Some(a) = anchor {
            cands.push(rescale(a, obj.n));      // one cycle of period n
        }
        if cands.is_empty() { cands.push(seed_from_policy(&obj)) }
        // Score the candidates *after a short polish*, not before it. Judged cold the BFS
        // parent always wins -- it is a solved schedule and the others are not -- but going up
        // in n it then polishes straight into chatter, while a tiled seed that starts lower
        // ends far higher. Judged cold, the n axis came back with a band from 350 to 530 losing
        // 0.71 blocks per 10 ticks at lag-1 -0.87, with n = 530 scoring 6.94 where n = 540,
        // which the tiled seed reached, scored 16.56.
        let init = if cands.len() == 1 {
            cands.pop().unwrap()
        } else {
            let probe = PolishOpts { max_passes: 2, tol: 0.0, ..opts };
            cands.into_iter()
                // Rank on the objective actually being optimized. Ranking on `r.j` alone
                // picks the roughest candidate whenever a price is on, which is the seed most
                // likely to hand the whole continuation chain a schedule it will be charged for.
                .map(|c| { let r = polish(&obj, &c, probe); (r.j - r.rough_cost, r.pitches) })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap().1
        };
        // Resume: a cell whose file already matches this objective and physics is not redone,
        // but its pitches still seed the neighbors, so a killed job costs one cell.
        let existing = (!force).then(|| std::fs::read_to_string(&path).ok()).flatten()
            .and_then(|t| Profile::parse(&t).ok())
            // The roughness price and the pitch margin are part of the utility function, so a
            // file written under a different one answers a different question and must be
            // resolved, not resumed.
            .filter(|p| p.trig == trig_mode() && p.flight == flight_mode()
                       && p.pitches.len() == obj.n
                        && p.rough == opts.rough
                        && if opts.steady {
                // A steady file's header `v0` is its own fixed point, not this cell's seed, so
                // the objectives never compare equal and a plain `==` would silently re-solve
                // the whole grid on every resume. What makes the file answer *this* cell is
                // that the horizon and price match and its stated `v0` really is the fixed
                // point of its own pitches -- which is checkable, so check it rather than
                // trusting the path it was found at.
                p.obj.n == obj.n && p.obj.lambda == obj.lambda
                    && dv_l1(steady_vel(&p.pitches, p.obj.v0), p.obj.v0) < 1e-9
            } else {
                p.obj == obj
            });
        let pitches = match existing {
            Some(p) => { skipped += 1; p.pitches }
            None => {
                let (prof, _) = solve(&obj, &init, opts);
                write_profile(&path, &prof);
                done += 1;
                if done % 25 == 0 { eprintln!("  vy{vy:+.3} vz{vz:+.3}: {done} solved, {skipped} resumed") }
                prof.pitches
            }
        };
        solved[i * nl + j] = Some(pitches);

        for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let (a, b) = (i as i64 + di, j as i64 + dj);
            if a < 0 || b < 0 || a as usize >= nn || b as usize >= nl { continue }
            let (a, b) = (a as usize, b as usize);
            if std::mem::replace(&mut seen[a * nl + b], true) { continue }
            queue.push_back((a, b, Some((i, j))));
        }
    }
    (done, skipped)
}

fn write_manifest(dir: &str, g: &Grid, opts: PolishOpts) {
    let list = |v: &[f64]| v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", ");
    let text = format!(
"{{
  \"commit\": \"{}\",
  \"trig\": \"{}\",
  \"flight\": \"{}\",
  \"objective\": \"J = TE(s_n) + w*z_n, TE in blocks (KE = |v|^2/(2g), PE = y), v_n free\",
  \"y_ref\": {Y_REF}, \"z_ref\": {Z_REF},
  \"axes\": {{
    \"n\": [{}],
    \"lambda\": [{}],
    \"vy0\": [{}],
    \"vz0\": [{}]
  }},
  \"cells\": {},
  \"polish\": {{ \"max_passes\": {}, \"global_every\": {}, \"global_step\": {}, \"local_span\": {}, \"local_step\": {}, \"tol\": {} }},
  \"control\": {{ \"mu\": {}, \"mu_tv\": {}, \"cap\": \"{}\", \"slew_cap\": \"{}\", \"limit\": {}, \"flick_at\": {}, \"flick_pitch\": {} }},
  \"jitter\": {{ \"sigma\": {}, \"draws\": {}, \"resample\": {}, \"seed\": {} }},
  \"fingerprint\": \"{:016x}\"
}}
", commit_hash(), trig_mode(), flight_mode(),
   g.ns.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
   list(&g.lams), list(&g.vys), list(&g.vzs),
   g.ns.len() * g.lams.len() * g.vys.len() * g.vzs.len(),
   opts.max_passes, opts.global_every, opts.global_step, opts.local_span, opts.local_step, opts.tol,
        // The price and the margin are part of the utility function; a manifest that omits them
        // describes a different sweep than the one that ran. Each profile carries them too, so
        // nothing was unverifiable -- but the index has to agree with the files.
        opts.rough.mu, opts.rough.mu_tv, opts.rough.cap, opts.rough.slew_cap, opts.rough.limit,
        opts.rough.flick_at.map_or_else(|| "null".to_string(), |t| t.to_string()),
        opts.rough.flick_pitch,
        opts.jitter.sigma, opts.jitter.draws, opts.jitter.resample, opts.jitter.seed,
   physics_fingerprint());
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(format!("{dir}/manifest.json"), text).unwrap();
}

fn cmd_run(dir: &str, g: Grid, opts: PolishOpts, force: bool, anchor: Option<&str>, shard: &[usize]) {
    let anchor: Option<Vec<f64>> = anchor.map(|f| {
        let t = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        let parsed = Profile::parse(&t);
        // The anchor propagates to every cell in the shard, so this one is worth stopping for
        // even more than `--init` is.
        if let Ok(p) = &parsed { require_physics(p, "--anchor") }
        parsed.map(|p| p.pitches).unwrap_or_else(|_| {
            t.lines().flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace())
             .map(|x| x.parse().unwrap()).collect()
        })
    });
    // A shard is one (vy0, vz0) cell, so it is a directory and an independent job.
    let mut shards: Vec<(f64, f64)> = vec![];
    for &vy in &g.vys { for &vz in &g.vzs { shards.push((vy, vz)) } }
    if !shard.is_empty() {
        shards = shard.iter().map(|&i| shards[i]).collect();
    }
    // Under `--steady` `v0` is an output, so the (vy0, vz0) axis is not a grid axis: every
    // shard would solve the same problems and race to write the same files. Refuse rather than
    // silently collapse them, because the request says the caller expects distinct answers.
    if opts.steady && shards.len() > 1 {
        panic!("--steady makes v0 an output -- the schedule's own fixed point -- so (vy0, vz0) \
                is not a grid axis and all {} shards would write the same files. Pass a single \
                --vys/--vzs pair; it seeds the fixed-point iteration and little else.",
               shards.len())
    }
    write_manifest(dir, &g, opts);
    eprintln!("sweep: {} shards x {} cells, {} passes, trig {}, flight {}, commit {}",
              shards.len(), g.ns.len() * g.lams.len(), opts.max_passes, trig_mode(),
              flight_mode(), commit_hash());
    let t0 = Instant::now();
    let totals: Vec<(usize, usize)> = shards.par_iter()
        .map(|&(vy, vz)| run_shard(dir, &g, vy, vz, opts, force, anchor.as_deref())).collect();
    let (d, s): (usize, usize) = totals.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    eprintln!("sweep: {d} solved, {s} resumed, {:.0}s", t0.elapsed().as_secs_f64());
}

fn cmd_fingerprint() {
    println!("{:016x}  trig {}  flight {}  commit {}",
             physics_fingerprint(), trig_mode(), flight_mode(), commit_hash());
}

// ---------------------------------------------------------------- main

fn main() {
    let cli = Cli::parse();
    if let Some(t) = cli.trig { set_trig_mode(t) }
    if let Some(f) = cli.flight { set_flight_mode(f) }
    match cli.cmd {
        Cmd::Polish { cell, polish, passes, init, premedian, presmooth, out } =>
            cmd_polish_cell(cell.obj(), polish.opts(passes), init.as_deref(), premedian, presmooth, out.as_deref()),
        Cmd::Verify { files } => cmd_verify(&files),
        Cmd::Bench { cell, every, passes } => cmd_bench(cell.obj(), every, passes),
        Cmd::Pilot { out, passes, ns, lams, vels, vys, vzs } => cmd_pilot(&out, passes, &ns, &lams, &vels, &vys, &vzs),
        Cmd::Run { grid, polish, passes, out, force, anchor, shard } =>
            cmd_run(&out, Grid::from(&grid), polish.opts(passes), force, anchor.as_deref(), &shard),
        Cmd::Fingerprint => cmd_fingerprint(),
        Cmd::Residuals { file } => {
            let f = &file;
            let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
            let pr = Profile::parse(&text).unwrap_or_else(|e| panic!("{f}: {e}"));
            set_trig_mode(pr.trig);
            set_flight_mode(pr.flight);
            let r = residuals_reg(&pr.obj, &pr.pitches, PolishOpts::default().global_step,
                                  pr.jitter, pr.rough);
            let (pos, neg) = (r.iter().filter(|x| x.0 > 1e-9).count(),
                              r.iter().filter(|x| x.0 < -1e-9).count());
            let sum: f64 = r.iter().map(|x| x.0).sum();
            let absum: f64 = r.iter().map(|x| x.0.abs()).sum();
            let gain: f64 = r.iter().map(|x| x.1).sum();
            eprintln!("{}: n {}, {} want more pitch, {} want less, {} still",
                      f.rsplit('/').next().unwrap(), r.len(), pos, neg, r.len() - pos - neg);
            eprintln!("  sum of moves {sum:+.3} deg, sum of |moves| {absum:.3} deg  ->  \
                       coherence {:.3}", if absum > 0.0 { sum / absum } else { 0.0 });
            eprintln!("  sum of per-tick gains {gain:.4} blocks, largest {:.2e}",
                      r.iter().map(|x| x.1).fold(0.0, f64::max));
            let deltas: Vec<f64> = r.iter().map(|x| x.0).collect();
            let (lag1, run_len, ks) = delta_structure(&deltas);
            eprintln!("  lag-1 correlation of the moves {lag1:+.3}, mean same-sign run {run_len:.1} ticks");
            eprint!("  energy in the first k cosine modes:");
            for (k, e) in &ks { eprint!("  k={k}: {:.0}%", 100.0 * e) }
            eprintln!();
            match jacobi_step_reg(&pr.obj, &pr.pitches, PolishOpts::default().global_step,
                                  pr.jitter, pr.rough) {
                Some((_, alpha, g)) => eprintln!(
                    "  best whole-schedule step: alpha {alpha:.3} worth {g:.4} blocks \
                     ({:.1}% of the per-tick total)", 100.0 * g / gain.max(1e-12)),
                None => eprintln!("  no whole-schedule step improves J"),
            }
            println!("tick,pitch,delta,gain");
            for (t, (d, g)) in r.iter().enumerate() {
                println!("{t},{:.5},{d:+.5},{g:.3e}", pr.pitches[t]);
            }
        }
        Cmd::Structure { files } => {
            for f in &files {
                let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
                let ps = Profile::parse(&text).map(|p| p.pitches).unwrap_or_else(|_|
                    text.lines().flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace())
                        .map(|s| s.parse().unwrap()).collect());
                let sh = shape(&ps);
                println!("{:<28} n {:>4}  {:>9}  pitch [{:>7.2}, {:>6.2}]  flat {:>3}",
                         f.rsplit('/').next().unwrap(), ps.len(), sh.structure.to_string(),
                         sh.pitch_min, sh.pitch_max, sh.flat_ticks);
            }
        }
    }
}
