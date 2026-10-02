//! Backward dynamic programming for flight over a floor: the value of every state, not one
//! schedule. `floor exit` climbs from a seed and finds a local optimum; this asks what the
//! optimum is.
//!
//! The state is `(h, v_y, v_z)`: clearance above the floor and velocity. The flight kernel reads
//! only the velocity and the pitch (`update_fall_flying_movement_cached`), and the floor only the
//! clearance, so nothing else matters; `z` enters range as a reward. The value is
//!
//!   V(s) = max_p [ r(s, p) + V(s') ]     s' = s flown one tick at pitch p
//!
//! with `r = 1` (endurance) or `r = v'_z` (range, the tick's `dz`), and at a step that goes under
//! the floor only the fraction `f = h / (h - h')` of it, with no future: exactly `exit_score`'s
//! interpolated exit. `V` lives on a grid and is read between nodes by trilinear interpolation
//! (clamped at the grid's edges), a semi-Lagrangian scheme.
//!
//! Solved by backward induction from `V_0 = 0`: sweep `n` is `V_n = T V_{n-1}`, the best flight
//! capped at `n` ticks, every state in parallel. It rises monotonically to the true value (every
//! reward is non-negative and interpolation is monotone), and a start state has converged once
//! `n` passes its flight's length. why not Gauss-Seidel (tried first, states in order of rising
//! energy): a tick moves a glide about 0.07 blocks against a grid cell of 0.25-0.5, so a backup
//! mostly reads its own cell, and the self-loop made it rise one tick per sweep all the same.
//! States whose value reaches `--cap` fly forever from there.
//!
//! The DP's value is the grid's opinion. The policy it implies -- each tick the pitch maximizing
//! `r + V(s')` over a fine grid, flown in the exact simulator from rest at `y0` -- is the check:
//! its `t*` or `z(t*)` is a real flight's.
//!
//! Usage: `floordp --mode time|dist [--y0 1,2,..] [--out <dir>] ...`; `floordp --help` has every
//! option. `--save <file>`/`--load <file>` keep V as raw little-endian f32, and `--load` skips
//! the sweeps.
//!
//! `--map` writes the policy on every velocity node at each `--map-h`: the chosen pitch, as
//! little-endian i16 quarter-degrees, then the *margin*, as f32, `h`-major then `v_y` then `v_z`.
//! The margin is how much the choice beats the best pitch more than `--gap` degrees from it. A
//! small margin is where the DP is nearly indifferent between two different pitches, which is
//! where a feedback policy can flip between them tick to tick. `<file>.spec` records the
//! layout. `--trace` writes, beside each flown policy, `<mode>_y<y0>.trace`: per tick the
//! state, the pitch, its `r + V(s')`, and that runner-up and its margin; and
//! `<mode>_y<y0>.q`, `r + V(s')` for every `--dp` pitch at each of the first `--trace-q 3000`
//! ticks, little-endian f32, tick-major.
//!
//! Physics is `floor`'s: `mth_lut` trig and `reference` flight.

use clap::{Parser, ValueEnum};
use elytrasim::sim::*;
use rayon::prelude::*;

/// Backward DP over (h, v_y, v_z) for flight over a floor, and its policy flown from rest.
#[derive(Parser)]
#[command(allow_negative_numbers = true)]
struct Args {
    /// Endurance (`time`) or range (`dist`).
    #[arg(long, value_enum, default_value_t = Mode::Time)]
    mode: Mode,
    /// Top of the height grid, blocks above the floor.
    #[arg(long, default_value_t = 34.0)]
    hmax: f64,
    /// Height-node spacing exponent; above 1 packs nodes toward the floor.
    #[arg(long, default_value_t = 1.0)]
    gamma: f64,
    /// Height nodes.
    #[arg(long, default_value_t = 137)]
    nh: usize,
    /// Velocity node spacing, blocks/tick, on both axes.
    #[arg(long, default_value_t = 0.05)]
    dv: f64,
    #[arg(long, default_value_t = -3.0)]
    vymin: f64,
    #[arg(long, default_value_t = 2.0)]
    vymax: f64,
    #[arg(long, default_value_t = 3.0)]
    vzmax: f64,
    /// Pitch step of the backup, degrees; must divide 170.
    #[arg(long, default_value_t = 1.0)]
    dp: f64,
    #[arg(long, default_value_t = 400)]
    sweeps: usize,
    /// Stop once no start state's value rises more than this in a sweep.
    #[arg(long, default_value_t = 1e-4)]
    tol: f64,
    /// Values are clipped here; a state that reaches it flies forever.
    #[arg(long, default_value_t = 1e5)]
    cap: f64,
    /// Start heights to fly the policy from (default 1..=32).
    #[arg(long, value_delimiter = ',')]
    y0: Vec<f64>,
    /// Write V (and `<file>.spec`) after the sweeps.
    #[arg(long)]
    save: Option<String>,
    /// Read V saved for the same grid and mode, and skip the sweeps.
    #[arg(long)]
    load: Option<String>,
    /// Report progress every this many sweeps.
    #[arg(long, default_value_t = 50)]
    every: usize,
    /// Pitch step of the flown policy, degrees.
    #[arg(long, default_value_t = 0.25)]
    fine: f64,
    /// Degrees from the best pitch that a runner-up must be.
    #[arg(long, default_value_t = 10.0)]
    gap: f64,
    /// Write `<mode>_y<y0>.trace` and `.q` beside each flown policy (needs `--out`).
    #[arg(long)]
    trace: bool,
    /// Ticks of `.q` to write under `--trace`.
    #[arg(long, default_value_t = 3000)]
    trace_q: usize,
    /// Write the policy map here (see the module docs).
    #[arg(long, requires = "map_h")]
    map: Option<String>,
    /// Heights the map is taken at.
    #[arg(long, value_delimiter = ',')]
    map_h: Vec<f64>,
    /// Longest flight of the policy, ticks.
    #[arg(long, default_value_t = 20000)]
    maxt: usize,
    /// Directory for the flown pitches.
    #[arg(long)]
    out: Option<String>,
}

const HBINS: usize = 4096;

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Mode { Time, Dist }

/// A grid over `(h, v_y, v_z)`, node `(i, j, k)` at `(hmax (i / (nh-1))^gamma, vy0 + j dvy,
/// k dvz)`. `gamma > 1` packs the heights toward the floor, where the value is sharpest: whether
/// a fast descent can still pull up is decided within a fraction of a block.
///
/// The node heights are precomputed, and a height is bracketed through `find`, a table over 4096
/// even bins of `[0, hmax]` giving the node at or below each bin's floor, then stepped up: `powf`
/// on every read made a sweep 4x slower.
struct Grid { hmax: f64, nh: usize, vy0: f64, dvy: f64, nvy: usize, dvz: f64, nvz: usize, hs: Vec<f64>, find: Vec<u32> }

impl Grid {
    fn len(&self) -> usize { self.nh * self.nvy * self.nvz }
    fn idx(&self, i: usize, j: usize, k: usize) -> usize { (i * self.nvy + j) * self.nvz + k }
    fn node(&self, n: usize) -> (f64, f64, f64) {
        let k = n % self.nvz;
        let j = (n / self.nvz) % self.nvy;
        let i = n / (self.nvz * self.nvy);
        (self.h(i), self.vy0 + j as f64 * self.dvy, k as f64 * self.dvz)
    }
    fn h(&self, i: usize) -> f64 { self.hs[i] }
    fn new(hmax: f64, gamma: f64, nh: usize, vy0: f64, dvy: f64, nvy: usize, dvz: f64, nvz: usize) -> Grid {
        let hs: Vec<f64> = (0..nh).map(|i| hmax * (i as f64 / (nh - 1) as f64).powf(gamma)).collect();
        let find = (0..HBINS).map(|b| {
            let x = hmax * b as f64 / HBINS as f64;
            (hs.partition_point(|&h| h <= x).saturating_sub(1)).min(nh - 2) as u32
        }).collect();
        Grid { hmax, nh, vy0, dvy, nvy, dvz, nvz, hs, find }
    }
    /// `V` at an arbitrary point, trilinear, each coordinate clamped to the grid.
    #[inline]
    fn at(&self, v: &[f32], h: f64, vy: f64, vz: f64) -> f64 {
        #[inline]
        fn split(x: f64, n: usize) -> (usize, f64) {
            let x = x.clamp(0.0, (n - 1) as f64);
            let i = (x.floor() as usize).min(n.saturating_sub(2));
            (i, x - i as f64)
        }
        // Linear in `h` between the two bracketing nodes, not in the stretched coordinate.
        let hc = h.clamp(0.0, self.hmax);
        let mut i = self.find[((hc / self.hmax * HBINS as f64) as usize).min(HBINS - 1)] as usize;
        while i + 2 < self.nh && self.hs[i + 1] <= hc { i += 1 }
        let a = ((hc - self.hs[i]) / (self.hs[i + 1] - self.hs[i])).clamp(0.0, 1.0);
        let (j, b) = split((vy - self.vy0) / self.dvy, self.nvy);
        let (k, c) = split(vz / self.dvz, self.nvz);
        let g = |di: usize, dj: usize, dk: usize| v[self.idx(i + di, j + dj, k + dk)] as f64;
        let l00 = g(0, 0, 0) * (1.0 - c) + g(0, 0, 1) * c;
        let l01 = g(0, 1, 0) * (1.0 - c) + g(0, 1, 1) * c;
        let l10 = g(1, 0, 0) * (1.0 - c) + g(1, 0, 1) * c;
        let l11 = g(1, 1, 0) * (1.0 - c) + g(1, 1, 1) * c;
        let l0 = l00 * (1.0 - b) + l01 * b;
        let l1 = l10 * (1.0 - b) + l11 * b;
        l0 * (1.0 - a) + l1 * a
    }
    fn inside(&self, h: f64, vy: f64, vz: f64) -> bool {
        h <= self.hmax && vy >= self.vy0 && vy <= self.vy0 + (self.nvy - 1) as f64 * self.dvy
            && vz >= 0.0 && vz <= (self.nvz - 1) as f64 * self.dvz
    }
}

/// One tick from `(h, v)` at `pitch`: the value of flying it, `r + V(s')`, or the exit's
/// fraction of `r` if it goes under. Also the new state.
#[inline]
fn backup(g: &Grid, v: &[f32], mode: Mode, h: f64, vel: Vec3, pitch: PitchTrig) -> (f64, f64, Vec3) {
    let nv = update_fall_flying_movement_cached(vel, pitch);
    let nh = h + nv.y;
    let r = match mode { Mode::Time => 1.0, Mode::Dist => nv.z };
    if nh < 0.0 { return (h / (h - nh) * r, nh, nv) }
    (r + g.at(v, nh, nv.y, nv.z), nh, nv)
}

/// The best pitch on `ps` from `(h, vel)` and its value, and the best more than `gap` degrees
/// from it (the runner-up) and its value; the runner-up is `None` when every pitch is within `gap`.
fn ranked(g: &Grid, v: &[f32], mode: Mode, h: f64, vel: Vec3, ps: &[f64], ts: &[PitchTrig], gap: f64)
          -> (usize, f64, Option<(usize, f64)>) {
    let q: Vec<f64> = ts.iter().map(|&t| backup(g, v, mode, h, vel, t).0).collect();
    let best = (0..q.len()).fold(0, |a, i| if q[i] > q[a] { i } else { a });
    let alt = (0..q.len()).filter(|&i| (ps[i] - ps[best]).abs() > gap)
        .fold(None, |a: Option<usize>, i| if a.map_or(true, |a| q[i] > q[a]) { Some(i) } else { a });
    (best, q[best], alt.map(|i| (i, q[i])))
}

fn main() {
    set_trig_mode(TrigMode::MthLut);
    set_flight_mode(FlightMode::Reference);
    let a = Args::parse();
    let Args { mode, hmax, gamma, nh, dv, vymin, vymax, vzmax, dp, tol, cap, every, fine, gap, trace, trace_q, maxt, .. } = a;
    // `v_y = 0` and `v_z = 0` are nodes, so the start state from rest is one.
    let jy0 = (-vymin / dv).round() as usize;
    let g = Grid::new(hmax, gamma, nh, -(jy0 as f64) * dv, dv, jy0 + (vymax / dv).round() as usize + 1,
                      dv, (vzmax / dv).round() as usize + 1);
    let lim = 85.0;
    assert!(((2.0 * lim / dp).round() * dp - 2.0 * lim).abs() < 1e-9, "--dp {dp} does not divide 170: +85 would be left out");
    let pitches: Vec<f64> = (0..=((2.0 * lim / dp).round() as i64)).map(|i| (-lim + dp * i as f64).min(lim)).collect();
    let trig: Vec<PitchTrig> = pitches.iter().map(|&p| PitchTrig::new(p as f32)).collect();
    let y0s: Vec<f64> = if a.y0.is_empty() { (1..=32).map(|y| y as f64).collect() } else { a.y0.clone() };
    assert!(y0s.iter().all(|&y| y <= hmax), "a --y0 above --hmax {hmax} would read V clamped to the grid's top");
    eprintln!("grid {} x {} x {} = {} states (gamma {gamma}: dh {:.4} at the floor, {:.3} at the top; dv {dv}; h 0..{hmax}, v_y {:.2}..{vymax}, v_z 0..{vzmax}); {} pitches",
              g.nh, g.nvy, g.nvz, g.len(), g.h(1), hmax - g.h(nh - 2), g.vy0, trig.len());

    // `--save` writes `<file>.spec` beside `V`, and `--load` refuses a `V` whose spec differs:
    // the byte count alone accepts a different `--gamma`, `--hmax`, `--dv` or `--dp`.
    let vspec = format!("{} grid {}x{}x{} hmax {hmax} gamma {gamma} vy {} dv {dv} dp {dp}",
                        match mode { Mode::Time => "time", Mode::Dist => "dist" }, g.nh, g.nvy, g.nvz, g.vy0);
    let mut v = vec![0.0f32; g.len()];
    if let Some(f) = &a.load {
        let saved = std::fs::read_to_string(format!("{f}.spec")).unwrap_or_else(|e| panic!("{f}.spec: {e}"));
        assert_eq!(saved.trim(), vspec, "{f} was saved for another grid or mode");
        let b = std::fs::read(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(b.len(), 4 * g.len(), "{f} is not a V for this grid");
        v = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    }
    let sweeps = if a.load.is_some() { 0 } else { a.sweeps };
    let clock = std::time::Instant::now();
    let start = |v: &[f32], y0: f64| g.at(v, y0, 0.0, 0.0);
    let mut done = 0;
    // The start values 100 sweeps back: a start still rising at the end flies longer than the
    // sweeps, or forever.
    let mut back: std::collections::VecDeque<Vec<f64>> = Default::default();
    for s in 1..=sweeps {
        back.push_back(y0s.iter().map(|&y| start(&v, y)).collect());
        if back.len() > 100 { back.pop_front(); }
        let new: Vec<f32> = (0..g.len()).into_par_iter().map(|n| {
            let (h, vy, vz) = g.node(n);
            let vel = Vec3::new(0.0, vy, vz);
            let best = trig.iter().map(|&t| backup(&g, &v, mode, h, vel, t).0).fold(0.0, f64::max);
            best.min(cap) as f32
        }).collect();
        // The largest rise at the start states is what says whether the answer has converged.
        let rise = y0s.iter().map(|&y| g.at(&new, y, 0.0, 0.0) - start(&v, y)).fold(0.0, f64::max);
        v = new;
        done = s;
        if s % every == 0 || rise < tol {
            let capped = v.iter().filter(|&&x| x as f64 >= cap).count();
            eprintln!("sweep {s:4}  start rise {rise:.3e}  capped {capped}  V(start) y0 8 {:.2} 16 {:.2} 24 {:.2} 29 {:.2} 32 {:.2}  {:.0}s",
                      start(&v, 8.0), start(&v, 16.0), start(&v, 24.0), start(&v, 29.0), start(&v, 32.0), clock.elapsed().as_secs_f64());
        }
        if rise < tol { break }
    }

    if let Some(f) = &a.save {
        std::fs::write(f, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        std::fs::write(format!("{f}.spec"), format!("{vspec}\n")).unwrap();
    }
    // The policy, flown exactly from rest.
    let fp: Vec<f64> = (0..=((2.0 * lim / fine).round() as i64)).map(|i| (-lim + fine * i as f64).min(lim)).collect();
    let ft: Vec<PitchTrig> = fp.iter().map(|&p| PitchTrig::new(p as f32)).collect();
    if let Some(f) = &a.map {
        let mh = &a.map_h;
        let cells: Vec<(f64, f64, f64)> = mh.iter().flat_map(|&h| (0..g.nvy).flat_map(move |j| (0..g.nvz).map(move |k| (h, j, k))))
            .map(|(h, j, k)| (h, g.vy0 + j as f64 * g.dvy, k as f64 * g.dvz)).collect();
        let pol: Vec<(i16, f32)> = cells.par_iter().map(|&(h, vy, vz)| {
            let (i, q, alt) = ranked(&g, &v, mode, h, Vec3::new(0.0, vy, vz), &fp, &ft, gap);
            ((fp[i] * 4.0).round() as i16, alt.map_or(f32::INFINITY, |(_, aq)| (q - aq) as f32))
        }).collect();
        let mut b: Vec<u8> = pol.iter().flat_map(|p| p.0.to_le_bytes()).collect();
        b.extend(pol.iter().flat_map(|p| p.1.to_le_bytes()));
        std::fs::write(f, b).unwrap();
        std::fs::write(format!("{f}.spec"), format!("{vspec} fine {fine} gap {gap}\nnvy {} vy0 {} nvz {} dv {dv}\nmap-h {}\n",
                                                      g.nvy, g.vy0, g.nvz, mh.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","))).unwrap();
        eprintln!("map: {} heights x {} x {} to {f}", mh.len(), g.nvy, g.nvz);
    }
    let out = a.out.as_deref();
    if let Some(d) = out { std::fs::create_dir_all(d).unwrap() }
    let tag = match mode { Mode::Time => "time", Mode::Dist => "dist" };
    let spec = format!("grid {}x{}x{} hmax {hmax} gamma {gamma} dv {dv} dp {dp}", g.nh, g.nvy, g.nvz);
    println!("# floordp {tag}: {spec}; {done} sweeps; policy on a {fine}-degree grid");
    println!("{:>5} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}", "y0", "V(start)", "rise/100", "t*", "z(t*)", "outside", "dips<1");
    for (yi, &y0) in y0s.iter().enumerate() {
        let rise100 = start(&v, y0) - back.front().map_or(f64::NAN, |b| b[yi]);
        let (mut h, mut vel, mut z) = (y0, Vec3::ZERO, 0.0);
        let mut p = Vec::new();
        let mut outside = 0;
        let (mut t_exit, mut z_exit) = (f64::NAN, f64::NAN);
        let mut hs = vec![h];
        let mut rows = vec!["t,h,vy,vz,pitch,q,alt_pitch,margin".to_string()];
        let mut qs: Vec<u8> = Vec::new();
        for t in 0..maxt {
            let (i, q, alt) = ranked(&g, &v, mode, h, vel, &fp, &ft, gap);
            if trace {
                let (ap, aq) = alt.map_or((f64::NAN, f64::NAN), |(j, aq)| (fp[j], aq));
                rows.push(format!("{t},{h},{},{},{},{q},{ap},{}", vel.y, vel.z, fp[i], q - aq));
                if t < trace_q { qs.extend(trig.iter().flat_map(|&tr| (backup(&g, &v, mode, h, vel, tr).0 as f32).to_le_bytes())) }
            }
            let nv = update_fall_flying_movement_cached(vel, ft[i]);
            let nh = h + nv.y;
            p.push(fp[i]);
            if nh < 0.0 {
                let f = h / (h - nh);
                t_exit = t as f64 + f;
                z_exit = z + f * nv.z;
                break;
            }
            h = nh; vel = nv; z += nv.z;
            hs.push(h);
            if !g.inside(h, vel.y, vel.z) { outside += 1 }
        }
        let dips = (1..hs.len().saturating_sub(1)).filter(|&j| hs[j] < 1.0 && hs[j] <= hs[j - 1] && hs[j] <= hs[j + 1]).count();
        println!("{y0:>5} {:>10.3} {rise100:>10.3} {:>10.3} {:>10.3} {outside:>8} {dips:>8}", start(&v, y0), t_exit, z_exit);
        if let Some(d) = out {
            let body: Vec<String> = p.iter().map(|x| format!("{}", *x as f32)).collect();
            std::fs::write(format!("{d}/{tag}_y{y0}.pitches"),
                           format!("# floordp {tag} y0 {y0} {spec} sweeps {done} fine {fine}\n# V(start) {:.4}  rise/100 {rise100:.4}  t* {t_exit:.4}  z(t*) {z_exit:.4}\n{}\n",
                                   start(&v, y0), body.join(" "))).unwrap();
            if trace {
                std::fs::write(format!("{d}/{tag}_y{y0}.trace"), rows.join("\n") + "\n").unwrap();
                std::fs::write(format!("{d}/{tag}_y{y0}.q"), &qs).unwrap();
            }
        }
    }
}
