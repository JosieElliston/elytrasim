//! A stationary pump policy, velocity -> pitch, as the optimum of an infinite-horizon objective
//! on ΔTE. Two objectives, both solved on a grid over `v = (v_y, v_z)`:
//!
//! * `--mode disc --discount γ`: maximize `Σ γ^t r_t`, `r_t` the tick's ΔTE in blocks.
//!   `V(v) = max_p [r(v, p) + γ V(v')]`. The effective horizon is `1 / (1 - γ)` ticks.
//! * `--mode avg`: maximize the gain rate `ρ = lim ΔTE_n / n`, and among policies with the same
//!   rate, the bias `h(v) = Σ_t (r_t - ρ)` (summed along the flight from `v`, Cesàro-averaged when
//!   the flight ends on a cycle). They satisfy `h(v) + ρ = max_p [r(v, p) + h(v')]`, `h` up to a
//!   constant, pinned here by `h(--ref) = 0`. `TE + h` rises by exactly `ρ` per tick along the
//!   optimal flight and by no more under any other pitch, so one-tick greedy on `TE + h` is the
//!   optimal policy.
//!
//! The state is velocity alone: the flight kernel reads nothing else, and ΔTE doesn't depend on
//! height. `V`/`h` live on the nodes and are read between them bilinearly (semi-Lagrangian).
//! A step leaving the grid is clamped onto its edge, and `r` is charged to the clamped state, so
//! the clamp only ever destroys energy (the grid holds the origin, so a clamp shrinks |v|). Were
//! `r` the true ΔTE, the energy a clamp destroys would be credited and never paid back: a pump.
//!
//! Solved by modified policy iteration: a sweep maximizes over `--dp`-degree pitches at every
//! node, then `--evals` sweeps evaluate that policy with its next states cached. `avg` damps every
//! update by `--tau` (the aperiodicity transform: the optimal flight is a cycle, and undamped
//! relative value iteration on a periodic chain oscillates with its period). In `avg`, every
//! improvement sweep brackets the grid problem's rate: `min (Th - h) <= ρ <= max (Th - h)`.
//!
//! The grid's answer is then flown exactly: from `--v0`, each tick the pitch maximizing the
//! backup over a `--fine`-degree grid, `--fly` ticks, and its rate is read over the last
//! `--window` ticks. That rate is a real flight's.
//!
//! Physics is `field`'s f64 `libm` kernel at yaw zero, pitch clamped to `--limit`, v_z >= 0.
//!
//! Writes into `--out`, as .npy: `vy`, `vz` (axes, blocks/tick); per node, indexed [vy][vz]: `V`
//! (`h` in avg; in disc, `V` minus `V(ref)`), `pitch` (the policy on the fine grid), `margin`
//! (how much the best beats the best pitch more than `--gap` degrees away, blocks), `q1` (the
//! one-tick ΔTE at the policy's pitch); and `fly`, rows of (t, vy, vz, pitch, TE) for the flown
//! policy. `spec.txt` records the arguments and results.

use std::io::Write;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

const GRAVITY: f64 = 0.08;
const DRAG_Y: f64 = 0.98_f32 as f64;
const DRAG_Z: f64 = 0.99_f32 as f64;
/// `--conv-curv`, set once in `main`; 1 is the real physics. See `step`.
static CONV_CURV: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
/// `--conv-lift-exp`, set once in `main`; 1 is the real physics. See `step`.
static CONV_EXP: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

#[derive(Clone, Copy, PartialEq, ValueEnum, Debug)]
enum Mode { Disc, Avg }

#[derive(Parser)]
#[command(allow_negative_numbers = true)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    /// Discount per tick, `disc` only.
    #[arg(long, default_value_t = 0.99)]
    discount: f64,
    /// Damping of every `avg` update, in (0, 1].
    #[arg(long, default_value_t = 0.5)]
    tau: f64,
    /// Node spacing, blocks/tick, both axes.
    #[arg(long, default_value_t = 0.01)]
    dv: f64,
    #[arg(long, default_value_t = -2.0)]
    vymin: f64,
    #[arg(long, default_value_t = 2.8)]
    vymax: f64,
    #[arg(long, default_value_t = 3.6)]
    vzmax: f64,
    /// Pitch step of the improvement sweeps, degrees; must divide 2 `--limit`.
    #[arg(long, default_value_t = 1.0)]
    dp: f64,
    /// Evaluation sweeps per improvement sweep.
    #[arg(long, default_value_t = 30)]
    evals: usize,
    /// Most improvement sweeps.
    #[arg(long, default_value_t = 2000)]
    sweeps: usize,
    /// Stop once an improvement sweep moves no node more than this (disc: |TV - V|; avg: the
    /// width of the rate bracket), blocks.
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,
    /// The node `h` (or `V`) is pinned to 0 at, vy,vz blocks/tick: the reference cycle's v0.
    #[arg(long, value_delimiter = ',', default_value = "0.167467,0.200887")]
    r#ref: Vec<f64>,
    /// Start of the flown policy, vy,vz blocks/tick.
    #[arg(long, value_delimiter = ',', default_value = "0.167467,0.200887")]
    v0: Vec<f64>,
    /// Pitch step of the flown policy and the written map, degrees.
    #[arg(long, default_value_t = 0.25)]
    fine: f64,
    /// Degrees from the best pitch that the margin's runner-up must be.
    #[arg(long, default_value_t = 10.0)]
    gap: f64,
    #[arg(long, default_value_t = 60000)]
    fly: usize,
    /// Ticks at the end of the flight the rate is read over, cut down to a whole number of the
    /// flight's final period when it has one. why: a window that isn't a whole number of cycles
    /// is off by up to the cycle's KE swing (~40 blocks) over its length, 0.016 b/s even on 50000
    /// ticks. The transient onto the cycle runs thousands of ticks, hence the long flight.
    #[arg(long, default_value_t = 50000)]
    window: usize,
    /// Velocity distance, blocks/tick, at which the flight counts as back where it was a period ago.
    #[arg(long, default_value_t = 1e-5)]
    close: f64,
    #[arg(long, default_value_t = 50)]
    every: usize,
    /// Pitch clamp, degrees either way. Minecraft's is 90; `field` uses 89, the sweeps 85.
    #[arg(long, default_value_t = 89.0)]
    limit: f64,
    /// Ablation, not physics: scales the `lift^2` coefficient of the descent-to-forward conversion
    /// (`-0.1 * 0.75 * GRAVITY`, from the gravity relief inside the conversion's amount) while
    /// holding the conversion's value at lift 0 and 1, so both bang outcomes are unchanged and
    /// only interior pitches move. That term is the only thing making a dive tick's outcome
    /// concave in lift; 0 makes the dive's kernel affine in lift, above 1 more concave. 1 is
    /// vanilla.
    #[arg(long, default_value_t = 1.0)]
    conv_curv: f64,
    /// Ablation, not physics: the conversion's outer lift factor becomes lift^k, so
    /// `conv = -0.1 * lift^k * vy_grav`, with the gravity relief inside `vy_grav` left alone. Both
    /// ends are held (lift 1 exactly; at the 89° clamp lift is 3e-4 and lift^k moves it a little).
    /// Unlike `--conv-curv`, the change scales with the fall speed. k < 1 converts more at interior
    /// pitches, k > 1 less. 1 is vanilla.
    #[arg(long, default_value_t = 1.0)]
    conv_lift_exp: f64,
    /// Ablation of the objective, not the physics: every backup is charged
    /// `s / (2 GRAVITY) * |v' - c|^2`, `c` the mean next state over the `--fine` pitches from that
    /// state (the middle of the tick's reachable arc), so `s` is in units of KE's own curvature.
    /// It adds concavity to `r + γ V(v')` along the arc. q1, the flight's TE and its rate stay true
    /// ΔTE. 0 is vanilla.
    #[arg(long, default_value_t = 0.0)]
    obj_concavity: f64,
    #[arg(long)]
    out: String,
}

/// One tick from (vy, vz) at a pitch given as (cos², -sin if nose up else 0): `field::g`'s kernel.
#[inline]
fn step(vy: f64, vz: f64, (lift, up_sin): (f64, f64)) -> (f64, f64) {
    let move_hor = vz.abs();
    let mut y = vy + GRAVITY * (-1.0 + lift * 0.75);
    let mut z = vz;
    let f = *CONV_CURV.get().unwrap_or(&1.0);
    let k = *CONV_EXP.get().unwrap_or(&1.0);
    let outer = if k == 1.0 { lift } else { lift.powf(k) };
    let conv = if y < 0.0 { y * -0.1 * outer + (f - 1.0) * 0.1 * 0.75 * GRAVITY * lift * (1.0 - lift) } else { 0.0 };
    y += conv;
    z += conv;
    let up = move_hor * up_sin * 0.04;
    y += up * 3.2;
    z -= up;
    z += (move_hor - z) * 0.1;
    (y * DRAG_Y, z * DRAG_Z)
}

fn trig(p: f64) -> (f64, f64) {
    let lean = p.to_radians();
    (lean.cos() * lean.cos(), if lean < 0.0 { -lean.sin() } else { 0.0 })
}

fn ke(vy: f64, vz: f64) -> f64 { (vy * vy + vz * vz) * 0.5 / GRAVITY }

struct Grid { vy0: f64, dv: f64, ny: usize, nz: usize }

/// Where a velocity lands on the grid: the lower-left node and the bilinear weights.
#[derive(Clone, Copy, Default)]
struct Cell { n: u32, a: f32, b: f32 }

impl Grid {
    fn len(&self) -> usize { self.ny * self.nz }
    fn node(&self, n: usize) -> (f64, f64) {
        (self.vy0 + (n / self.nz) as f64 * self.dv, (n % self.nz) as f64 * self.dv)
    }
    fn clamp(&self, vy: f64, vz: f64) -> (f64, f64) {
        (vy.clamp(self.vy0, self.vy0 + (self.ny - 1) as f64 * self.dv), vz.clamp(0.0, (self.nz - 1) as f64 * self.dv))
    }
    /// The clamped point's cell.
    fn cell(&self, vy: f64, vz: f64) -> Cell {
        let split = |x: f64, n: usize| {
            let i = (x.floor() as usize).min(n - 2);
            (i, x - i as f64)
        };
        let (vy, vz) = self.clamp(vy, vz);
        let (j, a) = split((vy - self.vy0) / self.dv, self.ny);
        let (k, b) = split(vz / self.dv, self.nz);
        Cell { n: (j * self.nz + k) as u32, a: a as f32, b: b as f32 }
    }
    #[inline]
    fn read(&self, v: &[f64], c: Cell) -> f64 {
        let n = c.n as usize;
        let (a, b) = (c.a as f64, c.b as f64);
        let lo = v[n] * (1.0 - b) + v[n + 1] * b;
        let hi = v[n + self.nz] * (1.0 - b) + v[n + self.nz + 1] * b;
        lo * (1.0 - a) + hi * a
    }
}

/// One tick from (vy, vz) at a pitch: the clamped next state's cell, the tick's reward, and the
/// `--obj-concavity` charge `s` (already scaled) against the arc's middle `ctr`.
#[inline]
fn transition(g: &Grid, vy: f64, vz: f64, t: (f64, f64), s: f64, ctr: (f64, f64)) -> (Cell, f64, f64) {
    let (ny, nz) = step(vy, vz, t);
    let (cy, cz) = g.clamp(ny, nz);
    let pen = if s == 0.0 { 0.0 } else { s * ((ny - ctr.0).powi(2) + (nz - ctr.1).powi(2)) };
    (g.cell(ny, nz), ny + ke(cy, cz) - ke(vy, vz), pen)
}

/// The middle of the tick's reachable arc from (vy, vz): the mean next state over `ts`.
fn arc_mid(vy: f64, vz: f64, ts: &[(f64, f64)]) -> (f64, f64) {
    let (y, z) = ts.iter().fold((0.0, 0.0), |(y, z), &t| { let (a, b) = step(vy, vz, t); (y + a, z + b) });
    (y / ts.len() as f64, z / ts.len() as f64)
}

fn write_npy(path: &std::path::Path, shape: &[usize], data: &[f64]) {
    let shape_s = match shape {
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_s}, }}");
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn main() {
    let a = Args::parse();
    let Args { mode, discount, tau, dv, vymin, vymax, vzmax, dp, evals, sweeps, tol, fine, gap, every, limit, .. } = a;
    assert!(tau > 0.0 && tau <= 1.0);
    CONV_CURV.set(a.conv_curv).unwrap();
    CONV_EXP.set(a.conv_lift_exp).unwrap();
    let jy0 = (-vymin / dv).round() as usize;
    let g = Grid { vy0: -(jy0 as f64) * dv, dv, ny: jy0 + (vymax / dv).round() as usize + 1, nz: (vzmax / dv).round() as usize + 1 };
    let pitches = |d: f64| -> Vec<f64> {
        let n = (2.0 * limit / d).round() as i64;
        assert!((n as f64 * d - 2.0 * limit).abs() < 1e-9, "a pitch step of {d} does not divide {}", 2.0 * limit);
        (0..=n).map(|i| (-limit + d * i as f64).min(limit)).collect()
    };
    let ps = pitches(dp);
    let ts: Vec<(f64, f64)> = ps.iter().map(|&p| trig(p)).collect();
    let fps = pitches(fine);
    let fts: Vec<(f64, f64)> = fps.iter().map(|&p| trig(p)).collect();
    let sc = a.obj_concavity * 0.5 / GRAVITY;
    let mid = |vy: f64, vz: f64| if sc == 0.0 { (0.0, 0.0) } else { arc_mid(vy, vz, &fts) };
    let mids: Vec<(f64, f64)> = (0..g.len()).into_par_iter().map(|n| { let (vy, vz) = g.node(n); mid(vy, vz) }).collect();
    let (gam, damp) = match mode { Mode::Disc => (discount, 1.0), Mode::Avg => (1.0, tau) };
    let rc = g.cell(a.r#ref[0], a.r#ref[1]);
    eprintln!("{mode:?}: grid {} x {} = {} nodes (dv {dv}, vy {:.2}..{vymax}, vz 0..{vzmax}), {} pitches; {}",
              g.ny, g.nz, g.len(), g.vy0, ps.len(),
              match mode { Mode::Disc => format!("discount {discount} (horizon {:.0} ticks)", 1.0 / (1.0 - discount)), Mode::Avg => format!("tau {tau}") });

    let clock = std::time::Instant::now();
    let mut v = vec![0.0f64; g.len()];
    let mut rate = (f64::NAN, f64::NAN);
    let mut done = 0;
    for s in 1..=sweeps {
        // Improvement: the best pitch at every node, and Tv.
        let best: Vec<(f64, Cell, f64)> = (0..g.len()).into_par_iter().map(|n| {
            let (vy, vz) = g.node(n);
            let mut top = (f64::NEG_INFINITY, Cell::default(), 0.0);
            for &t in &ts {
                let (c, r, pen) = transition(&g, vy, vz, t, sc, mids[n]);
                let q = r - pen + gam * g.read(&v, c);
                if q > top.0 { top = (q, c, r - pen) }
            }
            top
        }).collect();
        let (lo, hi) = best.par_iter().zip(&v).map(|(b, x)| b.0 - x).fold(|| (f64::INFINITY, f64::NEG_INFINITY), |(l, h), d| (l.min(d), h.max(d)))
            .reduce(|| (f64::INFINITY, f64::NEG_INFINITY), |p, q| (p.0.min(q.0), p.1.max(q.1)));
        let resid = match mode { Mode::Disc => lo.abs().max(hi.abs()), Mode::Avg => hi - lo };
        rate = (lo, hi);
        v = best.iter().zip(&v).map(|(b, x)| x + damp * (b.0 - x)).collect();
        // Evaluation of that policy, next states cached.
        for _ in 0..evals {
            let new: Vec<f64> = best.par_iter().zip(&v).map(|(&(_, c, r), x)| x + damp * (r + gam * g.read(&v, c) - x)).collect();
            v = new;
        }
        if mode == Mode::Avg {
            let pin = g.read(&v, rc);
            v.iter_mut().for_each(|x| *x -= pin);
        }
        done = s;
        if s % every == 0 || resid < tol {
            match mode {
                Mode::Disc => eprintln!("sweep {s:5}  |TV - V| {resid:.3e}  V(ref) {:.4}  {:.0}s", g.read(&v, rc), clock.elapsed().as_secs_f64()),
                Mode::Avg => eprintln!("sweep {s:5}  rate in [{lo:.7}, {hi:.7}] b/tick  {:.0}s", clock.elapsed().as_secs_f64()),
            }
        }
        if resid < tol { break }
    }

    // The policy on the fine grid, its margin, and its flight.
    let rank = |vy: f64, vz: f64| -> (usize, f64, f64) {
        let m = mid(vy, vz);
        let q: Vec<f64> = fts.iter().map(|&t| { let (c, r, pen) = transition(&g, vy, vz, t, sc, m); r - pen + gam * g.read(&v, c) }).collect();
        let i = (0..q.len()).fold(0, |a, i| if q[i] > q[a] { i } else { a });
        let alt = (0..q.len()).filter(|&j| (fps[j] - fps[i]).abs() > gap).map(|j| q[j]).fold(f64::NEG_INFINITY, f64::max);
        (i, q[i] - alt, transition(&g, vy, vz, fts[i], 0.0, m).1)
    };
    let map: Vec<(usize, f64, f64)> = (0..g.len()).into_par_iter().map(|n| { let (vy, vz) = g.node(n); rank(vy, vz) }).collect();

    let (mut vy, mut vz) = (a.v0[0], a.v0[1]);
    let mut te = 0.0;
    let mut rows = Vec::with_capacity(5 * a.fly);
    let mut tes = vec![ke(vy, vz)];
    let mut vs = vec![(vy, vz)];
    let mut clamped = 0;
    for t in 0..a.fly {
        let (i, _, _) = rank(vy, vz);
        rows.extend([t as f64, vy, vz, fps[i], te + ke(vy, vz)]);
        let (ny, nz) = step(vy, vz, fts[i]);
        if (ny, nz) != g.clamp(ny, nz) { clamped += 1 }
        te += ny;
        (vy, vz) = (ny, nz);
        tes.push(te + ke(vy, vz));
        vs.push((vy, vz));
    }
    let w = a.window.min(a.fly);
    let last = vs[a.fly];
    let period = (1..=w).find(|&k| { let (y, z) = vs[a.fly - k]; (y - last.0).hypot(z - last.1) < a.close });
    let w = period.map_or(w, |p| w / p * p);
    let flown = (tes[a.fly] - tes[a.fly - w]) / w as f64;
    let half = (tes[a.fly - w / 2] - tes[a.fly - w]) / (w / 2) as f64;

    let out = std::path::Path::new(&a.out);
    std::fs::create_dir_all(out).unwrap();
    let shape = [g.ny, g.nz];
    let pin = match mode { Mode::Disc => g.read(&v, rc), Mode::Avg => 0.0 };
    write_npy(&out.join("vy.npy"), &[g.ny], &(0..g.ny).map(|j| g.vy0 + j as f64 * dv).collect::<Vec<_>>());
    write_npy(&out.join("vz.npy"), &[g.nz], &(0..g.nz).map(|k| k as f64 * dv).collect::<Vec<_>>());
    write_npy(&out.join("V.npy"), &shape, &v.iter().map(|x| x - pin).collect::<Vec<_>>());
    write_npy(&out.join("pitch.npy"), &shape, &map.iter().map(|m| fps[m.0]).collect::<Vec<_>>());
    write_npy(&out.join("margin.npy"), &shape, &map.iter().map(|m| m.1).collect::<Vec<_>>());
    write_npy(&out.join("q1.npy"), &shape, &map.iter().map(|m| m.2).collect::<Vec<_>>());
    write_npy(&out.join("fly.npy"), &[a.fly, 5], &rows);
    let objective = match mode {
        Mode::Disc => format!("disc discount {discount} horizon {:.1}", 1.0 / (1.0 - discount)),
        Mode::Avg => format!("avg tau {tau}"),
    };
    let summary = format!(
        "pumpdp {objective} limit {limit}{}\ngrid dv {dv} vy {}..{vymax} vz 0..{vzmax} ({} x {}) dp {dp} evals {evals} fine {fine} gap {gap}\n\
         ref {:?} v0 {:?}\nsweeps {done} tol {tol}\nbracket (min, max of Tv - v) {:.8} {:.8}\n\
         period {} ticks\nflown rate over last {w} ticks {flown:.6} b/tick ({:.4} b/s); first half of that window {half:.6}; clamped ticks {clamped}\n\
         time {:.0}s\n",
        format!("{}{}", if a.conv_curv == 1.0 { String::new() } else { format!(" conv_curv {}", a.conv_curv) },
                if a.conv_lift_exp == 1.0 { String::new() } else { format!(" conv_lift_exp {}", a.conv_lift_exp) })
            + &if a.obj_concavity == 0.0 { String::new() } else { format!(" obj_concavity {}", a.obj_concavity) },
        g.vy0, g.ny, g.nz, a.r#ref, a.v0, rate.0, rate.1, period.map_or("none".to_string(), |p| p.to_string()), flown * 20.0, clock.elapsed().as_secs_f64());
    std::fs::write(out.join("spec.txt"), &summary).unwrap();
    print!("{summary}");
}
