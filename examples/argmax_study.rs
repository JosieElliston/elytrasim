//! How precise does a pitch argmax need to be, and how cheaply can we get it? Scores pitch
//! refiners against a dense reference (0.01-degree sweep, then golden section on every sampled
//! peak) over the reference cycle's states and a velocity grid. `opt::argmax`'s constants come
//! from here. `cargo run --release --example argmax_study [lut]`; ~10 s.
//!
//! Columns: mean evaluations of f; p99 and max |pitch - reference|; max regret, f(reference) -
//! f(pitch), in the objective's units (blocks for dTE); and `wrong`, the states where the method
//! is more than 0.5 degrees off *and* scores worse by more than 1e-9 -- a wrong peak, not a tie.
use elytrasim::opt::*;
use elytrasim::replay_pitches::REPLAY_PITCHES_300;
use elytrasim::sim::*;
use rayon::prelude::*;
use std::cell::Cell;

const GOLD: f64 = 0.618_033_988_749_894_9;

fn local_maxima(ps: &[f64], vs: &[f64], k: usize) -> Vec<usize> {
    let n = vs.len();
    let mut c: Vec<usize> = (0..n)
        .filter(|&i| (i == 0 || vs[i] >= vs[i - 1]) && (i == n - 1 || vs[i] >= vs[i + 1]))
        .collect();
    c.sort_by(|&a, &b| vs[b].partial_cmp(&vs[a]).unwrap());
    c.dedup_by(|a, b| (ps[*a] - ps[*b]).abs() < 1e-12);
    c.truncate(k);
    c
}

fn sweep(f: &dyn Fn(f64) -> f64, step: f64, lo: f64, hi: f64) -> (Vec<f64>, Vec<f64>) {
    let n = ((hi - lo) / step).ceil() as i64;
    let ps: Vec<f64> = (0..=n).map(|i| (lo + step * i as f64).min(hi)).collect();
    let vs = ps.iter().map(|&p| f(p)).collect();
    (ps, vs)
}

fn golden(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64) -> (f64, f64) {
    let mut c = b - GOLD * (b - a);
    let mut d = a + GOLD * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    while b - a > tol {
        if fc >= fd { b = d; d = c; fd = fc; c = b - GOLD * (b - a); fc = f(c) }
        else { a = c; c = d; fc = fd; d = a + GOLD * (b - a); fd = f(d) }
    }
    if fc >= fd { (c, fc) } else { (d, fd) }
}

/// Brent's method for a maximum on [a, b] (Numerical Recipes' `brent`, negated).
fn brent(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64) -> (f64, f64) {
    let cg = 1.0 - GOLD;
    let g = |p: f64| -f(p);
    let mut x = a + cg * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = g(x);
    let (mut fw, mut fv) = (fx, fx);
    let (mut d, mut e): (f64, f64) = (0.0, 0.0);
    for _ in 0..100 {
        let xm = 0.5 * (a + b);
        let tol1 = tol * 0.5;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) { break }
        let mut golden_step = true;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 { p = -p }
            q = q.abs();
            let etemp = e;
            e = d;
            if !(p.abs() >= (0.5 * q * etemp).abs() || p <= q * (a - x) || p >= q * (b - x)) {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 { d = tol1.copysign(xm - x) }
                golden_step = false;
            }
        }
        if golden_step {
            e = if x >= xm { a - x } else { b - x };
            d = cg * e;
        }
        let u = if d.abs() >= tol1 { x + d } else { x + tol1.copysign(d) };
        let fu = g(u);
        if fu <= fx {
            if u >= x { a = x } else { b = x }
            (v, w, x) = (w, x, u);
            (fv, fw, fx) = (fw, fx, fu);
        } else {
            if u < x { a = u } else { b = u }
            if fu <= fw || w == x { (v, w) = (w, u); (fv, fw) = (fw, fu) }
            else if fu <= fv || v == x || v == w { v = u; fv = fu }
        }
    }
    (x, -fx)
}

/// Finite-difference Newton ascent from `p0`, safeguarded to [a, b]: the user's "grad ascent".
fn newton(f: &dyn Fn(f64) -> f64, p0: f64, a: f64, b: f64, tol: f64, h: f64) -> (f64, f64) {
    let mut p = p0;
    let mut fp = f(p);
    for _ in 0..30 {
        let (fl, fr) = (f((p - h).max(a)), f((p + h).min(b)));
        let g = (fr - fl) / (2.0 * h);
        let c = (fr - 2.0 * fp + fl) / (h * h);
        let step = if c < 0.0 { -g / c } else { g.signum() * 0.5 * (b - a) };
        let q = (p + step).clamp(a, b);
        let fq = f(q);
        if fq <= fp { break }
        let moved = (q - p).abs();
        p = q; fp = fq;
        if moved < tol { break }
    }
    (p, fp)
}

/// The `opt::argmax_in` this study replaced: a 0.125-degree sweep, then 60 ternary steps in the
/// winning cell.
fn old_argmax_in(f: &dyn Fn(f64) -> f64, step: f64, lo: f64, hi: f64) -> f64 {
    let (mut bp, mut bs) = (0.0_f64.clamp(lo, hi), f64::NEG_INFINITY);
    let n = ((hi - lo) / step).round() as i64;
    for i in 0..=n {
        let p = (lo + step * i as f64).min(hi);
        let v = f(p);
        if v > bs { bs = v; bp = p }
    }
    let (mut a, mut b) = ((bp - step).max(lo), (bp + step).min(hi));
    for _ in 0..60 {
        let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
        if f(m1) < f(m2) { a = m1 } else { b = m2 }
    }
    let p = 0.5 * (a + b);
    if f(p) > bs { p } else { bp }
}

#[derive(Clone, Copy)]
enum M { Old, Lib, Golden, Brent, Newton, NewtonWide }

fn run(m: M, f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, step: f64, tol: f64, k: usize) -> f64 {
    match m {
        M::Old => return old_argmax_in(f, 0.125, lo, hi),
        M::Lib => return argmax_in(f, lo, hi),
        _ => {}
    }
    let (ps, vs) = sweep(f, step, lo, hi);
    let mut best = (ps[0], f64::NEG_INFINITY);
    for i in local_maxima(&ps, &vs, k) {
        if vs[i] > best.1 { best = (ps[i], vs[i]) }
        let (a, b) = ((ps[i] - step).max(lo), (ps[i] + step).min(hi));
        let r = match m {
            M::Golden => golden(f, a, b, tol),
            M::Brent => brent(f, a, b, tol),
            M::Newton => newton(f, ps[i], a, b, tol, 1e-3),
            M::NewtonWide => newton(f, ps[i], a, b, tol, 0.05),
            M::Old | M::Lib => unreachable!(),
        };
        if r.1 > best.1 { best = r }
    }
    best.0
}

fn reference(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64) -> (f64, f64, f64) {
    let (ps, vs) = sweep(f, 0.01, lo, hi);
    let mut best = (0.0, f64::NEG_INFINITY);
    let mut second = f64::NEG_INFINITY; // best local max more than 1 degree away
    let refined: Vec<(f64, f64)> = local_maxima(&ps, &vs, 6)
        .into_iter()
        .map(|i| golden(f, (ps[i] - 0.01).max(lo), (ps[i] + 0.01).min(hi), 1e-9))
        .collect();
    for &r in &refined { if r.1 > best.1 { best = r } }
    for &r in &refined { if (r.0 - best.0).abs() > 1.0 && r.1 > second { second = r.1 } }
    (best.0, best.1, best.1 - second)
}

fn main() {
    let lut = std::env::args().any(|a| a == "lut");
    if lut { set_trig_mode(TrigMode::MthLut) }
    let ps: Vec<f64> = REPLAY_PITCHES_300.iter().map(|&p| p as f64).collect();
    let mut states: Vec<(String, State)> = replay(&ps).into_iter().take(300).map(|s| ("cycle".into(), s)).collect();
    for i in 0..15 { for j in 0..15 {
        let vel = Vec3::new(0.0, -3.0 + 4.5 * i as f64 / 14.0, 3.5 * j as f64 / 14.0);
        states.push(("grid".into(), State { pos: Vec3::ZERO, vel }));
    }}

    type Obj = (String, f64, f64, Box<dyn Fn(&State, f64) -> f64 + Sync>);
    let mut objs: Vec<Obj> = vec![];
    for n in [1usize, 4, 20, 48] {
        objs.push((format!("dTE n={n}"), -90.0, 90.0, Box::new(move |s: &State, p| run_n(s, p, n).total_energy() - s.total_energy())));
    }
    objs.push(("dJ n=20 w=.01 |p|<85".into(), -85.0, 85.0, Box::new(|s: &State, p| {
        let t = run_n(s, p, 20); t.total_energy() + 0.01 * t.pos.z - s.total_energy() - 0.01 * s.pos.z })));
    objs.push(("next speed".into(), -90.0, 90.0, Box::new(|s: &State, p| ticked(s, p).vel.length())));
    objs.push(("-|next v_y|".into(), -90.0, 90.0, Box::new(|s: &State, p| -ticked(s, p).vel.y.abs())));

    let methods: Vec<(String, M, f64, f64, usize)> = vec![
        ("old (0.125 sweep + 60 ternary)".into(), M::Old, 0.0, 0.0, 0),
        ("opt::argmax_in".into(), M::Lib, 0.0, 0.0, 0),
        ("golden  S=1 tol=1e-3 k=3".into(), M::Golden, 1.0, 1e-3, 3),
        ("golden  S=2 tol=1e-3 k=3".into(), M::Golden, 2.0, 1e-3, 3),
        ("golden  S=5 tol=1e-3 k=3".into(), M::Golden, 5.0, 1e-3, 3),
        ("golden  S=2 tol=1e-2 k=3".into(), M::Golden, 2.0, 1e-2, 3),
        ("golden  S=2 tol=1e-4 k=3".into(), M::Golden, 2.0, 1e-4, 3),
        ("golden  S=2 tol=1e-3 k=1".into(), M::Golden, 2.0, 1e-3, 1),
        ("brent   S=2 tol=1e-3 k=3".into(), M::Brent, 2.0, 1e-3, 3),
        ("brent   S=2 tol=1e-4 k=3".into(), M::Brent, 2.0, 1e-4, 3),
        ("newton  S=2 tol=1e-3 k=3 h=1e-3".into(), M::Newton, 2.0, 1e-3, 3),
        ("newton  S=2 tol=1e-3 k=3 h=0.05".into(), M::NewtonWide, 2.0, 1e-3, 3),
    ];

    println!("trig = {}, {} states ({} cycle + grid)", trig_mode(), states.len(), 300);
    for (name, lo, hi, obj) in &objs {
        let refs: Vec<(f64, f64, f64)> = states.par_iter().map(|(_, s)| reference(&|p| obj(s, p), *lo, *hi)).collect();
        let mut gaps: Vec<f64> = refs.iter().map(|r| r.2).collect();
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let near_tie = gaps.iter().filter(|&&g| g < 1e-6).count();
        println!("\n== {name}   (states whose runner-up mode is within 1e-6 of the max: {near_tie})");
        println!("{:<36} {:>9} {:>9} {:>11} {:>9} {:>7}", "method", "evals", "p99 |dp|", "max |dp|", "max regret", "wrong");
        for (mname, m, step, tol, k) in &methods {
            let rows: Vec<(f64, f64, usize)> = states.par_iter().zip(&refs).map(|((_, s), r)| {
                let cnt = Cell::new(0usize);
                let f = |p: f64| { cnt.set(cnt.get() + 1); obj(s, p) };
                let p = run(*m, &f, *lo, *hi, *step, *tol, *k);
                ((p - r.0).abs(), r.1 - obj(s, p), cnt.get())
            }).collect();
            let mut dp: Vec<f64> = rows.iter().map(|r| r.0).collect();
            dp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let evals = rows.iter().map(|r| r.2).sum::<usize>() as f64 / rows.len() as f64;
            let regret = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
            let wrong = rows.iter().filter(|r| r.0 > 0.5 && r.1 > 1e-9).count();
            println!("{mname:<36} {evals:>9.0} {:>9.1e} {:>11.1e} {regret:>9.1e} {wrong:>7}",
                     dp[dp.len() * 99 / 100], dp[dp.len() - 1]);
        }
    }
}
