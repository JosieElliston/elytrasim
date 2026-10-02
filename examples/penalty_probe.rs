//! Why does the polish stall under a large floor penalty while its certificate still finds a big
//! single-tick move? For one schedule: the certificate's residuals at a fine and at the polish's
//! global step, and the per-tick score curve at the tick that gains most.
//!
//! cargo run --release --example penalty_probe -- <pitches> <n> <lambda> <y0> <quad> <mu>
use elytrasim::opt::*;
use elytrasim::sim::*;

fn main() {
    set_trig_mode(TrigMode::MthLut);
    set_flight_mode(FlightMode::Reference);
    let a: Vec<String> = std::env::args().collect();
    let p = read_pitches(&a[1]);
    let n: usize = a[2].parse().unwrap();
    let lambda: f64 = a[3].parse().unwrap();
    let y0: f64 = a[4].parse().unwrap();
    let quad: f64 = a[5].parse().unwrap();
    let mu: f64 = a[6].parse().unwrap();
    assert_eq!(p.len(), n);
    let obj = Objective { v0: Vec3::ZERO, n, lambda };
    let floor = Floor { depth: y0, margin: 0.0, weight: 0.0, wall: 0.0, quad };
    let rough = Rough { mu, shape: PriceShape::L2(2.0), limit: 85.0, ..Default::default() };
    let score_all = |q: &[f64]| obj.eval(q) - floor.cost(&obj.replay(q)) - rough.cost(q);
    let base = score_all(&p);
    println!("J(priced) {base:.6}  floor cost {:.6e}", floor.cost(&obj.replay(&p)));
    for step in [0.25, 2.0] {
        let r = residuals_floored(&obj, &p, step, Jitter::default(), rough, floor);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| r[j].1.total_cmp(&r[i].1));
        println!("step {step}: top moves (tick, dpitch, gain):");
        for &i in idx.iter().take(5) { println!("  t {i:4}  {:+9.4}  {:.6e}", r[i].0, r[i].1) }
    }
    let r = residuals_floored(&obj, &p, 0.25, Jitter::default(), rough, floor);
    let t = (0..n).max_by(|&i, &j| r[i].1.total_cmp(&r[j].1)).unwrap();
    let best = p[t] + r[t].0;
    println!("tick {t}: current {:.4}, best {best:.4}; the score of that one pitch, others fixed:", p[t]);
    let mut q = p.clone();
    let mut f = |x: f64| { q[t] = x as f32 as f64; score_all(&q) - base };
    for k in -40..=40 {
        let x = best + k as f64 * 0.05;
        if x.abs() <= 85.0 { println!("  {x:9.4}  {:+.6e}", f(x)) }
    }
    println!("coarse, every 2 degrees:");
    for k in -45..=45 { let x = k as f64 * 2.0; if x.abs() <= 85.0 { println!("  {x:9.4}  {:+.6e}", f(x)) } }
}
