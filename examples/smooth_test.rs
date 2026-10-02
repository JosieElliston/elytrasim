//! Is a zigzag optimal? Replace pitches `a..b` of a schedule by their running mean (3 and 5
//! ticks) and by the zigzag's own 2-tick pair means, and report J (lambda 0 here: TE at the end)
//! and the least clearance before and after.
//!
//! cargo run --release --example smooth_test -- <pitches> <y0> <a> <b>
use elytrasim::opt::*;
use elytrasim::sim::*;

fn main() {
    set_trig_mode(TrigMode::MthLut);
    set_flight_mode(FlightMode::Reference);
    let a: Vec<String> = std::env::args().collect();
    let p = read_pitches(&a[1]);
    let y0: f64 = a[2].parse().unwrap();
    let (lo, hi): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let obj = Objective { v0: Vec3::ZERO, n: p.len(), lambda: 0.0 };
    let floor = Floor { depth: y0, ..Default::default() };
    let show = |name: &str, q: &[f64]| {
        let st = obj.replay(q);
        println!("{name:>10}  J {:+.6}  clearance {:+.5}", obj.eval(q), floor.clearance(&st));
    };
    show("as is", &p);
    for k in [3usize, 5] {
        let mut q = p.clone();
        for t in lo..hi { let w = k / 2; q[t] = p[t - w..=t + w].iter().sum::<f64>() / k as f64 }
        show(&format!("mean {k}"), &q);
        if let Some(o) = a.get(5) {
            let s: Vec<String> = q.iter().map(|x| format!("{}", *x as f32)).collect();
            std::fs::write(format!("{o}_mean{k}.pitches"), s.join("\n") + "\n").unwrap();
        }
    }
    // Same total `sin^2` dump per pair, spread evenly: each pair set to the pitch whose sin^2 is
    // the pair's mean sin^2.
    let mut q = p.clone();
    let mut t = lo;
    while t + 1 < hi {
        let s2 = (p[t].to_radians().sin().powi(2) + p[t + 1].to_radians().sin().powi(2)) / 2.0;
        let m = s2.sqrt().asin().to_degrees() * p[t].signum();
        q[t] = m; q[t + 1] = m; t += 2;
    }
    show("sin2 pairs", &q);
}
