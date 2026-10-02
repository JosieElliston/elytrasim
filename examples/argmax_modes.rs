//! How many modes do the pitch objectives have? Counts local maxima of a 0.05-degree sweep whose
//! prominence clears a threshold, per objective, over the reference cycle's states (and, for the
//! optimizer's per-tick objective, over every tick of a corpus profile). Also: at a 2-degree
//! coarse sweep, how many sampled peaks there are, and what rank the true winner's cell has.
//! `cargo run --release --example argmax_modes [lut]`. The per-tick part reads
//! `runs/gainlaw/n0300_lamP0.cyc`, which lives in the untracked `runs/`; any profile will do.
use elytrasim::opt::*;
use elytrasim::replay_pitches::REPLAY_PITCHES_300;
use elytrasim::sim::*;
use rayon::prelude::*;

/// Prominence of each local max of `vs`: its height above the higher of the two lowest points
/// separating it from higher ground (or from the ends).
fn prominences(vs: &[f64]) -> Vec<(usize, f64)> {
    let n = vs.len();
    let mut out = vec![];
    for i in 0..n {
        let up = i == 0 || vs[i] > vs[i - 1];
        let down = i + 1 == n || vs[i] >= vs[i + 1];
        if !(up && down) { continue }
        let (mut ml, mut j) = (vs[i], i);
        while j > 0 { j -= 1; if vs[j] > vs[i] { break } ml = ml.min(vs[j]) }
        let (mut mr, mut j) = (vs[i], i);
        while j + 1 < n { j += 1; if vs[j] > vs[i] { break } mr = mr.min(vs[j]) }
        out.push((i, vs[i] - ml.max(mr)));
    }
    out
}

fn hist(name: &str, counts: &[usize]) {
    let mut h = [0usize; 8];
    for &c in counts { h[c.min(7)] += 1 }
    print!("{name:<44}");
    for (k, v) in h.iter().enumerate() { if *v > 0 { print!(" {k}{}:{v}", if k == 7 { "+" } else { "" }) } }
    println!();
}

fn study(name: &str, fs: &[Box<dyn Fn(f64) -> f64 + Sync + '_>]) {
    let fine: Vec<f64> = (0..=3600).map(|i| -90.0 + 0.05 * i as f64).collect();
    let coarse: Vec<f64> = (0..=90).map(|i| -90.0 + 2.0 * i as f64).collect();
    let rows: Vec<(Vec<usize>, usize, usize)> = fs.par_iter().map(|f| {
        let vs: Vec<f64> = fine.iter().map(|&p| f(p)).collect();
        let pr = prominences(&vs);
        let modes = [1e-7, 1e-5, 1e-3].iter().map(|&e| pr.iter().filter(|x| x.1 >= e).count()).collect();
        // The winner, and the rank (by sample value) of the coarse peak whose cell holds it.
        let win = fine[(0..vs.len()).max_by(|&a, &b| vs[a].total_cmp(&vs[b])).unwrap()];
        let cv: Vec<f64> = coarse.iter().map(|&p| f(p)).collect();
        let mut peaks: Vec<usize> = (0..cv.len())
            .filter(|&i| (i == 0 || cv[i] >= cv[i - 1]) && (i + 1 == cv.len() || cv[i] >= cv[i + 1])).collect();
        peaks.sort_by(|&a, &b| cv[b].total_cmp(&cv[a]));
        let rank = peaks.iter().position(|&i| (coarse[i] - win).abs() <= 2.0).map_or(99, |r| r + 1);
        (modes, peaks.len(), rank)
    }).collect();
    println!("\n== {name} ({} cases)", rows.len());
    for (k, e) in ["1e-7", "1e-5", "1e-3"].iter().enumerate() {
        hist(&format!("modes, prominence >= {e}"), &rows.iter().map(|r| r.0[k]).collect::<Vec<_>>());
    }
    hist("peaks among the 2-degree samples", &rows.iter().map(|r| r.1).collect::<Vec<_>>());
    hist("rank of the winner's sampled peak (7+ = none)", &rows.iter().map(|r| r.2).collect::<Vec<_>>());
}

fn main() {
    if std::env::args().any(|a| a == "lut") { set_trig_mode(TrigMode::MthLut) }
    println!("trig = {}", trig_mode());
    let ps: Vec<f64> = REPLAY_PITCHES_300.iter().map(|&p| p as f64).collect();
    let st = replay(&ps);
    let mut states: Vec<State> = st[..300].to_vec();
    for i in 0..15 { for j in 0..15 {
        states.push(State { pos: Vec3::ZERO, vel: Vec3::new(0.0, -3.0 + 4.5 * i as f64 / 14.0, 3.5 * j as f64 / 14.0) });
    }}
    for n in [1usize, 4, 10, 20, 48] {
        let fs: Vec<Box<dyn Fn(f64) -> f64 + Sync>> = states.iter().map(|s| {
            Box::new(move |p: f64| run_n(s, p, n).total_energy() - s.total_energy()) as Box<dyn Fn(f64) -> f64 + Sync>
        }).collect();
        study(&format!("dTE held {n} ticks"), &fs);
    }
    let fs: Vec<Box<dyn Fn(f64) -> f64 + Sync>> = states.iter().map(|s| {
        Box::new(move |p: f64| -(gamma(ticked(s, p).vel) - gamma(s.vel)).abs()) as Box<dyn Fn(f64) -> f64 + Sync>
    }).collect();
    study("-|gamma' - gamma| (hold the velocity's direction)", &fs);

    // The optimizer's own per-tick objective: J with one tick's pitch replaced.
    let path = "runs/gainlaw/n0300_lamP0.cyc";
    let prof = Profile::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let pp = read_pitches(path);
    let obj = prof.obj;
    let fs: Vec<Box<dyn Fn(f64) -> f64 + Sync>> = (0..pp.len()).map(|t| {
        let (pp, obj) = (pp.clone(), obj);
        Box::new(move |p: f64| { let mut q = pp.clone(); q[t] = p; obj.eval(&q) }) as Box<dyn Fn(f64) -> f64 + Sync>
    }).collect();
    study(&format!("polish: J with one tick moved ({path})"), &fs);
}
