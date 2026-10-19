//! Tied-weights polish: one lap's pitches, scored by flying that same lap `k` times back to back.
//!
//! The objective is TE after `k` laps from `v0`, terminal velocity free. One-lap polish has a
//! horizon effect at the end of the lap (it spends the next lap's setup on height); unrolling the
//! schedule over two laps removes it from the first lap but doubles the coordinates. Tying the
//! laps keeps one lap's coordinates and dilutes the end effect to the last of `k` copies, so as
//! `k` grows the answer tends to the repeatable cycle's.
//!
//! usage: tied <pitches> --vy <vy> --vz <vz> --k <k> [--limit 89] [--passes 2000] [--out file]
use elytrasim::opt::*;
use elytrasim::sim::*;

fn arg<T: std::str::FromStr>(a: &[String], name: &str, def: Option<T>) -> T {
    a.iter().position(|x| x == name).map(|i| a[i + 1].parse().ok().unwrap())
        .or(def).unwrap_or_else(|| panic!("{name} is required"))
}

/// States of `k` laps of `trig` from `s0`: `k * n + 1` of them.
fn rollout(s0: &State, trig: &[PitchTrig], k: usize) -> Vec<State> {
    let mut v = Vec::with_capacity(k * trig.len() + 1);
    v.push(s0.clone());
    for _ in 0..k { for &p in trig { let s = v.last().unwrap().ticked_cached(p); v.push(s) } }
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pitches = read_pitches(&a[1]);
    let (vy, vz): (f64, f64) = (arg(&a, "--vy", None), arg(&a, "--vz", None));
    let k: usize = arg(&a, "--k", None);
    let limit: f64 = arg(&a, "--limit", Some(89.0));
    let max_passes: usize = arg(&a, "--passes", Some(2000));
    let out: String = arg(&a, "--out", Some(String::new()));
    let n = pitches.len();
    let s0 = State { pos: Vec3::ZERO, vel: Vec3::new(0.0, vy, vz) };
    let te0 = s0.total_energy();
    let mut trig: Vec<PitchTrig> = pitches.iter().map(|&p| PitchTrig::new(p as f32)).collect();
    let mut states = rollout(&s0, &trig, k);
    let score0 = states.last().unwrap().total_energy() - te0;
    let mut quiet = 0;
    let t0 = std::time::Instant::now();
    for pass in 0..max_passes {
        let before = states.last().unwrap().total_energy();
        let global = pass % 4 == 0;
        for t in 0..n {
            // Flying from the first copy of tick t onward with the candidate in every later copy.
            let st = states[t].clone();
            let tr = &trig;
            let score = |x: f64| {
                let px = PitchTrig::new(x as f32);
                let mut s = st.ticked_cached(px);
                for i in t + 1..k * n { s = s.ticked_cached(if i % n == t { px } else { tr[i % n] }) }
                s.total_energy()
            };
            let cur = pitches[t];
            let (lo, hi, step) = if global { (-limit, limit, 2.0) }
                                 else { ((cur - 8.0).max(-limit), (cur + 8.0).min(limit), 0.5) };
            let (np, j) = coordinate_argmax(&score, cur, lo, hi, step, 1e-4);
            let np = np as f32 as f64;
            if j > score(cur) {
                pitches[t] = np;
                trig[t] = PitchTrig::new(np as f32);
                states = rollout(&s0, &trig, k);
            }
        }
        let gain = states.last().unwrap().total_energy() - before;
        if pass % 10 == 0 { eprintln!("pass {pass}: dTE over {k} laps {:.6} (+{gain:.2e})", states.last().unwrap().total_energy() - te0) }
        // Two passes in a row with nothing, one of them global.
        if gain < 1e-10 { quiet += 1; if quiet >= 4 { break } } else { quiet = 0 }
    }
    let laps: Vec<f64> = (0..k).map(|i| states[(i + 1) * n].total_energy() - states[i * n].total_energy()).collect();
    let b = &states[n].vel;
    println!("k {k}  n {n}  seed {score0:.6}  total {:.6}  per lap {:?}  lap-1 end v ({:.6}, {:.6})  {:.1}s",
             states.last().unwrap().total_energy() - te0, laps.iter().map(|x| (x * 1e6).round() / 1e6).collect::<Vec<_>>(),
             b.y, b.z, t0.elapsed().as_secs_f64());
    if !out.is_empty() {
        let body: Vec<String> = pitches.iter().map(|p| format!("{p}")).collect();
        std::fs::write(&out, format!("# tied k {k} v0 {vy} {vz}\n{}\n", body.join("\n"))).unwrap();
    }
}
