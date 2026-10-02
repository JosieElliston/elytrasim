//! Finite-difference schedule ascent. Run with --help for options.
use elytrasim::opt::*;
use elytrasim::sim::*;
use rayon::prelude::*;
use std::{collections::HashMap, time::Instant};

const LIMIT: f64 = 90.0 - 1e-4;

struct Args(HashMap<String, String>);
impl Args {
    fn new() -> Self {
        let mut args = std::env::args().skip(1);
        let mut values = HashMap::new();
        while let Some(key) = args.next() {
            if key == "--help" {
                println!(
                    "gd [--n 300 --lambda 0 --vy 0 --vz 0 --init reference --passes 2000\n    --lr 0.5 --opt adam|sgd --fd-step 0.05 --jitter 0 --draws 8 --seed 1\n    --fixed-draws --l2-curv 0 --l1-curv 0 --decay 1 --report 25 --out FILE]\nJ includes curvature penalties; raw_J is the unpenalized draw average.\nReports use fixed pass-0 draws for comparable scores; gradients use pass draws.\nseconds is cumulative optimization/report time, excluding initialization and certification."
                );
                std::process::exit(0);
            }
            assert!(
                [
                    "--n",
                    "--lambda",
                    "--vy",
                    "--vz",
                    "--init",
                    "--passes",
                    "--lr",
                    "--opt",
                    "--fd-step",
                    "--jitter",
                    "--draws",
                    "--seed",
                    "--fixed-draws",
                    "--l2-curv",
                    "--l1-curv",
                    "--decay",
                    "--report",
                    "--out"
                ]
                .contains(&key.as_str()),
                "unknown option {key}"
            );
            let value = if key == "--fixed-draws" {
                "true".into()
            } else {
                args.next()
                    .unwrap_or_else(|| panic!("missing value for {key}"))
            };
            assert!(
                values.insert(key.clone(), value).is_none(),
                "duplicate option {key}"
            );
        }
        Self(values)
    }
    fn get<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.0.get(key).map(String::as_str).unwrap_or(default)
    }
    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.0.get(key).map_or(default, |v| {
            v.parse().unwrap_or_else(|_| panic!("invalid {key}: {v}"))
        })
    }
}

// The two perturbations share their entire prefix and every draw within a pass.
// Central differences deliberately remain symmetric near the box boundaries;
// only optimizer updates are projected into the pitch box.
fn gradient(obj: &Objective, p: &[f64], draws: &[Vec3], h: f64) -> Vec<f64> {
    let states: Vec<_> = draws.iter().map(|d| replay_from(obj.v0 + *d, p)).collect();
    (0..p.len())
        .into_par_iter()
        .map(|t| {
            states
                .iter()
                .map(|st| {
                    let mut plus = ticked(&st[t], p[t] + h);
                    let mut minus = ticked(&st[t], p[t] - h);
                    for &q in &p[t + 1..] {
                        plus = ticked(&plus, q);
                        minus = ticked(&minus, q);
                    }
                    (obj.j(&plus) - obj.j(&minus)) / (2.0 * h)
                })
                .sum::<f64>()
                / draws.len() as f64
        })
        .collect()
}

fn penalty(p: &[f64], l2: f64, l1: f64, g: &mut [f64]) -> f64 {
    let mut cost = 0.0;
    for (i, w) in p.windows(3).enumerate() {
        let d = w[2] - 2.0 * w[1] + w[0];
        cost += l2 * d * d + l1 * d.abs();
        let sign = if d > 0.0 {
            1.0
        } else if d < 0.0 {
            -1.0
        } else {
            0.0
        };
        let a = -2.0 * l2 * d - l1 * sign;
        g[i] += a;
        g[i + 1] -= 2.0 * a;
        g[i + 2] += a;
    }
    cost
}

// Reconstruct before clamping: clamping intermediate values would alter the
// requested second differences. The final box projection can reintroduce curvature.
fn decay_curvature(p: &mut [f64], decay: f64) {
    if decay == 1.0 {
        return;
    }
    let d2: Vec<_> = p
        .windows(3)
        .map(|w| decay * (w[2] - 2.0 * w[1] + w[0]))
        .collect();
    for (i, d) in d2.into_iter().enumerate() {
        p[i + 2] = d + 2.0 * p[i + 1] - p[i];
    }
}

fn project(p: &mut [f64]) {
    for x in p {
        assert!(x.is_finite(), "nonfinite optimizer step");
        // Canonical f32 controls, also used for all scores and saved profiles.
        let bound = f32::from_bits((LIMIT as f32).to_bits() - 1) as f64;
        *x = x.clamp(-bound, bound) as f32 as f64;
    }
}

#[derive(Default)]
struct Moments {
    m: Vec<f64>,
    v: Vec<f64>,
    b1: f64,
    b2: f64,
}
impl Moments {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            b1: 1.0,
            b2: 1.0,
        }
    }
    fn step(&mut self, p: &mut [f64], g: &[f64], lr: f64, adam: bool) {
        self.b1 *= 0.9;
        self.b2 *= 0.999;
        for i in 0..p.len() {
            let update = if adam {
                self.m[i] = 0.9 * self.m[i] + 0.1 * g[i];
                self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] * g[i];
                (self.m[i] / (1.0 - self.b1)) / ((self.v[i] / (1.0 - self.b2)).sqrt() + 1e-8)
            } else {
                g[i]
            };
            p[i] += lr * update;
        }
    }
}

fn report(
    obj: &Objective,
    p: &[f64],
    draws: &[Vec3],
    h: f64,
    l2: f64,
    l1: f64,
    pass: usize,
    start: Instant,
    residual: Option<f64>,
) {
    let mut g = gradient(obj, p, draws, h);
    let cost = penalty(p, l2, l1, &mut g);
    let raw = obj.eval_jittered(p, draws);
    let j = raw - cost;
    let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut d2: Vec<_> = p
        .windows(3)
        .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
        .collect();
    d2.sort_by(f64::total_cmp);
    let max = d2.last().copied().unwrap_or(0.0);
    let p99 = if d2.is_empty() {
        0.0
    } else {
        d2[(0.99 * d2.len() as f64).ceil() as usize - 1]
    };
    println!(
        "{pass},{j:.9},{:.9},{norm:.9},{:.6},{:.6},{max:.6},{p99:.6},{:.6},{raw:.9},{cost:.9},{}",
        j - obj.eval_jittered(&[], draws),
        total_variation(p),
        lag1(p),
        start.elapsed().as_secs_f64(),
        residual.map_or(String::new(), |r| format!("{r:.12e}"))
    );
}

fn main() {
    let a = Args::new();
    let obj = Objective {
        n: a.num("--n", 300),
        lambda: a.num("--lambda", 0.0),
        v0: Vec3::new(0.0, a.num("--vy", 0.0), a.num("--vz", 0.0)),
    };
    let passes = a.num("--passes", 2000usize);
    let lr: f64 = a.num("--lr", 0.5);
    let h: f64 = a.num("--fd-step", 0.05);
    let l2: f64 = a.num("--l2-curv", 0.0);
    let l1: f64 = a.num("--l1-curv", 0.0);
    let decay: f64 = a.num("--decay", 1.0);
    let every = a.num("--report", 25usize);
    let jitter = Jitter {
        sigma: a.num("--jitter", 0.0),
        draws: a.num("--draws", 8),
        seed: a.num("--seed", 1),
        resample: !a.0.contains_key("--fixed-draws"),
    };
    assert!(
        obj.n > 0 && every > 0 && jitter.draws > 0,
        "n, report, and draws must be positive"
    );
    assert!(
        [
            obj.lambda,
            obj.v0.y,
            obj.v0.z,
            lr,
            h,
            l2,
            l1,
            decay,
            jitter.sigma
        ]
        .iter()
        .all(|x| x.is_finite()),
        "parameters must be finite"
    );
    assert!(
        lr > 0.0
            && h > 0.0
            && l2 >= 0.0
            && l1 >= 0.0
            && jitter.sigma >= 0.0
            && (0.0..=1.0).contains(&decay),
        "invalid learning rate, finite-difference step, penalty, jitter, or decay"
    );
    let adam = match a.get("--opt", "adam") {
        "adam" => true,
        "sgd" => false,
        v => panic!("unknown optimizer {v}"),
    };
    let mut p = match a.get("--init", "reference") {
        "policy" => seed_from_policy(&obj),
        "reference" => seed_from_reference(obj.n),
        path => {
            let seed = read_pitches(path);
            assert!(
                !seed.is_empty() && seed.iter().all(|p| p.is_finite()),
                "empty or nonfinite seed"
            );
            stretch(&seed, obj.n)
        }
    };
    project(&mut p);
    let mut moments = Moments::new(p.len());
    let fixed = jitter.draws_at_0();
    println!(
        "# command: {}",
        std::env::args().collect::<Vec<_>>().join(" ")
    );
    println!(
        "# trig={} rayon_threads={} commit={}",
        trig_mode(),
        rayon::current_num_threads(),
        commit_hash()
    );
    println!(
        "# J=raw_J-penalty; reporting/grad_norm use fixed pass-0 draws; training uses draws_at(pass), starting at pass=0"
    );
    println!(
        "# p99 uses nearest rank; seconds excludes initialization/certification; residual certifies unpenalized J with pass-0 draws"
    );
    println!("pass,J,dJ,grad_norm,TV,lag1,max_abs_d2p,p99_abs_d2p,seconds,raw_J,penalty,residual");
    let start = Instant::now();
    report(&obj, &p, &fixed, h, l2, l1, 0, start, None);
    let mut step_seconds = 0.0;
    for pass in 0..passes {
        let timer = Instant::now();
        let mut g = gradient(&obj, &p, &jitter.draws_at(pass as u64), h);
        penalty(&p, l2, l1, &mut g);
        moments.step(&mut p, &g, lr, adam);
        project(&mut p);
        decay_curvature(&mut p, decay);
        project(&mut p);
        step_seconds += timer.elapsed().as_secs_f64();
        if (pass + 1) % every == 0 && pass + 1 != passes {
            report(&obj, &p, &fixed, h, l2, l1, pass + 1, start, None);
        }
    }
    let cert_start = Instant::now();
    let residual = certify(&obj, &p, CERTIFY_STEP, jitter);
    let cert_seconds = cert_start.elapsed().as_secs_f64();
    // Shift the origin so the final row has the same timing definition as earlier rows.
    report(
        &obj,
        &p,
        &fixed,
        h,
        l2,
        l1,
        passes,
        start + cert_start.elapsed(),
        Some(residual),
    );
    println!(
        "# seconds_per_pass={:.9} optimization_seconds={step_seconds:.6} certification_seconds={cert_seconds:.6}",
        step_seconds / passes.max(1) as f64
    );
    if let Some(path) = a.0.get("--out") {
        let profile = Profile {
            obj,
            trig: trig_mode(),
            flight: flight_mode(),
            jitter,
            // The curvature penalties gd applies live in its own flags, not in `Rough`; a
            // profile it writes is a stationary point of `J` plus those, so the header records
            // no roughness price and `verify --optimality` will certify it against plain `J`.
            rough: Rough::default(),
            commit: commit_hash().into(),
            pitches: p,
            residual,
            passes,
        };
        if let Some(dir) = std::path::Path::new(path)
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
        {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(path, profile.to_string()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_gradient_matches_full_replay_with_jitter() {
        let obj = Objective {
            n: 12,
            lambda: 0.7,
            v0: V0,
        };
        let p: Vec<_> = (0..12).map(|i| -35.0 + i as f64 * 7.3).collect();
        for resample in [false, true] {
            let jit = Jitter {
                sigma: 0.1,
                draws: 3,
                seed: 3,
                resample,
            };
            for pass in [0, 4] {
                let draws = jit.draws_at(pass);
                let g = gradient(&obj, &p, &draws, 0.05);
                for t in 0..p.len() {
                    let mut q = p.clone();
                    q[t] += 0.05;
                    let plus = obj.eval_jittered(&q, &draws);
                    q[t] = p[t] - 0.05;
                    let minus = obj.eval_jittered(&q, &draws);
                    assert!((g[t] - (plus - minus) / 0.1).abs() < 1e-12);
                }
            }
        }
    }
    #[test]
    fn curvature_gradient_matches_finite_difference() {
        let p = vec![1.0, 4.0, -2.0, 3.0, 8.0, 0.0];
        for (l2, l1) in [(0.003, 0.0), (0.0, 0.02), (0.003, 0.02)] {
            let mut g = vec![0.0; p.len()];
            penalty(&p, l2, l1, &mut g);
            for t in 0..p.len() {
                let mut q = p.clone();
                q[t] += 1e-5;
                let plus = -penalty(&q, l2, l1, &mut vec![0.0; p.len()]);
                q[t] = p[t] - 1e-5;
                let minus = -penalty(&q, l2, l1, &mut vec![0.0; p.len()]);
                assert!((g[t] - (plus - minus) / 2e-5).abs() < 1e-9);
            }
        }
    }
    #[test]
    fn decay_preserves_anchors_and_scales_second_differences() {
        let p = vec![1.0, 2.0, -4.0, 7.0, 3.0];
        for d in [0.0, 0.98, 1.0] {
            let mut q = p.clone();
            decay_curvature(&mut q, d);
            assert_eq!(&q[..2], &p[..2]);
            for (a, b) in p.windows(3).zip(q.windows(3)) {
                assert!((b[2] - 2.0 * b[1] + b[0] - d * (a[2] - 2.0 * a[1] + a[0])).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn adam_bias_correction_and_sgd() {
        let mut p = vec![0.0, 0.0];
        let mut m = Moments::new(2);
        for k in 1..=3 {
            m.step(&mut p, &[2.0, -4.0], 0.5, true);
            assert!((p[0] - k as f64 * 0.5).abs() < 1e-8);
            assert!((p[1] + k as f64 * 0.5).abs() < 1e-8);
        }
        m.step(&mut p, &[2.0, -4.0], 200.0, false);
        project(&mut p);
        assert!(p.iter().all(|p| p.abs() <= LIMIT));
    }
}
