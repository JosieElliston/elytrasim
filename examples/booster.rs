//! The gain phase alone, from a booster's start: a fixed speed pointed along the look direction.
//!
//! The gain markers were fitted to the speeds a climb cycle hands its climb (`v_z` up to ~2.4
//! b/t at the flick). A booster sets `|v|` far above that, so this asks whether they still agree
//! with the optimum there. Two optima, because what the climb is for is a choice:
//!
//! * `height`: maximize the apex height, nothing priced after it. Solved directly -- coordinate
//!   ascent on `y_n`, scanned over `n` -- and cross-checked against `bvp` with a zero apex price.
//! * `cycle`: the climb-to-apex problem with the reference cycle's apex prices
//!   `(mu_y, kappa) = (0.80, 11.53)` (README-myopic.md, polished w = 0), i.e. the climb that
//!   hands over to climb cycles. Solved with `gain_bvp`, copied from `myopic`.
//!
//! Markers are scored two ways: pointwise along the optimum's own trajectory, and flown
//! closed-loop from the start to their own apex.
//!
//! A start aimed below horizontal (positive pitch) is not a gain phase yet: the optimum holds pitch 0 until `v_z`
//! peaks (the cycle's snap, `vz_peaked`), flicks, and only then climbs. The `snap+` markers put
//! that hold in front of a gain rule.
//!
//! Run as `TRIG=mth_lut LIM=89 cargo run --release --example booster -- <mode> ...`, which is what
//! elytra-vario runs; `FLIGHT=algebraic` gives the same answers 3.6x faster. Speeds are b/t, and
//! lists are comma-separated. Every angle is a Minecraft pitch in degrees, negative nose-up;
//! `<pitch>` for a launch is the look pitch the booster reads, so the boost's flight path is
//! `-pitch` above horizontal. Modes:
//!
//!     [speeds] [pitches]           the marker-vs-optimum grid (no mode word)
//!     explore <speed> <pitch> <dir> one explorer cell to <dir>, for tools/plot_booster_explorer.py
//!     cell <speed> <pitch>         the apex optimum, one line: speed,pitch,apex,apex_tick
//!     launch [speeds]              best approach pitch per speed (grid + golden section)
//!     ceiling <speed> <h> [pitches] fewest ticks to reach h blocks up, per approach pitch
//!     race <v_y> <v_z> <n> <held>  most height by tick n, against the markers and a held pitch
//!     flights <v_y> <v_z>          optimum and three markers flown, for tools/plot_booster.py
//!     lag <v_y> <v_z> <pitch0>     apex when following n=20 late or slowly
//!     replay                       stdin "v_y v_z p0 p1 ...": the states, for checking logs
//!     trace <speed> <pitch>        the optimum's pitch per tick beside n=12, n=20 and both laws
//!
//! See README-booster.md for what the results say and how EMC's boosters work.

use elytrasim::opt::*;
use elytrasim::sim::*;
use rayon::prelude::*;

const CYCLE_MU: (f64, f64) = (0.80, 11.53);
/// Exactly -90 is not flyable physics: under libm `cos(-90f32)` is -4.4e-8, so the look vector
/// points backward and the redirect eats ~17% of `v_z` a tick; under `mth_lut` it is exactly 0
/// and the flight is ballistic. Every rule and the optimizer stay inside this. `LIM=89` in the
/// environment matches elytra-vario's sweep; `TRIG=mth_lut` matches vanilla's trig.
static LIM_CELL: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
fn lim() -> f64 { *LIM_CELL.get_or_init(|| std::env::var("LIM").map_or(89.9, |s| s.parse().unwrap())) }

fn dte(s: &State, n: usize) -> f64 {
    let te = s.total_energy();
    argmax_in(|p| run_n(s, p, n).total_energy() - te, -lim(), lim())
}
fn law(vz: f64, k: f64) -> f64 { gain_law_pitch(vz, k).max(-lim()) }

/// `myopic`'s `gain_bvp`, returning the whole schedule instead of its first pitch.
fn gain_bvp(s: &State, w: f64, mu_t: (f64, f64), p_init: f64, cap: usize) -> Vec<f64> {
    let c = (1.0, w);
    let mut ps = vec![p_init; cap];
    let mut n = cap;
    for _ in 0..200 {
        let mut st = vec![s.clone()];
        n = cap;
        let mut up = false;
        for k in 0..cap {
            up |= st[k].vel.y > 0.1;
            if k > 0 && up && !dive_branch_off(st[k].vel, ps[k]) { n = k; break }
            st.push(ticked(&st[k], ps[k]));
        }
        let mut mu = vec![(0.0, 0.0); n + 1];
        mu[n] = mu_t;
        for k in (0..n).rev() {
            let am = mt_vec(&jac(st[k].vel, ps[k]), mu[k + 1]);
            mu[k] = (c.0 + am.0, c.1 + am.1);
        }
        let mut worst: f64 = 0.0;
        for k in 0..n {
            let q = argmax_in_tol(|p| { let f = update_fall_flying_movement(st[k].vel, rot(p));
                                        mu[k + 1].0 * f.y + mu[k + 1].1 * f.z }, -lim(), lim(), 1e-5);
            worst = worst.max((q - ps[k]).abs());
            ps[k] = 0.5 * ps[k] + 0.5 * q;
        }
        if worst < 1e-4 { break }
    }
    ps.truncate(n);
    ps
}

/// Coordinate ascent on `y_n`, the first pass global. The tail is re-simulated per candidate.
fn polish_height(v0: Vec3, init: &[f64], passes: usize) -> (f64, Vec<f64>) {
    let mut ps = init.to_vec();
    let n = ps.len();
    let tail = |s: &State, ps: &[f64], i: usize, p: f64| {
        let mut s = ticked(s, p);
        for k in i + 1..n { s = ticked(&s, ps[k]) }
        s.pos.y
    };
    let mut best = replay_from(v0, &ps)[n].pos.y;
    for pass in 0..passes {
        let before = best;
        let mut st = replay_from(v0, &ps);
        for i in 0..n {
            let (lo, hi) = if pass % 4 == 0 { (-lim(), lim()) } else { ((ps[i] - 6.0).max(-lim()), (ps[i] + 6.0).min(lim())) };
            let q = argmax_in(|p| tail(&st[i], &ps, i, p), lo, hi);
            if tail(&st[i], &ps, i, q) > tail(&st[i], &ps, i, ps[i]) { ps[i] = q }
            st[i + 1] = ticked(&st[i], ps[i]);
        }
        best = replay_from(v0, &ps)[n].pos.y;
        if pass > 4 && best - before < 1e-7 { break }
    }
    (best, ps)
}

/// The apex height, solved: `max_n max y_n`, warm-started from whichever seed climbs highest,
/// cut at its own apex, and scanned over `n` around that apex.
fn solve_height(v0: Vec3, seeds: &[Vec<f64>]) -> (f64, Vec<f64>) {
    let seed = seeds.iter().map(|ps| {
        let st = replay_from(v0, ps);
        let a = (0..st.len()).max_by(|&i, &j| st[i].pos.y.total_cmp(&st[j].pos.y)).unwrap();
        (st[a].pos.y, ps[..a].to_vec())
    }).max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1;
    let n0 = seed.len();
    let lens: Vec<usize> = (n0.saturating_sub(10).max(2)..=n0 + 10).collect();
    lens.par_iter().map(|&n| {
        let mut init = seed.to_vec();
        init.resize(n, 0.0);
        polish_height(v0, &init, 60)
    }).max_by(|a, b| a.0.total_cmp(&b.0)).unwrap()
}

/// A booster's launch: `speed` b/t along a look at Minecraft `pitch` (negative is up).
fn launch(speed: f64, pitch: f64) -> Vec3 {
    Vec3::new(0.0, -speed * pitch.to_radians().sin(), speed * pitch.to_radians().cos())
}

/// The best apex from a booster launch at `speed` b/t and look `pitch`.
fn apex_opt(speed: f64, pitch: f64) -> (f64, Vec<f64>) {
    let v0 = launch(speed, pitch);
    let s0 = State { pos: Vec3::ZERO, vel: v0 };
    let seeds = vec![gain_bvp(&s0, 0.0, (0.0, 0.0), -45.0, 600),
                     fly(v0, &|s: &State| law(s.vel.z, 0.771)).0,
                     fly(v0, &|s: &State| dte(s, 20)).0];
    solve_height(v0, &seeds)
}

/// The fewest ticks in which a launch at (`speed`, `pitch`) can be `h` blocks up, and the
/// schedule that does it: bisection on `n` over `max y_n`, warm-started from the apex climb.
fn soonest(speed: f64, pitch: f64, h: f64) -> Option<(usize, Vec<f64>)> {
    let v0 = launch(speed, pitch);
    let (apex, ps) = apex_opt(speed, pitch);
    if apex < h { return None }
    let st = replay_from(v0, &ps);
    let mut hi = (0..st.len()).find(|&t| st[t].pos.y >= h).unwrap();
    let mut best = ps[..hi].to_vec();
    let mut lo = 1;
    while hi - lo > 1 {
        let n = (lo + hi) / 2;
        let tries = [best[..n].to_vec(), ps[..n].to_vec(), vec![-60.0; n]];
        let r = tries.iter().map(|i| polish_height(v0, i, 60)).max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
        if r.0 >= h { hi = n; best = r.1 } else { lo = n }
    }
    Some((hi, best))
}

/// What each objective pays for a trajectory ending at `s`: apex height, or the cycle's price.
fn value(s: &State, cycle: bool) -> f64 {
    if cycle { s.pos.y + CYCLE_MU.0 * s.vel.y + CYCLE_MU.1 * s.vel.z } else { s.pos.y }
}

/// Fly a marker closed-loop to its own apex: the tick the down-to-forward branch switches on
/// once the climb has started, which is where the gain problem ends. A start with `v_y < 0` has
/// the branch on from tick 0, so "started" means `v_y` has been positive. Returns the pitches
/// and the state there.
fn fly(v0: Vec3, rule: &dyn Fn(&State) -> f64) -> (Vec<f64>, State) {
    let mut s = State { pos: Vec3::ZERO, vel: v0 };
    let mut ps = vec![];
    let mut up = false;
    for _ in 0..600 {
        let p = rule(&s);
        up |= s.vel.y > 0.1;
        if !ps.is_empty() && up && !dive_branch_off(s.vel, p) { break }
        s = ticked(&s, p);
        ps.push(p);
    }
    (ps, s)
}

struct Marker { name: &'static str, f: Box<dyn Fn(&State) -> f64 + Sync> }

fn markers() -> Vec<Marker> {
    let mut m: Vec<Marker> = vec![];
    for n in [1usize, 12, 16, 20, 24, 32] {
        m.push(Marker { name: Box::leak(format!("dTE n={n}").into_boxed_str()), f: Box::new(move |s| dte(s, n)) });
    }
    m.push(Marker { name: "law K=0.771", f: Box::new(|s| law(s.vel.z, 0.771)) });
    m.push(Marker { name: "law K=0.670", f: Box::new(|s| law(s.vel.z, 0.670)) });
    // The cycle's snap in front of the gain phase: sinking, hold pitch 0 while it still raises v_z.
    let snap = |s: &State| s.vel.y < 0.0 && !vz_peaked(s);
    m.push(Marker { name: "snap+n=20", f: Box::new(move |s| if snap(s) { 0.0 } else { dte(s, 20) }) });
    m.push(Marker { name: "snap+K=0.771", f: Box::new(move |s| if snap(s) { 0.0 } else { law(s.vel.z, 0.771) }) });
    m
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if let Ok(m) = std::env::var("TRIG") { set_trig_mode(m.parse().unwrap_or_else(|e| panic!("TRIG: {e:?}"))) }
    if let Ok(m) = std::env::var("FLIGHT") { set_flight_mode(m.parse().unwrap_or_else(|e| panic!("FLIGHT: {e:?}"))) }
    let list = |i: usize, d: &str| -> Vec<f64> {
        a.get(i).map_or(d, |s| s.as_str()).split(',').map(|x| x.parse().unwrap()).collect()
    };
    if a.get(1).map(|s| s.as_str()) == Some("flights") {
        let v0 = Vec3::new(0.0, a[2].parse().unwrap(), a[3].parse().unwrap());
        let ms = markers();
        let mut seeds = vec![gain_bvp(&State { pos: Vec3::ZERO, vel: v0 }, 0.0, (0.0, 0.0), -45.0, 600)];
        seeds.extend(ms.iter().map(|m| fly(v0, &*m.f).0));
        let opt = solve_height(v0, &seeds).1;
        let len = opt.len() + 20;
        let mut runs = vec![("optimum".to_string(), opt)];
        // A fixed window rather than `fly`'s apex test, so a rule that levels off early is shown
        // doing whatever it does next instead of being cut off there.
        for m in &ms {
            if !matches!(m.name, "dTE n=1" | "dTE n=20" | "law K=0.771") { continue }
            let mut s = State { pos: Vec3::ZERO, vel: v0 };
            let ps: Vec<f64> = (0..len).map(|_| { let p = (m.f)(&s); s = ticked(&s, p); p }).collect();
            runs.push((m.name.to_string(), ps));
        }
        println!("rule,t,pitch,y,z,vy,vz");
        for (name, ps) in runs {
            let st = replay_from(v0, &ps);
            for t in 0..=ps.len() {
                let p = ps.get(t).map_or(String::new(), |p| format!("{p:.4}"));
                println!("{name},{t},{p},{:.5},{:.5},{:.5},{:.5}", st[t].pos.y, st[t].pos.z, st[t].vel.y, st[t].vel.z);
            }
        }
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("lag") {
        // A pilot who reacts `d` ticks late, holding the boost's look pitch until then, and turns
        // at most `r` deg/tick toward the rule's pitch plus `bias` degrees nose-up. Apex height.
        let v0 = Vec3::new(0.0, a[2].parse().unwrap(), a[3].parse().unwrap());
        let p0: f64 = a[4].parse().unwrap();
        let apex = |d: usize, r: f64, bias: f64| {
            let (mut s, mut p, mut top) = (State { pos: Vec3::ZERO, vel: v0 }, p0, 0.0_f64);
            for t in 0..200 {
                if t >= d { let q = (dte(&s, 20) - bias).max(-lim()); p += (q - p).clamp(-r, r) }
                s = ticked(&s, p);
                top = top.max(s.pos.y);
            }
            top
        };
        println!("# start pitch {p0}; apex height following n=20, by reaction delay (rows) and turn rate (cols)");
        let rates = [10.0, 20.0, 40.0, 180.0];
        println!("delay  {}", rates.iter().map(|r| format!("{r:>7}/t")).collect::<String>());
        for d in [0usize, 2, 4, 6, 8, 10, 15] {
            println!("{d:>5}  {}", rates.iter().map(|&r| format!("{:>9.1}", apex(d, r, 0.0))).collect::<String>());
        }
        println!("# bias nose-up over the marker, delay 6, 20 deg/t");
        for b in [0.0, 5.0, 10.0, 20.0] { println!("bias {b:>4}: {:.1}", apex(6, 20.0, b)) }
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("replay") {
        // Replay logged pitches from a logged velocity: stdin is "v_y v_z" then one pitch per
        // line; prints y, v_y, v_z after each tick. v_z is the horizontal speed along the look.
        use std::io::Read;
        let mut inp = String::new();
        std::io::stdin().read_to_string(&mut inp).unwrap();
        let mut it = inp.split_whitespace().map(|x| x.parse::<f64>().unwrap());
        let v0 = Vec3::new(0.0, it.next().unwrap(), it.next().unwrap());
        let ps: Vec<f64> = it.collect();
        for (k, s) in replay_from(v0, &ps).iter().enumerate().skip(1) {
            println!("{k} {:.5} {:.5} {:.5}", s.pos.y, s.vel.y, s.vel.z);
        }
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("race") {
        // Reaching a ceiling soon, not a high apex: from (v_y, v_z), the schedule maximizing y at
        // tick n, against the markers and a held pitch, all flown n ticks. Prints y and the
        // horizontal distance at every tick for each.
        let v0 = Vec3::new(0.0, a[2].parse().unwrap(), a[3].parse().unwrap());
        let n: usize = a[4].parse().unwrap();
        let held: f64 = a[5].parse().unwrap();
        let ms = markers();
        let flown = |f: &dyn Fn(&State) -> f64| {
            let mut s = State { pos: Vec3::ZERO, vel: v0 };
            (0..n).map(|_| { let p = f(&s); s = ticked(&s, p); p }).collect::<Vec<f64>>()
        };
        let mut best = (f64::MIN, vec![]);
        for init in [-89.0, -60.0, -30.0, 0.0, held] {
            let r = polish_height(v0, &vec![init; n], 80);
            if r.0 > best.0 { best = r }
        }
        let mut runs = vec![(format!("opt y@{n}"), best.1), (format!("held {held}"), vec![held; n])];
        for m in &ms {
            if matches!(m.name, "dTE n=1" | "dTE n=20" | "law K=0.771") { runs.push((m.name.to_string(), flown(&*m.f))) }
        }
        println!("rule,t,pitch,y,z");
        for (name, ps) in runs {
            let st = replay_from(v0, &ps);
            for t in 0..=n {
                println!("{name},{t},{},{:.4},{:.4}", ps.get(t).map_or(String::new(), |p| format!("{p:.3}")), st[t].pos.y, st[t].pos.z);
            }
        }
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("launch") {
        // The best launch pitch for the apex at each booster speed: a 2.5-degree grid, then a
        // golden-section refine around its best. CSV rows: the grid, then the refined optimum.
        let speeds = list(2, "1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5");
        println!("kind,speed,pitch,apex,apex_tick");
        let rows: Vec<String> = speeds.par_iter().flat_map(|&v| {
            let grid: Vec<(f64, f64, usize)> = (0..=40).into_par_iter().map(|i| {
                let g = 20.0 - 2.5 * i as f64;
                let (h, ps) = apex_opt(v, g);
                (g, h, ps.len())
            }).collect();
            let &(g0, _, _) = grid.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            let (mut lo, mut hi) = (g0 - 2.5, g0 + 2.5);
            let f = |g: f64| apex_opt(v, g).0;
            for _ in 0..12 {
                let (m1, m2) = (lo + 0.382 * (hi - lo), hi - 0.382 * (hi - lo));
                if f(m1) < f(m2) { lo = m1 } else { hi = m2 }
            }
            let g = 0.5 * (lo + hi);
            let (h, ps) = apex_opt(v, g);
            let mut out: Vec<String> = grid.iter().map(|(g, h, n)| format!("grid,{v},{g},{h:.4},{n}")).collect();
            out.push(format!("best,{v},{g:.3},{h:.4},{}", ps.len()));
            out
        }).collect();
        for r in rows { println!("{r}") }
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("explore") {
        // One explorer cell, written to <dir>/v<speed>_p<pitch>.csv: the apex optimum from a
        // launch at (speed, pitch), and n=20 and the gain law flown from the same launch over
        // the optimum's length plus 30 ticks, so each is seen past its own apex.
        let (v, g): (f64, f64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
        let dir = &a[4];
        let v0 = launch(v, g);
        let (h, opt) = apex_opt(v, g);
        let len = opt.len() + 30;
        let flown = |f: &dyn Fn(&State) -> f64| {
            let mut s = State { pos: Vec3::ZERO, vel: v0 };
            (0..len).map(|_| { let p = f(&s); s = ticked(&s, p); p }).collect::<Vec<f64>>()
        };
        let runs = [("optimum", opt), ("dTE n=20", flown(&|s: &State| dte(s, 20))),
                    ("law K=0.771", flown(&|s: &State| law(s.vel.z, 0.771)))];
        let mut out = format!("# speed {v} pitch {g} apex {h:.5} trig {:?} flight {:?} lim {}\nrule,t,pitch,y,z,vy,vz\n",
                              trig_mode(), flight_mode(), lim());
        for (name, ps) in &runs {
            for (t, s) in replay_from(v0, ps).iter().enumerate() {
                let p = ps.get(t).map_or(String::new(), |p| format!("{p:.4}"));
                out += &format!("{name},{t},{p},{:.5},{:.5},{:.6},{:.6}\n", s.pos.y, s.pos.z, s.vel.y, s.vel.z);
            }
        }
        std::fs::write(format!("{dir}/v{v:.1}_p{g}.csv"), out).unwrap();
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("cell") {
        // One (speed, launch pitch) cell: the apex optimum. For timing and for the explorer.
        let (v, g): (f64, f64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
        let (h, ps) = apex_opt(v, g);
        println!("{v},{g},{h:.4},{}", ps.len());
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("ceiling") {
        // The launch pitch that reaches `h` blocks up soonest, at one booster speed.
        let v: f64 = a[2].parse().unwrap();
        let h: f64 = a[3].parse().unwrap();
        let pitches = list(4, "-15,-17.5,-20,-22.5,-25,-27.5,-30,-32.5,-35");
        println!("pitch,ticks,z,pitches");
        let rows: Vec<String> = pitches.par_iter().map(|&g| match soonest(v, g, h) {
            None => format!("{g},,,"),
            Some((n, ps)) => {
                let st = replay_from(launch(v, g), &ps);
                format!("{g},{n},{:.3},{}", st[n].pos.z, ps.iter().map(|p| format!("{p:.2}")).collect::<Vec<_>>().join(" "))
            }
        }).collect();
        for r in rows { println!("{r}") }
        return;
    }
    if a.get(1).map(|s| s.as_str()) == Some("trace") {
        let (v, g): (f64, f64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
        let v0 = launch(v, g);
        let s0 = State { pos: Vec3::ZERO, vel: v0 };
        let bvp0 = gain_bvp(&s0, 0.0, (0.0, 0.0), -45.0, 600);
        let ms = markers();
        let mut seeds = vec![bvp0];
        seeds.extend(ms.iter().map(|m| fly(v0, &*m.f).0));
        let (h, ph) = solve_height(v0, &seeds);
        let st = replay_from(v0, &ph);
        println!("# apex {h:.3}; tick opt n12 n20 law.771 law.670 vy vz");
        for t in 0..ph.len() {
            let s = &st[t];
            println!("{t:3} {:7.2} {:7.2} {:7.2} {:7.2} {:7.2}  {:.3} {:.3}", ph[t], dte(s, 12), dte(s, 20), law(s.vel.z, 0.771), law(s.vel.z, 0.670), s.vel.y, s.vel.z);
        }
        return;
    }
    let speeds = list(1, "1.5,2,2.5,3,3.5,4");
    let pitches = list(2, "0,-10,-20,-30");
    let ms = markers();

    println!("# speed b/t, pitch deg = the look the booster reads (Minecraft, negative is up)");
    println!("# height: pts = pointwise RMS deg on the optimum's climb (first 20 ticks | all); flown = apex height lost, blocks");
    println!("# cycle:  the same against the climb-to-apex optimum at the cycle's apex prices; flown = J lost, blocks");
    for &v in &speeds {
        for &g in &pitches {
            let v0 = launch(v, g);
            let s0 = State { pos: Vec3::ZERO, vel: v0 };

            let bvp0 = gain_bvp(&s0, 0.0, (0.0, 0.0), -45.0, 600);
            let y_bvp0 = replay_from(v0, &bvp0).last().unwrap().pos.y;
            let mut seeds = vec![bvp0.clone()];
            seeds.extend(ms.par_iter().map(|m| fly(v0, &*m.f).0).collect::<Vec<_>>());
            let (h, ph) = solve_height(v0, &seeds);
            let sth = replay_from(v0, &ph);
            let apex_h = (0..=ph.len()).max_by(|&i, &j| sth[i].pos.y.total_cmp(&sth[j].pos.y)).unwrap();

            let pc = gain_bvp(&s0, 0.0, CYCLE_MU, -45.0, 600);
            let stc = replay_from(v0, &pc);
            let jc = value(stc.last().unwrap(), true);

            println!("\n== |v| {v:.2} b/t ({:.1} b/s), pitch {g:.0}: v_y {:.3} v_z {:.3}", v * 20.0, v0.y, v0.z);
            println!("   height optimum: apex {h:.3} b at tick {apex_h} of {}; bvp(mu=0) {y_bvp0:.3} b in {} ticks", ph.len(), bvp0.len());
            println!("     pitches[0..10]: {}", ph.iter().take(10).map(|p| format!("{p:.1}")).collect::<Vec<_>>().join(" "));
            println!("   cycle optimum:  J {jc:.3}, apex y {:.3} v_z {:.3} at tick {}", stc.last().unwrap().pos.y, stc.last().unwrap().vel.z, pc.len());
            println!("     pitches[0..10]: {}", pc.iter().take(10).map(|p| format!("{p:.1}")).collect::<Vec<_>>().join(" "));
            println!("   {:<13} {:>7} {:>7} {:>9} | {:>7} {:>7} {:>9}   first pitches (height opt)", "marker", "pts20", "ptsall", "flown", "pts20", "ptsall", "flown");

            let rows: Vec<String> = ms.par_iter().map(|m| {
                let rms = |ps: &[f64], st: &[State], k: usize| {
                    let k = k.min(ps.len());
                    ((0..k).map(|t| ((m.f)(&st[t]) - ps[t]).powi(2)).sum::<f64>() / k as f64).sqrt()
                };
                let n_h = apex_h.min(ph.len());
                let (fp, fs) = fly(v0, &*m.f);
                let hy = { let st = replay_from(v0, &fp); st.iter().map(|s| s.pos.y).fold(f64::MIN, f64::max) };
                let first: Vec<String> = (0..5).map(|t| format!("{:.1}", (m.f)(&sth[t]))).collect();
                format!("   {:<13} {:>7.2} {:>7.2} {:>9.3} | {:>7.2} {:>7.2} {:>9.3}   {}",
                        m.name, rms(&ph[..n_h], &sth, 20), rms(&ph[..n_h], &sth, n_h), h - hy,
                        rms(&pc, &stc, 20), rms(&pc, &stc, pc.len()), jc - value(&fs, true), first.join(" "))
            }).collect();
            for r in rows { println!("{r}") }
        }
    }
}
