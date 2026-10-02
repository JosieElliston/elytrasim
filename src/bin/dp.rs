//! Finite-horizon dynamic programming for the unregularized elytra objective.
//!
//! Position is deliberately absent from the state. With yaw fixed at zero and `v_x = 0`, the
//! movement update depends only on `(v_y, v_z)`, while position is the sum of post-tick
//! velocities. Thus each tick pays `v_y' + w v_z'` and the terminal value is kinetic energy.

use elytrasim::opt::{Objective, replay_from, w_of_lambda};
use elytrasim::sim::{
    FlightMode, GRAVITY, Rot, TrigMode, Vec3, flight_mode, set_flight_mode, set_trig_mode,
    trig_mode,
};
use clap::Parser;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_ALLOCATION: u64 = 6 * 1024 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Transition {
    y: f32,
    z: f32,
    reward: f32,
}

#[derive(Clone, Copy)]
struct Grid {
    n: usize,
    vy_lo: f64,
    vy_hi: f64,
    vz_lo: f64,
    vz_hi: f64,
}

impl Grid {
    fn states(self) -> usize {
        self.n.checked_mul(self.n).expect("grid size overflow")
    }

    fn vy(self, iy: usize) -> f64 {
        self.vy_lo + iy as f64 * (self.vy_hi - self.vy_lo) / (self.n - 1) as f64
    }

    fn vz(self, iz: usize) -> f64 {
        self.vz_lo + iz as f64 * (self.vz_hi - self.vz_lo) / (self.n - 1) as f64
    }

    fn frac(self, vy: f64, vz: f64) -> (f64, f64) {
        let y = (vy - self.vy_lo) * (self.n - 1) as f64 / (self.vy_hi - self.vy_lo);
        let z = (vz - self.vz_lo) * (self.n - 1) as f64 / (self.vz_hi - self.vz_lo);
        (
            y.clamp(0.0, (self.n - 1) as f64),
            z.clamp(0.0, (self.n - 1) as f64),
        )
    }

    fn contains(self, vy: f64, vz: f64) -> bool {
        vy >= self.vy_lo && vy <= self.vy_hi && vz >= self.vz_lo && vz <= self.vz_hi
    }

    fn nearest(self, vy: f64, vz: f64) -> usize {
        let (y, z) = self.frac(vy, vz);
        (y.round() as usize) * self.n + z.round() as usize
    }
}

/// Finite-horizon DP over (v_y, v_z) for the unregularized objective; writes `value.csv` and the
/// flown policy from each probe into `--out`.
#[derive(Parser)]
#[command(allow_negative_numbers = true)]
struct Args {
    /// Horizon, ticks.
    #[arg(long = "n")]
    horizon: usize,
    #[arg(long)]
    lambda: f64,
    #[arg(long, default_value_t = TrigMode::MthLut)]
    trig: TrigMode,
    #[arg(long, default_value_t = FlightMode::Reference)]
    flight: FlightMode,
    /// Largest |pitch|, degrees.
    #[arg(long, default_value_t = 85.0)]
    limit: f64,
    #[arg(long, default_value_t = 0.5)]
    pitch_step: f64,
    #[arg(long, default_value_t = -4.5)]
    vy_lo: f64,
    #[arg(long, default_value_t = 1.5)]
    vy_hi: f64,
    #[arg(long, default_value_t = -1.0)]
    vz_lo: f64,
    #[arg(long, default_value_t = 4.5)]
    vz_hi: f64,
    /// Nodes per velocity axis.
    #[arg(long = "grid", default_value_t = 513)]
    grid_n: usize,
    #[arg(long, default_value = "dp-out")]
    out: PathBuf,
    /// Start velocities to fly the policy from, `vy:vz,...`.
    #[arg(long = "probe", value_delimiter = ',', value_parser = parse_probe, allow_hyphen_values = true,
          default_value = "0:0")]
    probes: Vec<(f64, f64)>,
}

fn parse_probe(point: &str) -> Result<(f64, f64), String> {
    let (vy, vz) = point.split_once(':').ok_or_else(|| format!("want vy:vz, got {point:?}"))?;
    let num = |x: &str| x.parse::<f64>().map_err(|e| format!("{x:?}: {e}"));
    Ok((num(vy)?, num(vz)?))
}

impl Args {
    fn grid(&self) -> Grid {
        Grid { n: self.grid_n, vy_lo: self.vy_lo, vy_hi: self.vy_hi, vz_lo: self.vz_lo, vz_hi: self.vz_hi }
    }

    fn validate(&self) {
        assert!(self.horizon > 0, "--n must be positive");
        assert!(self.lambda.is_finite(), "--lambda must be finite");
        assert!(
            self.limit.is_finite() && self.limit > 0.0 && self.limit <= 90.0,
            "--limit must be in (0, 90]"
        );
        assert!(
            self.pitch_step.is_finite() && self.pitch_step > 0.0,
            "--pitch-step must be positive"
        );
        assert!(self.grid_n >= 2, "--grid must be at least 2");
        assert!(
            self.grid().vy_lo.is_finite()
                && self.grid().vy_hi.is_finite()
                && self.grid().vz_lo.is_finite()
                && self.grid().vz_hi.is_finite(),
            "velocity bounds must be finite"
        );
        assert!(
            self.grid().vy_lo < self.grid().vy_hi,
            "--vy-lo must be below --vy-hi"
        );
        assert!(
            self.grid().vz_lo < self.grid().vz_hi,
            "--vz-lo must be below --vz-hi"
        );
        assert!(
            !self.probes.is_empty(),
            "--probe must contain at least one point"
        );
        for &(vy, vz) in &self.probes {
            assert!(
                vy.is_finite() && vz.is_finite(),
                "probe velocities must be finite"
            );
            assert!(
                self.grid().contains(vy, vz),
                "probe ({vy}, {vz}) lies outside the velocity grid"
            );
        }
    }
}

fn controls(limit: f64, step: f64) -> Vec<f64> {
    let intervals = (2.0 * limit / step).round() as usize;
    let represented = intervals as f64 * step;
    assert!(
        (represented - 2.0 * limit).abs() <= 1e-9 * (2.0 * limit).max(1.0),
        "--pitch-step must divide the full pitch interval [-limit, limit]"
    );
    (0..=intervals).map(|i| -limit + i as f64 * step).collect()
}

#[inline]
fn bilinear(values: &[f64], n: usize, y: f32, z: f32) -> f64 {
    // Put a point exactly on the upper boundary in the final cell with weight one. This keeps
    // all four array accesses valid without moving or special-casing the physical point.
    let yf = y as f64;
    let zf = z as f64;
    let y0 = (yf.floor() as usize).min(n - 2);
    let z0 = (zf.floor() as usize).min(n - 2);
    let ty = yf - y0 as f64;
    let tz = zf - z0 as f64;
    let a = values[y0 * n + z0] * (1.0 - tz) + values[y0 * n + z0 + 1] * tz;
    let b = values[(y0 + 1) * n + z0] * (1.0 - tz) + values[(y0 + 1) * n + z0 + 1] * tz;
    a * (1.0 - ty) + b * ty
}

fn value_at(values: &[f64], grid: Grid, vy: f64, vz: f64) -> f64 {
    let (y, z) = grid.frac(vy, vz);
    bilinear(values, grid.n, y as f32, z as f32)
}

#[derive(Default)]
struct BoundaryStats {
    pairs: u64,
    worst_y: f64,
    worst_z: f64,
}

fn overshoot(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo - x
    } else if x > hi {
        x - hi
    } else {
        0.0
    }
}

fn allocation_bytes(states: usize, controls: usize, horizon: usize) -> (u64, u64, u64) {
    let transitions = states
        .checked_mul(controls)
        .and_then(|x| x.checked_mul(size_of::<Transition>()))
        .expect("transition size overflow") as u64;
    let policy = states
        .checked_mul(horizon)
        .and_then(|x| x.checked_mul(size_of::<u16>()))
        .expect("policy size overflow") as u64;
    let values = states
        .checked_mul(2)
        .and_then(|x| x.checked_mul(size_of::<f64>()))
        .expect("value size overflow") as u64;
    (transitions, policy, values)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn build_transitions(grid: Grid, pitches: &[f64], w: f64) -> (Vec<Transition>, BoundaryStats) {
    let mut table = vec![
        Transition {
            y: 0.0,
            z: 0.0,
            reward: 0.0
        };
        grid.states() * pitches.len()
    ];
    let stats = table
        .par_chunks_mut(pitches.len())
        .enumerate()
        .map(|(state, row)| {
            let iy = state / grid.n;
            let iz = state % grid.n;
            let vel = Vec3::new(0.0, grid.vy(iy), grid.vz(iz));
            let mut local = BoundaryStats::default();
            for (slot, &pitch) in row.iter_mut().zip(pitches) {
                let next = elytrasim::sim::update_fall_flying_movement(
                    vel,
                    Rot {
                        x: pitch as f32,
                        y: 0.0,
                    },
                );
                let oy = overshoot(next.y, grid.vy_lo, grid.vy_hi);
                let oz = overshoot(next.z, grid.vz_lo, grid.vz_hi);
                if oy > 0.0 || oz > 0.0 {
                    local.pairs += 1;
                }
                local.worst_y = local.worst_y.max(oy);
                local.worst_z = local.worst_z.max(oz);
                let (y, z) = grid.frac(next.y, next.z);
                *slot = Transition {
                    y: y as f32,
                    z: z as f32,
                    reward: (next.y + w * next.z) as f32,
                };
            }
            local
        })
        // The transition map is time-homogeneous, so this single count is the count for every
        // Bellman sweep. Recording it while building avoids running the physics a second time.
        .reduce(BoundaryStats::default, |a, b| BoundaryStats {
            pairs: a.pairs + b.pairs,
            worst_y: a.worst_y.max(b.worst_y),
            worst_z: a.worst_z.max(b.worst_z),
        });
    (table, stats)
}

fn write_schedule(path: &Path, pitches: &[f64]) {
    let file = File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = BufWriter::new(file);
    for pitch in pitches {
        writeln!(out, "{pitch:.10}").unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

fn rollout_policy(
    probe: (f64, f64),
    horizon: usize,
    grid: Grid,
    pitch_values: &[f64],
    policy: &[u16],
) -> (Vec<f64>, usize, usize, f64, f64) {
    let mut vel = Vec3::new(0.0, probe.0, probe.1);
    let mut schedule = Vec::with_capacity(horizon);
    let mut grid_oob = 0;
    let mut exact_oob = 0;
    let (mut worst_y, mut worst_z): (f64, f64) = (0.0, 0.0);
    for remaining in (1..=horizon).rev() {
        // The policy is defined at grid nodes. Nearest-node lookup preserves its discrete
        // bang-bang choices; interpolating pitch itself would invent controls the DP did not
        // evaluate and would particularly misrepresent an intentionally chattering policy.
        let state = grid.nearest(vel.y, vel.z);
        let control = policy[(remaining - 1) * grid.states() + state] as usize;
        let pitch = pitch_values[control];
        let grid_vel = Vec3::new(0.0, grid.vy(state / grid.n), grid.vz(state % grid.n));
        let grid_next = elytrasim::sim::update_fall_flying_movement(
            grid_vel,
            Rot {
                x: pitch as f32,
                y: 0.0,
            },
        );
        if !grid.contains(grid_next.y, grid_next.z) {
            grid_oob += 1
        }

        let next = elytrasim::sim::update_fall_flying_movement(
            vel,
            Rot {
                x: pitch as f32,
                y: 0.0,
            },
        );
        let oy = overshoot(next.y, grid.vy_lo, grid.vy_hi);
        let oz = overshoot(next.z, grid.vz_lo, grid.vz_hi);
        if oy > 0.0 || oz > 0.0 {
            exact_oob += 1
        }
        worst_y = worst_y.max(oy);
        worst_z = worst_z.max(oz);
        vel = next;
        schedule.push(pitch);
    }
    (schedule, grid_oob, exact_oob, worst_y, worst_z)
}

fn main() {
    let args = Args::parse();
    args.validate();
    let grid = args.grid();
    set_trig_mode(args.trig);
    set_flight_mode(args.flight);
    let pitch_values = controls(args.limit, args.pitch_step);
    assert!(
        pitch_values.len() <= u16::MAX as usize + 1,
        "too many controls for the u16 policy representation"
    );
    let states = grid.states();
    let (transition_bytes, policy_bytes, value_bytes) =
        allocation_bytes(states, pitch_values.len(), args.horizon);
    let total_bytes = transition_bytes + policy_bytes + value_bytes;
    eprintln!(
        "dp: grid {}x{} ({} states), {} controls, n {}, trig {}, flight {}",
        grid.n,
        grid.n,
        states,
        pitch_values.len(),
        args.horizon,
        trig_mode(),
        flight_mode()
    );
    eprintln!(
        "allocation before allocating: transitions {:.3} GiB, policy {:.3} GiB, \
         values {:.3} GiB; total {:.3} GiB",
        gib(transition_bytes),
        gib(policy_bytes),
        gib(value_bytes),
        gib(total_bytes)
    );
    if total_bytes > MAX_ALLOCATION {
        eprintln!("refusing allocation above 6 GiB; reduce --grid, --pitch-step, or --n");
        std::process::exit(2);
    }

    fs::create_dir_all(&args.out).unwrap_or_else(|e| panic!("{}: {e}", args.out.display()));
    let value_path = args.out.join("value.csv");
    let mut csv = BufWriter::new(
        File::create(&value_path).unwrap_or_else(|e| panic!("{}: {e}", value_path.display())),
    );
    writeln!(csv, "n,vy0,vz0,V").unwrap();

    let w = w_of_lambda(args.lambda);
    let precompute_start = Instant::now();
    let (transitions, boundary) = build_transitions(grid, &pitch_values, w);
    let precompute_secs = precompute_start.elapsed().as_secs_f64();
    eprintln!("transition precompute: {precompute_secs:.3}s");

    let mut previous: Vec<f64> = (0..states)
        .into_par_iter()
        .map(|state| {
            let vy = grid.vy(state / grid.n);
            let vz = grid.vz(state % grid.n);
            (vy * vy + vz * vz) * 0.5 / GRAVITY
        })
        .collect();
    let mut current = vec![0.0; states];
    let mut policy = vec![0u16; states * args.horizon];
    let sweep_start = Instant::now();
    for k in 1..=args.horizon {
        let layer = &mut policy[(k - 1) * states..k * states];
        current
            .par_iter_mut()
            .zip(layer.par_iter_mut())
            .enumerate()
            .for_each(|(state, (value, action))| {
                let row =
                    &transitions[state * pitch_values.len()..(state + 1) * pitch_values.len()];
                let mut best = f64::NEG_INFINITY;
                let mut best_control = 0usize;
                for (control, tr) in row.iter().enumerate() {
                    let candidate = tr.reward as f64 + bilinear(&previous, grid.n, tr.y, tr.z);
                    if candidate > best {
                        best = candidate;
                        best_control = control;
                    }
                }
                *value = best;
                *action = best_control as u16;
            });
        for &(vy, vz) in &args.probes {
            writeln!(
                csv,
                "{k},{vy:.10},{vz:.10},{:.12}",
                value_at(&current, grid, vy, vz)
            )
            .unwrap();
        }
        std::mem::swap(&mut previous, &mut current);
    }
    csv.flush().unwrap();
    let sweep_secs = sweep_start.elapsed().as_secs_f64();
    eprintln!(
        "backward induction: {sweep_secs:.3}s ({:.6}s/sweep)",
        sweep_secs / args.horizon as f64
    );

    for &(vy, vz) in &args.probes {
        let predicted = value_at(&previous, grid, vy, vz);
        let (schedule, grid_oob, exact_oob, path_oy, path_oz) =
            rollout_policy((vy, vz), args.horizon, grid, &pitch_values, &policy);
        let filename = format!("dp_n{}_vy{vy:+.6}_vz{vz:+.6}.pitches", args.horizon);
        let path = args.out.join(filename);
        write_schedule(&path, &schedule);

        let states = replay_from(Vec3::new(0.0, vy, vz), &schedule);
        let objective = Objective {
            v0: Vec3::new(0.0, vy, vz),
            n: args.horizon,
            lambda: args.lambda,
        };
        let realized = objective.j(states.last().unwrap());
        let gap = realized - predicted;
        eprintln!("SELF-CHECK probe ({vy:.6}, {vz:.6}):");
        eprintln!("  DP predicted V = {predicted:.12}");
        eprintln!("  simulator J    = {realized:.12}");
        eprintln!("  replay - DP    = {gap:+.12} (absolute {:.3e})", gap.abs());
        eprintln!("  schedule       = {}", path.display());
        if grid_oob > 0 || exact_oob > 0 {
            eprintln!(
                "WARNING: OPTIMAL PATH LEFT THE VELOCITY BOX: {grid_oob} selected \
                       grid transitions, {exact_oob} exact rollout transitions; worst \
                       overshoot vy {path_oy:.6}, vz {path_oz:.6}"
            );
        } else {
            eprintln!("  optimal path boundary check: no out-of-box transitions");
        }
    }

    let aggregate = boundary.pairs.saturating_mul(args.horizon as u64);
    eprintln!(
        "out-of-box transitions: {} state/control pairs PER SWEEP; {} across {} sweeps; \
         worst overshoot vy {:.6}, vz {:.6} (all such transitions were clamped)",
        boundary.pairs, aggregate, args.horizon, boundary.worst_y, boundary.worst_z
    );
    eprintln!(
        "total measured compute time: {:.3}s (precompute {:.3}s + sweeps {:.3}s)",
        precompute_secs + sweep_secs,
        precompute_secs,
        sweep_secs
    );
}
