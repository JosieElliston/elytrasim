//! Exact replay data for the interactive floor-profile figure.

use clap::Parser;
use elytrasim::opt::{Objective, read_pitches};
use elytrasim::sim::{FlightMode, State, TrigMode, Vec3, set_flight_mode, set_trig_mode};

/// Replay one schedule from rest and print it per tick, as CSV.
#[derive(Parser)]
#[command(group = clap::ArgGroup::new("schedule").required(true).args(["file", "hold", "best_hold"]))]
struct Args {
    /// A pitch file to replay.
    #[arg(long)]
    file: Option<String>,
    /// Hold this pitch for `--n` ticks.
    #[arg(long, allow_negative_numbers = true)]
    hold: Option<f64>,
    /// Hold the constant pitch that exits the floor at `--y0` latest (or furthest, `--mode dist`).
    #[arg(long, requires = "y0")]
    best_hold: bool,
    /// Ticks to hold for; ignored with `--file`.
    #[arg(long, default_value_t = 450)]
    n: usize,
    /// Height above the floor, for `--best-hold`.
    #[arg(long)]
    y0: Option<f64>,
    /// What `--best-hold` maximizes: `time` or `dist`.
    #[arg(long, default_value = "time", value_parser = ["time", "dist"])]
    mode: String,
}

fn replay(pitches: &[f64]) {
    let obj = Objective { v0: Vec3::ZERO, n: pitches.len(), lambda: 0.0 };
    let states = obj.replay(pitches);
    println!("t,pitch,y,z,vy,vz,te");
    for (t, state) in states.iter().enumerate() {
        let pitch = pitches.get(t).copied().unwrap_or(f64::NAN);
        println!("{t},{pitch},{},{},{},{},{}", state.pos.y, state.pos.z,
                 state.vel.y, state.vel.z, state.total_energy());
    }
}

fn exit_value(states: &[State], depth: f64, distance: bool) -> f64 {
    for k in 1..states.len() {
        let (a, b) = (states[k - 1].pos.y + depth, states[k].pos.y + depth);
        if b < 0.0 {
            let f = a / (a - b);
            return if distance {
                states[k - 1].pos.z + f * (states[k].pos.z - states[k - 1].pos.z)
            } else {
                (k - 1) as f64 + f
            };
        }
    }
    f64::NEG_INFINITY
}

fn best_hold(depth: f64, n: usize, distance: bool) -> f64 {
    let obj = Objective { v0: Vec3::ZERO, n, lambda: 0.0 };
    let score = |pitch: f64| exit_value(&obj.replay(&vec![pitch; n]), depth, distance);
    let (mut best_pitch, mut best_score) = (0.0, f64::NEG_INFINITY);
    for i in -900..=900 {
        let pitch = i as f64 / 10.0;
        let value = score(pitch);
        if value > best_score { best_pitch = pitch; best_score = value; }
    }
    for i in -100..=100 {
        let pitch = best_pitch + i as f64 / 1000.0;
        let value = score(pitch);
        if value > best_score { best_pitch = pitch; best_score = value; }
    }
    best_pitch
}

fn main() {
    set_trig_mode(TrigMode::MthLut);
    set_flight_mode(FlightMode::Reference);
    let a = Args::parse();
    let pitches = if let Some(path) = &a.file {
        read_pitches(path)
    } else if a.best_hold {
        let pitch = best_hold(a.y0.unwrap(), a.n, a.mode == "dist");
        eprintln!("best_hold={pitch}");
        vec![pitch; a.n]
    } else {
        vec![a.hold.unwrap(); a.n]
    };
    replay(&pitches);
}
