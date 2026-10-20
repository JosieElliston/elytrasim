use elytrasim::sim::*;
use std::sync::{Mutex, MutexGuard};

/// The trig and flight modes are process-global and cargo runs a binary's tests on parallel
/// threads, so every test that sets a mode holds this for its whole body. Without it one test
/// flips the mode under another and `cached_pitch_terms_are_bit_identical_in_every_physics_mode`
/// fails about 1 run in 10.
static MODES: Mutex<()> = Mutex::new(());

fn hold_modes() -> MutexGuard<'static, ()> {
    MODES.lock().unwrap_or_else(|e| e.into_inner())
}

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn between(state: &mut u64, lo: f64, hi: f64) -> f64 {
    let unit = (next_u64(state) >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
    lo + (hi - lo) * unit
}

fn max_component_error(a: Vec3, b: Vec3) -> f64 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

#[test]
fn yaw_zero_algebra_matches_reference_in_both_trig_modes() {
    let _modes = hold_modes();
    let mut rng = 0x72d2_19bb_047e_920d_u64;
    for trig in [TrigMode::Libm, TrigMode::MthLut] {
        set_trig_mode(trig);
        let mut worst = 0.0_f64;
        for _ in 0..100_000 {
            let vel = Vec3::new(
                between(&mut rng, -5.0, 5.0),
                between(&mut rng, -5.0, 5.0),
                between(&mut rng, -5.0, 5.0),
            );
            let pitch = between(&mut rng, -90.0, 90.0) as f32;
            let reference = update_fall_flying_movement_reference(
                vel,
                Rot { x: pitch, y: 0.0 },
            );
            let algebraic = update_fall_flying_movement_yaw_zero(vel, pitch);
            worst = worst.max(max_component_error(reference, algebraic));
        }
        for pitch in [-90.0, -89.999, -0.0, 0.0, 89.999, 90.0] {
            let vel = Vec3::new(0.3, -0.7, 2.1);
            let reference = update_fall_flying_movement_reference(
                vel,
                Rot { x: pitch, y: 0.0 },
            );
            let algebraic = update_fall_flying_movement_yaw_zero(vel, pitch);
            worst = worst.max(max_component_error(reference, algebraic));
        }
        assert!(worst < 1e-12, "{trig}: worst component error was {worst:.3e}");
    }
    set_trig_mode(TrigMode::default());
}

#[test]
fn cached_pitch_terms_are_bit_identical_in_every_physics_mode() {
    let _modes = hold_modes();
    assert_eq!(std::mem::size_of::<PitchTrig>(), 16);
    let mut rng = 0xb438_0f5a_d17c_c923_u64;
    for trig in [TrigMode::Libm, TrigMode::MthLut] {
        set_trig_mode(trig);
        for flight in [FlightMode::Reference, FlightMode::Algebraic] {
            set_flight_mode(flight);
            for _ in 0..100_000 {
                let vel = Vec3::new(
                    between(&mut rng, -5.0, 5.0),
                    between(&mut rng, -5.0, 5.0),
                    between(&mut rng, -5.0, 5.0),
                );
                let pitch = between(&mut rng, -90.0, 90.0) as f32;
                let uncached = update_fall_flying_movement(vel, Rot { x: pitch, y: 0.0 });
                let cached = update_fall_flying_movement_cached(vel, PitchTrig::new(pitch));
                assert_eq!(uncached, cached, "{trig} {flight} at pitch {pitch}");
            }
            for pitch in [-90.0, -89.999, -0.001, -0.0, 0.0, 0.001, 89.999, 90.0] {
                let vel = Vec3::new(0.3, -0.7, 2.1);
                let uncached = update_fall_flying_movement(vel, Rot { x: pitch, y: 0.0 });
                let cached = update_fall_flying_movement_cached(vel, PitchTrig::new(pitch));
                assert_eq!(uncached, cached, "{trig} {flight} at pitch {pitch}");
            }
        }
    }
    set_trig_mode(TrigMode::default());
    set_flight_mode(FlightMode::default());
}

#[test]
fn flight_mode_parses_and_defaults_to_reference() {
    assert_eq!(FlightMode::default(), FlightMode::Reference);
    assert_eq!("reference".parse(), Ok(FlightMode::Reference));
    assert_eq!("ref".parse(), Ok(FlightMode::Reference));
    assert_eq!("algebraic".parse(), Ok(FlightMode::Algebraic));
    assert_eq!(FlightMode::Algebraic.to_string(), "algebraic");
}
