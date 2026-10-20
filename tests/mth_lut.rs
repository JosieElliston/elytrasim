//! The `Mth` lookup table is a portability decision: Minecraft's trig is a 65536-entry table,
//! not libm, so using it makes a profile replay the same on a laptop and on the cluster. These
//! tests check it is really that table, and really vanilla's indexing.

use elytrasim::sim::*;
use std::sync::{Mutex, MutexGuard};

/// The trig mode is process-global and cargo runs a binary's tests on parallel threads, so every
/// test that sets it holds this for its whole body. Otherwise
/// `the_two_modes_disagree_and_libm_is_the_default` can flip the mode to libm mid-way through a
/// test that expects the table.
static MODES: Mutex<()> = Mutex::new(());

fn hold_modes() -> MutexGuard<'static, ()> {
    MODES.lock().unwrap_or_else(|e| e.into_inner())
}

/// The table is built from `f64::sin`, so a platform whose libm rounded one entry differently
/// would silently replay every profile in the sweep differently. This makes that a test failure
/// instead of a mystery.
#[test]
fn sin_table_is_bit_identical_everywhere() {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in sin_table() {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    assert_eq!(h, 0x21cb003ccaa34307, "Mth.SIN differs from the reference table");
}

/// `Mth.cos(-90 deg)` is *exactly* zero under the table where libm gives -4.4e-8. This is the
/// porting caveat in README-myopic.md: at pitch -90 the lift force vanishes exactly.
#[test]
fn cos_of_minus_ninety_degrees_is_exactly_zero() {
    let _modes = hold_modes();
    let x = -90.0f32 * (std::f64::consts::PI / 180.0) as f32;
    set_trig_mode(TrigMode::MthLut);
    assert_eq!(Mth::cos(x), 0.0, "Mth.cos(-90 deg) must be exactly 0");
    assert_eq!(Mth::sin(x), -1.0, "Mth.sin(-90 deg) must be exactly -1");
}

/// It is the right table: the index truncates rather than rounds, so the error is bounded by
/// one full step of `2*pi/65536`, and it really does reach that -- if it were much smaller we
/// would be looking at libm by mistake.
#[test]
fn lut_tracks_libm_to_within_one_step() {
    let _modes = hold_modes();
    set_trig_mode(TrigMode::MthLut);
    let mut worst = 0.0f64;
    for i in -400_000..=400_000i32 {
        let x = i as f32 * 1e-5;
        worst = worst
            .max((Mth::sin(x) as f64 - (x as f64).sin()).abs())
            .max((Mth::cos(x) as f64 - (x as f64).cos()).abs());
    }
    let step = std::f64::consts::TAU / 65536.0;
    assert!(worst < 1.05 * step, "LUT off by {worst:.3e}, more than one step {step:.3e}");
    assert!(worst > step / 2.0, "LUT off by only {worst:.3e} -- is this really the table?");
}

/// Java's float-to-int cast truncates toward zero and the index then wraps with `& 0xFFFF`, so
/// a negative angle reads from the top of the table. Half the pitch domain is negative, and
/// getting this wrong (rounding, or wrapping the wrong way) would be silent, so check the index
/// arithmetic directly rather than trusting a symmetry -- the table is not exactly odd, because
/// its two mirrored entries come from two different `f64::sin` arguments.
#[test]
fn negative_angles_index_from_the_top_of_the_table() {
    let _modes = hold_modes();
    set_trig_mode(TrigMode::MthLut);
    for k in 1..20_000i32 {
        let x = k as f32 * 1e-4;
        let i = (x * 10430.378) as i32;
        assert_eq!(Mth::sin(-x), sin_table()[((-i) & 0xFFFF) as usize], "bad index at -{x}");
        assert_eq!(Mth::sin(x), sin_table()[(i & 0xFFFF) as usize], "bad index at {x}");
    }
}

/// Switching modes must actually switch the physics, and `libm` must stay the default so the
/// numbers in README-myopic.md keep meaning what they say.
#[test]
fn the_two_modes_disagree_and_libm_is_the_default() {
    let _modes = hold_modes();
    assert_eq!(TrigMode::default(), TrigMode::Libm);
    let r = Rot { x: 37.25, y: 0.0 };
    set_trig_mode(TrigMode::Libm);
    let a = r.look_angle();
    set_trig_mode(TrigMode::MthLut);
    let b = r.look_angle();
    assert_ne!(a.y.to_bits(), b.y.to_bits(), "mth_lut is not doing anything");
    assert!((a.y - b.y).abs() < 1e-4, "the modes disagree by more than a table step");
}
