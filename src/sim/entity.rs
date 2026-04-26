use std::f64::consts::PI;

use super::{GRAVITY, Mth, Rot, Vec3};

#[derive(Debug, Default, Clone)]
pub struct Entity {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: Rot,
}

impl Entity {
    pub fn travel(&mut self) {
        self.vel = update_fall_flying_movement(self.vel, self.rot);
        self.mov();
    }

    pub fn mov(&mut self) {
        self.pos += self.vel;
    }
}

pub fn update_fall_flying_movement(mut vel: Vec3, rot: Rot) -> Vec3 {
    let look_angle: Vec3 = rot.look_angle();
    let lean_angle: f32 = rot.x * (PI / 180.0) as f32;

    let look_hor_length: f64 = (look_angle.x * look_angle.x + look_angle.z * look_angle.z).sqrt();
    let move_hor_length: f64 = vel.horizontal_distance();
    let gravity: f64 = GRAVITY;
    let lift_force: f64 = Mth::square((lean_angle as f64).cos());
    vel.y += gravity * (-1.0 + lift_force * 0.75);
    if vel.y < 0.0 && look_hor_length > 0.0 {
        let convert: f64 = vel.y * -0.1 * lift_force;
        vel += Vec3::new(
            look_angle.x * convert / look_hor_length,
            convert,
            look_angle.z * convert / look_hor_length,
        );
    }

    if lean_angle < 0.0 && look_hor_length > 0.0 {
        let convert: f64 = move_hor_length * -Mth::sin(lean_angle) as f64 * 0.04;
        vel += Vec3::new(
            -look_angle.x * convert / look_hor_length,
            convert * 3.2,
            -look_angle.z * convert / look_hor_length,
        );
    }

    if look_hor_length > 0.0 {
        vel += Vec3::new(
            (look_angle.x / look_hor_length * move_hor_length - vel.x) * 0.1,
            0.0,
            (look_angle.z / look_hor_length * move_hor_length - vel.z) * 0.1,
        );
    }

    vel * Vec3::new(0.99_f32 as f64, 0.98_f32 as f64, 0.99_f32 as f64)
}
