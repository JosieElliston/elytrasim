mod entity;
mod mth;
mod rot;
mod vec3;
mod state;

pub use entity::Entity;
pub use entity::update_fall_flying_movement;
pub use state::State;
pub use mth::Mth;
pub use rot::Rot;
pub use vec3::Vec3;

pub const GRAVITY: f64 = 0.08; // m/tick/tick
