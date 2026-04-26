use super::{Entity, GRAVITY, Rot, Vec3};

#[derive(Debug, Default, Clone)]
pub struct State {
    pub pos: Vec3,
    pub vel: Vec3,
}
impl State {
    pub fn ticked(&self, rot: Rot) -> Self {
        let mut entity = Entity {
            pos: self.pos,
            vel: self.vel,
            rot,
        };
        entity.travel();
        entity.into()
    }

    /// kilograms * blocks^2 / ticks^2
    pub fn kinetic_energy(&self) -> f64 {
        self.vel.length_sq() * 0.5
    }

    /// kilograms * blocks^2 / ticks^2
    pub fn potential_energy(&self) -> f64 {
        self.pos.y * GRAVITY
    }

    /// kilograms * blocks^2 / ticks^2
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }
}
impl From<Entity> for State {
    fn from(entity: Entity) -> Self {
        Self {
            pos: entity.pos,
            vel: entity.vel,
        }
    }
}
