use crate::sim::*;

type Goodness = f64;

fn approx_eq_f32(a: f32, b: f32) -> bool {
    assert!(a.is_finite());
    assert!(b.is_finite());
    (a - b).abs() < 0.001
}

fn approx_eq_f64(a: f64, b: f64) -> bool {
    assert!(a.is_finite());
    assert!(b.is_finite());
    (a - b).abs() < 0.001
}

fn clamp_pitch(pitch: &mut f32) {
    *pitch = pitch.clamp(-90.0, 90.0);
}

fn clamped_pitch(pitch: f32) -> f32 {
    pitch.clamp(-90.0, 90.0)
}

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

    fn goodness(&self) -> Goodness {
        // real optimization targets
        // self.pos.y
        self.total_energy()

        // mental illnesses
        // self.vel.y
        // self.pos.y / self.pos.z
        // z vel is only interesting for not steady state
        // self.vel.z
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

#[derive(Debug, Clone)]
pub struct Pitches(pub Vec<f32>);
impl Pitches {
    pub fn new(ticks: usize) -> Self {
        Self(vec![0.0; ticks])
    }

    /// positive for the first half, negative for the second.
    pub fn new_4040(ticks: usize) -> Self {
        let mid = ticks / 2;
        Self(
            (0..mid)
                .map(|_| 40.0)
                .chain((mid..ticks).map(|_| -40.0))
                .collect(),
        )
    }

    // /// of len self.0.len() + 1
    // pub fn cycle(&self, init_vel: Vec3) -> Vec<State> {
    //     let mut states = Vec::with_capacity(self.0.len() + 1);
    //     let mut cur = State {
    //         pos: Vec3::ZERO,
    //         vel: init_vel,
    //     };
    //     states.push(cur.clone());
    //     for pitch in self.0.iter() {
    //         let rot = Rot { x: *pitch, y: 0.0 };
    //         cur = cur.ticked(rot);
    //         states.push(cur.clone());
    //     }
    //     states
    // }
    /// of len self.0.len()
    pub fn cycle(&self, init_vel: Vec3) -> Vec<State> {
        let mut states = Vec::with_capacity(self.0.len() + 1);
        let mut cur = State {
            pos: Vec3::ZERO,
            vel: init_vel,
        };
        for pitch in self.0.iter() {
            let rot = Rot { x: *pitch, y: 0.0 };
            cur = cur.ticked(rot);
            states.push(cur.clone());
        }
        states
    }

    /// given this init velocity, return the state after applying the pitches.
    pub fn after_cycle(&self, vel: Vec3) -> State {
        let mut cur = State {
            pos: Vec3::ZERO,
            vel,
        };
        for pitch in self.0.iter() {
            let rot = Rot { x: *pitch, y: 0.0 };
            cur = cur.ticked(rot);
        }
        cur
    }

    /// init vel is a guess at the stead state velocity.
    pub fn steady_vel_guessed(&self, steady_vel_guess: Vec3) -> Vec3 {
        let mut state = self.after_cycle(steady_vel_guess);
        loop {
            let next = self.after_cycle(state.vel);
            if approx_eq_f64(state.vel.x, next.vel.x)
                && approx_eq_f64(state.vel.y, next.vel.y)
                && approx_eq_f64(state.vel.z, next.vel.z)
            {
                break;
            }
            state = next;
        }
        state.vel
    }

    // /// factor out bc we may have better heuristics in the future.
    // pub fn steady_vel(&self) -> Vec3 {
    //     self.steady_vel_guessed(Vec3::ZERO)
    // }

    fn clamp(&mut self) {
        for pitch in self.0.iter_mut() {
            debug_assert!(pitch.is_finite());
            clamp_pitch(pitch);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Optimizer {
    /// doesn't need to be exactly to maintain the invariant of the type,
    /// but should only diverge when inside the api boundary.
    /// outside the api, it should be exact.
    pub steady_vel: Vec3,
    pub pitches: Pitches,
}
impl Optimizer {
    /// not gradient decent
    fn optimization_step_pitch_not_gradient_decent(&mut self, pitch_i: usize) {
        const EPSILON: f64 = 0.1;

        let cur_pitch = self.pitches.0[pitch_i];
        let cur_goodness = self.pitches.after_cycle(self.steady_vel).goodness();

        // TODO: do goodness after steady state instead of assuming it's the same for the delta
        let right_pitch = cur_pitch + EPSILON as f32;
        if right_pitch == clamped_pitch(right_pitch) {
            self.pitches.0[pitch_i] = right_pitch;
            let right_goodness = self.pitches.after_cycle(self.steady_vel).goodness();
            if right_goodness > cur_goodness {
                return;
            }
        }

        let left_pitch = cur_pitch - EPSILON as f32;
        if left_pitch == clamped_pitch(left_pitch) {
            self.pitches.0[pitch_i] = left_pitch;
            let left_goodness = self.pitches.after_cycle(self.steady_vel).goodness();
            if left_goodness > cur_goodness {
                return;
            }
        }

        self.pitches.0[pitch_i] = cur_pitch;
    }

    /// gradient decent
    fn optimization_step_pitch_gradient_descent(&mut self, pitch_i: usize) {
        const EPSILON: f64 = 0.1;
        const LEARNING_RATE: f64 = 100.0;

        let cur_pitch = self.pitches.0[pitch_i];

        let right_pitch = cur_pitch + EPSILON as f32;
        let right_goodness = if right_pitch == clamped_pitch(right_pitch) {
            self.pitches.0[pitch_i] = right_pitch;
            Some(self.pitches.after_cycle(self.steady_vel).goodness())
            // let mut slf = self.clone();
            // slf.pitches.0[pitch_i] = right_pitch;
            // slf.steady_vel = slf.pitches.steady_vel_guessed(slf.steady_vel);
            // Some(slf.pitches.after_cycle(slf.steady_vel).goodness())
        } else {
            None
        };

        let left_pitch = cur_pitch - EPSILON as f32;
        let left_goodness = if left_pitch == clamped_pitch(left_pitch) {
            self.pitches.0[pitch_i] = left_pitch;
            Some(self.pitches.after_cycle(self.steady_vel).goodness())
            // let mut slf = self.clone();
            // slf.pitches.0[pitch_i] = left_pitch;
            // slf.steady_vel = slf.pitches.steady_vel_guessed(slf.steady_vel);
            // Some(slf.pitches.after_cycle(slf.steady_vel).goodness())
        } else {
            None
        };

        // only compute this if we need to, otherwise we can use central difference
        let cur_goodness = if left_goodness.is_none() || right_goodness.is_none() {
            self.pitches.0[pitch_i] = cur_pitch;
            Some(self.pitches.after_cycle(self.steady_vel).goodness())
        } else {
            None
        };

        let grad = match (left_goodness, right_goodness) {
            // central difference if we can
            (Some(left_goodness), Some(right_goodness)) => {
                (right_goodness - left_goodness) / (2.0 * EPSILON)
            }
            (None, Some(right_goodness)) => (right_goodness - cur_goodness.unwrap()) / EPSILON,
            (Some(left_goodness), None) => (cur_goodness.unwrap() - left_goodness) / EPSILON,
            (None, None) => unreachable!(),
        };

        let delta_pitch = ((LEARNING_RATE * grad) as f32).clamp(-5.0, 5.0);
        self.pitches.0[pitch_i] = clamped_pitch(cur_pitch + delta_pitch);
    }

    /// apply one step of optimization to the pitches.
    pub fn optimization_step(&mut self) {
        for i in 0..self.pitches.0.len() {
            // self.optimization_step_pitch_not_gradient_decent(i);
            self.optimization_step_pitch_gradient_descent(i);
            self.pitches.clamp();
        }
        self.steady_vel = self.pitches.steady_vel_guessed(self.steady_vel);
    }

    // fn show(&self, ui: &mut egui::Ui) {}
}
impl From<Pitches> for Optimizer {
    fn from(pitches: Pitches) -> Self {
        let steady_vel = pitches.steady_vel_guessed(Vec3::ZERO);
        Self {
            steady_vel,
            pitches,
        }
    }
}
