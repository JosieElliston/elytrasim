mod optimizer;
mod sim;

use sim::*;

use crate::optimizer::*;

pub const TICKS_PER_SECOND: u8 = 20;
pub const TICK_DURATION: std::time::Duration = std::time::Duration::from_millis(50); // 20 per second

fn main() -> eframe::Result {
    // let ticks = 10;
    // let ticks = 20;
    // let ticks = 50;
    // let ticks = 100; // like -6 delta y
    // let ticks = 150; // like 2 delta y
    // let ticks = 200;
    // let ticks = 250;
    let ticks = 300; // like 20 delta y
    // let ticks = 500;

    let mut optimizer = {
        // let pitches = Pitches::new_uniform(ticks, 0.0);
        // let pitches = Pitches::new_4040(ticks, 0.5);
        // let pitches = Pitches::new_4040(ticks, 0.65);
        // let pitches = Pitches::new_40zero40(ticks, 0.65, 0.70);
        // close to the optimal curve with four lines
        let pitches = {
            // let left_left_cut = 0.05;
            let left_cut = 0.65;
            let right_cut = 0.70;
            let right_right_cut = 0.80;
            // let left_left = (ticks as f64 * left_left_cut) as usize;
            let left = (ticks as f64 * left_cut) as usize;
            let right = (ticks as f64 * right_cut) as usize;
            let right_right = (ticks as f64 * right_right_cut) as usize;
            Pitches(
                // Pitches::new_lerp(left_left, 0.0, 10.0)
                //     .0
                //     .iter()
                //     .chain(Pitches::new_lerp(left - left_left, 10.0, 50.0).0.iter())
                Pitches::new_lerp(left, 10.0, 50.0)
                    .0
                    .iter()
                    .chain(Pitches::new_uniform(right - left, 0.0).0.iter())
                    .chain(
                        Pitches::new_lerp(right_right - right, -85.0, -30.0)
                            .0
                            .iter(),
                    )
                    .chain(Pitches::new_lerp(ticks - right_right, -30.0, -10.0).0.iter())
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        };

        OptimizerSteadyState::new(pitches)

        // // let vel = Vel::ZERO;
        // // the optimal steady state vel
        // let vel = Vel::new(0.0, 0.17, 0.2);

        // OptimizerInitState::new(vel, pitches)
    };

    let mut optimization_strategy = OptimizationStrategy::GradientDescent;
    // for the gradient descent strategy
    let mut learning_rate = 500.0;
    // for the fixed delta strategy
    let mut fixed_delta = 0.1;

    let mut optimizing = false;
    let mut optimization_steps_per_frame: usize = 10;

    fn get_neighboring_optimizers<const N: usize>(
        base_vel: Vel,
        base_pitches: &Pitches,
        delta_vel: f64,
    ) -> [[OptimizerInitState; N]; N] {
        (0..N)
            .map(|i| {
                let delta_y = (i as f64 - (N as f64 - 1.0) / 2.0) * delta_vel;
                (0..N)
                    .map(|j| {
                        let delta_z = (j as f64 - (N as f64 - 1.0) / 2.0) * delta_vel;
                        let vel = base_vel + Vel::new(0.0, delta_y, delta_z);
                        OptimizerInitState::new(vel, base_pitches.clone())
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    // OptimizerInitState in a grid around the optimal steady state vel, for comparison
    const N: usize = 3;
    // const N: usize = 5;
    let mut neighboring_delta_vel = 0.5;
    let mut neighboring_optimizers: Option<[[OptimizerInitState; N]; N]> = None;

    eframe::run_ui_native(
        "Elytra Sim",
        eframe::NativeOptions::default(),
        move |ui, _frame| {
            ui.request_repaint();
            egui::Panel::left("side_panel").show_inside(ui, |ui| {
                // increase / decrease ticks (and perhaps other parameters later)
                ui.group(|ui| {
                    // increase / decrease ticks
                    ui.horizontal(|ui| {
                        let mul = if ui.ctx().input(|i| i.modifiers.shift) {
                            10
                        } else {
                            1
                        };

                        ui.label(format!("ticks: {}", optimizer.pitches.0.len()));
                        if ui.button("-").on_hover_text("hold shift for 10x").clicked() {
                            for _ in 0..mul {
                                optimizer.pitches.0.pop();
                                if let Some(optimizers) = neighboring_optimizers.as_mut() {
                                    for line in optimizers.iter_mut() {
                                        for optimizer in line.iter_mut() {
                                            optimizer.pitches.0.pop();
                                        }
                                    }
                                }
                            }
                        }
                        if ui.button("+").on_hover_text("hold shift for 10x").clicked() {
                            for _ in 0..mul {
                                optimizer
                                    .pitches
                                    .0
                                    .push(*optimizer.pitches.0.last().unwrap_or(&0.0));
                                if let Some(optimizers) = neighboring_optimizers.as_mut() {
                                    for line in optimizers.iter_mut() {
                                        for optimizer in line.iter_mut() {
                                            optimizer
                                                .pitches
                                                .0
                                                .push(*optimizer.pitches.0.last().unwrap_or(&0.0));
                                        }
                                    }
                                }
                            }
                        }
                    });

                    if ui.button("double").clicked() {
                        optimizer = OptimizerSteadyState::new(Pitches(
                            optimizer
                                .pitches
                                .0
                                .iter()
                                .chain(optimizer.pitches.0.iter())
                                .cloned()
                                .collect(),
                        ));
                    }
                });

                // neighboring optimizers
                ui.group(|ui| {
                    if ui.button("set neighboring optimizers").clicked() {
                        neighboring_optimizers = Some(get_neighboring_optimizers::<N>(
                            optimizer.init_vel(),
                            &optimizer.pitches,
                            neighboring_delta_vel,
                        ));
                    }
                    if ui.button("clear neighboring optimizers").clicked() {
                        neighboring_optimizers = None;
                    }
                    ui.label("neighboring optimizers delta vel:");
                    ui.add(egui::Slider::new(&mut neighboring_delta_vel, 0.0..=1.0));
                });

                // optimizer parameters
                ui.group(|ui| {
                    ui.label("optimization strategy:");
                    egui::ComboBox::from_id_salt(egui::Id::new("optimization strategy"))
                        .selected_text(match optimization_strategy {
                            OptimizationStrategy::GradientDescent => "gradient descent",
                            OptimizationStrategy::FixedDelta => "fixed delta",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut optimization_strategy,
                                OptimizationStrategy::GradientDescent,
                                "gradient descent",
                            );
                            ui.selectable_value(
                                &mut optimization_strategy,
                                OptimizationStrategy::FixedDelta,
                                "fixed delta",
                            );
                        });

                    ui.label("learning rate:");
                    ui.add(egui::Slider::new(&mut learning_rate, 10.0..=10000.0).logarithmic(true));

                    ui.label("fixed delta:");
                    ui.add(egui::Slider::new(&mut fixed_delta, 0.0..=1.0));
                });

                // optimization on / off
                {
                    let mut do_optimization_step = || match optimization_strategy {
                        OptimizationStrategy::GradientDescent => {
                            optimizer.gradient_descent_step(learning_rate);
                            if let Some(optimizers) = neighboring_optimizers.as_mut() {
                                for line in optimizers.iter_mut() {
                                    for optimizer in line.iter_mut() {
                                        optimizer.gradient_descent_step(learning_rate);
                                    }
                                }
                            }
                        }
                        OptimizationStrategy::FixedDelta => {
                            optimizer.fixed_delta_step(fixed_delta);
                            if let Some(optimizers) = neighboring_optimizers.as_mut() {
                                for line in optimizers.iter_mut() {
                                    for optimizer in line.iter_mut() {
                                        optimizer.fixed_delta_step(fixed_delta);
                                    }
                                }
                            }
                        }
                    };

                    // optimization on / off
                    ui.group(|ui| {
                        if ui.button("optimization step").clicked() {
                            do_optimization_step();
                        }
                        ui.checkbox(&mut optimizing, "optimizing");
                        ui.label("optimization steps per frame:");
                        ui.add(egui::Slider::new(
                            &mut optimization_steps_per_frame,
                            0..=100,
                        ));
                    });

                    // do the optimization steps
                    if optimizing {
                        for _ in 0..optimization_steps_per_frame {
                            do_optimization_step();
                        }
                    }
                }

                ui.group(|ui| {
                    let init_vel = optimizer.init_vel();
                    ui.label(format!("before vel.y: {:.06}", init_vel.y));
                    ui.label(format!("before vel.z: {:.06}", init_vel.z));
                    let after = optimizer.pitches.after_cycle(init_vel);
                    ui.label(format!("after vel.y: {:.06}", after.vel.y));
                    ui.label(format!("after vel.z: {:.06}", after.vel.z));
                    ui.label(format!("after pos.y: {:.06}", after.pos.y));
                    ui.label(format!("after pos.z: {:.06}", after.pos.z));
                });
            });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                // square
                let rect = {
                    let rect = ui.max_rect();
                    let size = rect.size();
                    egui::Rect::from_min_size(rect.min, size)
                };

                // horizontal center line
                ui.painter().line_segment(
                    [
                        egui::pos2(rect.left(), rect.center().y),
                        egui::pos2(rect.right(), rect.center().y),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::from_gray(250)),
                );

                // vertical lines for seconds
                {
                    let seconds = optimizer.pitches.0.len() as f32 / TICKS_PER_SECOND as f32;
                    for second in 0..=seconds.ceil() as usize {
                        let x = rect.left()
                            + (second as f32 * TICKS_PER_SECOND as f32
                                / optimizer.pitches.0.len() as f32)
                                * rect.width();
                        ui.painter().line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            egui::Stroke::new(1.0, egui::Color32::from_gray(150)),
                        );
                    }
                }

                let value_to_y = |value: f32, approx_max_value: f32| {
                    rect.center().y - (value / approx_max_value) * (rect.height() / 2.0)
                };

                let mut dot_at = |x, y: f32, rad: f32, color: egui::Color32| {
                    let dot_rect = egui::Rect::from_center_size(
                        egui::Pos2::new(x, y),
                        egui::Vec2::splat(10.0),
                    );
                    let r = ui.allocate_rect(dot_rect, egui::Sense::hover());
                    ui.painter().circle_filled(egui::pos2(x, y), rad, color);
                    r
                };

                // show all the stuff for the main optimizer
                for (tick, (state, pitch)) in optimizer
                    .pitches
                    .cycle(optimizer.init_vel())
                    .zip(optimizer.pitches.0.iter())
                    .enumerate()
                {
                    let x = rect.left()
                        + (tick as f32 / optimizer.pitches.0.len() as f32) * rect.width();

                    // pitch (pink)
                    {
                        let y = value_to_y(-*pitch, 90.0);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(252, 3, 198))
                            .on_hover_text(format!("tick: {}, pitch: {}", tick, pitch));
                    }

                    // pitch gradient (purple)
                    // actually this just goes to zero, so it's not very interesting
                    #[cfg(false)]
                    {
                        // this is for the ui, just clone it.
                        let mut pitches = optimizer.pitches.clone();
                        let grad = pitches.grad_at_tick(optimizer.init_vel(), tick);
                        let approx_max_grad = 0.01;
                        let grad = grad.clamp(-approx_max_grad, approx_max_grad);
                        let y = value_to_y(-grad, approx_max_grad);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(128, 0, 128))
                            .on_hover_text(format!("tick: {}, pitch gradient: {}", tick, grad));
                    }

                    // pos.y (dark green)
                    {
                        let y = value_to_y(state.pos.y as f32, 100.0);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(0, 100, 0))
                            .on_hover_text(format!("tick: {}, pos.y: {}", tick, state.pos.y));
                    }

                    // pos.z (dark blue)
                    {
                        let y = value_to_y(state.pos.z as f32, 100.0);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(52, 61, 235))
                            .on_hover_text(format!("tick: {}, pos.z: {}", tick, state.pos.z));
                    }

                    // vel.y (light green)
                    {
                        let y = value_to_y(state.vel.y as f32, 5.0);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(144, 238, 144))
                            .on_hover_text(format!("tick: {}, vel.y: {}", tick, state.vel.y));
                    }

                    // vel.z (light blue)
                    {
                        let y = value_to_y(state.vel.z as f32, 5.0);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(52, 165, 235))
                            .on_hover_text(format!("tick: {}, vel.z: {}", tick, state.vel.z));
                    }

                    let approx_max_energy = 4.0;
                    // kinetic energy (yellow)
                    {
                        let ke = state.kinetic_energy();
                        let y = value_to_y(ke as f32, approx_max_energy);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(235, 214, 52))
                            .on_hover_text(format!("tick: {}, kinetic energy: {}", tick, ke));
                    }

                    // potential energy (red)
                    {
                        let pe = state.potential_energy();
                        let y = value_to_y(pe as f32, approx_max_energy);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(255, 0, 0))
                            .on_hover_text(format!("tick: {}, potential energy: {}", tick, pe));
                    }

                    // total energy (orange)
                    {
                        let energy = state.total_energy();
                        let y = value_to_y(energy as f32, approx_max_energy);
                        dot_at(x, y, 4.0, egui::Color32::from_rgb(235, 143, 52))
                            .on_hover_text(format!("tick: {}, total energy: {}", tick, energy));
                    }
                }

                // show the optimizer grid
                if let Some(optimizers) = neighboring_optimizers.as_mut() {
                    for line in optimizers.iter() {
                        for optimizer in line.iter() {
                            for (tick, (state, pitch)) in optimizer
                                .pitches
                                .cycle(optimizer.init_vel())
                                .zip(optimizer.pitches.0.iter())
                                .enumerate()
                            {
                                let x = rect.left()
                                    + (tick as f32 / optimizer.pitches.0.len() as f32)
                                        * rect.width();

                                // pitch (pink)
                                {
                                    let y = value_to_y(-*pitch, 90.0);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(252, 3, 198))
                                        .on_hover_text(format!(
                                            "tick: {}, pitch: {}, init_vel: {:?}",
                                            tick,
                                            pitch,
                                            optimizer.init_vel()
                                        ));
                                }

                                // pitch gradient (purple)
                                // actually this just goes to zero, so it's not very interesting
                                #[cfg(false)]
                                {
                                    // this is for the ui, just clone it.
                                    let mut pitches = optimizer.pitches.clone();
                                    let grad = pitches.grad_at_tick(optimizer.init_vel(), tick);
                                    let approx_max_grad = 0.01;
                                    let grad = grad.clamp(-approx_max_grad, approx_max_grad);
                                    let y = value_to_y(-grad, approx_max_grad);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(128, 0, 128))
                                        .on_hover_text(format!(
                                            "tick: {}, pitch gradient: {}, init_vel: {:?}",
                                            tick,
                                            grad,
                                            optimizer.init_vel()
                                        ));
                                }

                                // pos.y (dark green)
                                {
                                    let y = value_to_y(state.pos.y as f32, 100.0);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(0, 100, 0))
                                        .on_hover_text(format!(
                                            "tick: {}, pos.y: {}, init_vel: {:?}",
                                            tick,
                                            state.pos.y,
                                            optimizer.init_vel()
                                        ));
                                }

                                // pos.z (dark blue)
                                {
                                    let y = value_to_y(state.pos.z as f32, 100.0);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(52, 61, 235))
                                        .on_hover_text(format!(
                                            "tick: {}, pos.z: {}, init_vel: {:?}",
                                            tick,
                                            state.pos.z,
                                            optimizer.init_vel()
                                        ));
                                }

                                // vel.y (light green)
                                {
                                    let y = value_to_y(state.vel.y as f32, 5.0);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(144, 238, 144))
                                        .on_hover_text(format!(
                                            "tick: {}, vel.y: {}, init_vel: {:?}",
                                            tick,
                                            state.vel.y,
                                            optimizer.init_vel()
                                        ));
                                }

                                // vel.z (light blue)
                                {
                                    let y = value_to_y(state.vel.z as f32, 5.0);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(52, 165, 235))
                                        .on_hover_text(format!(
                                            "tick: {}, vel.z: {}, init_vel: {:?}",
                                            tick,
                                            state.vel.z,
                                            optimizer.init_vel()
                                        ));
                                }

                                let approx_max_energy = 4.0;
                                // kinetic energy (yellow)
                                {
                                    let ke = state.kinetic_energy();
                                    let y = value_to_y(ke as f32, approx_max_energy);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(235, 214, 52))
                                        .on_hover_text(format!(
                                            "tick: {}, kinetic energy: {}, init_vel: {:?}",
                                            tick,
                                            ke,
                                            optimizer.init_vel()
                                        ));
                                }

                                // potential energy (red)
                                {
                                    let pe = state.potential_energy();
                                    let y = value_to_y(pe as f32, approx_max_energy);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(255, 0, 0))
                                        .on_hover_text(format!(
                                            "tick: {}, potential energy: {}, init_vel: {:?}",
                                            tick,
                                            pe,
                                            optimizer.init_vel()
                                        ));
                                }

                                // total energy (orange)
                                {
                                    let energy = state.total_energy();
                                    let y = value_to_y(energy as f32, approx_max_energy);
                                    dot_at(x, y, 2.0, egui::Color32::from_rgb(235, 143, 52))
                                        .on_hover_text(format!(
                                            "tick: {}, total energy: {}, init_vel: {:?}",
                                            tick,
                                            energy,
                                            optimizer.init_vel()
                                        ));
                                }
                            }
                        }
                    }
                }
            });
        },
    )
}

// pub const TICK_DURATION: std::time::Duration = std::time::Duration::from_millis(50); // 20 per second

// fn main() -> eframe::Result {
//     let mut entity = Entity {
//         pos: Vec3::ZERO,
//         vel: Vec3::ZERO,
//         rot: Rot { x: 0.0, y: 0.0 },
//     };

//     let mut running = false;
//     let mut next_tick = std::time::Instant::now();

//     eframe::run_ui_native(
//         "Elytra Sim",
//         eframe::NativeOptions::default(),
//         move |ui, _frame| {
//             let now = std::time::Instant::now();

//             egui::CentralPanel::default().show_inside(ui, |ui| {
//                 ui.group(|ui| {
//                     ui.checkbox(&mut running, "Running");
//                     if ui.button("Step").clicked() {
//                         entity.travel();
//                         running = false;
//                     } else if running {
//                         if next_tick <= now {
//                             entity.travel();
//                             next_tick += TICK_DURATION;
//                         }
//                         ui.request_repaint_after(next_tick.saturating_duration_since(now));
//                     }
//                     if ui.button("Reset").clicked() {
//                         entity = Entity::default();
//                     }
//                 });

//                 ui.group(|ui| {
//                     ui.strong("Position");
//                     ui.label("X");
//                     ui.add(pos_slider(&mut entity.pos.x));
//                     ui.label("Y");
//                     ui.add(pos_slider(&mut entity.pos.y));
//                     ui.label("Z");
//                     ui.add(pos_slider(&mut entity.pos.z));
//                 });

//                 ui.group(|ui| {
//                     ui.strong("Velocity");
//                     ui.label(format!("X = {:.3}", entity.vel.x * 20.0));
//                     ui.add(vel_slider(&mut entity.vel.x));
//                     ui.label(format!("Y = {:.3}", entity.vel.y * 20.0));
//                     ui.add(vel_slider(&mut entity.vel.y));
//                     ui.label(format!("Z = {:.3}", entity.vel.z * 20.0));
//                     ui.add(vel_slider(&mut entity.vel.z));
//                 });

//                 ui.group(|ui| {
//                     ui.strong("Rotation");
//                     ui.label("X");
//                     ui.add(
//                         egui::Slider::new(&mut entity.rot.x, -180.0..=180.0)
//                             .clamping(egui::SliderClamping::Never),
//                     );
//                     ui.label("Y");
//                     ui.add(
//                         egui::Slider::new(&mut entity.rot.y, -90.0..=90.0)
//                             .clamping(egui::SliderClamping::Never),
//                     );
//                 });
//             });
//         },
//     )
// }

pub fn pos_slider(value: &mut f64) -> egui::Slider<'_> {
    egui::Slider::new(value, -100.0..=100.0).clamping(egui::SliderClamping::Never)
}

pub fn vel_slider(value: &mut f64) -> egui::Slider<'_> {
    egui::Slider::new(value, -5.0..=5.0).clamping(egui::SliderClamping::Never)
}
