mod optimizer;
mod sim;

use sim::*;

use crate::optimizer::*;

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

    // let mut optimizer = Optimizer::<false>::from(Pitches::new(ticks));
    // let mut optimizer = Optimizer::<false>::from(Pitches::new_4040(ticks));
    // let mut optimizer = Optimizer::<true>::from(Pitches::new(ticks));
    let mut optimizer = Optimizer::<true>::from(Pitches::new_4040(ticks));
    let mut optimizing = false;
    let mut optimization_steps_per_frame: usize = 10;
    let mut learning_rate = 500.0;
    eframe::run_ui_native(
        "Elytra Sim",
        eframe::NativeOptions::default(),
        move |ui, _frame| {
            ui.request_repaint();
            egui::Panel::left("side_panel").show_inside(ui, |ui| {
                if ui.button("optimization step").clicked() {
                    optimizer.optimization_step(learning_rate);
                }
                ui.checkbox(&mut optimizing, "optimizing");
                ui.label("optimization steps per frame:");
                ui.add(egui::Slider::new(
                    &mut optimization_steps_per_frame,
                    0..=100,
                ));
                ui.label("learning rate:");
                ui.add(egui::Slider::new(&mut learning_rate, 10.0..=10000.0).logarithmic(true));
                if optimizing {
                    for _ in 0..optimization_steps_per_frame {
                        optimizer.optimization_step(learning_rate);
                    }
                }

                let after = optimizer.pitches.after_cycle(optimizer.steady_vel);
                ui.label(format!("after pos.y: {:.06}", after.pos.y));
                ui.label(format!("after pos.z: {:.06}", after.pos.z));
                ui.label(format!("before vel.y: {:.06}", optimizer.steady_vel.y));
                ui.label(format!("before vel.z: {:.06}", optimizer.steady_vel.z));
                ui.label(format!("after vel.y: {:.06}", after.vel.y));
                ui.label(format!("after vel.z: {:.06}", after.vel.z));

                // // debug steady state
                // {
                //     let steady_vel = optimizer.pitches.steady_vel_guessed(optimizer.steady_vel);
                //     let cycled = optimizer.pitches.after_cycle(steady_vel);
                //     ui.label(format!(
                //         "steady_vel_guessed: ({:.06}, {:.06}, {:.06}), after cycle pos: ({:.06}, {:.06}, {:.06}), vel: ({:.06}, {:.06}, {:.06})",
                //         steady_vel.x, steady_vel.y, steady_vel.z,
                //         cycled.pos.x, cycled.pos.y, cycled.pos.z,
                //         cycled.vel.x, cycled.vel.y, cycled.vel.z,
                //     ));
                // }
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
                    let ticks_per_second = 20;
                    let seconds = optimizer.pitches.0.len() as f32 / ticks_per_second as f32;
                    for second in 0..=seconds.ceil() as usize {
                        let x = rect.left()
                            + (second as f32 * ticks_per_second as f32
                                / optimizer.pitches.0.len() as f32)
                                * rect.width();
                        ui.painter().line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            egui::Stroke::new(1.0, egui::Color32::from_gray(150)),
                        );
                    }
                }

                for (tick, (state, pitch)) in optimizer
                    .pitches
                    .cycle(optimizer.steady_vel)
                    .iter()
                    .zip(optimizer.pitches.0.iter())
                    .enumerate()
                {
                    let x = rect.left()
                        + (tick as f32 / optimizer.pitches.0.len() as f32) * rect.width();
                    let mut dot_at = |y: f32, rad: f32, color: egui::Color32| {
                        let dot_rect = egui::Rect::from_center_size(
                            egui::Pos2::new(x, y),
                            egui::Vec2::splat(10.0),
                        );
                        let response = ui.allocate_rect(dot_rect, egui::Sense::hover());
                        ui.painter().circle_filled(egui::pos2(x, y), rad, color);
                        // response.on_hover_text(format!(
                        //     "tick {}:\napplied pitch {:.2}\nresulting state: {:#?}",
                        //     i, pitch, state
                        // ));
                        response
                    };

                    // pitch (pink)
                    {
                        let y = rect.center().y + (*pitch / 90.0) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(252, 3, 198))
                            .on_hover_text(format!("tick: {}, pitch: {}", tick, pitch));
                    }

                    // pos.y (dark green)
                    {
                        let y =
                            rect.center().y - (state.pos.y as f32 / 100.0) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(0, 100, 0))
                            .on_hover_text(format!("tick: {}, pos.y: {}", tick, state.pos.y));
                    }

                    // pos.z (dark blue)
                    {
                        let y =
                            rect.center().y - (state.pos.z as f32 / 100.0) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(52, 61, 235))
                            .on_hover_text(format!("tick: {}, pos.z: {}", tick, state.pos.z));
                    }

                    // vel.y (light green)
                    {
                        let y =
                            rect.center().y - (state.vel.y as f32 / 5.0) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(144, 238, 144))
                            .on_hover_text(format!("tick: {}, vel.y: {}", tick, state.vel.y));
                    }

                    // vel.z (light blue)
                    {
                        let y =
                            rect.center().y - (state.vel.z as f32 / 5.0) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(52, 165, 235))
                            .on_hover_text(format!("tick: {}, vel.z: {}", tick, state.vel.z));
                    }

                    let energy_scale = 1.0 / 4.0;
                    // kinetic energy (yellow)
                    {
                        let ke = state.kinetic_energy();
                        let y =
                            rect.center().y - (ke as f32 * energy_scale) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(235, 214, 52))
                            .on_hover_text(format!("tick: {}, kinetic energy: {}", tick, ke));
                    }

                    // potential energy (red)
                    {
                        let pe = state.potential_energy();
                        let y =
                            rect.center().y - (pe as f32 * energy_scale) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(255, 0, 0))
                            .on_hover_text(format!("tick: {}, potential energy: {}", tick, pe));
                    }

                    // total energy (orange)
                    {
                        let energy = state.total_energy();
                        let y = rect.center().y
                            - (energy as f32 * energy_scale) * (rect.height() / 2.0);
                        dot_at(y, 4.0, egui::Color32::from_rgb(235, 143, 52))
                            .on_hover_text(format!("tick: {}, total energy: {}", tick, energy));
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
