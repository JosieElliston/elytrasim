mod replay_pitches;
mod sim;

use egui;
use sim::*;

use crate::replay_pitches::REPLAY_PITCHES;

pub const TICK_DURATION: std::time::Duration = std::time::Duration::from_millis(50); // 20 per second

fn main() -> eframe::Result {
    let mut grid_width = 100;

    const Y_VEL_LO: f64 = -3.;
    const Y_VEL_HI: f64 = 3.;
    const Z_VEL_LO: f64 = 0.;
    const Z_VEL_HI: f64 = 4.;

    let mut mag_scale = 8.;
    let mut arrow_scale = 0.9;

    let mut rot = Rot::new(0., 0.);

    let mut clicked_cell = None;

    let mut state_index: usize = 0;
    let replay_states = {
        let mut replay_states = vec![State {
            pos: Vec3 {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            vel: Vec3 {
                x: 0.,
                y: 0.167467,
                z: 0.200887,
            },
        }];
        for p in REPLAY_PITCHES {
            let state = replay_states.last().expect("wtf");
            replay_states.push(state.ticked(Rot { x: *p, y: 0. }));
        }
        replay_states
    };

    eframe::run_ui_native(
        "Elytra Sim",
        eframe::NativeOptions::default(),
        move |ui, _frame| {
            egui::Panel::left("left").show_inside(ui, |ui| {
                ui.group(|ui| {
                    ui.label("Grid Width");
                    ui.add(
                        egui::Slider::new(&mut grid_width, 1..=500)
                            .clamping(egui::SliderClamping::Never),
                    );

                    ui.label("Mag Scale");
                    ui.add(
                        egui::Slider::new(&mut mag_scale, 0.1..=100.0)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true),
                    );

                    ui.label("Arrow Scale");
                    ui.add(
                        egui::Slider::new(&mut arrow_scale, 0.0..=2.0)
                            .clamping(egui::SliderClamping::Never),
                    );
                });
                ui.group(|ui| {
                    ui.strong("Rotation");
                    ui.label("Pitch");

                    ui.add(
                        egui::Slider::new(&mut rot.x, -90.0..=90.0)
                            .clamping(egui::SliderClamping::Never),
                    );
                    ui.label("Yaw");
                    ui.add(
                        egui::Slider::new(&mut rot.y, -90.0..=90.0)
                            .clamping(egui::SliderClamping::Never),
                    );
                    let a = rot.x * std::f32::consts::PI / 180.;
                    ui.allocate_ui(egui::vec2(50., 50.), |ui| {
                        ui.painter().arrow(
                            ui.available_rect_before_wrap().left_top(),
                            40. * egui::vec2(a.cos(), a.sin()),
                            (3., egui::Color32::GRAY),
                        );
                    });
                });
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let mut changed = ui
                            .add(egui::Slider::new(
                                &mut state_index,
                                0..=replay_states.len() - 1,
                            ))
                            .changed();
                        if ui.button("-").clicked() {
                            state_index = state_index.saturating_sub(1);
                            changed = true;
                        }
                        if ui.button("+").clicked() {
                            state_index = std::cmp::min(state_index + 1, replay_states.len() - 1);
                            changed = true;
                        }
                        if changed {
                            rot.x = REPLAY_PITCHES[state_index % REPLAY_PITCHES.len()];
                        }
                    })
                })
            });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                let rect = ui.available_rect_before_wrap();

                let grid_to_vel = |(x, y): (usize, usize)| Vec3 {
                    x: 0.,
                    y: lerp_f64(
                        Y_VEL_LO,
                        Y_VEL_HI,
                        1.0 - inv_lerp_f64(
                            0.,
                            grid_width as f64 * (rect.height() / rect.width()) as f64,
                            y as f64,
                        ),
                    ),
                    z: lerp_f64(
                        Z_VEL_LO,
                        Z_VEL_HI,
                        inv_lerp_f64(0., grid_width as f64, x as f64),
                    ),
                };

                let vel_to_grid = move |vel: Vec3| {
                    assert_eq!(vel.x, 0., "not a hard error, but probably should have this");
                    (
                        lerp_f64(
                            0.,
                            grid_width as f64,
                            inv_lerp_f64(Z_VEL_LO, Z_VEL_HI, vel.z),
                        ) as f32,
                        lerp_f64(
                            0.,
                            grid_width as f64 * (rect.height() / rect.width()) as f64,
                            1.0 - inv_lerp_f64(Y_VEL_LO, Y_VEL_HI, vel.y),
                        ) as f32,
                    )
                };

                let grid_v = (grid_width as f32 * rect.height() / rect.width()) as usize;

                let step = rect.width() / grid_width as f32;

                for x in 0..grid_width {
                    for y in 0..grid_v {
                        let init_state = State {
                            pos: Vec3::ZERO,
                            vel: grid_to_vel((x, y)),
                        };

                        let new_state = init_state.ticked(rot);

                        let dv = new_state.vel - init_state.vel;
                        let de = new_state.total_energy() - init_state.total_energy();

                        let norm = dv.length();
                        let mag = norm / de;
                        let vec = egui::vec2(dv.z as f32, -dv.y as f32) / norm as f32;

                        let cen = rect.left_top() + egui::vec2(x as f32, y as f32) * step;

                        // draw arrow
                        {
                            let color = if mag >= 0. {
                                egui::Color32::lerp_to_gamma(
                                    &egui::Color32::PURPLE,
                                    egui::Color32::RED,
                                    (mag / mag_scale) as f32,
                                )
                            } else {
                                egui::Color32::lerp_to_gamma(
                                    &egui::Color32::PURPLE,
                                    egui::Color32::BLUE,
                                    (-mag / mag_scale) as f32,
                                )
                            };
                            ui.painter().arrow(
                                cen,
                                vec * arrow_scale * step,
                                egui::Stroke::new(0.2 * step, color),
                            );
                        }

                        // toggle clicked cell on click, show tooltip on hover
                        if ui
                            .allocate_rect(
                                egui::Rect::from_center_size(cen, egui::Vec2::splat(step)),
                                egui::Sense::HOVER | egui::Sense::CLICK,
                            )
                            .on_hover_text(format!(
                                "zVel: {:?} bpt\nyVel: {:?} bpt\ndv/de: {:?}",
                                init_state.vel.z, init_state.vel.y, mag
                            ))
                            .clicked()
                        {
                            if clicked_cell == Some((x, y)) {
                                clicked_cell = None;
                            } else {
                                clicked_cell = Some((x, y));
                            }
                        }
                    }
                }

                // draw the path from the clicked cell
                if let Some((x, y)) = clicked_cell {
                    let mut start = rect.left_top() + egui::vec2(x as f32, y as f32) * step;
                    ui.painter().circle_filled(start, 4., egui::Color32::GOLD);
                    let mut state = State {
                        pos: Vec3::ZERO,
                        vel: grid_to_vel((x, y)),
                    };
                    const PATH_LEN: usize = 10;
                    for _ in 0..PATH_LEN {
                        state = state.ticked(rot);
                        let (x, y) = vel_to_grid(state.vel);
                        let end = rect.left_top() + egui::vec2(x, y) * step;
                        ui.painter()
                            .line_segment([start, end], (3., egui::Color32::GOLD));
                        ui.painter().circle_filled(end, 4., egui::Color32::GOLD);
                        start = end;
                    }
                }

                // replay path
                for i in 0..=state_index {
                    let state = &replay_states[i];
                    let (x, y) = vel_to_grid(state.vel);
                    let start = rect.left_top() + egui::vec2(x, y) * step;
                    ui.painter().circle_filled(start, 4., egui::Color32::GOLD);
                    if i < state_index {
                        let end_state = &replay_states[i + 1];
                        let (x2, y2) = vel_to_grid(end_state.vel);
                        let end = rect.left_top() + egui::vec2(x2, y2) * step;
                        ui.painter()
                            .line_segment([start, end], (3., egui::Color32::GOLD));
                    }
                    if i == state_index {
                        let a = rot.x * std::f32::consts::PI / 180.;
                        ui.allocate_ui(egui::vec2(50., 50.), |ui| {
                            ui.painter().arrow(
                                start,
                                40. * egui::vec2(a.cos(), a.sin()),
                                (3., egui::Color32::GRAY),
                            );
                        });
                    }
                }
            });
        },
    )
}

pub fn pos_slider(value: &mut f64) -> egui::Slider<'_> {
    egui::Slider::new(value, -100.0..=100.0).clamping(egui::SliderClamping::Never)
}

pub fn vel_slider(value: &mut f64) -> egui::Slider<'_> {
    egui::Slider::new(value, -5.0..=5.0).clamping(egui::SliderClamping::Never)
}

pub fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn inv_lerp_f32(a: f32, b: f32, v: f32) -> f32 {
    (v - a) / (b - a)
}

pub fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    assert!((0.0..=1.0).contains(&t));
    a + (b - a) * t
}

pub fn inv_lerp_f64(a: f64, b: f64, v: f64) -> f64 {
    assert!((a..=b).contains(&v));
    (v - a) / (b - a)
}
