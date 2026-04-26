mod replay_pitches;
mod sim;

use sim::*;

use crate::replay_pitches::REPLAY_PITCHES;

pub const TICK_DURATION: std::time::Duration = std::time::Duration::from_millis(50); // 20 per second

fn main() -> eframe::Result {
    #[cfg(false)]
    {
        let vel = Vec3 {
            x: 0.,
            y: -0.6,
            z: 2.4,
        };
        let state = State {
            pos: Vec3::ZERO,
            vel,
        };

        let next_plus = state.ticked(Rot { x: 1., y: 0. });
        println!("plus: {:#?}", next_plus.sub(&state));

        let next_zero = state.ticked(Rot { x: 0., y: 0. });
        println!("zero: {:#?}", next_zero.sub(&state));

        let next_minus = state.ticked(Rot { x: -1., y: 0. });
        println!("minus: {:#?}", next_minus.sub(&state));

        panic!();
    }

    let mut grid_width = 100;

    const Y_VEL_LO: f64 = -3.;
    const Y_VEL_HI: f64 = 3.;
    const Z_VEL_LO: f64 = 0.;
    const Z_VEL_HI: f64 = 4.;

    let mut mag_scale = 0.04;
    let mut arrow_scale = 0.9;

    let mut draw_arrow_type = DrawArrowType::OptimalPitch;

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
                        egui::Slider::new(&mut mag_scale, 0.01..=100.0)
                            .clamping(egui::SliderClamping::Never)
                            .logarithmic(true),
                    );

                    ui.label("Arrow Scale");
                    ui.add(
                        egui::Slider::new(&mut arrow_scale, 0.0..=2.0)
                            .clamping(egui::SliderClamping::Never),
                    );

                    // draw_arrow_type
                    ui.label("Draw Arrow Type");
                    egui::ComboBox::from_id_salt("Draw Arrow Type")
                        .selected_text(match draw_arrow_type {
                            DrawArrowType::GlobalPitch => "Global Pitch",
                            DrawArrowType::OptimalPitch => "Optimal Pitch",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut draw_arrow_type,
                                DrawArrowType::GlobalPitch,
                                "Global Pitch",
                            );
                            ui.selectable_value(
                                &mut draw_arrow_type,
                                DrawArrowType::OptimalPitch,
                                "Optimal Pitch",
                            );
                        });
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
                    ui.allocate_ui(egui::vec2(50., 50.), |ui| {
                        ui.painter().arrow(
                            ui.available_rect_before_wrap().left_top(),
                            40. * egui::Vec2::angled(rot.x * std::f32::consts::PI / 180.),
                            (3., egui::Color32::from_rgb(252, 3, 198)),
                        );
                    });
                });
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let mut changed = ui
                            .add(egui::Slider::new(
                                &mut state_index,
                                // TODO: should this be 0..=replay_states.len() - 2,
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

                let color_of_energy = |delta_energy: f64| {
                    // fade to slightly different purples to show off 0
                    if delta_energy >= 0. {
                        egui::Color32::lerp_to_gamma(
                            &egui::Color32::from_rgb(130, 0, 100),
                            egui::Color32::RED,
                            (delta_energy / mag_scale) as f32,
                        )
                    } else {
                        egui::Color32::lerp_to_gamma(
                            &egui::Color32::from_rgb(100, 0, 130),
                            egui::Color32::BLUE,
                            (-delta_energy / mag_scale) as f32,
                        )
                    }
                };

                for x in 0..grid_width {
                    for y in 0..grid_v {
                        let cen = rect.left_top() + egui::vec2(x as f32, y as f32) * step;

                        let init_state = State {
                            pos: Vec3::ZERO,
                            vel: grid_to_vel((x, y)),
                        };

                        let rot_new_state = init_state.ticked(rot);
                        let rot_delta_vel = rot_new_state.vel - init_state.vel;
                        let rot_delta_kinetic =
                            rot_new_state.kinetic_energy() - init_state.kinetic_energy();
                        let rot_delta_potential =
                            rot_new_state.potential_energy() - init_state.potential_energy();
                        let rot_delta_energy =
                            rot_new_state.total_energy() - init_state.total_energy();

                        // stuff for argmax_{pitch} (delta_energy)
                        let optimal_pitch = get_argmax_over_pitch_of_delta_energy(init_state.vel);
                        let optimal_new_state = init_state.ticked(Rot {
                            x: optimal_pitch,
                            y: 0.,
                        });
                        let optimal_delta_vel = optimal_new_state.vel - init_state.vel;
                        let optimal_delta_kinetic =
                            optimal_new_state.kinetic_energy() - init_state.kinetic_energy();
                        let optimal_delta_potential =
                            optimal_new_state.potential_energy() - init_state.potential_energy();
                        let optimal_delta_energy =
                            optimal_new_state.total_energy() - init_state.total_energy();

                        // label energy components

                        match draw_arrow_type {
                            DrawArrowType::GlobalPitch => {
                                // delta vel along global pitch (colored by delta energy)
                                {
                                    let color = color_of_energy(rot_delta_energy);
                                    ui.painter().arrow(
                                        cen,
                                        egui::vec2(rot_delta_vel.z as f32, -rot_delta_vel.y as f32)
                                            .normalized()
                                            * arrow_scale
                                            * step,
                                        egui::Stroke::new(0.2 * step, color),
                                    );
                                }
                            }
                            DrawArrowType::OptimalPitch => {
                                // delta vel along optimal pitch (colored by delta energy)
                                {
                                    let color = color_of_energy(
                                        optimal_new_state.total_energy()
                                            - init_state.total_energy(),
                                    );
                                    ui.painter().arrow(
                                        cen,
                                        egui::vec2(
                                            optimal_delta_vel.z as f32,
                                            -optimal_delta_vel.y as f32,
                                        )
                                        .normalized()
                                            * arrow_scale
                                            * step,
                                        egui::Stroke::new(0.2 * step, color),
                                    );
                                }

                                // optimal pitch (pink)
                                {
                                    let color =
                                        egui::Color32::from_rgba_unmultiplied(252, 3, 198, 150);
                                    ui.painter().arrow(
                                        cen,
                                        egui::Vec2::angled(
                                            optimal_pitch * std::f32::consts::PI / 180.,
                                        ) * arrow_scale
                                            * step,
                                        egui::Stroke::new(0.1 * step, color),
                                    );
                                }
                            }
                        }

                        // show tooltip on hover, toggle clicked cell on click
                        if ui
                            .allocate_rect(
                                egui::Rect::from_center_size(cen, egui::Vec2::splat(step)),
                                egui::Sense::HOVER | egui::Sense::CLICK,
                            )
                            .on_hover_ui(|ui| {
                                ui.group(|ui| {
                                    ui.label(format!("z vel: {:?} bpt", init_state.vel.z));
                                    ui.label(format!("y vel: {:?} bpt", init_state.vel.y));
                                });
                                ui.group(|ui| {
                                    ui.label(format!("pitch: {:?} deg", rot.x));
                                    ui.label(format!("|dv|: {:?}", rot_delta_vel.length()));
                                    ui.label(format!("dk: {:?}", rot_delta_kinetic));
                                    ui.label(format!("dp: {:?}", rot_delta_potential));
                                    ui.label(format!("de: {:?}", rot_delta_energy));
                                });
                                ui.group(|ui| {
                                    ui.label(format!("optimal pitch: {:?} deg", optimal_pitch));
                                    ui.label(format!("|odv|: {:?}", optimal_delta_vel.length()));
                                    ui.label(format!("odk: {:?}", optimal_delta_kinetic));
                                    ui.label(format!("odp: {:?}", optimal_delta_potential));
                                    ui.label(format!("ode: {:?}", optimal_delta_energy));
                                });
                            })
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
                for i in 0..state_index {
                    let state = &replay_states[i];
                    let next = &replay_states[i + 1];

                    // draw dot at state
                    let (x0, y0) = vel_to_grid(state.vel);
                    let start = rect.left_top() + egui::vec2(x0, y0) * step;
                    ui.painter().circle_filled(start, 4., egui::Color32::GOLD);

                    // draw line to next state
                    let (x1, y1) = vel_to_grid(next.vel);
                    let end = rect.left_top() + egui::vec2(x1, y1) * step;
                    ui.painter()
                        .line_segment([start, end], (3., egui::Color32::GOLD));

                    // let state = &replay_states[i];
                    // let (x, y) = vel_to_grid(state.vel);
                    // let start = rect.left_top() + egui::vec2(x, y) * step;
                    // ui.painter().circle_filled(start, 4., egui::Color32::GOLD);
                    // if i < state_index {
                    //     let end_state = &replay_states[i + 1];
                    //     let (x2, y2) = vel_to_grid(end_state.vel);
                    //     let end = rect.left_top() + egui::vec2(x2, y2) * step;
                    //     ui.painter()
                    //         .line_segment([start, end], (3., egui::Color32::GOLD));
                    // }
                    // if i == state_index {
                    //     let a = rot.x * std::f32::consts::PI / 180.;
                    //     ui.allocate_ui(egui::vec2(50., 50.), |ui| {
                    //         ui.painter().arrow(
                    //             start,
                    //             40. * egui::vec2(a.cos(), a.sin()),
                    //             (3., egui::Color32::GRAY),
                    //         );
                    //     });
                    // }
                }

                // at last state draw dot and pitch arrow and delta vel arrow
                {
                    let state = &replay_states[state_index];
                    let (x, y) = vel_to_grid(state.vel);
                    let start = rect.left_top() + egui::vec2(x, y) * step;
                    ui.painter().circle_filled(start, 4., egui::Color32::GOLD);

                    // pitch arrow (pink)
                    ui.painter().arrow(
                        start,
                        40. * egui::Vec2::angled(
                            REPLAY_PITCHES[state_index % REPLAY_PITCHES.len()]
                                * std::f32::consts::PI
                                / 180.,
                        ),
                        (3., egui::Color32::from_rgb(252, 3, 198)),
                    );

                    let vel_scale = 60.;

                    // vel arrow (green)
                    {
                        ui.painter().arrow(
                            start,
                            vel_scale * egui::vec2(state.vel.z as f32, -state.vel.y as f32),
                            (3., egui::Color32::from_rgb(0, 170, 0)),
                        );
                    }

                    // TODO: recolor this, add vel arrow, don't normalize, arrow for (potential, kinetic)
                    // TODO: arrow direction for best energy?

                    // arrow of argmax_over_pitch_of_delta_energy (fancy color)
                    {
                        let best_pitch = get_argmax_over_pitch_of_delta_energy(state.vel);
                        let rot = Rot {
                            x: best_pitch,
                            y: 0.,
                        };
                        let new_state = state.ticked(rot);
                        let delta_vel = new_state.vel - state.vel;
                        let color = egui::Color32::lerp_to_gamma(
                            &color_of_energy(new_state.total_energy() - state.total_energy()),
                            egui::Color32::WHITE,
                            0.5,
                        );
                        ui.painter().arrow(
                            start,
                            vel_scale * 18. * egui::vec2(delta_vel.z as f32, -delta_vel.y as f32),
                            egui::Stroke::new(3., color),
                        );
                    }

                    // delta vel arrow (light green)
                    {
                        let next = state.ticked(Rot {
                            x: REPLAY_PITCHES[state_index % REPLAY_PITCHES.len()],
                            y: 0.,
                        });
                        let delta_vel = next.vel - state.vel;
                        ui.painter().arrow(
                            start,
                            vel_scale * 20. * egui::vec2(delta_vel.z as f32, -delta_vel.y as f32),
                            (3., egui::Color32::from_rgb(100, 238, 100)),
                        );
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

// TODO: longer time horizon
// TODO: flow field which the optimal path is following by definition
fn get_argmax_over_pitch_of_delta_energy(vel: Vec3) -> f32 {
    let mut best_pitch = 0.;
    let mut best_delta_energy = f64::NEG_INFINITY;
    for pitch in -90..=90 {
        let rot = Rot {
            x: pitch as f32,
            y: 0.,
        };
        let state = State {
            pos: Vec3::ZERO,
            vel,
        };
        let new_state = state.ticked(rot);
        let delta_energy = new_state.total_energy() - state.total_energy();
        if delta_energy > best_delta_energy {
            best_delta_energy = delta_energy;
            best_pitch = pitch as f32;
        }
    }
    best_pitch
}

#[derive(Debug, PartialEq)]
enum DrawArrowType {
    GlobalPitch,
    OptimalPitch,
}
