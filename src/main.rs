mod sim;

use egui::vec2;
use sim::*;

pub const TICK_DURATION: std::time::Duration = std::time::Duration::from_millis(50); // 20 per second

fn main() -> eframe::Result {
    let pitches = vec![
        2.2041235,
        2.9893193,
        4.074002,
        5.578016,
        7.6685104,
        10.512784,
        13.433644,
        13.528389,
        13.573162,
        13.582421,
        13.568344,
        13.540985,
        13.508449,
        13.4771595,
        13.452049,
        13.436776,
        13.433888,
        13.445045,
        13.471154,
        13.512545,
        13.569151,
        13.640649,
        13.726605,
        13.858109,
        14.103367,
        14.365466,
        14.642613,
        14.933199,
        15.235708,
        15.548787,
        15.871189,
        16.20175,
        16.539377,
        16.883091,
        17.231901,
        17.584986,
        17.941479,
        18.300648,
        18.661722,
        19.02406,
        19.387075,
        19.750092,
        20.11266,
        20.474201,
        20.83436,
        21.192682,
        21.548737,
        21.902246,
        22.252848,
        22.60035,
        22.944395,
        23.284863,
        23.621557,
        23.954332,
        24.283033,
        24.607628,
        24.927933,
        25.243956,
        25.555662,
        25.863018,
        26.166016,
        26.464724,
        26.75912,
        27.049223,
        27.335136,
        27.616919,
        27.89459,
        28.168287,
        28.43809,
        28.70398,
        28.966185,
        29.224733,
        29.479721,
        29.731274,
        29.979477,
        30.224405,
        30.466183,
        30.704916,
        30.940664,
        31.173552,
        31.403685,
        31.631092,
        31.85592,
        32.078243,
        32.298103,
        32.515667,
        32.730877,
        32.94389,
        33.154808,
        33.363594,
        33.570408,
        33.775215,
        33.978172,
        34.1793,
        34.378593,
        34.576115,
        34.771973,
        34.966194,
        35.158726,
        35.349648,
        35.53905,
        35.72698,
        35.91335,
        36.09826,
        36.281788,
        36.463863,
        36.64453,
        36.823875,
        37.00182,
        37.178444,
        37.353794,
        37.527817,
        37.70055,
        37.87208,
        38.042336,
        38.21134,
        38.379147,
        38.545757,
        38.711098,
        38.875317,
        39.038326,
        39.200207,
        39.360905,
        39.520485,
        39.678894,
        39.836174,
        39.992397,
        40.14746,
        40.301456,
        40.454357,
        40.60614,
        40.756897,
        40.906536,
        41.055122,
        41.202686,
        41.349117,
        41.494583,
        41.638943,
        41.782284,
        41.924587,
        42.06585,
        42.206104,
        42.345264,
        42.48343,
        42.620567,
        42.75665,
        42.89172,
        43.025723,
        43.158756,
        43.290707,
        43.421726,
        43.551704,
        43.68068,
        43.80877,
        43.935925,
        44.062225,
        44.187782,
        44.312637,
        44.437008,
        44.560963,
        44.68482,
        44.808796,
        44.933197,
        45.05854,
        45.185375,
        45.31429,
        45.44614,
        45.581997,
        45.72013,
        45.856422,
        45.99768,
        46.14178,
        46.27799,
        46.390766,
        46.49076,
        46.589626,
        46.68601,
        46.778362,
        46.86499,
        46.94406,
        47.01364,
        47.071777,
        47.11652,
        47.146008,
        47.158558,
        47.152737,
        47.127506,
        47.082355,
        47.017467,
        46.93391,
        46.43348,
        45.47536,
        44.13551,
        42.29089,
        39.79308,
        1.5233016,
        1.1420771,
        0.72004724,
        0.4557087,
        0.365286,
        0.28655583,
        0.22103214,
        0.16957241,
        0.13287868,
        0.11149618,
        0.10455809,
        0.103193514,
        0.07538817,
        -17.628214,
        -36.381187,
        -64.5585,
        -82.631546,
        -82.625275,
        -82.55319,
        -79.43314,
        -75.416595,
        -71.351906,
        -67.56552,
        -64.19418,
        -61.236485,
        -58.633514,
        -56.318954,
        -54.236233,
        -52.34138,
        -50.60108,
        -48.990517,
        -47.49101,
        -46.087605,
        -44.768566,
        -43.524544,
        -42.347637,
        -41.231358,
        -40.170036,
        -39.159103,
        -38.19427,
        -37.271732,
        -36.388557,
        -35.541786,
        -34.728996,
        -33.947807,
        -33.197018,
        -32.473717,
        -31.776304,
        -31.103443,
        -30.453566,
        -29.825502,
        -29.217987,
        -28.629969,
        -28.060482,
        -27.508501,
        -26.973194,
        -26.453753,
        -25.94946,
        -25.459549,
        -24.983358,
        -24.520266,
        -24.069754,
        -23.631186,
        -23.204111,
        -22.788,
        -22.382378,
        -21.986855,
        -21.600971,
        -21.22437,
        -20.856604,
        -20.497389,
        -20.146328,
        -19.803099,
        -19.467426,
        -19.138927,
        -18.817366,
        -18.502365,
        -18.19373,
        -17.891094,
        -17.59425,
        -17.302841,
        -17.016619,
        -16.735308,
        -16.458517,
        -16.185974,
        -15.917335,
        -15.652224,
        -15.390199,
        -15.130762,
        -14.873377,
        -14.617364,
        -14.3619175,
        -14.106107,
        -13.848643,
        -13.587981,
        -13.322128,
        -13.048467,
        -12.76348,
        -12.462531,
        -12.139348,
        -11.785178,
        -11.388125,
        -10.931468,
        -10.391628,
        -9.735042,
    ];
    let mut state_index: usize = 0;

    let mut replay_states = vec![State { pos: Vec3 { x: 0., y: 0., z: 0. }, vel: Vec3 { x: 0., y: 0.167467, z: 0.200887 } }];
    for &p in &pitches {
        let state = replay_states.last().expect("wtf");
        replay_states.push(state.ticked(Rot { x: p, y: 0. }));
    }
    let replay_states = replay_states;

    let mut base_vel = (0,0);

    let mut rot = Rot::new(0., 0.);


    let mut mag_scale = 1.;
    let mut arrow_length = 5.;

    const GRID_H: usize = 200;
    const GRID_V: usize = 150;

    let h_factor = 6.;
    let v_factor = h_factor * GRID_H as f64/GRID_V as f64;

    let grid_to_vel = move |(x,y): (usize, usize)| {
        (h_factor * (x as f64 / GRID_H as f64), v_factor * (0.5 - (y as f64 / GRID_V as f64)))
    };
    let vel_to_grid = move |(z,y): (f64, f64)| {
        ((z * (GRID_H as f64) / h_factor), (y - v_factor / 2.) * (GRID_V as f64) / -v_factor)
    };
    let mut grid_states: Vec<State> = vec![Entity::default().into(); GRID_H * GRID_V];
    for x in 0..GRID_H {
        for y in 0..GRID_V {
            let s = &mut grid_states[x+GRID_H*y];
            (s.vel.z, s.vel.y) = grid_to_vel((x,y));
        }
    }

    let mut tick = false;

    eframe::run_ui_native(
        "Elytra Sim",
        eframe::NativeOptions::default(),
        move |ui, _frame| {

            egui::Panel::left("left").show_inside(ui, |ui| {
                tick = false;
                ui.group(|ui| {
                    ui.label("Mag Scale");
                    ui.add(
                        egui::Slider::new(&mut mag_scale, 0.0..=100.0).clamping(egui::SliderClamping::Never)
                    );
                    ui.label("Arrow length");
                    ui.add(
                        egui::Slider::new(&mut arrow_length, 0.0..=20.0)
                    );
                });
                ui.group(|ui| {
                    ui.strong("Rotation");
                    ui.label("Pitch");

                    ui.add(
                        egui::Slider::new(&mut rot.x, -90.0..=90.0)
                        .clamping(egui::SliderClamping::Never)
                    );
                    ui.label("Yaw");
                    ui.add(
                        egui::Slider::new(&mut rot.y, -90.0..=90.0)
                        .clamping(egui::SliderClamping::Never)
                    );
                    let a = rot.x * std::f32::consts::PI / 180.;
                    ui.allocate_ui(vec2(50.,50.), |ui| {
                        ui.painter().arrow(ui.available_rect_before_wrap().left_top(), 40. * vec2(a.cos(), a.sin()), (3.,egui::Color32::GRAY));
                    } );
                });
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let mut changed = ui.add(egui::Slider::new(&mut state_index, 0..=replay_states.len()-1)).changed();
                        if ui.button("-").clicked() {
                            state_index = state_index.saturating_sub(1);
                            changed = true;
                        }
                        if ui.button("+").clicked() {
                            state_index = std::cmp::min(state_index+1, replay_states.len()-1);
                            changed = true;
                        }
                        if changed {
                            rot.x = pitches[state_index % pitches.len()];
                        }
                    })
                })
            });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let egui::Vec2{x: width,y: height} = rect.size();


                let stepx = width/GRID_H as f32;
                let stepy = height/GRID_V as f32;

                let step = f32::min(stepx,stepy);

                for x in 0..GRID_H {
                    for y in 0..GRID_V {
                        let init_state = &grid_states[x + GRID_H*y];

                        let new_state = init_state.ticked(rot);

                        let dv = new_state.vel - init_state.vel;
                        let de = new_state.total_energy() - init_state.total_energy();

                        let norm = dv.length_sq().sqrt();
                        let mag = norm / de;
                        let vec = vec2((dv.z/norm) as f32,-(dv.y/norm) as f32);

                        let cen = rect.left_top() + egui::vec2(x as f32 * step, y as f32 * step);

                        let col = if mag >= 0. {
                            egui::Color32::lerp_to_gamma(&egui::Color32::PURPLE, egui::Color32::RED, (mag / mag_scale) as f32)
                        } else {
                            egui::Color32::lerp_to_gamma(&egui::Color32::PURPLE, egui::Color32::BLUE, (-mag / mag_scale) as f32)
                        };
                        ui.painter().arrow(cen, vec * arrow_length, egui::Stroke::new(2., col));
                        let r = ui.allocate_rect(egui::Rect::from_center_size(cen, egui::Vec2::splat(step)), egui::Sense::HOVER | egui::Sense::CLICK);
                        if r.on_hover_text(format!("zVel: {:?}bpt\nyVel: {:?}bpt\ndv/de: {:?}", init_state.vel.z, init_state.vel.y, mag)).clicked() {
                            base_vel = (x,y);
                        }
                    }
                }
                // let (x,y) = base_vel;
                // let mut start = rect.left_top() + egui::vec2(x as f32 * step, y as f32 * step);
                // ui.painter().circle_filled(start, 4., egui::Color32::GOLD);
                // let mut state = states[x + GRID_H*y].clone();
                // for _ in 0..10 {
                //     state = state.ticked(rot);
                //     let (x,y) = vel_to_grid((state.vel.z, state.vel.y));
                //     let end = rect.left_top() + egui::vec2(
                //         x as f32 * step,
                //         y as f32 * step
                //     );
                //     ui.painter().line_segment([start, end], (3., egui::Color32::GOLD));
                //     ui.painter().circle_filled(end, 4., egui::Color32::GOLD);
                //     start = end;
                // }

                for i in 0..=state_index {
                    let state = &replay_states[i];
                    let (x,y) = vel_to_grid((state.vel.z,state.vel.y));
                    let start = rect.left_top() + egui::vec2(x as f32 * step, y as f32 * step);
                    ui.painter().circle_filled(start, 4., egui::Color32::GOLD);
                    if i < state_index {
                        let end_state = &replay_states[i+1];
                        let (x2,y2) = vel_to_grid((end_state.vel.z,end_state.vel.y));
                        let end = rect.left_top() + egui::vec2(
                            x2 as f32 * step,
                            y2 as f32 * step
                        );
                        ui.painter().line_segment([start, end], (3., egui::Color32::GOLD));
                    }
                    if i == state_index {
                        let a = rot.x * std::f32::consts::PI / 180.;
                        ui.allocate_ui(vec2(50.,50.), |ui| {
                            ui.painter().arrow(start, 40. * vec2(a.cos(), a.sin()), (3.,egui::Color32::GRAY));
                        } );
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


