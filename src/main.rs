fn main() -> eframe::Result {
    eframe::run_ui_native(
        "Eltyra Sim",
        eframe::NativeOptions::default(),
        |ui, _frame| {
            ui.label("Hello, world!");
        },
    )
}
