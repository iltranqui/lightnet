//! Right-click menu: the replacement for DarkMark's View/Marks menus and for
//! the old left panel and top bar.
//!
//! DarkMark puts classes, sort, filter, view, labels, predictions and the
//! windows in the right-click popup (`DMContent.cpp:4100-4160`; the layout plan
//! lists the required items). Every entry either calls the same [`Action`] the
//! keyboard uses or opens a window, so no feature exists only here.

use eframe::egui::{self, Ui};
use lightnet_gui_project::filter::SortKey;
use lightnet_gui_project::hotkeys::Action;

use crate::{App, View};

/// Draws the class list, the display toggles, the editing actions and the
/// window shortcuts. The caller owns the popup (`Response::context_menu`).
pub fn menu(ui: &mut Ui, app: &mut App) {
    classes(ui, app);
    ui.separator();
    display(ui, app);
    ui.separator();
    editing(ui, app);
    ui.separator();
    if ui.button("Filter…").clicked() {
        app.show_filter = true;
        ui.close();
    }
    if ui.button("Jump…").clicked() {
        app.show_jump = true;
        ui.close();
    }
    if ui.button("Class editor…").clicked() {
        app.open_class_editor();
        ui.close();
    }
    if ui.button("Settings…").clicked() {
        app.show_settings = true;
        ui.close();
    }
    if ui.button("Load model…").clicked() {
        app.show_model = true;
        ui.close();
    }
    if ui.button("Projects…").clicked() {
        app.show_launcher = true;
        ui.close();
    }
    if ui.button("Hotkeys (F2)").clicked() {
        app.show_help = true;
        ui.close();
    }
}

/// Classes as DarkMark's popup lists them; clicking one reclassifies the
/// selection exactly like a digit key (`Action::SetClass`).
fn classes(ui: &mut Ui, app: &mut App) {
    let Some(editor) = app.editor.as_ref() else {
        ui.label("No image open");
        return;
    };
    let current = editor.class_id;
    let names: Vec<(usize, String)> = (0..editor.class_count)
        .map(|id| (id, editor.class_name(id)))
        .collect();
    ui.menu_button("Classes", |ui| {
        egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
            for (id, name) in names {
                let label = format!("{id}: {name}");
                if ui.selectable_label(current == id, label).clicked() {
                    app.run_action(Action::SetClass(id));
                    ui.close();
                }
            }
        });
    });
}

fn display(ui: &mut Ui, app: &mut App) {
    ui.menu_button("Sort", |ui| {
        if ui.selectable_label(app.sort == SortKey::Name, "alphabetical").clicked() {
            app.run_action(Action::SortAlphabetical);
            ui.close();
        }
        if ui.selectable_label(app.sort == SortKey::MarkCount, "mark count").clicked() {
            app.sort = SortKey::MarkCount;
            app.random_order = None;
            ui.close();
        }
        if ui.selectable_label(app.random_order.is_some(), "random").clicked() {
            app.run_action(Action::SortRandom);
            ui.close();
        }
    });
    ui.menu_button("View", |ui| {
        for view in [
            View::Normal,
            View::BlackWhite,
            View::Canny,
            View::Dot,
            View::Bold,
            View::Shade,
        ] {
            if ui.selectable_label(app.view == view, view.label()).clicked() {
                app.view = view;
                ui.close();
            }
        }
    });
    ui.menu_button("Labels", |ui| {
        for (mode, name) in [(0u8, "off"), (1, "on"), (2, "auto")] {
            if ui.selectable_label(app.label_mode == mode, name).clicked() {
                app.set_label_mode(mode);
                ui.close();
            }
        }
    });
    ui.menu_button("Predictions", |ui| {
        for (mode, name) in [(0u8, "off"), (1, "on"), (2, "auto")] {
            if ui.selectable_label(app.prediction_mode == mode, name).clicked() {
                app.run_action(Action::SetPredictions(mode));
                ui.close();
            }
        }
    });
    let marks = if app.show_marks { "Hide user marks" } else { "Show user marks" };
    if ui.button(marks).clicked() {
        app.run_action(Action::ToggleMarks);
        ui.close();
    }
}

fn editing(ui: &mut Ui, app: &mut App) {
    let dirty = app.dirty;
    if ui.button(if dirty { "Save *" } else { "Save" }).clicked() {
        app.run_action(Action::Save);
        ui.close();
    }
    if ui.button("Accept all predictions").clicked() {
        app.run_action(Action::AcceptAll);
        ui.close();
    }
    if ui.button("Accept selected prediction").clicked() {
        app.run_action(Action::AcceptSelected);
        ui.close();
    }
    if ui.button("Undo").clicked() {
        app.run_action(Action::Undo);
        ui.close();
    }
    if ui.button("Redo").clicked() {
        app.run_action(Action::Redo);
        ui.close();
    }
}
