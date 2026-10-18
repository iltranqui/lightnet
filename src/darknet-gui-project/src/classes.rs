//! Class list editing with a dry-run plan and an atomic apply (`T5.1`).
//!
//! `apply_remap` validates and stages everything before touching the project:
//! originals are copied to the backup directory, every sidecar is rewritten to a
//! temporary file, and only when all writes succeeded are the temporaries
//! renamed into place. The `.names` file is written last. Any failure restores
//! the originals from the backup.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::project::Project;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassEdit {
    Rename(usize, String),
    Add(String),
    Delete(usize),
    Move { from: usize, to: usize },
}

#[derive(Clone, Debug)]
pub struct RemapPlan {
    pub old_names: Vec<String>,
    pub new_names: Vec<String>,
    /// Old class id -> new class id (`None` removes the class).
    pub mapping: Vec<Option<usize>>,
    /// Per-image marks before/after, for the dry-run table.
    pub counts_before: Vec<usize>,
    pub counts_after: Vec<usize>,
    /// Absolute paths that would be rewritten.
    pub files: Vec<PathBuf>,
}

pub fn plan_remap(project: &Project, edits: &[ClassEdit]) -> Result<RemapPlan, String> {
    let mut names = project.classes.clone();
    // Old class id -> its current position in `names`; `None` once deleted.
    let mut old_to_current: Vec<Option<usize>> = (0..names.len()).map(Some).collect();

    for edit in edits {
        match edit {
            ClassEdit::Rename(index, name) => {
                if *index >= names.len() {
                    return Err(format!("class {index} does not exist"));
                }
                if name.trim().is_empty() {
                    return Err("class name cannot be empty".to_owned());
                }
                if names
                    .iter()
                    .enumerate()
                    .any(|(other, existing)| other != *index && existing == name)
                {
                    return Err(format!("duplicate class name `{name}`"));
                }
                names[*index] = name.clone();
            }
            ClassEdit::Add(name) => {
                if name.trim().is_empty() {
                    return Err("class name cannot be empty".to_owned());
                }
                if names.contains(name) {
                    return Err(format!("duplicate class name `{name}`"));
                }
                names.push(name.clone());
            }
            ClassEdit::Delete(index) => {
                if *index >= names.len() {
                    return Err(format!("class {index} does not exist"));
                }
                names.remove(*index);
                for slot in &mut old_to_current {
                    *slot = match *slot {
                        Some(position) if position == *index => None,
                        Some(position) if position > *index => Some(position - 1),
                        other => other,
                    };
                }
            }
            ClassEdit::Move { from, to } => {
                if *from >= names.len() || *to >= names.len() {
                    return Err("move endpoints must exist".to_owned());
                }
                let name = names.remove(*from);
                names.insert(*to, name);
                let shift = |position: usize| -> usize {
                    if position == *from {
                        *to
                    } else if *from < *to && position > *from && position <= *to {
                        position - 1
                    } else if *to < *from && position >= *to && position < *from {
                        position + 1
                    } else {
                        position
                    }
                };
                for slot in &mut old_to_current {
                    *slot = slot.map(shift);
                }
            }
        }
    }

    // Final ids are the current positions; deleted classes stay `None`.
    let mapping: Vec<Option<usize>> = old_to_current;

    let mut files = Vec::new();
    let mut counts_before = Vec::with_capacity(project.images.len());
    let mut counts_after = Vec::with_capacity(project.images.len());
    let mut touched: BTreeMap<PathBuf, (usize, usize)> = BTreeMap::new();
    for (index, entry) in project.images.iter().enumerate() {
        let annotation = project.load_annotation(index).unwrap_or_default();
        let before = annotation.real_marks().count();
        let after = annotation
            .real_marks()
            .filter(|mark| mapping.get(mark.class_id).copied().flatten().is_some())
            .count();
        counts_before.push(before);
        counts_after.push(after);
        if entry.json_path.is_file() || entry.txt_path.is_file() {
            touched.insert(entry.json_path.clone(), (before, after));
        }
    }
    files.extend(touched.keys().cloned());

    Ok(RemapPlan {
        old_names: project.classes.clone(),
        new_names: names,
        mapping,
        counts_before,
        counts_after,
        files,
    })
}

pub fn apply_remap(project: &Project, plan: &RemapPlan, backup_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(backup_dir).map_err(|error| format!("{}: {error}", backup_dir.display()))?;

    // Stage 1: read and remap every annotation, and back up the originals.
    let mut staged: Vec<(PathBuf, PathBuf, String)> = Vec::new();
    for (index, entry) in project.images.iter().enumerate() {
        let has_json = entry.json_path.is_file();
        let has_txt = entry.txt_path.is_file();
        if !has_json && !has_txt {
            continue;
        }
        let mut annotation = project.load_annotation(index)?;
        let mut marks = Vec::new();
        for mark in annotation.real_marks() {
            if let Some(Some(new_id)) = plan.mapping.get(mark.class_id) {
                let mut mark = mark.clone();
                mark.class_id = *new_id;
                mark.name = plan
                    .new_names
                    .get(*new_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class {new_id}"));
                marks.push(mark);
            }
        }
        annotation.marks = marks;
        annotation.completely_empty = annotation.completely_empty && !annotation.has_real_marks();

        let stem = entry.stem();
        if has_json {
            let backup = backup_dir.join(format!("{stem}.json"));
            fs::copy(&entry.json_path, &backup)
                .map_err(|error| format!("{}: {error}", entry.json_path.display()))?;
            staged.push((entry.json_path.clone(), backup, annotation.to_json()));
        }
        if has_txt {
            let backup = backup_dir.join(format!("{stem}.txt"));
            fs::copy(&entry.txt_path, &backup)
                .map_err(|error| format!("{}: {error}", entry.txt_path.display()))?;
            // The empty-and-unflagged rule also applies to the txt rewrite.
            let body = if annotation.has_real_marks() || annotation.completely_empty {
                annotation.to_yolo()
            } else {
                String::new()
            };
            staged.push((entry.txt_path.clone(), backup, body));
        }
    }

    // Stage 2: write temporaries next to the targets.
    let mut temporaries = Vec::new();
    for (target, _, body) in &staged {
        let temp = target.with_extension(format!(
            "{}.tmp",
            target.extension().and_then(|value| value.to_str()).unwrap_or("tmp")
        ));
        if let Err(error) = fs::write(&temp, body) {
            cleanup(&temporaries);
            return Err(format!("{}: {error}", temp.display()));
        }
        temporaries.push((temp, target.clone()));
    }

    // Stage 3: rename into place; on failure restore every original.
    for (position, (temp, target)) in temporaries.iter().enumerate() {
        if let Err(error) = fs::rename(temp, target) {
            cleanup(&temporaries[position..]);
            for (target, backup, _) in &staged {
                let _ = fs::copy(backup, target);
            }
            return Err(format!("{}: {error}", target.display()));
        }
    }

    // Stage 4: the class list last, so a crash leaves the old names valid.
    let names_path = project.root.join(format!("{}.names", project.name));
    let mut names = String::new();
    for name in &plan.new_names {
        names.push_str(name);
        names.push('\n');
    }
    fs::write(&names_path, names).map_err(|error| format!("{}: {error}", names_path.display()))?;
    Ok(())
}

fn cleanup(temporaries: &[(PathBuf, PathBuf)]) {
    for (temp, _) in temporaries {
        let _ = fs::remove_file(temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::{Annotation, Mark};

    fn temp(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("darknet-gui-classes-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn project_with_marks(name: &str) -> Project {
        let root = temp(name);
    /// A valid 1x1 PNG, so `image::image_dimensions` succeeds.
    fn write_png(path: &std::path::Path) {
        const PNG: [u8; 70] = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99,
            0x3d, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(path, PNG).unwrap();
    }
        write_png(&root.join("a.png"));
        std::fs::write(root.join("a.names"), "zero\none\ntwo\n").unwrap();
        let mut project = Project::load(&root).unwrap();
        let mut annotation = Annotation::new(1, 1);
        annotation.marks = vec![
            Mark::from_corners(0, "zero", 0.0, 0.0, 1.0, 1.0),
            Mark::from_corners(2, "two", 0.0, 0.0, 1.0, 1.0),
        ];
        project.save_annotation(0, &annotation).unwrap();
        project
    }

    #[test]
    fn plan_does_not_touch_disk() {
        let project = project_with_marks("plan");
        let before = std::fs::read(project.images[0].json_path.clone()).unwrap();
        let plan = plan_remap(&project, &[ClassEdit::Delete(1)]).unwrap();
        assert_eq!(plan.new_names, vec!["zero", "two"]);
        assert_eq!(plan.mapping, vec![Some(0), None, Some(1)]);
        assert_eq!(plan.counts_after, vec![2]);
        let after = std::fs::read(project.images[0].json_path.clone()).unwrap();
        assert_eq!(before, after);
        std::fs::remove_dir_all(project.root).unwrap();
    }

    #[test]
    fn delete_class_removes_its_marks_and_shifts_ids() {
        let project = project_with_marks("delete");
        let backup = project.root.join("backup");
        let plan = plan_remap(&project, &[ClassEdit::Delete(0)]).unwrap();
        apply_remap(&project, &plan, &backup).unwrap();
        assert!(backup.join("a.json").is_file());
        let reloaded = Project::load(&project.root).unwrap();
        let annotation = reloaded.load_annotation(0).unwrap();
        assert_eq!(annotation.real_marks().count(), 1);
        assert_eq!(annotation.real_marks().next().unwrap().class_id, 1);
        assert_eq!(reloaded.classes, vec!["one", "two"]);
        std::fs::remove_dir_all(project.root).unwrap();
    }

    #[test]
    fn rename_keeps_mark_ids_and_updates_names() {
        let project = project_with_marks("rename");
        let backup = project.root.join("backup");
        let plan = plan_remap(&project, &[ClassEdit::Rename(0, "zed".into())]).unwrap();
        apply_remap(&project, &plan, &backup).unwrap();
        let reloaded = Project::load(&project.root).unwrap();
        assert_eq!(reloaded.classes, vec!["zed", "one", "two"]);
        assert_eq!(reloaded.mark_count(0), 2);
        std::fs::remove_dir_all(project.root).unwrap();
    }
}
