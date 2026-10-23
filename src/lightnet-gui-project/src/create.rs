//! Project creation with DarkMark's on-disk layout (`G1`).
//!
//! A project is a directory; the name is the directory name. The created files
//! are `<name>.names` and `<name>.data`, so `Project::load` finds them first
//! (`D5`).

use std::fs;
use std::path::{Path, PathBuf};

use crate::project::Project;

/// Refuses a non-empty directory, an existing project file, or empty/duplicate
/// class names.
pub fn create_project(dir: &Path, class_names: &[String]) -> Result<Project, String> {
    if class_names.is_empty() {
        return Err("a project needs at least one class".to_owned());
    }
    for (index, name) in class_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(format!("class {index} has an empty name"));
        }
        if class_names[..index].contains(name) {
            return Err(format!("duplicate class name `{name}`"));
        }
    }

    fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let mut existing = fs::read_dir(dir).map_err(|error| error.to_string())?;
    if existing.next().is_some() {
        return Err(format!("{} is not empty", dir.display()));
    }

    let name = dir
        .canonicalize()
        .map_err(|error| format!("{}: {error}", dir.display()))?
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| "project directory needs a name".to_owned())?;

    let names_path = dir.join(format!("{name}.names"));
    let data_path = dir.join(format!("{name}.data"));
    let mut names = String::new();
    for class in class_names {
        names.push_str(class.trim());
        names.push('\n');
    }
    let data = format!(
        "classes = {}\ntrain = {name}_train.txt\nvalid = {name}_valid.txt\nnames = {name}.names\nbackup = backup\n",
        class_names.len()
    );
    fs::write(&names_path, names).map_err(|error| format!("{}: {error}", names_path.display()))?;
    fs::write(&data_path, data).map_err(|error| format!("{}: {error}", data_path.display()))?;

    // An empty project cannot be loaded (no images), so validate the files
    // directly rather than through `Project::load`.
    Ok(Project {
        root: dir.to_path_buf(),
        name,
        images: Vec::new(),
        classes: class_names.to_vec(),
        data_file: Some(data_path),
        cfg_file: None,
        weights_file: None,
        mark_counts: Vec::new(),
    })
}

/// Paths a new project writes, for the launcher's confirmation text.
pub fn planned_files(dir: &Path) -> (PathBuf, PathBuf) {
    let name = dir
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "project".to_owned());
    (dir.join(format!("{name}.names")), dir.join(format!("{name}.data")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("lightnet-gui-create-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn creates_files_and_refuses_a_second_project() {
        let root = temp("basic");
        let project = create_project(&root, &["person".into(), "car".into()]).unwrap();
        assert_eq!(project.classes, vec!["person", "car"]);
        assert!(root.join(format!("{}.names", project.name)).is_file());
        assert!(root.join(format!("{}.data", project.name)).is_file());
        assert!(create_project(&root, &["person".into()]).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_empty_and_duplicate_class_names() {
        let root = temp("names");
        assert!(create_project(&root, &[]).is_err());
        assert!(create_project(&root, &["a".into(), "a".into()]).is_err());
        assert!(create_project(&root, &["  ".into()]).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
