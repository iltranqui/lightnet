use std::fs;
use std::path::Path;

use crate::error::Result;

pub fn load_names_file(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}
