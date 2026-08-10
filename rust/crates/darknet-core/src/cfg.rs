//! Darknet `.cfg` parsing.
//!
//! Darknet configuration files are a sequence of named sections followed by
//! `key=value` pairs.  This parser preserves the compatibility details that
//! matter for existing configs: case-insensitive names, comma or whitespace
//! separated lists, and `#`/`;` inline comments.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawSection {
    pub name: String,
    pub line: usize,
    pub values: BTreeMap<String, String>,
}

impl RawSection {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn get_str(&self, key: &str, default: &str) -> String {
        self.get(key).unwrap_or(default).trim().to_owned()
    }

    pub fn get_usize(&self, key: &str, default: usize) -> Result<usize> {
        match self.get(key) {
            Some(value) => value.trim().parse::<usize>().map_err(|_| {
                Error::Parse(format!(
                    "expected an unsigned integer for key \"{key}\" in [{}] at line {}",
                    self.name, self.line
                ))
            }),
            None => Ok(default),
        }
    }

    pub fn get_isize(&self, key: &str, default: isize) -> Result<isize> {
        match self.get(key) {
            Some(value) => value.trim().parse::<isize>().map_err(|_| {
                Error::Parse(format!(
                    "expected an integer for key \"{key}\" in [{}] at line {}",
                    self.name, self.line
                ))
            }),
            None => Ok(default),
        }
    }

    pub fn get_i64(&self, key: &str, default: i64) -> Result<i64> {
        match self.get(key) {
            Some(value) => value.trim().parse::<i64>().map_err(|_| {
                Error::Parse(format!(
                    "expected an integer for key \"{key}\" in [{}] at line {}",
                    self.name, self.line
                ))
            }),
            None => Ok(default),
        }
    }

    pub fn get_f32(&self, key: &str, default: f32) -> Result<f32> {
        match self.get(key) {
            Some(value) => value.trim().parse::<f32>().map_err(|_| {
                Error::Parse(format!(
                    "expected a float for key \"{key}\" in [{}] at line {}",
                    self.name, self.line
                ))
            }),
            None => Ok(default),
        }
    }

    pub fn get_bool_int(&self, key: &str, default: bool) -> Result<bool> {
        Ok(self.get_i64(key, default as i64)? != 0)
    }

    pub fn get_isize_list(&self, key: &str) -> Result<Vec<isize>> {
        parse_list(self.get(key), self.name.as_str(), self.line, key)
    }

    pub fn get_usize_list(&self, key: &str) -> Result<Vec<usize>> {
        parse_list::<usize>(self.get(key), self.name.as_str(), self.line, key)
    }

    pub fn get_f32_list(&self, key: &str) -> Result<Vec<f32>> {
        parse_list::<f32>(self.get(key), self.name.as_str(), self.line, key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigFile {
    pub path: PathBuf,
    pub sections: Vec<RawSection>,
}

impl ConfigFile {
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        Self::parse_str(path, &text)
    }

    pub fn parse_str(path: impl AsRef<Path>, text: &str) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut sections = Vec::new();
        let mut current: Option<RawSection> = None;

        for (idx, raw_line) in text.lines().enumerate() {
            // Remove comments before interpreting a line.  Darknet accepts a
            // comment after a value, not only on a line by itself.
            let line_number = idx + 1;
            let line = raw_line
                .split_once('#')
                .map_or(raw_line, |(value, _)| value);
            let line = line.split_once(';').map_or(line, |(value, _)| value).trim();

            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }

            if line.starts_with('[') && line.ends_with(']') {
                if let Some(section) = current.take() {
                    sections.push(section);
                }

                current = Some(RawSection {
                    name: line[1..line.len() - 1].trim().to_ascii_lowercase(),
                    line: line_number,
                    values: BTreeMap::new(),
                });
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                return Err(Error::Parse(format!(
                    "invalid config line {} in {}: expected key=value",
                    line_number,
                    path.display()
                )));
            };

            let Some(section) = current.as_mut() else {
                return Err(Error::Parse(format!(
                    "encountered key/value before any section at line {} in {}",
                    line_number,
                    path.display()
                )));
            };

            section
                .values
                .insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
        }

        if let Some(section) = current.take() {
            sections.push(section);
        }

        if sections.is_empty() {
            return Err(Error::Parse(format!(
                "no Darknet sections found in {}",
                path.display()
            )));
        }

        Ok(Self { path, sections })
    }
}

fn parse_list<T>(raw: Option<&str>, section_name: &str, line: usize, key: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
{
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };

    raw.split(|character: char| character == ',' || character.is_ascii_whitespace())
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.parse::<T>().map_err(|_| {
                Error::Parse(format!(
                    "failed to parse list item \"{item}\" for key \"{key}\" in [{}] at line {}",
                    section_name, line
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::ConfigFile;

    #[test]
    fn parses_basic_darknet_cfg() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=416
                height=416
                channels=3

                [convolutional]
                filters=16
                size=3
                stride=1
                pad=1
                activation=leaky
                # an ordinary comment
                route_hint=1 ### an inline comment
            "#,
        )
        .expect("cfg should parse");

        assert_eq!(cfg.sections.len(), 2);
        assert_eq!(cfg.sections[0].name, "net");
        assert_eq!(cfg.sections[1].get("filters"), Some("16"));
    }
}
