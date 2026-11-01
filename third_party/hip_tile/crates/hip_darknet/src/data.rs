//! Dataset plumbing: the `.data` file, image lists, and Darknet label files.
//!
//! Label semantics: `_shared/darknet-semantics.md` (Training data and augmentation) —
//! one `class cx cy w h` per line, all coordinates normalized to the image.

use crate::error::{Error, Result};
use crate::image::Image;
use std::path::{Path, PathBuf};

/// One annotated object, normalized like Darknet's `box_label`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxLabel {
    pub class: usize,
    /// Center and size as fractions of the image.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// One dataset item: an image file and its labels.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub image: PathBuf,
    pub labels: Vec<BoxLabel>,
}

/// A parsed Darknet `.data` file plus the lists it points at.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub classes: usize,
    /// `.names` contents (empty when the `.data` file has no `names` key).
    pub names: Vec<String>,
    pub train: Vec<Sample>,
    pub valid: Vec<Sample>,
    /// The `.data` file's own directory, used to resolve relative paths.
    pub root: PathBuf,
}

/// Raw `.data` key/values, before the lists are read.
#[derive(Debug, Clone, Default)]
pub struct DataFile {
    pub classes: usize,
    pub train: Option<PathBuf>,
    pub valid: Option<PathBuf>,
    pub names: Option<PathBuf>,
    pub backup: Option<PathBuf>,
}

impl DataFile {
    pub fn parse(text: &str) -> Result<DataFile> {
        let mut d = DataFile::default();
        for (i, raw) in text.lines().enumerate() {
            let t = raw.trim();
            if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
                continue;
            }
            let (k, v) = t
                .split_once('=')
                .ok_or(Error::Cfg { line: i + 1, msg: format!("expected `key=value`, got `{t}`") })?;
            let v = v.trim();
            match k.trim() {
                "classes" => {
                    d.classes = v
                        .parse()
                        .map_err(|_| Error::Cfg { line: i + 1, msg: format!("`classes` is not an integer: `{v}`") })?
                }
                "train" => d.train = Some(PathBuf::from(v)),
                "valid" => d.valid = Some(PathBuf::from(v)),
                "names" => d.names = Some(PathBuf::from(v)),
                "backup" => d.backup = Some(PathBuf::from(v)),
                // `eval`, `top`, `labels`, `results`: not used by training.
                _ => {}
            }
        }
        if d.classes == 0 {
            return Err(Error::Model("`.data` has no `classes`".into()));
        }
        Ok(d)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<DataFile> {
        let p = path.as_ref();
        let text = read_text(p)?;
        DataFile::parse(&text)
    }
}

fn read_text(p: &Path) -> Result<String> {
    std::fs::read_to_string(p).map_err(|source| Error::Io { path: p.display().to_string(), source })
}

/// Resolve `p` against `root` when it is relative (Darknet resolves against the cwd; ours is
/// the `.data` file's directory, which is what the DarkSuite datasets assume).
fn resolve(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() { p.to_path_buf() } else { root.join(p) }
}

/// Darknet's `replace_image_to_label`: same path with a `.txt` extension.
pub fn label_path(image: &Path) -> PathBuf {
    image.with_extension("txt")
}

/// Read one label file. Empty lines are skipped; anything Darknet treats as fatal is an error.
pub fn read_labels(path: &Path, classes: usize) -> Result<Vec<BoxLabel>> {
    let text = read_text(path)?;
    let mut out = vec![];
    for (i, raw) in text.lines().enumerate() {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        let mut it = t.split_whitespace();
        let mut next = || -> Result<f32> {
            it.next()
                .ok_or_else(|| Error::Model(format!("{}:{}: short label line `{t}`", path.display(), i + 1)))?
                .parse::<f32>()
                .map_err(|_| Error::Model(format!("{}:{}: bad number in `{t}`", path.display(), i + 1)))
        };
        let class = next()?;
        let (x, y, w, h) = (next()?, next()?, next()?, next()?);
        if class < 0.0 || class.fract() != 0.0 || class as usize >= classes {
            return Err(Error::Model(format!(
                "{}:{}: class {} outside 0..{classes}",
                path.display(),
                i + 1,
                class
            )));
        }
        if !(x > 0.0 && x <= 1.0 && y > 0.0 && y <= 1.0) {
            return Err(Error::Model(format!(
                "{}:{}: center ({x}, {y}) outside (0, 1]",
                path.display(),
                i + 1
            )));
        }
        if !(w <= 1.0 && h <= 1.0 && w > 0.0 && h > 0.0) {
            return Err(Error::Model(format!(
                "{}:{}: size ({w}, {h}) outside (0, 1]",
                path.display(),
                i + 1
            )));
        }
        out.push(BoxLabel { class: class as usize, x, y, w, h });
    }
    Ok(out)
}

/// Read an image list: one path per line, resolved against the list's own directory when relative.
pub fn read_list(path: &Path, classes: usize) -> Result<Vec<Sample>> {
    let text = read_text(path)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out = vec![];
    for raw in text.lines() {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        let image = resolve(dir, Path::new(t));
        let labels = read_labels(&label_path(&image), classes)?;
        out.push(Sample { image, labels });
    }
    if out.is_empty() {
        return Err(Error::Model(format!("{}: empty image list", path.display())));
    }
    Ok(out)
}

impl Dataset {
    pub fn load(path: impl AsRef<Path>) -> Result<Dataset> {
        let data_path = path.as_ref();
        let root = data_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let d = DataFile::load(data_path)?;
        let names = match &d.names {
            Some(p) => crate::load_names(resolve(&root, p))?,
            None => vec![],
        };
        if !names.is_empty() && names.len() != d.classes {
            return Err(Error::Model(format!(
                "{} classes but {} names",
                d.classes,
                names.len()
            )));
        }
        let train_path = d
            .train
            .as_ref()
            .ok_or_else(|| Error::Model("`.data` has no `train` list".into()))?;
        let train = read_list(&resolve(&root, train_path), d.classes)?;
        let valid = match &d.valid {
            Some(p) => read_list(&resolve(&root, p), d.classes)?,
            None => vec![],
        };
        Ok(Dataset { classes: d.classes, names, train, valid, root })
    }

    /// Load one sample's image at its own size.
    pub fn image(&self, sample: &Sample) -> Result<Image> {
        Image::load(&sample.image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATA: &str = "classes = 5\ntrain = train.txt\nvalid = valid.txt\nnames = n.names\nbackup = .\n";

    #[test]
    fn data_file_parses() {
        let d = DataFile::parse(DATA).unwrap();
        assert_eq!(d.classes, 5);
        assert_eq!(d.train.as_deref(), Some(Path::new("train.txt")));
        assert_eq!(d.valid.as_deref(), Some(Path::new("valid.txt")));
        assert_eq!(d.names.as_deref(), Some(Path::new("n.names")));
        assert!(DataFile::parse("classes = 0\n").is_err());
        assert!(DataFile::parse("classes 5\n").is_err());
    }

    #[test]
    fn label_path_swaps_extension() {
        assert_eq!(label_path(Path::new("/a/b/frame.png")), PathBuf::from("/a/b/frame.txt"));
    }

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn dataset_reads_real_files() {
        let dir = std::env::temp_dir().join(format!("hipd_data_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "img0.txt", "0 0.5 0.5 0.25 0.25\n4 0.1 0.9 0.02 0.04\n\n");
        write(&dir, "img1.txt", "2 0.2 0.3 0.1 0.1\n");
        write(&dir, "train.txt", "img0.jpg\nimg1.png\n");
        write(&dir, "valid.txt", "img1.png\n");
        write(&dir, "n.names", "a\nb\nc\nd\ne\n");
        write(&dir, "t.data", DATA);
        let ds = Dataset::load(dir.join("t.data")).unwrap();
        assert_eq!(ds.classes, 5);
        assert_eq!(ds.names, vec!["a", "b", "c", "d", "e"]);
        assert_eq!(ds.train.len(), 2);
        assert_eq!(ds.train[0].labels.len(), 2);
        assert_eq!(ds.train[0].labels[1], BoxLabel { class: 4, x: 0.1, y: 0.9, w: 0.02, h: 0.04 });
        assert_eq!(ds.valid.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bad_labels_are_errors() {
        let dir = std::env::temp_dir().join(format!("hipd_data_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = write(&dir, "x.txt", "5 0.5 0.5 0.1 0.1\n");
        assert!(read_labels(&p, 5).unwrap_err().to_string().contains("class 5"));
        let p = write(&dir, "y.txt", "1 0.0 0.5 0.1 0.1\n");
        assert!(read_labels(&p, 5).unwrap_err().to_string().contains("outside"));
        let p = write(&dir, "z.txt", "1 0.5 0.5\n");
        assert!(read_labels(&p, 5).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
