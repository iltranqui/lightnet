use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::model::{Network, TensorShape};
use crate::runtime::Tensor;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataConfig {
    pub source: PathBuf,
    pub classes: Option<usize>,
    pub train: Option<PathBuf>,
    pub valid: Option<PathBuf>,
    pub names: Option<PathBuf>,
    pub backup: Option<PathBuf>,
    pub options: BTreeMap<String, String>,
}

impl DataConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let source = path.to_path_buf();
        let parent = source
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let mut options = BTreeMap::new();

        for (index, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                return Err(Error::Parse(format!(
                    "failed to parse {} line {}: expected key=value",
                    source.display(),
                    index + 1
                )));
            };
            options.insert(key.trim().to_owned(), value.trim().to_owned());
        }

        Ok(Self {
            source,
            classes: options
                .get("classes")
                .map(|value| {
                    value.parse::<usize>().map_err(|_| {
                        Error::Parse(format!("invalid classes value in {}", path.display()))
                    })
                })
                .transpose()?,
            train: resolve_optional_path(&options, &parent, "train"),
            valid: resolve_optional_path(&options, &parent, "valid"),
            names: resolve_optional_path(&options, &parent, "names"),
            backup: resolve_optional_path(&options, &parent, "backup"),
            options,
        })
    }

    pub fn required_train_list(&self) -> Result<&Path> {
        self.train.as_deref().ok_or_else(|| {
            Error::InvalidConfig(format!(
                "{} is missing train=...",
                self.source.display()
            ))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloLabel {
    pub class_id: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingBatch {
    pub input: Tensor,
    pub image_paths: Vec<PathBuf>,
    pub label_paths: Vec<PathBuf>,
    pub labels: Vec<Vec<YoloLabel>>,
}

impl TrainingBatch {
    pub fn label_count(&self) -> usize {
        self.labels.iter().map(Vec::len).sum()
    }
}

pub fn load_paths_file(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        out.push(resolve_path(parent, &path));
    }

    if out.is_empty() {
        return Err(Error::InvalidConfig(format!(
            "failed to read any paths from {}",
            path.display()
        )));
    }

    Ok(out)
}

pub fn replace_image_to_label(path: impl AsRef<Path>) -> PathBuf {
    let mut out = path.as_ref().to_path_buf();
    out.set_extension("txt");
    out
}

pub fn load_yolo_labels(
    path: impl AsRef<Path>,
    classes: Option<usize>,
) -> Result<Vec<YoloLabel>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let mut labels = Vec::new();

    for (index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 5 {
            return Err(Error::Parse(format!(
                "failed to parse {} line {}: expected 5 YOLO fields",
                path.display(),
                index + 1
            )));
        }

        let class_id = parts[0].parse::<usize>().map_err(|_| {
            Error::Parse(format!(
                "invalid class id in {} line {}",
                path.display(),
                index + 1
            ))
        })?;
        if let Some(classes) = classes && class_id >= classes {
            return Err(Error::InvalidConfig(format!(
                "class id {} exceeds classes={} in {}",
                class_id,
                classes,
                path.display()
            )));
        }

        let x = parse_unit_f32(parts[1], path, index + 1, "x")?;
        let y = parse_unit_f32(parts[2], path, index + 1, "y")?;
        let w = parse_unit_f32(parts[3], path, index + 1, "w")?;
        let h = parse_unit_f32(parts[4], path, index + 1, "h")?;
        if w <= 0.0 || h <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "invalid box size in {} line {}",
                path.display(),
                index + 1
            )));
        }

        labels.push(YoloLabel {
            class_id,
            x,
            y,
            w,
            h,
        });
    }

    Ok(labels)
}

pub fn load_training_batch(
    network: &Network,
    data: &DataConfig,
    start_index: usize,
) -> Result<TrainingBatch> {
    let train_list = data.required_train_list()?;
    let image_paths = load_paths_file(train_list)?;
    load_training_batch_from_paths(network, data, &image_paths, start_index)
}

pub fn load_training_batch_from_paths(
    network: &Network,
    data: &DataConfig,
    image_paths: &[PathBuf],
    start_index: usize,
) -> Result<TrainingBatch> {
    if image_paths.is_empty() {
        return Err(Error::InvalidConfig(
            "training image list is empty".to_owned(),
        ));
    }
    let batch_size = network.net.batch.max(1);
    let channels = network.net.channels;
    let width = network.net.width;
    let height = network.net.height;
    let shape = TensorShape::new(width, height, channels, batch_size);
    let per_image = shape.element_count();
    let mut input = Tensor::zeros(shape);
    let mut batch_image_paths = Vec::with_capacity(batch_size);
    let mut batch_label_paths = Vec::with_capacity(batch_size);
    let mut batch_labels = Vec::with_capacity(batch_size);

    for batch_index in 0..batch_size {
        let image_path = &image_paths[(start_index + batch_index) % image_paths.len()];
        let label_path = replace_image_to_label(image_path);
        if !label_path.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing label file {}", label_path.display()),
            )));
        }

        let image = PlanarRgbImage::load(image_path)?;
        let prepared = image.resize(width, height)?;
        if prepared.shape.channels != channels {
            return Err(Error::InvalidConfig(format!(
                "image {} has {} channels, expected {}",
                image_path.display(),
                prepared.shape.channels,
                channels
            )));
        }

        let labels = load_yolo_labels(&label_path, data.classes)?;
        let dst = batch_index * per_image;
        input.data[dst..dst + per_image].copy_from_slice(&prepared.data);
        batch_image_paths.push(image_path.clone());
        batch_label_paths.push(label_path);
        batch_labels.push(labels);
    }

    Ok(TrainingBatch {
        input,
        image_paths: batch_image_paths,
        label_paths: batch_label_paths,
        labels: batch_labels,
    })
}

fn parse_unit_f32(value: &str, path: &Path, line: usize, field: &str) -> Result<f32> {
    let parsed = value.parse::<f32>().map_err(|_| {
        Error::Parse(format!(
            "invalid {} in {} line {}",
            field,
            path.display(),
            line
        ))
    })?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err(Error::InvalidConfig(format!(
            "invalid {}={} in {} line {}",
            field,
            parsed,
            path.display(),
            line
        )));
    }
    Ok(parsed)
}

fn resolve_optional_path(
    options: &BTreeMap<String, String>,
    parent: &Path,
    key: &str,
) -> Option<PathBuf> {
    options.get(key).map(|value| resolve_path(parent, Path::new(value)))
}

fn resolve_path(parent: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        parent.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::{DataConfig, load_training_batch, load_yolo_labels, replace_image_to_label};
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn parses_data_cfg_and_resolves_relative_paths() {
        let dir = temp_path("darknet-train-data-cfg");
        let _ = fs::create_dir_all(&dir);
        let data_path = dir.join("sample.data");
        fs::write(
            &data_path,
            "classes=2\ntrain=train.txt\nvalid=valid.txt\nnames=labels.names\nbackup=backup\n",
        )
        .expect("data cfg writes");

        let parsed = DataConfig::from_path(&data_path).expect("data cfg parses");
        assert_eq!(parsed.classes, Some(2));
        assert_eq!(parsed.train.as_deref(), Some(dir.join("train.txt").as_path()));
        assert_eq!(parsed.valid.as_deref(), Some(dir.join("valid.txt").as_path()));
        assert_eq!(parsed.names.as_deref(), Some(dir.join("labels.names").as_path()));
        assert_eq!(parsed.backup.as_deref(), Some(dir.join("backup").as_path()));
    }

    #[test]
    fn loads_yolo_labels_and_accepts_empty_files() {
        let path = temp_path("darknet-yolo-labels.txt");
        fs::write(&path, "0 0.5 0.25 0.1 0.2\n").expect("label writes");
        let labels = load_yolo_labels(&path, Some(1)).expect("labels load");
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].class_id, 0);

        fs::write(&path, "").expect("label truncates");
        let empty = load_yolo_labels(&path, Some(1)).expect("empty labels load");
        assert!(empty.is_empty());
    }

    #[test]
    fn replaces_image_extension_with_txt() {
        assert_eq!(
            replace_image_to_label("/tmp/example.jpg"),
            PathBuf::from("/tmp/example.txt")
        );
        assert_eq!(
            replace_image_to_label("/tmp/example.png"),
            PathBuf::from("/tmp/example.txt")
        );
    }

    #[test]
    fn loads_training_batch_from_supported_images() {
        let dir = temp_path("darknet-train-batch");
        let _ = fs::create_dir_all(&dir);
        let data_path = dir.join("sample.data");
        let train_list = dir.join("train.txt");
        let image_path = dir.join("sample.ppm");
        let label_path = dir.join("sample.txt");

        fs::write(
            &image_path,
            b"P6\n1 1\n255\n\xff\x80\x00",
        )
        .expect("ppm writes");
        fs::write(&label_path, "0 0.5 0.5 0.25 0.25\n").expect("labels write");
        fs::write(&train_list, format!("{}\n", image_path.display())).expect("train list writes");
        fs::write(
            &data_path,
            format!("classes=1\ntrain={}\n", train_list.display()),
        )
        .expect("data cfg writes");

        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=1
                height=1
                channels=3

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let data = DataConfig::from_path(&data_path).expect("data cfg parses");
        let batch = load_training_batch(&network, &data, 0).expect("batch loads");

        assert_eq!(batch.image_paths, vec![image_path]);
        assert_eq!(batch.label_paths, vec![label_path]);
        assert_eq!(batch.label_count(), 1);
        assert_eq!(batch.input.shape, network.net.input_shape());
    }
}
