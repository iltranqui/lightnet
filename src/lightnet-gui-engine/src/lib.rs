//! Child-process bridge to `lightnet-rs detect-server`.
//!
//! One request per line on stdin, one JSON reply per line on stdout. Any child
//! failure, malformed reply or timeout becomes an `Err` so the UI can show a
//! banner and keep editing; this module never panics on model output.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub class: usize,
    pub conf: f32,
    /// Normalized centre/size, the same space as the `.txt` labels.
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    boxes: Vec<Prediction>,
    #[serde(default)]
    error: Option<String>,
}

pub struct DetectWorker {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Result<String, String>>,
    reader: Option<JoinHandle<()>>,
}

impl DetectWorker {
    pub fn spawn(bin: &Path, cfg: &Path, weights: &Path) -> Result<Self, String> {
        let mut child = Command::new(bin)
            .arg("detect-server")
            .arg(cfg)
            .arg(weights)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("{}: {error}", bin.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| "child stdin unavailable".to_owned())?;
        let stdout = child.stdout.take().ok_or_else(|| "child stdout unavailable".to_owned())?;
        let (sender, replies) = channel();
        let reader = std::thread::spawn(move || read_lines(stdout, sender));
        Ok(Self {
            child,
            stdin: Some(stdin),
            replies,
            reader: Some(reader),
        })
    }

    pub fn detect(&mut self, image: &Path, thresh: f32) -> Result<Vec<Prediction>, String> {
        let request = serde_json::json!({
            "image": image.to_string_lossy(),
            "thresh": thresh,
        });
        {
            let stdin = self
                .stdin
                .as_mut()
                .ok_or_else(|| "detect-server is not running".to_owned())?;
            writeln!(stdin, "{request}").map_err(|error| format!("write request: {error}"))?;
            stdin.flush().map_err(|error| format!("flush request: {error}"))?;
        }
        match self.replies.recv_timeout(TIMEOUT) {
            Ok(Ok(line)) => parse_reply(&line),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => {
                let _ = self.child.kill();
                Err(format!("detect-server timed out after {}s", TIMEOUT.as_secs()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err("detect-server exited".to_owned())
            }
        }
    }
}

fn read_lines(stdout: ChildStdout, sender: std::sync::mpsc::Sender<Result<String, String>>) {
    for line in BufReader::new(stdout).lines() {
        match line {
            Ok(line) => {
                if sender.send(Ok(line)).is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = sender.send(Err(format!("read reply: {error}")));
                return;
            }
        }
    }
    let _ = sender.send(Err("detect-server closed stdout".to_owned()));
}

fn parse_reply(line: &str) -> Result<Vec<Prediction>, String> {
    let reply: Reply = serde_json::from_str(line).map_err(|error| format!("bad reply: {error}"))?;
    if let Some(error) = reply.error {
        return Err(error);
    }
    Ok(reply.boxes)
}

impl Drop for DetectWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.reader.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn script(body: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "lightnet-gui-fake-server-{}-{}.sh",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&path, permissions).unwrap();
        }
        path
    }

    #[test]
    fn returns_predictions_from_a_fake_server() {
        let server = script(
            r#"while read line; do echo '{"boxes":[{"class":1,"conf":0.9,"cx":0.5,"cy":0.5,"w":0.2,"h":0.2}]}'; done"#,
        );
        let mut worker = DetectWorker::spawn(&server, Path::new("cfg"), Path::new("w")).unwrap();
        let boxes = worker.detect(Path::new("image.png"), 0.25).unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].class, 1);
        std::fs::remove_file(server).unwrap();
    }

    #[test]
    fn error_object_is_a_string_error() {
        let server = script(r#"while read line; do echo '{"error":"no such image"}'; done"#);
        let mut worker = DetectWorker::spawn(&server, Path::new("cfg"), Path::new("w")).unwrap();
        assert_eq!(
            worker.detect(Path::new("image.png"), 0.25),
            Err("no such image".to_owned())
        );
        std::fs::remove_file(server).unwrap();
    }

    #[test]
    fn child_exit_is_an_error_not_a_panic() {
        let server = script("exit 3");
        let mut worker = DetectWorker::spawn(&server, Path::new("cfg"), Path::new("w")).unwrap();
        assert!(worker.detect(Path::new("image.png"), 0.25).is_err());
        std::fs::remove_file(server).unwrap();
    }

    #[test]
    fn malformed_json_is_an_error() {
        let server = script(r#"while read line; do echo 'not json'; done"#);
        let mut worker = DetectWorker::spawn(&server, Path::new("cfg"), Path::new("w")).unwrap();
        assert!(worker.detect(Path::new("image.png"), 0.25).is_err());
        std::fs::remove_file(server).unwrap();
    }
}
