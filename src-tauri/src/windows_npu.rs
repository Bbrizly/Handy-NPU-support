//! Opt-in Windows x64 Intel NPU transcription through OpenVINO GenAI.
//!
//! The existing transcribe-cpp worker remains authoritative and is always the
//! fallback. This feature intentionally never treats DirectML/Vulkan as NPU.
//! See docs/windows-intel-npu.md for configuration and hardware requirements.

use log::{info, warn};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::Duration;

const MAX_AUDIO_SAMPLES: usize = 16_000 * 600; // 10 minutes at 16 kHz.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

static JOB_SENDER: OnceLock<mpsc::Sender<Job>> = OnceLock::new();
static DISABLED: AtomicBool = AtomicBool::new(false);
static WARNED: AtomicBool = AtomicBool::new(false);

struct Config {
    model_dir: PathBuf,
    python: PathBuf,
}

struct Job {
    samples: Vec<f32>,
    language: Option<String>,
    translate: bool,
    reply: mpsc::Sender<Result<String, String>>,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn config_from_env() -> Option<Config> {
    let model_dir = std::env::var_os("HANDY_OPENVINO_NPU_MODEL_DIR")?;
    if model_dir.is_empty() {
        return None;
    }
    let python = std::env::var_os("HANDY_OPENVINO_PYTHON")
        .unwrap_or_else(|| "python".into());
    Some(Config {
        model_dir: PathBuf::from(model_dir),
        python: PathBuf::from(python),
    })
}

fn language_token(language: Option<&str>) -> Option<String> {
    let language = language?;
    // Only ISO 639-1 codes are accepted as Whisper language tokens. "auto"
    // means automatic language detection, not the literal token <|auto|>.
    if language.len() == 2 && language.bytes().all(|b| b.is_ascii_lowercase()) {
        Some(format!("<|{language}|>"))
    } else {
        None
    }
}

impl Worker {
    fn spawn(config: &Config) -> Result<Self, String> {
        if !config.model_dir.is_dir() {
            return Err(format!(
                "OpenVINO model directory does not exist: {}",
                config.model_dir.display()
            ));
        }

        let mut child = Command::new(&config.python)
            .arg("-u")
            .arg("-c")
            .arg(include_str!("windows_npu_worker.py"))
            .arg(&config.model_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Python writes structured errors on stdout; suppress native
            // provider noise so it cannot corrupt the line protocol.
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Could not launch OpenVINO Python: {e}"))?;

        let stdin = child.stdin.take().ok_or("Missing Python stdin")?;
        let stdout = child.stdout.take().ok_or("Missing Python stdout")?;
        let mut worker = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = worker.read_response()?;
        if ready.get("ready").and_then(Value::as_bool) != Some(true) {
            return Err(format!(
                "Intel NPU unavailable: {}",
                ready.get("error").and_then(Value::as_str).unwrap_or("unknown error")
            ));
        }
        info!("Handy OpenVINO Whisper NPU worker initialized (device=NPU)");
        Ok(worker)
    }

    fn read_response(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        let bytes = self.stdout.read_line(&mut line)
            .map_err(|e| format!("NPU worker read failure: {e}"))?;
        if bytes == 0 {
            return Err("NPU worker exited unexpectedly".to_string());
        }
        serde_json::from_str(&line)
            .map_err(|e| format!("NPU worker returned invalid JSON: {e}"))
    }

    fn transcribe(&mut self, job: &Job) -> Result<String, String> {
        let header = json!({
            "samples": job.samples.len(),
            "language": language_token(job.language.as_deref()),
            "translate": job.translate
        });
        self.stdin.write_all(header.to_string().as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .map_err(|e| format!("NPU request header write failed: {e}"))?;

        // One binary write (not thousands of small Windows pipe writes).
        let mut pcm = Vec::with_capacity(job.samples.len() * 4);
        for &sample in &job.samples {
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
        self.stdin.write_all(&pcm)
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("NPU PCM write failed: {e}"))?;

        let response = self.read_response()?;
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(response.get("error")
                .and_then(Value::as_str)
                .unwrap_or("OpenVINO inference failed")
                .to_string());
        }
        response.get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or("NPU worker returned no transcript text".into())
    }
}

fn run_worker(jobs: mpsc::Receiver<Job>, config: Config) {
    let mut worker: Option<Worker> = None;
    for job in jobs {
        if DISABLED.load(Ordering::Acquire) {
            let _ = job.reply.send(Err("NPU backend disabled for this session".into()));
            continue;
        }
        if worker.is_none() {
            match Worker::spawn(&config) {
                Ok(new_worker) => worker = Some(new_worker),
                Err(error) => {
                    DISABLED.store(true, Ordering::Release);
                    let _ = job.reply.send(Err(error));
                    continue;
                }
            }
        }
        let result = worker.as_mut().unwrap().transcribe(&job);
        if result.is_err() {
            // An error can leave the protocol/model state ambiguous. Never reuse
            // this worker; the following request starts with a fresh session.
            worker = None;
        }
        let _ = job.reply.send(result);
    }
}

/// Returns None when not configured, unsupported, timed out, or inference failed.
/// Caller must then run the existing transcribe-cpp backend with the SAME audio.
pub fn try_transcribe(audio: &[f32], language: Option<&str>, translate: bool) -> Option<String> {
    if DISABLED.load(Ordering::Acquire) || audio.is_empty() || audio.len() > MAX_AUDIO_SAMPLES {
        return None;
    }
    let config = config_from_env()?;
    // Do not route invalid samples into the NPU pipeline.
    if audio.iter().any(|x| !x.is_finite()) {
        return None;
    }

    let sender = JOB_SENDER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("handy-openvino-npu".into())
            .spawn(move || run_worker(receiver, config))
            .expect("Failed to start OpenVINO NPU worker thread");
        sender
    });
    let (reply_tx, reply_rx) = mpsc::channel();
    let job = Job {
        samples: audio.to_vec(),
        language: language.map(str::to_owned),
        translate,
        reply: reply_tx,
    };
    if sender.send(job).is_err() {
        DISABLED.store(true, Ordering::Release);
        return None;
    }

    match reply_rx.recv_timeout(REQUEST_TIMEOUT) {
        Ok(Ok(text)) => Some(text),
        Ok(Err(error)) => {
            if !WARNED.swap(true, Ordering::AcqRel) {
                warn!("Intel NPU transcription unavailable ({error}); using Handy's original engine");
            }
            None
        }
        Err(error) => {
            DISABLED.store(true, Ordering::Release);
            if !WARNED.swap(true, Ordering::AcqRel) {
                warn!("Intel NPU worker timed out/disconnected ({error}); disabled for this session, using original engine");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::language_token;

    #[test]
    fn whisper_languages_are_validated() {
        assert_eq!(language_token(Some("en")), Some("<|en|>".into()));
        assert_eq!(language_token(Some("ar")), Some("<|ar|>".into()));
        assert_eq!(language_token(Some("auto")), None);
        assert_eq!(language_token(Some("eng")), None);
        assert_eq!(language_token(Some("en\n")), None);
    }
}
