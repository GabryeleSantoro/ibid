//! Sidecar supervision.
//!
//! The shell owns the lifetime of every child process. It picks a free port,
//! mints a per-session token, starts the process, waits for it to report
//! healthy, restarts it with backoff if it dies, and kills it on exit. The
//! webview never learns the port or the token.
//!
//! Only `ragcore` is supervised here. It starts and restarts the embedder and
//! reranker `llama-server` instances itself (`model_servers.py`), so the shell
//! needs no knowledge of them.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::AppHandle;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::watch;

use crate::hardware::HardwareInfo;

const LOG_CAPACITY: usize = 2000;
/// The log file is moved aside once it passes this, keeping one previous copy.
// Generous: ragcore re-reads the whole library before it reports healthy.
const READY_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_POLL: Duration = Duration::from_millis(250);
const BACKOFF_MAX: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProcState {
    Stopped,
    Starting,
    Ready,
    Restarting,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct SidecarStatus {
    pub name: String,
    pub role: String,
    pub state: ProcState,
    pub pid: Option<u32>,
    pub port: Option<u16>,
    pub uptime_s: f64,
    pub restarts: u32,
    pub detail: Option<String>,
}

#[derive(Default)]
struct Inner {
    state: Option<ProcState>,
    pid: Option<u32>,
    restarts: u32,
    ready_since: Option<Instant>,
    detail: Option<String>,
    logs: VecDeque<String>,
}

pub struct Supervisor {
    pub port: u16,
    pub token: String,
    inner: Mutex<Inner>,
    shutdown: watch::Sender<bool>,
    log_file: Mutex<Option<(PathBuf, File)>>,
}

/// "2026-09-29 08:15:02Z". UTC, so no timezone database is needed; ragcore's
/// own lines carry no timestamp and get this one too.
fn utc_timestamp(now: SystemTime) -> String {
    let secs = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Civil-from-days (Howard Hinnant), valid for any date after 1970.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn day_file(day: &str) -> String {
    format!("ibid-{day}.log")
}

fn today() -> String {
    utc_timestamp(SystemTime::now())[..10].to_string()
}

fn open_day_log(dir: &std::path::Path, day: &str) -> std::io::Result<(PathBuf, File)> {
    let path = dir.join(day_file(day));
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    Ok((path, file))
}

/// Claim a free port by binding to :0 and letting the OS choose, then release
/// it. There is a race window before the child binds; it is small and the
/// alternative is a fixed port that collides with whatever else is running.
fn free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn session_token() -> String {
    let bytes: [u8; 32] = rand::random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// In development the sidecar runs from the workspace through uv. A packaged
/// build runs the PyInstaller binary that ships next to the app instead.
#[cfg(debug_assertions)]
fn ragcore_command(port: u16, token: &str, hw: &HardwareInfo, _app: &AppHandle) -> Command {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("."));

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("--directory")
        .arg(workspace.join("core/ragcore"))
        .arg("ragcore");
    push_serve_args(&mut command, port, token, hw);
    command.current_dir(workspace);
    command
}

#[cfg(not(debug_assertions))]
fn ragcore_command(port: u16, token: &str, hw: &HardwareInfo, app: &AppHandle) -> Command {
    use tauri::Manager;

    let binary = app
        .path()
        .resource_dir()
        .map(|dir| dir.join("binaries/ragcore"))
        .unwrap_or_else(|_| PathBuf::from("ragcore"));

    let mut command = Command::new(binary);
    push_serve_args(&mut command, port, token, hw);
    command.arg("--prod");
    command
}

fn push_serve_args(command: &mut Command, port: u16, token: &str, hw: &HardwareInfo) {
    command
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .arg("--token")
        .arg(token)
        .arg("--ram-mb")
        .arg(hw.ram_mb.to_string())
        .arg("--vram-mb")
        .arg(hw.vram_mb.to_string())
        .arg("--gpu-backend")
        .arg(&hw.gpu_backend)
        .arg("--backend")
        .arg("real");
}

impl Supervisor {
    pub fn new() -> Result<Arc<Self>, String> {
        let port = free_port().map_err(|e| format!("no free port: {e}"))?;
        let (shutdown, _) = watch::channel(false);
        Ok(Arc::new(Self {
            port,
            token: session_token(),
            inner: Mutex::new(Inner::default()),
            shutdown,
            log_file: Mutex::new(None),
        }))
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn status(&self) -> SidecarStatus {
        let inner = self.inner.lock().expect("sidecar state poisoned");
        SidecarStatus {
            name: "ragcore".into(),
            role: "ragcore".into(),
            state: inner.state.unwrap_or(ProcState::Stopped),
            pid: inner.pid,
            port: Some(self.port),
            uptime_s: inner
                .ready_since
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0),
            restarts: inner.restarts,
            detail: inner.detail.clone(),
        }
    }

    pub fn logs(&self) -> Vec<String> {
        let inner = self.inner.lock().expect("sidecar state poisoned");
        inner.logs.iter().cloned().collect()
    }

    /// Where every log line is also written, so logs survive a restart.
    pub fn log_path(&self) -> Option<PathBuf> {
        let file = self.log_file.lock().expect("log file poisoned");
        file.as_ref().map(|(path, _)| path.clone())
    }

    fn open_log_file(&self, dir: PathBuf) -> std::io::Result<()> {
        std::fs::create_dir_all(&dir)?;
        *self.log_file.lock().expect("log file poisoned") = Some(open_day_log(&dir, &today())?);
        Ok(())
    }

    /// The one funnel for logs: the shell's own events, the proxy's and every
    /// line ragcore prints. Kept in memory for the Logs page and appended to
    /// disk, one `ibid-YYYY-MM-DD.log` per UTC day.
    pub fn log(&self, line: String) {
        let stamp = utc_timestamp(SystemTime::now());
        let line = format!("{stamp} {line}");
        if let Some(current) = self.log_file.lock().expect("log file poisoned").as_mut() {
            if !current.0.ends_with(day_file(&stamp[..10])) {
                if let Some(dir) = current.0.parent() {
                    if let Ok(next) = open_day_log(dir, &stamp[..10]) {
                        *current = next;
                    }
                }
            }
            let _ = writeln!(current.1, "{line}");
        }
        let mut inner = self.inner.lock().expect("sidecar state poisoned");
        if inner.logs.len() == LOG_CAPACITY {
            inner.logs.pop_front();
        }
        inner.logs.push_back(line);
    }

    fn set_state(&self, state: ProcState, detail: Option<String>) {
        let mut inner = self.inner.lock().expect("sidecar state poisoned");
        inner.state = Some(state);
        inner.detail = detail;
        if state == ProcState::Ready {
            inner.ready_since = Some(Instant::now());
        } else {
            inner.ready_since = None;
        }
    }

    /// Hold a UI call until ragcore is up. The window
    /// opens seconds before the frozen binary is healthy, and a query that fails
    /// then leaves the UI empty until something refetches it. Gives up after
    /// READY_TIMEOUT and lets the call fail on its own.
    pub async fn wait_ready(&self) {
        // ponytail: polls the state; a watch channel if this ever shows up in profiles.
        let deadline = Instant::now() + READY_TIMEOUT;
        while self.status().state != ProcState::Ready && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub fn request_shutdown(&self) {
        let _ = self.shutdown.send(true);
    }

    /// Restart on demand, used by Diagnostics. The supervise loop treats it as
    /// an ordinary crash, so backoff and the restart counter still apply.
    pub fn kill_child(&self) {
        let pid = self.inner.lock().expect("sidecar state poisoned").pid;
        if let Some(pid) = pid {
            #[cfg(unix)]
            unsafe {
                // Negative pid: the whole process group spawned above.
                libc_kill(-(pid as i32));
            }
            #[cfg(not(unix))]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/T", "/F"])
                    .status();
            }
        }
    }

    pub fn spawn(self: &Arc<Self>, app: AppHandle, hw: HardwareInfo, http: reqwest::Client) {
        {
            use tauri::Manager;
            match app.path().app_log_dir() {
                Ok(dir) => {
                    if let Err(err) = self.open_log_file(dir) {
                        self.log(format!("[shell] log file unavailable: {err}"));
                    }
                }
                Err(err) => self.log(format!("[shell] no log directory: {err}")),
            }
        }
        self.log(format!("[shell] Ibid {} starting", env!("CARGO_PKG_VERSION")));
        let supervisor = Arc::clone(self);
        tauri::async_runtime::spawn(async move {
            supervisor.supervise(app, hw, http).await;
        });
    }

    async fn supervise(self: Arc<Self>, app: AppHandle, hw: HardwareInfo, http: reqwest::Client) {
        let mut shutdown_rx = self.shutdown.subscribe();
        let mut attempt: u32 = 0;

        loop {
            if *shutdown_rx.borrow() {
                break;
            }

            self.set_state(ProcState::Starting, Some("spawning".into()));
            let mut command = ragcore_command(self.port, &self.token, &hw, &app);
            command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            // Its own process group, so killing it kills everything it started.
            // In dev the direct child is `uv`, which execs Python as a grandchild:
            // signalling only the child would leave that Python holding the port.
            #[cfg(unix)]
            command.process_group(0);

            let mut child: Child = match command.spawn() {
                Ok(child) => child,
                Err(err) => {
                    let message = format!("failed to spawn ragcore: {err}");
                    self.log(message.clone());
                    self.set_state(ProcState::Error, Some(message));
                    if self.backoff(&mut attempt, &mut shutdown_rx).await {
                        break;
                    }
                    continue;
                }
            };

            {
                let mut inner = self.inner.lock().expect("sidecar state poisoned");
                inner.pid = child.id();
            }
            self.pipe_logs(&mut child);

            match self.await_ready(&http, &mut shutdown_rx).await {
                Ok(true) => {
                    attempt = 0;
                    // Ready first: reading the keychain can sit behind a macOS
                    // password prompt, and the UI must not wait on that.
                    self.set_state(ProcState::Ready, None);
                    if let Err(err) = self.restore_secrets(&http).await {
                        self.log(format!("could not restore API keys: {err}"));
                    }
                }
                Ok(false) => {} // shutting down
                Err(message) => {
                    self.log(message.clone());
                    self.set_state(ProcState::Error, Some(message));
                }
            }

            tokio::select! {
                status = child.wait() => {
                    let detail = match status {
                        Ok(status) => format!("ragcore exited: {status}"),
                        Err(err) => format!("ragcore wait failed: {err}"),
                    };
                    self.log(detail.clone());
                    if *shutdown_rx.borrow() {
                        self.set_state(ProcState::Stopped, Some(detail));
                        break;
                    }
                    {
                        let mut inner = self.inner.lock().expect("sidecar state poisoned");
                        inner.restarts += 1;
                        inner.pid = None;
                    }
                    self.set_state(ProcState::Restarting, Some(detail));
                    if self.backoff(&mut attempt, &mut shutdown_rx).await {
                        break;
                    }
                }
                _ = shutdown_rx.changed() => {
                    let _ = child.kill().await;
                    self.set_state(ProcState::Stopped, Some("shutting down".into()));
                    break;
                }
            }
        }
    }

    /// Mirror the child's stdout and stderr into the ring buffer Diagnostics reads.
    fn pipe_logs(self: &Arc<Self>, child: &mut Child) {
        if let Some(stdout) = child.stdout.take() {
            let supervisor = Arc::clone(self);
            tauri::async_runtime::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    supervisor.log(format!("[out] {line}"));
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            let supervisor = Arc::clone(self);
            tauri::async_runtime::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    supervisor.log(format!("[err] {line}"));
                }
            });
        }
    }

    async fn await_ready(
        &self,
        http: &reqwest::Client,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) -> Result<bool, String> {
        let url = format!("{}/health", self.base_url());
        let deadline = Instant::now() + READY_TIMEOUT;

        while Instant::now() < deadline {
            if *shutdown_rx.borrow() {
                return Ok(false);
            }
            if let Ok(response) = http.get(&url).timeout(Duration::from_secs(2)).send().await {
                if response.status().is_success() {
                    return Ok(true);
                }
            }
            tokio::time::sleep(HEALTH_POLL).await;
        }
        Err(format!(
            "ragcore did not become healthy within {}s",
            READY_TIMEOUT.as_secs()
        ))
    }

    /// ragcore keeps API keys in memory only, so every fresh process gets them
    /// back from the keychain before the UI is told it is ready.
    async fn restore_secrets(&self, http: &reqwest::Client) -> Result<(), String> {
        let connections: Vec<serde_json::Value> = http
            .get(format!("{}/connections", self.base_url()))
            .bearer_auth(&self.token)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;

        for connection in connections {
            let Some(id) = connection["id"].as_str() else { continue };
            if connection["has_api_key"] != true {
                continue;
            }
            let Some(secret) = crate::keychain::get(id)? else { continue };
            http.put(format!("{}/connections/{id}/secret", self.base_url()))
                .bearer_auth(&self.token)
                .json(&serde_json::json!({ "api_key": secret }))
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Exponential backoff, capped. Returns true when shutdown was requested
    /// while waiting.
    async fn backoff(&self, attempt: &mut u32, shutdown_rx: &mut watch::Receiver<bool>) -> bool {
        *attempt = (*attempt + 1).min(6);
        let delay = Duration::from_millis(250 * (1 << *attempt)).min(BACKOFF_MAX);
        tokio::select! {
            _ = tokio::time::sleep(delay) => false,
            _ = shutdown_rx.changed() => true,
        }
    }
}

#[cfg(unix)]
unsafe fn libc_kill(pid: i32) {
    // SIGKILL, so a wedged sidecar cannot ignore it. The supervise loop sees
    // the exit and restarts it like any other crash.
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    unsafe { kill(pid, 9) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_utc_calendar_dates() {
        let at = |secs| utc_timestamp(UNIX_EPOCH + Duration::from_secs(secs));

        assert_eq!(at(0), "1970-01-01 00:00:00Z");
        assert_eq!(at(951_782_400), "2000-02-29 00:00:00Z");
        assert_eq!(at(1_790_669_702), "2026-09-29 08:15:02Z");
    }

    #[test]
    fn a_session_token_is_long_and_never_repeats() {
        let first = session_token();

        assert_eq!(first.len(), 64);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, session_token());
    }

    #[test]
    fn the_reserved_port_is_free_and_local() {
        let port = free_port().expect("the OS always has a free ephemeral port");

        assert!(port > 0);
        // Released again, so the child can bind it.
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn a_new_supervisor_starts_stopped_on_its_own_port() {
        let supervisor = Supervisor::new().expect("a port");
        let status = supervisor.status();

        assert_eq!(status.state, ProcState::Stopped);
        assert_eq!(status.port, Some(supervisor.port));
        assert_eq!(status.restarts, 0);
        assert!(status.pid.is_none());
        assert_eq!(supervisor.base_url(), format!("http://127.0.0.1:{}", supervisor.port));
    }

    #[test]
    fn the_log_ring_buffer_drops_the_oldest_line() {
        let supervisor = Supervisor::new().expect("a port");
        for i in 0..LOG_CAPACITY + 10 {
            supervisor.log(format!("line {i}"));
        }

        let logs = supervisor.logs();

        assert_eq!(logs.len(), LOG_CAPACITY);
        assert!(logs[0].ends_with("Z line 10"));
        assert!(logs[LOG_CAPACITY - 1].ends_with(&format!("Z line {}", LOG_CAPACITY + 9)));
    }

    #[test]
    fn uptime_is_counted_only_while_ready() {
        let supervisor = Supervisor::new().expect("a port");

        supervisor.set_state(ProcState::Ready, None);
        assert!(supervisor.status().uptime_s >= 0.0);

        supervisor.set_state(ProcState::Restarting, Some("crashed".into()));
        let status = supervisor.status();
        assert_eq!(status.uptime_s, 0.0);
        assert_eq!(status.detail.as_deref(), Some("crashed"));
    }

    #[test]
    fn killing_a_child_that_never_started_is_a_no_op() {
        Supervisor::new().expect("a port").kill_child();
    }

    #[test]
    fn the_serve_args_carry_the_port_token_and_hardware_budget() {
        let hw = HardwareInfo {
            os: "macos".into(),
            arch: "aarch64".into(),
            cpu_count: 10,
            ram_mb: 32768,
            vram_mb: 32768,
            gpu_backend: "metal".into(),
            gpu_name: None,
            profile: "gpu".into(),
        };
        let mut command = Command::new("true");
        push_serve_args(&mut command, 8765, "deadbeef", &hw);

        let args: Vec<String> = command
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();

        assert_eq!(args[0], "serve");
        assert!(args.windows(2).any(|w| w == ["--port", "8765"]));
        assert!(args.windows(2).any(|w| w == ["--token", "deadbeef"]));
        assert!(args.windows(2).any(|w| w == ["--ram-mb", "32768"]));
        assert!(args.windows(2).any(|w| w == ["--vram-mb", "32768"]));
        assert!(args.windows(2).any(|w| w == ["--gpu-backend", "metal"]));
        assert!(args.windows(2).any(|w| w == ["--backend", "real"]));
    }

    #[test]
    fn a_status_serialises_its_state_in_lowercase_for_the_ui() {
        let supervisor = Supervisor::new().expect("a port");
        supervisor.set_state(ProcState::Restarting, None);

        let json = serde_json::to_value(supervisor.status()).unwrap();

        assert_eq!(json["state"], "restarting");
        assert_eq!(json["role"], "ragcore");
    }
}
