//! Hardware probe.
//!
//! Runs once at startup. The result picks the performance profile, sizes the
//! model download fit badges, and is handed to `ragcore` at spawn time so the
//! Python side never has to guess what the machine can run.

use serde::Serialize;
use std::process::Command;
use sysinfo::System;

#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub ram_mb: u64,
    pub vram_mb: u64,
    pub gpu_backend: String,
    pub gpu_name: Option<String>,
    pub profile: String,
}

/// Ask `nvidia-smi` for the first GPU. Absent driver means no CUDA, which is
/// the only thing we need to learn from it.
fn nvidia() -> Option<(String, u64)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let first = line.lines().next()?;
    let (name, mem) = first.split_once(',')?;
    Some((name.trim().to_string(), mem.trim().parse().ok()?))
}

fn has_vulkan() -> bool {
    Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn detect() -> HardwareInfo {
    let mut system = System::new();
    system.refresh_memory();

    let ram_mb = system.total_memory() / (1024 * 1024);
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let (gpu_backend, gpu_name, vram_mb) = if cfg!(all(target_os = "macos", target_arch = "aarch64"))
    {
        // Apple silicon shares one pool between CPU and GPU, so the usable
        // budget for a model is the system memory, not a separate VRAM figure.
        (
            "metal".to_string(),
            Some("Apple Silicon (unified memory)".to_string()),
            ram_mb,
        )
    } else if let Some((name, vram)) = nvidia() {
        ("cuda".to_string(), Some(name), vram)
    } else if has_vulkan() {
        ("vulkan".to_string(), None, 0)
    } else {
        ("cpu".to_string(), None, 0)
    };

    let profile = if vram_mb >= 8192 {
        "gpu"
    } else if gpu_backend != "cpu" {
        "balanced"
    } else {
        "cpu"
    };

    HardwareInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count,
        ram_mb,
        vram_mb,
        gpu_backend,
        gpu_name,
        profile: profile.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Probes shell out, so only the host-independent invariants are asserted.
    // ponytail: nvidia() parsing is untested; extract a pure parse fn if it ever breaks.
    #[test]
    fn detect_is_self_consistent() {
        let hw = detect();
        assert!(hw.cpu_count >= 1);
        assert!(["gpu", "balanced", "cpu"].contains(&hw.profile.as_str()));
        assert_eq!(hw.profile == "cpu", hw.gpu_backend == "cpu");
    }
}
