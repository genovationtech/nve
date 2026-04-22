//! Hardware device abstraction for NVE inference.
//!
//! Provides a unified `Device` enum that selects between all supported
//! compute backends at runtime, with graceful CPU fallback.
//!
//! ## Supported backends
//!
//! | Variant      | Hardware          | Feature flag  | Prerequisite              |
//! |--------------|-------------------|---------------|---------------------------|
//! | `Cpu`        | Any x86-64 / ARM  | (always)      | AVX2+FMA SIMD via rayon   |
//! | `Cuda(idx)`  | NVIDIA GPU        | `--features cuda`  | CUDA Toolkit ≥ 11.8  |
//! | `Hip(idx)`   | AMD GPU (ROCm)    | `--features hip`   | ROCm ≥ 5.0           |
//! | `Metal`      | Apple GPU         | `--features metal` | macOS 13+ / Apple Silicon|
//! | `Vulkan(idx)`| Any Vulkan GPU    | `--features vulkan`| Vulkan SDK / mesa     |
//!
//! ## Build recipes
//!
//! ```sh
//! # NVIDIA CUDA (requires CUDA_PATH / CUDA_HOME set)
//! cargo build --release --features cuda
//!
//! # AMD ROCm (requires ROCM_PATH set, e.g. /opt/rocm)
//! cargo build --release --features hip
//!
//! # Apple Silicon (macOS only)
//! cargo build --release --features metal
//!
//! # Cross-platform Vulkan (AMD/Intel/NVIDIA without CUDA)
//! cargo build --release --features vulkan
//!
//! # Intel MKL CPU acceleration (no GPU)
//! cargo build --release --features mkl
//! ```
//!
//! ## Runtime selection
//!
//! ```sh
//! nve generate --device auto          # pick best available
//! nve generate --device cuda:0        # first NVIDIA GPU
//! nve generate --device hip:0         # first AMD GPU
//! nve generate --device metal         # Apple GPU
//! nve generate --device vulkan:0      # first Vulkan GPU
//! nve generate --device cpu           # CPU (always works)
//! ```
//!
//! ## Weight placement with GPU
//!
//! Hot layers → VRAM (`--hot-budget-mb`).
//! Warm layers → CPU RAM (`--warm-budget-mb`).
//! Cold layers → SSD (mmap).
//!
//! On CPU-only hosts the hot budget maps to pinned (mlocked) RAM.

use std::fmt;
use std::str::FromStr;

// ── Device enum ───────────────────────────────────────────────────────────────

/// Compute device for inference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// Pure CPU inference — AVX2+FMA SIMD (rayon multi-thread). Always available.
    Cpu,

    /// NVIDIA GPU via CUDA.
    /// Requires `--features cuda` at compile time and CUDA Toolkit ≥ 11.8.
    #[allow(dead_code)]
    Cuda(usize),

    /// AMD GPU via ROCm / HIP.
    /// Requires `--features hip` and ROCm ≥ 5.0 (`ROCM_PATH` env set).
    #[allow(dead_code)]
    Hip(usize),

    /// Apple GPU via Metal.
    /// Requires `--features metal` and macOS 13+ with Apple Silicon.
    #[allow(dead_code)]
    Metal,

    /// Cross-platform Vulkan GPU (AMD, Intel, NVIDIA without CUDA, etc.).
    /// Requires `--features vulkan` (wgpu backend).
    #[allow(dead_code)]
    Vulkan(usize),
}

impl Device {
    /// Resolve a device string to the best concrete `Device`, falling back
    /// to CPU with a warning if unavailable.
    ///
    /// Accepted strings:
    ///   `"cpu"`, `"auto"`, `"cuda"`, `"cuda:N"`, `"hip"`, `"hip:N"`,
    ///   `"metal"`, `"vulkan"`, `"vulkan:N"`, `"rocm"`, `"rocm:N"`.
    pub fn resolve(requested: &str) -> Self {
        match requested.to_lowercase().trim() {
            "cpu" => Device::Cpu,
            "auto" => Self::auto_select(),
            s if s.starts_with("cuda") => {
                let idx = s.strip_prefix("cuda:").and_then(|n| n.parse().ok()).unwrap_or(0);
                Self::try_cuda(idx)
            }
            // "rocm" is an alias for "hip" (common in Python tooling)
            s if s.starts_with("hip") || s.starts_with("rocm") => {
                let idx = s
                    .strip_prefix("hip:")
                    .or_else(|| s.strip_prefix("rocm:"))
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                Self::try_hip(idx)
            }
            "metal" => Self::try_metal(),
            s if s.starts_with("vulkan") => {
                let idx = s.strip_prefix("vulkan:").and_then(|n| n.parse().ok()).unwrap_or(0);
                Self::try_vulkan(idx)
            }
            other => {
                log::warn!("Unknown device '{}' — falling back to CPU", other);
                Device::Cpu
            }
        }
    }

    /// Select the best available device at runtime.
    ///
    /// Priority: CUDA → HIP → Metal → Vulkan → CPU.
    pub fn auto_select() -> Self {
        // NVIDIA CUDA
        #[cfg(feature = "cuda")]
        {
            if let Some(d) = Self::best_cuda() {
                log::info!("Auto-selected {}", d);
                return d;
            }
        }

        // AMD ROCm / HIP
        #[cfg(feature = "hip")]
        {
            if let Some(d) = Self::best_hip() {
                log::info!("Auto-selected {}", d);
                return d;
            }
        }

        // Apple Metal
        #[cfg(feature = "metal")]
        {
            if Self::metal_available() {
                log::info!("Auto-selected Metal");
                return Device::Metal;
            }
        }

        // Cross-platform Vulkan
        #[cfg(feature = "vulkan")]
        {
            if let Some(d) = Self::best_vulkan() {
                log::info!("Auto-selected {}", d);
                return d;
            }
        }

        log::info!("Auto-selected CPU (no GPU features compiled in or no GPU detected)");
        Device::Cpu
    }

    // ── CUDA (NVIDIA) ─────────────────────────────────────────────────────────

    fn try_cuda(idx: usize) -> Self {
        #[cfg(feature = "cuda")]
        {
            match candle_core::Device::new_cuda(idx) {
                Ok(_) => {
                    log::info!("CUDA device {} available", idx);
                    return Device::Cuda(idx);
                }
                Err(e) => {
                    log::warn!("CUDA device {} not available: {} — falling back to CPU", idx, e);
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!(
                "CUDA requested but NVE was not compiled with --features cuda. \
                 Rebuild: cargo build --release --features cuda"
            );
        }
        Device::Cpu
    }

    #[cfg(feature = "cuda")]
    fn best_cuda() -> Option<Device> {
        use candle_core::cuda::CudaDevice;
        // candle-core 0.8+: probe indices until failure (count() removed).
        // memory_info() also removed from CudaDevice wrapper; just pick first.
        for i in 0usize..8 {
            if CudaDevice::new_with_stream(i).is_ok() {
                return Some(Device::Cuda(i));
            }
        }
        None
    }

    // ── HIP / ROCm (AMD) ─────────────────────────────────────────────────────

    fn try_hip(idx: usize) -> Self {
        #[cfg(feature = "hip")]
        {
            // candle-core with HIP uses the same CUDA API surface via HIP.
            match candle_core::Device::new_cuda(idx) {
                Ok(_) => {
                    log::info!("HIP/ROCm device {} available", idx);
                    return Device::Hip(idx);
                }
                Err(e) => {
                    log::warn!("HIP/ROCm device {} not available: {} — falling back to CPU", idx, e);
                }
            }
        }
        #[cfg(not(feature = "hip"))]
        {
            log::warn!(
                "HIP/ROCm requested but NVE was not compiled with --features hip. \
                 Rebuild: cargo build --release --features hip  (requires ROCm ≥ 5.0)"
            );
        }
        Device::Cpu
    }

    #[cfg(feature = "hip")]
    fn best_hip() -> Option<Device> {
        use candle_core::cuda::CudaDevice;
        for i in 0usize..8 {
            if CudaDevice::new_with_stream(i).is_ok() {
                return Some(Device::Hip(i));
            }
        }
        None
    }

    // ── Metal (Apple) ─────────────────────────────────────────────────────────

    fn try_metal() -> Self {
        #[cfg(feature = "metal")]
        {
            match candle_core::Device::new_metal(0) {
                Ok(_) => {
                    log::info!("Metal device available");
                    return Device::Metal;
                }
                Err(e) => {
                    log::warn!("Metal device not available: {} — falling back to CPU", e);
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            log::warn!(
                "Metal requested but NVE was not compiled with --features metal. \
                 Rebuild: cargo build --release --features metal"
            );
        }
        Device::Cpu
    }

    #[cfg(feature = "metal")]
    fn metal_available() -> bool {
        candle_core::Device::new_metal(0).is_ok()
    }

    // ── Vulkan (cross-platform via wgpu) ──────────────────────────────────────

    fn try_vulkan(idx: usize) -> Self {
        #[cfg(feature = "vulkan")]
        {
            if Self::vulkan_adapter_count() > idx {
                log::info!("Vulkan adapter {} available", idx);
                return Device::Vulkan(idx);
            } else {
                log::warn!(
                    "Vulkan adapter {} not found ({} adapter(s) detected) — falling back to CPU",
                    idx,
                    Self::vulkan_adapter_count()
                );
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            log::warn!(
                "Vulkan requested but NVE was not compiled with --features vulkan. \
                 Rebuild: cargo build --release --features vulkan"
            );
        }
        Device::Cpu
    }

    #[cfg(feature = "vulkan")]
    fn best_vulkan() -> Option<Device> {
        let count = Self::vulkan_adapter_count();
        if count > 0 { Some(Device::Vulkan(0)) } else { None }
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_count() -> usize {
        // wgpu adapter enumeration is async; we use a blocking poll here.
        // This is called only during startup, so it is acceptable.
        use wgpu::{Backends, Instance, InstanceDescriptor};
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });
        instance.enumerate_adapters(Backends::VULKAN).count()
    }

    // ── candle Device conversion ──────────────────────────────────────────────

    /// Convert to a `candle_core::Device`.
    ///
    /// Only available when compiled with at least one candle feature
    /// (`cuda`, `hip`, `metal`, `mkl`, or `accelerate`).
    /// CPU always resolves; GPU variants fall back to CPU if the matching
    /// feature flag is not compiled in.
    #[cfg(any(feature = "cuda", feature = "hip", feature = "metal",
              feature = "mkl", feature = "accelerate"))]
    pub fn to_candle(&self) -> candle_core::Result<candle_core::Device> {
        match self {
            Device::Cpu => Ok(candle_core::Device::Cpu),

            #[cfg(feature = "cuda")]
            Device::Cuda(idx) => candle_core::Device::new_cuda(*idx),

            #[cfg(feature = "hip")]
            Device::Hip(idx) => {
                // HIP reuses the CUDA API surface in candle.
                candle_core::Device::new_cuda(*idx)
            }

            #[cfg(feature = "metal")]
            Device::Metal => candle_core::Device::new_metal(0),

            // Vulkan: no candle_core backend yet — fall back to CPU.
            // Actual Vulkan compute is handled by the wgpu path in nve::vulkan_backend.
            #[cfg(feature = "vulkan")]
            Device::Vulkan(_) => {
                log::warn!(
                    "candle_core does not yet have a Vulkan backend; \
                     using CPU for tensor ops."
                );
                Ok(candle_core::Device::Cpu)
            }

            // Catch-all: variant exists but matching feature not compiled in.
            #[allow(unreachable_patterns)]
            _ => Ok(candle_core::Device::Cpu),
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    pub fn is_hip(&self) -> bool {
        matches!(self, Device::Hip(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal)
    }

    pub fn is_vulkan(&self) -> bool {
        matches!(self, Device::Vulkan(_))
    }

    /// Human-readable name for logging and CLI display.
    pub fn name(&self) -> String {
        match self {
            Device::Cpu       => "cpu".to_string(),
            Device::Cuda(i)   => format!("cuda:{}", i),
            Device::Hip(i)    => format!("hip:{}", i),
            Device::Metal     => "metal".to_string(),
            Device::Vulkan(i) => format!("vulkan:{}", i),
        }
    }

    /// Approximate free memory in bytes on this device.
    /// Returns `None` on CPU or when the query fails.
    pub fn free_memory_bytes(&self) -> Option<usize> {
        match self {
            Device::Cpu | Device::Metal | Device::Vulkan(_) => None,

            // memory_info() was removed from candle_core::CudaDevice in 0.8.
            // Free memory is not queryable through the candle wrapper API.
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => None,

            #[cfg(feature = "hip")]
            Device::Hip(_) => None,

            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for Device {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::resolve(s))
    }
}

// ── Device capability info ────────────────────────────────────────────────────

/// Runtime information about a compute device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device: Device,
    pub name: String,
    pub vendor: String,
    pub vram_bytes: Option<usize>,
    pub free_vram_bytes: Option<usize>,
    pub driver_version: Option<String>,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
}

/// Enumerate all available compute devices (CPU + any GPUs).
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let mut devices = vec![DeviceInfo {
        device: Device::Cpu,
        name: "CPU (AVX2+FMA SIMD)".to_string(),
        vendor: "Generic".to_string(),
        vram_bytes: None,
        free_vram_bytes: None,
        driver_version: None,
        supports_fp16: true,
        supports_bf16: cfg!(target_arch = "aarch64"), // ARM64 (Apple M, Graviton)
    }];

    // ── NVIDIA CUDA ──────────────────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda::CudaDevice;
        for i in 0usize..8 {
            match CudaDevice::new_with_stream(i) {
                Ok(_) => {
                    devices.push(DeviceInfo {
                        device: Device::Cuda(i),
                        name: format!("NVIDIA GPU {} (CUDA)", i),
                        vendor: "NVIDIA".to_string(),
                        vram_bytes: None, // memory_info() removed in candle 0.8
                        free_vram_bytes: None,
                        driver_version: None,
                        supports_fp16: true,
                        supports_bf16: true,
                    });
                }
                Err(_) => break,
            }
        }
    }

    // ── AMD ROCm / HIP ───────────────────────────────────────────────────────
    #[cfg(feature = "hip")]
    {
        use candle_core::cuda::CudaDevice;
        for i in 0usize..8 {
            match CudaDevice::new_with_stream(i) {
                Ok(_) => {
                    devices.push(DeviceInfo {
                        device: Device::Hip(i),
                        name: format!("AMD GPU {} (ROCm/HIP)", i),
                        vendor: "AMD".to_string(),
                        vram_bytes: None,
                        free_vram_bytes: None,
                        driver_version: None,
                        supports_fp16: true,
                        supports_bf16: true,
                    });
                }
                Err(_) => break,
            }
        }
    }

    // ── Apple Metal ───────────────────────────────────────────────────────────
    #[cfg(feature = "metal")]
    {
        if candle_core::Device::new_metal(0).is_ok() {
            devices.push(DeviceInfo {
                device: Device::Metal,
                name: "Apple GPU (Metal)".to_string(),
                vendor: "Apple".to_string(),
                vram_bytes: None,       // Unified memory — not separately queryable
                free_vram_bytes: None,
                driver_version: None,
                supports_fp16: true,
                supports_bf16: false,   // MPS does not support bf16 natively as of 2024
            });
        }
    }

    // ── Vulkan (cross-platform: AMD without ROCm, Intel, etc.) ───────────────
    #[cfg(feature = "vulkan")]
    {
        use wgpu::{Backends, Instance, InstanceDescriptor};
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });
        for (i, adapter) in instance.enumerate_adapters(Backends::VULKAN).enumerate() {
            let info = adapter.get_info();
            devices.push(DeviceInfo {
                device: Device::Vulkan(i),
                name: info.name.clone(),
                vendor: format!("{:?}", info.vendor),
                vram_bytes: None,
                free_vram_bytes: None,
                driver_version: Some(format!("{}", info.driver)),
                supports_fp16: true,
                supports_bf16: false,
            });
        }
    }

    devices
}
