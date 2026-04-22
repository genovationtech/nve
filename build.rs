// build.rs — compile NVE custom CUDA kernels when the `cuda` feature is enabled.
//
// When `cargo build --features cuda` is run:
//  1. If NVE_KERNELS_PREBUILT=<dir> is set, skip nvcc and link the pre-built lib in <dir>.
//  2. Otherwise, finds nvcc and compiles cuda/nve_kernels.cu → libnve_kernels.a in OUT_DIR.
//
// The CUDA_COMPUTE_CAP env var (e.g. "75" for T4) controls the PTX target.
// Defaults to 75 if unset.

fn main() {
    // Only compile CUDA kernels when the cuda feature is active.
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    println!("cargo:rerun-if-changed=cuda/nve_kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVE_KERNELS_PREBUILT");

    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Link CUDA runtime from cuda_path.
    if !cuda_path.is_empty() {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-lib=cudart");

    // ── Fast path: use a pre-built shared library ────────────────────────────
    // Static CUDA archives (.a) contain CUDA fat-binary ELF sections that lld
    // cannot resolve.  A shared library (.so) compiled by nvcc is lld-safe.
    if let Ok(prebuilt_dir) = std::env::var("NVE_KERNELS_PREBUILT") {
        println!("cargo:warning=build.rs: using pre-built libnve_kernels.so from {}", prebuilt_dir);
        println!("cargo:rustc-link-search=native={}", prebuilt_dir);
        println!("cargo:rustc-link-lib=nve_kernels");   // dynamic link
        return;
    }

    // ── Full path: compile with nvcc ─────────────────────────────────────────
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernel_src = format!("{}/cuda/nve_kernels.cu", manifest_dir);

    if !std::path::Path::new(&kernel_src).exists() {
        panic!("CUDA kernel source not found at {}", kernel_src);
    }

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let compute_cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "75".to_string());

    let obj_path = format!("{}/nve_kernels.o", out_dir);
    let lib_path = format!("{}/libnve_kernels.a", out_dir);

    println!("cargo:warning=build.rs: nvcc={} compute_cap={}", nvcc, compute_cap);

    let nvcc_out = std::process::Command::new(&nvcc)
        .args([
            "-O3",
            "--use_fast_math",
            &format!("-arch=compute_{}", compute_cap),
            &format!("-code=sm_{}", compute_cap),
            "-Xcompiler", "-fPIC",
            "-c", &kernel_src,
            "-o", &obj_path,
        ])
        .output()
        .unwrap_or_else(|e| panic!("nvcc not found at {}: {}", nvcc, e));

    if !nvcc_out.status.success() {
        let stderr = String::from_utf8_lossy(&nvcc_out.stderr);
        let stdout = String::from_utf8_lossy(&nvcc_out.stdout);
        panic!(
            "nvcc compilation failed (exit={}).\nSTDOUT:\n{}\nSTDERR:\n{}",
            nvcc_out.status, stdout, stderr
        );
    }

    let ar_out = std::process::Command::new("ar")
        .args(["rcs", &lib_path, &obj_path])
        .output()
        .expect("ar not found");

    if !ar_out.status.success() {
        let stderr = String::from_utf8_lossy(&ar_out.stderr);
        panic!("ar failed: {}", stderr);
    }

    println!("cargo:warning=build.rs: libnve_kernels.a created OK");
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=nve_kernels");
}
