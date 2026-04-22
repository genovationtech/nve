//! Safetensors file format parser with memory-mapped I/O.
//!
//! Loads model weights directly from disk via mmap for zero-copy access.
//! Supports both single-file and sharded (multi-file) safetensors models.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;
use thiserror::Error;

use crate::tensor::{DType, Tensor, TensorError};

#[derive(Debug, Error)]
pub enum SafetensorsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid header: {0}")]
    InvalidHeader(String),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("invalid file: header_size={header_size} but file is only {file_size} bytes")]
    FileTooSmall { header_size: u64, file_size: u64 },
}

/// Metadata for a single tensor in the safetensors file.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// A memory-mapped safetensors file.
pub struct SafetensorsFile {
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsFile {
    /// Open and parse a single safetensors file.
    pub fn open(path: &Path) -> Result<Self, SafetensorsError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(SafetensorsError::InvalidHeader(
                "file too small for header".into(),
            ));
        }

        // First 8 bytes: header size as u64 LE.
        let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap());

        if header_size > 100 * 1024 * 1024 {
            return Err(SafetensorsError::InvalidHeader(format!(
                "header size {} exceeds 100MB limit",
                header_size
            )));
        }

        let total_header = 8 + header_size as usize;
        if mmap.len() < total_header {
            return Err(SafetensorsError::FileTooSmall {
                header_size,
                file_size: mmap.len() as u64,
            });
        }

        // Parse JSON header.
        let header_json = std::str::from_utf8(&mmap[8..total_header])
            .map_err(|e| SafetensorsError::InvalidHeader(format!("invalid UTF-8: {}", e)))?;

        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;

        let mut tensors = HashMap::new();
        for (key, value) in raw {
            if key == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value)?;
            tensors.insert(key, info);
        }

        Ok(SafetensorsFile {
            mmap,
            header_size: total_header,
            tensors,
        })
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor metadata without loading data.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get raw bytes for a tensor (zero-copy from mmap).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let start = self.header_size + info.data_offsets[0];
        let end = self.header_size + info.data_offsets[1];

        if end > self.mmap.len() {
            return Err(SafetensorsError::InvalidHeader(format!(
                "tensor '{}' data range [{}, {}) exceeds file size {}",
                name,
                start,
                end,
                self.mmap.len()
            )));
        }

        Ok(&self.mmap[start..end])
    }

    /// Load a tensor as f32 (converting from bf16/f16 if needed).
    pub fn load_tensor(&self, name: &str) -> Result<Tensor, SafetensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let bytes = self.tensor_bytes(name)?;
        let dtype = DType::from_safetensors_str(&info.dtype)?;
        let tensor = Tensor::from_bytes(bytes, info.shape.clone(), dtype)?;
        Ok(tensor)
    }

    /// Load a tensor in its native dtype without f32 conversion (compact storage).
    pub fn load_compact(&self, name: &str) -> Result<crate::tensor::CompactTensor, SafetensorsError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let bytes = self.tensor_bytes(name)?;
        let dtype = DType::from_safetensors_str(&info.dtype)?;
        Ok(crate::tensor::CompactTensor::new(bytes.to_vec(), info.shape.clone(), dtype))
    }

    /// Total number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Index file for sharded safetensors models.
#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub weight_map: HashMap<String, String>,
}

/// A collection of safetensors files (handles sharded models).
pub struct ModelWeights {
    files: HashMap<String, SafetensorsFile>,
    weight_map: HashMap<String, String>,
    model_dir: PathBuf,
}

impl ModelWeights {
    /// Load model weights from a directory.
    ///
    /// Handles both single-file (`model.safetensors`) and sharded models
    /// (`model.safetensors.index.json` + `model-00001-of-00002.safetensors`).
    pub fn load(model_dir: &Path) -> Result<Self, SafetensorsError> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        if index_path.exists() {
            Self::load_sharded(model_dir, &index_path)
        } else if single_path.exists() {
            Self::load_single(model_dir, &single_path)
        } else {
            Err(SafetensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no safetensors files found in {:?}", model_dir),
            )))
        }
    }

    fn load_single(model_dir: &Path, path: &Path) -> Result<Self, SafetensorsError> {
        let file = SafetensorsFile::open(path)?;
        let filename = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let weight_map: HashMap<String, String> = file
            .tensor_names()
            .iter()
            .map(|name| (name.to_string(), filename.clone()))
            .collect();

        let mut files = HashMap::new();
        files.insert(filename, file);

        Ok(ModelWeights {
            files,
            weight_map,
            model_dir: model_dir.to_path_buf(),
        })
    }

    fn load_sharded(model_dir: &Path, index_path: &Path) -> Result<Self, SafetensorsError> {
        let index_json = std::fs::read_to_string(index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_json)?;

        // Collect unique shard filenames.
        let shard_files: std::collections::HashSet<&str> =
            index.weight_map.values().map(|s| s.as_str()).collect();

        let mut files = HashMap::new();
        for shard_name in shard_files {
            let shard_path = model_dir.join(shard_name);
            let file = SafetensorsFile::open(&shard_path)?;
            files.insert(shard_name.to_string(), file);
        }

        Ok(ModelWeights {
            files,
            weight_map: index.weight_map,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Load a tensor by name as f32 (resolves to correct shard file).
    pub fn load_tensor(&self, name: &str) -> Result<Tensor, SafetensorsError> {
        let filename = self
            .weight_map
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let file = self.files.get(filename).ok_or_else(|| {
            SafetensorsError::TensorNotFound(format!(
                "shard file '{}' not loaded for tensor '{}'",
                filename, name
            ))
        })?;

        file.load_tensor(name)
    }

    /// Load a tensor in its native dtype (compact, ~half memory for bf16).
    pub fn load_compact(&self, name: &str) -> Result<crate::tensor::CompactTensor, SafetensorsError> {
        let filename = self
            .weight_map
            .get(name)
            .ok_or_else(|| SafetensorsError::TensorNotFound(name.to_string()))?;

        let file = self.files.get(filename).ok_or_else(|| {
            SafetensorsError::TensorNotFound(format!(
                "shard file '{}' not loaded for tensor '{}'",
                filename, name
            ))
        })?;

        file.load_compact(name)
    }

    /// Get tensor metadata without loading.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let filename = self.weight_map.get(name)?;
        let file = self.files.get(filename)?;
        file.tensor_info(name)
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    /// List all tensor names across all shards.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.weight_map.keys().map(|s| s.as_str()).collect()
    }

    /// Total number of tensors.
    pub fn len(&self) -> usize {
        self.weight_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weight_map.is_empty()
    }

    /// Get the model directory path.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Compute total size of all tensors in bytes (on disk).
    pub fn total_size_bytes(&self) -> usize {
        let mut total = 0;
        for name in self.weight_map.keys() {
            if let Some(info) = self.tensor_info(name) {
                total += info.data_offsets[1] - info.data_offsets[0];
            }
        }
        total
    }

    /// List all tensor names matching a prefix (e.g., "model.layers.0.").
    pub fn tensors_with_prefix(&self, prefix: &str) -> Vec<&str> {
        self.weight_map
            .keys()
            .filter(|k| k.starts_with(prefix))
            .map(|s| s.as_str())
            .collect()
    }
}

impl fmt::Debug for ModelWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelWeights(dir={:?}, tensors={}, shards={})",
            self.model_dir,
            self.weight_map.len(),
            self.files.len(),
        )
    }
}

use std::fmt;

impl fmt::Debug for SafetensorsFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SafetensorsFile(tensors={}, data_bytes={})",
            self.tensors.len(),
            self.mmap.len() - self.header_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a minimal valid safetensors file for testing.
    fn create_test_safetensors(dir: &Path, filename: &str) -> PathBuf {
        let path = dir.join(filename);

        // Create a simple tensor: "test_weight" with shape [2, 3], dtype F32
        // 6 floats * 4 bytes = 24 bytes of data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let header = serde_json::json!({
            "test_weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24]
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut file = File::create(&path).unwrap();
        file.write_all(&header_size.to_le_bytes()).unwrap();
        file.write_all(header_bytes).unwrap();
        file.write_all(&data_bytes).unwrap();

        path
    }

    #[test]
    fn test_open_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_safetensors(dir.path(), "test.safetensors");

        let sf = SafetensorsFile::open(&path).unwrap();
        assert_eq!(sf.len(), 1);
        assert!(sf.tensor_info("test_weight").is_some());
    }

    #[test]
    fn test_load_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_safetensors(dir.path(), "test.safetensors");

        let sf = SafetensorsFile::open(&path).unwrap();
        let tensor = sf.load_tensor("test_weight").unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = create_test_safetensors(dir.path(), "test.safetensors");

        let sf = SafetensorsFile::open(&path).unwrap();
        assert!(sf.load_tensor("nonexistent").is_err());
    }

    #[test]
    fn test_model_weights_single_file() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(dir.path(), "model.safetensors");

        let weights = ModelWeights::load(dir.path()).unwrap();
        assert_eq!(weights.len(), 1);
        assert!(weights.has_tensor("test_weight"));

        let tensor = weights.load_tensor("test_weight").unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }
}
