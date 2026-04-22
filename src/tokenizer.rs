//! BPE Tokenizer for Llama 3 models.
//!
//! Parses HuggingFace `tokenizer.json` format and implements tiktoken-compatible
//! byte-pair encoding with pre-tokenization regex and special token handling.

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid tokenizer format: {0}")]
    InvalidFormat(String),
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),
}

/// A BPE tokenizer compatible with Llama 3 models.
pub struct Tokenizer {
    /// Token string → ID.
    encoder: HashMap<Vec<u8>, u32>,
    /// ID → token bytes.
    decoder: Vec<Vec<u8>>,
    /// Merge pairs with their rank (lower = higher priority).
    merges: HashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Pre-tokenization regex pattern.
    pattern: Regex,
    /// Special tokens.
    special_tokens: HashMap<String, u32>,
    /// Reverse special tokens: ID → string.
    special_decoder: HashMap<u32, String>,
    /// BOS token ID.
    pub bos_id: u32,
    /// EOS token ID.
    pub eos_id: u32,
    /// Vocab size.
    vocab_size: usize,
}

// ── HuggingFace tokenizer.json structures ──

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelSection,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(rename = "type")]
    model_type: Option<String>,
    vocab: HashMap<String, u32>,
    merges: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

impl Tokenizer {
    /// Load tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Load tokenizer from a model directory (looks for tokenizer.json).
    pub fn from_model_dir(dir: &Path) -> Result<Self, TokenizerError> {
        Self::from_file(&dir.join("tokenizer.json"))
    }

    /// Parse tokenizer from JSON string.
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let parsed: TokenizerJson = serde_json::from_str(json)?;

        // Build encoder: map token strings to byte sequences → IDs.
        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut max_id = 0u32;

        for (token_str, &id) in &parsed.model.vocab {
            let bytes = Self::decode_token_str(token_str);
            encoder.insert(bytes, id);
            max_id = max_id.max(id);
        }

        // Add special tokens.
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut special_decoder: HashMap<u32, String> = HashMap::new();

        for token in &parsed.added_tokens {
            special_tokens.insert(token.content.clone(), token.id);
            special_decoder.insert(token.id, token.content.clone());
            max_id = max_id.max(token.id);
            if token.special {
                encoder.insert(token.content.as_bytes().to_vec(), token.id);
            }
        }

        // Build decoder (ID → bytes).
        let vocab_size = (max_id + 1) as usize;
        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];
        for (bytes, &id) in &encoder {
            decoder[id as usize] = bytes.clone();
        }

        // Parse merges (handles both string "a b" and array ["a", "b"] formats).
        let mut merges: HashMap<(Vec<u8>, Vec<u8>), u32> = HashMap::new();
        for (rank, merge_val) in parsed.model.merges.iter().enumerate() {
            let (part_a, part_b) = match merge_val {
                serde_json::Value::String(s) => {
                    let parts: Vec<&str> = s.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        (parts[0].to_string(), parts[1].to_string())
                    } else {
                        continue;
                    }
                }
                serde_json::Value::Array(arr) if arr.len() == 2 => {
                    if let (Some(a), Some(b)) = (arr[0].as_str(), arr[1].as_str()) {
                        (a.to_string(), b.to_string())
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            let a = Self::decode_token_str(&part_a);
            let b = Self::decode_token_str(&part_b);
            merges.insert((a, b), rank as u32);
        }

        // Pre-tokenization regex.
        // Llama 3 pattern (simplified for Rust regex crate compatibility).
        let pattern = Self::build_pretokenize_pattern(&parsed.pre_tokenizer)?;

        // Find BOS/EOS — check multiple conventions.
        let bos_id = special_tokens
            .get("<|begin_of_text|>")
            .or_else(|| special_tokens.get("<|endoftext|>"))
            .or_else(|| special_tokens.get("<s>"))
            .copied()
            .unwrap_or(0);
        let eos_id = special_tokens
            .get("<|end_of_text|>")
            .or_else(|| special_tokens.get("<|endoftext|>"))
            .or_else(|| special_tokens.get("</s>"))
            .copied()
            .unwrap_or(0);

        Ok(Tokenizer {
            encoder,
            decoder,
            merges,
            pattern,
            special_tokens,
            special_decoder,
            bos_id,
            eos_id,
            vocab_size,
        })
    }

    /// Decode a token string from the vocab (handles unicode escapes and byte-level encoding).
    fn decode_token_str(s: &str) -> Vec<u8> {
        // HuggingFace byte-level BPE uses a unicode mapping where bytes 0-255
        // are mapped to printable unicode characters. We need to reverse this.
        let mut bytes = Vec::new();
        for ch in s.chars() {
            let b = BYTE_DECODER.get(&ch);
            match b {
                Some(&byte) => bytes.push(byte),
                None => {
                    // Fall back to UTF-8 encoding of the character.
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(encoded.as_bytes());
                }
            }
        }
        bytes
    }

    fn build_pretokenize_pattern(
        pre_tokenizer: &Option<serde_json::Value>,
    ) -> Result<Regex, TokenizerError> {
        // Try to extract pattern from pre_tokenizer config.
        if let Some(pt) = pre_tokenizer {
            if let Some(pattern) = Self::extract_pattern(pt) {
                if let Ok(re) = Regex::new(&pattern) {
                    return Ok(re);
                }
            }
        }

        // Fallback: GPT-4 / Llama 3 compatible pattern.
        // Simplified for Rust's regex crate (no lookbehind, limited Unicode classes).
        let pattern = concat!(
            r"'(?:s|t|re|ve|m|ll|d)|",
            r"[^\r\n\p{L}\p{N}]?\p{L}+|",
            r"\p{N}{1,3}|",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
            r"\s*[\r\n]+|",
            r"\s+",
        );
        Ok(Regex::new(pattern)?)
    }

    fn extract_pattern(value: &serde_json::Value) -> Option<String> {
        // Navigate the pre_tokenizer JSON to find a Split pattern.
        if let Some(pretokenizers) = value.get("pretokenizers").and_then(|v| v.as_array()) {
            for pt in pretokenizers {
                if let Some(pattern) = pt
                    .get("pattern")
                    .and_then(|p| p.get("Regex").or(p.get("String")))
                    .and_then(|s| s.as_str())
                {
                    return Some(pattern.to_string());
                }
            }
        }
        // Direct pattern.
        if let Some(pattern) = value
            .get("pattern")
            .and_then(|p| p.get("Regex").or(p.get("String")))
            .and_then(|s| s.as_str())
        {
            return Some(pattern.to_string());
        }
        None
    }

    // ── Encode ──

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Pre-tokenize: split text into chunks using regex.
        for mat in self.pattern.find_iter(text) {
            let chunk = mat.as_str();
            let chunk_bytes: Vec<u8> = chunk.bytes().collect();

            // Check if the whole chunk is a single token.
            if let Some(&id) = self.encoder.get(&chunk_bytes) {
                tokens.push(id);
                continue;
            }

            // BPE: start with individual bytes as tokens.
            let mut parts: Vec<Vec<u8>> = chunk_bytes.iter().map(|&b| vec![b]).collect();

            // Iteratively merge the highest-priority pair.
            loop {
                if parts.len() < 2 {
                    break;
                }

                // Find the pair with lowest merge rank.
                let mut best_rank = u32::MAX;
                let mut best_idx = usize::MAX;

                for i in 0..parts.len() - 1 {
                    if let Some(&rank) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }

                if best_idx == usize::MAX {
                    break; // No more merges possible.
                }

                // Merge the pair.
                let mut merged = parts[best_idx].clone();
                merged.extend_from_slice(&parts[best_idx + 1]);
                parts.splice(best_idx..best_idx + 2, std::iter::once(merged));
            }

            // Look up IDs for resulting tokens.
            for part in &parts {
                if let Some(&id) = self.encoder.get(part) {
                    tokens.push(id);
                } else {
                    // Unknown token — encode individual bytes.
                    for &b in part {
                        if let Some(&id) = self.encoder.get(&vec![b]) {
                            tokens.push(id);
                        }
                    }
                }
            }
        }

        tokens
    }

    /// Encode text with BOS token prepended.
    pub fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let tokens = self.encode(text);
        // Skip BOS for models where BOS == EOS (GPT-2 style: <|endoftext|> serves both roles)
        if self.bos_id == self.eos_id {
            return tokens;
        }
        let mut result = vec![self.bos_id];
        result.extend(tokens);
        result
    }

    // ── Decode ──

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in tokens {
            if let Some(special) = self.special_decoder.get(&id) {
                bytes.extend_from_slice(special.as_bytes());
            } else if (id as usize) < self.decoder.len() {
                bytes.extend_from_slice(&self.decoder[id as usize]);
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token to its string representation.
    pub fn decode_token(&self, id: u32) -> String {
        if let Some(special) = self.special_decoder.get(&id) {
            return special.clone();
        }
        if (id as usize) < self.decoder.len() {
            return String::from_utf8_lossy(&self.decoder[id as usize]).into_owned();
        }
        format!("<unk:{}>", id)
    }

    // ── Accessors ──

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_decoder.contains_key(&id)
    }

    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Get EOS token IDs (Llama 3 can have multiple).
    pub fn eos_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.eos_id];
        if let Some(&id) = self.special_tokens.get("<|eot_id|>") {
            ids.push(id);
        }
        ids
    }
}

// ── Byte-level BPE unicode mapping ──
// Maps byte values 0-255 to unicode characters used in the vocab.
// This is the standard GPT-2 / tiktoken byte encoder.

static BYTE_DECODER: std::sync::LazyLock<HashMap<char, u8>> =
    std::sync::LazyLock::new(|| {
        let enc = build_byte_encoder();
        enc.into_iter().map(|(k, v)| (v, k)).collect()
    });

fn build_byte_encoder() -> HashMap<u8, char> {
    let mut mapping = HashMap::new();
    let mut n = 0u32;

    // Printable ASCII and Latin-1 supplement ranges map to themselves.
    let ranges: Vec<std::ops::RangeInclusive<u8>> =
        vec![b'!'..=b'~', 0xA1..=0xAC, 0xAE..=0xFF];

    let mut in_range = [false; 256];
    for range in ranges {
        for b in range {
            in_range[b as usize] = true;
        }
    }

    for b in 0u16..=255 {
        let byte = b as u8;
        if in_range[byte as usize] {
            mapping.insert(byte, char::from_u32(byte as u32).unwrap());
        } else {
            mapping.insert(byte, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    mapping
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tokenizer() -> Tokenizer {
        // Create a minimal tokenizer with byte-level vocab for testing.
        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::new();

        // Map individual bytes 0-255 to IDs 0-255.
        for b in 0u8..=255 {
            encoder.insert(vec![b], b as u32);
        }

        // Add some merged tokens.
        encoder.insert(b"th".to_vec(), 256);
        encoder.insert(b"the".to_vec(), 257);
        encoder.insert(b" the".to_vec(), 258);
        encoder.insert(b"he".to_vec(), 259);

        let vocab_size = 260;
        let mut decoder = vec![Vec::new(); vocab_size];
        for (bytes, &id) in &encoder {
            if (id as usize) < vocab_size {
                decoder[id as usize] = bytes.clone();
            }
        }

        let mut merges = HashMap::new();
        merges.insert((vec![b't'], vec![b'h']), 0); // t+h -> th (rank 0)
        merges.insert((vec![b't', b'h'], vec![b'e']), 1); // th+e -> the (rank 1)
        merges.insert((vec![b' '], vec![b't', b'h', b'e']), 2); // ' '+the -> ' the' (rank 2)
        merges.insert((vec![b'h'], vec![b'e']), 3); // h+e -> he (rank 3)

        let pattern = Regex::new(r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .unwrap();

        Tokenizer {
            encoder,
            decoder,
            merges,
            pattern,
            special_tokens: HashMap::new(),
            special_decoder: HashMap::new(),
            bos_id: 0,
            eos_id: 1,
            vocab_size,
        }
    }

    #[test]
    fn test_encode_simple() {
        let tok = make_simple_tokenizer();
        let tokens = tok.encode("the");
        // "the" should match the pre-tokenizer, then BPE merges t+h→th, th+e→the
        assert_eq!(tokens, vec![257]); // "the" = 257
    }

    #[test]
    fn test_decode() {
        let tok = make_simple_tokenizer();
        let text = tok.decode(&[257]);
        assert_eq!(text, "the");
    }

    #[test]
    fn test_byte_encoder_roundtrip() {
        let enc = build_byte_encoder();
        let dec: HashMap<char, u8> = enc.iter().map(|(&k, &v)| (v, k)).collect();
        // Every byte should map to a unique char and back.
        for b in 0u8..=255 {
            let ch = enc[&b];
            assert_eq!(dec[&ch], b);
        }
    }
}
