use std::collections::{HashMap, HashSet};
use std::io::BufReader;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::config::{DiffusionHeadConfig, KugelAudioConfig};
use crate::error::{KugelAudioError, Result};
use crate::model::acoustic_decoder::AcousticDecoder;
use crate::model::connector::SpeechConnector;
use crate::model::diffusion_head::DiffusionHead;
use crate::model::lm::Lm;
use crate::model::quantized_qwen2::QuantizedQwen2Model;
use crate::model::qwen2::{Qwen2Config, Qwen2Model};

// ---------------------------------------------------------------------------
// KugelAudioModel
// ---------------------------------------------------------------------------

/// Complete KugelAudio model for inference.
pub struct KugelAudioModel {
    /// Qwen2 language model backbone (full-precision or quantized via [`Lm`] enum).
    pub lm: Lm,
    /// Speech connector: acoustic features → LM hidden space.
    pub acoustic_connector: SpeechConnector,
    /// Diffusion prediction head.
    pub prediction_head: DiffusionHead,
    /// Acoustic tokenizer decoder (latent → audio).
    pub acoustic_decoder: AcousticDecoder,
    /// Scaling factor for speech latents.
    pub speech_scaling_factor: f32,
    /// Bias factor for speech latents.
    pub speech_bias_factor: f32,
    /// Number of diffusion inference steps.
    pub ddpm_inference_steps: i32,
    /// Acoustic VAE dimensionality (typically 64).
    pub vae_dim: usize,
    /// Diffusion head configuration (needed by the generation pipeline).
    pub diffusion_head_config: DiffusionHeadConfig,
    /// Compute device.
    pub device: Device,
    /// Model dtype (BF16 for inference).
    pub dtype: DType,
}

// ---------------------------------------------------------------------------
// Safetensor shard index
// ---------------------------------------------------------------------------

/// Minimal representation of `model.safetensors.index.json`.
#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// Read `model.safetensors.index.json` and return the sorted list of unique
/// shard file paths found in the same directory.
fn find_safetensor_shards(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let file = std::fs::File::open(&index_path).map_err(|e| {
        KugelAudioError::WeightLoading(format!("Cannot open {}: {e}", index_path.display()))
    })?;

    let index: SafetensorIndex = serde_json::from_reader(file)?;

    // Collect unique shard filenames in deterministic order.
    let mut seen = HashSet::new();
    let mut shards: Vec<String> = index
        .weight_map
        .into_values()
        .filter(|name| seen.insert(name.clone()))
        .collect();
    shards.sort();

    if shards.is_empty() {
        return Err(KugelAudioError::WeightLoading(
            "weight_map in index.json is empty".to_string(),
        ));
    }

    Ok(shards
        .into_iter()
        .map(|name| model_dir.join(name))
        .collect())
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

fn load_config(model_dir: &Path) -> Result<KugelAudioConfig> {
    let path = model_dir.join("config.json");
    let file = std::fs::File::open(&path)
        .map_err(|e| KugelAudioError::Config(format!("Cannot open {}: {e}", path.display())))?;
    serde_json::from_reader(file)
        .map_err(|e| KugelAudioError::Config(format!("Failed to parse config.json: {e}")))
}

// ---------------------------------------------------------------------------
// GGUF helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the GGUF tensor name belongs to the Qwen2 LM backbone.
///
/// LM tensors follow the llama.cpp naming convention:
/// - `token_embd.weight` — token embedding
/// - `output.weight` — LM head (may be tied to embedding)
/// - `output_norm.weight` — final RMS norm
/// - `blk.{i}.*` — decoder layer weights (attention, MLP, norms)
fn is_lm_tensor(name: &str) -> bool {
    name.starts_with("blk.")
        || name == "token_embd.weight"
        || name == "output.weight"
        || name == "output_norm.weight"
}

/// Extract a scalar f32 value from a dequantized tensor map.
///
/// Handles both 0-d scalar tensors and 1-d single-element tensors by
/// flattening before extraction.
fn extract_scalar_tensor(map: &HashMap<String, Tensor>, name: &str) -> Result<f32> {
    let t = map.get(name).ok_or_else(|| {
        KugelAudioError::WeightLoading(format!("{name} not found in GGUF tensors"))
    })?;
    let values: Vec<f32> = t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| KugelAudioError::WeightLoading(format!("{name} read error: {e}")))?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| KugelAudioError::WeightLoading(format!("{name} tensor is empty")))
}

/// Load [`KugelAudioConfig`] from GGUF metadata or a fallback `config.json`.
///
/// Tries the `kugelaudio.config` GGUF metadata key (JSON string) first.
/// Falls back to `config.json` in `gguf_dir` if the metadata key is absent.
fn load_kugelaudio_config_from_gguf(
    ct: &gguf_file::Content,
    gguf_dir: &Path,
) -> Result<KugelAudioConfig> {
    // Try GGUF metadata first
    if let Some(gguf_file::Value::String(json_str)) = ct.metadata.get("kugelaudio.config") {
        return serde_json::from_str(json_str).map_err(|e| {
            KugelAudioError::Config(format!(
                "Failed to parse 'kugelaudio.config' GGUF metadata: {e}"
            ))
        });
    }

    // Fallback: config.json alongside the GGUF file
    let config_path = gguf_dir.join("config.json");
    let file = std::fs::File::open(&config_path).map_err(|e| {
        KugelAudioError::Config(format!(
            "GGUF missing 'kugelaudio.config' metadata and no config.json at {}: {e}",
            config_path.display()
        ))
    })?;
    serde_json::from_reader(file).map_err(|e| {
        KugelAudioError::Config(format!("Failed to parse {}: {e}", config_path.display()))
    })
}

// ---------------------------------------------------------------------------
// Safetensors loading (directory)
// ---------------------------------------------------------------------------

/// Load the KugelAudio model from a safetensors model directory.
///
/// The directory must contain:
/// - `config.json`
/// - `model.safetensors.index.json`
/// - `model-0000X-of-00004.safetensors` shard files
///
/// # Safety
///
/// Uses `VarBuilder::from_mmaped_safetensors` which memory-maps the shard
/// files. The files must not be modified while this function is running.
fn load_model_safetensors(model_dir: &Path, device: &Device) -> Result<KugelAudioModel> {
    // Resolve symlinks and normalize the path before memory-mapping to
    // prevent path-traversal attacks.
    let model_dir = model_dir.canonicalize().map_err(|e| {
        KugelAudioError::WeightLoading(format!(
            "Cannot resolve model directory {}: {e}",
            model_dir.display()
        ))
    })?;
    if !model_dir.is_dir() {
        return Err(KugelAudioError::WeightLoading(format!(
            "{} is not a directory",
            model_dir.display()
        )));
    }

    let config = load_config(&model_dir)?;
    // BF16 for GPU (Metal/CUDA), F32 for CPU (which lacks BF16 matmul).
    let dtype = match device {
        Device::Cpu => DType::F32,
        _ => DType::BF16,
    };

    // Build VarBuilder over all shards.
    let shard_paths = find_safetensor_shards(&model_dir)?;
    // SAFETY: files are resolved through canonicalize() above and must not
    // be modified while mapped. This is inherent to memory-mapped I/O.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, device).map_err(|e| {
            KugelAudioError::WeightLoading(format!("Failed to mmap safetensors: {e}"))
        })?
    };

    // LM backbone — only the TTS subset of layers (e.g. 20 of 28).
    let mut qwen2_cfg = Qwen2Config::from_decoder_config(&config.decoder_config);
    qwen2_cfg.num_hidden_layers = config.tts_layers() as usize;
    let lm_head_vb = if qwen2_cfg.tie_word_embeddings {
        None
    } else {
        Some(vb.clone())
    };
    let lm = Lm::Full(
        Qwen2Model::new(&qwen2_cfg, vb.pp("model").pp("language_model"), lm_head_vb)
            .map_err(|e| KugelAudioError::Model(format!("Failed to load language model: {e}")))?,
    );

    // Speech connector.
    let acoustic_connector = SpeechConnector::new(
        config.vae_dim() as usize,
        config.decoder_config.hidden_size as usize,
        1e-6,
        vb.pp("model").pp("acoustic_connector"),
    )
    .map_err(|e| KugelAudioError::Model(format!("Failed to load acoustic_connector: {e}")))?;

    // Diffusion prediction head.
    let prediction_head = DiffusionHead::new(
        &config.diffusion_head_config,
        vb.pp("model").pp("prediction_head"),
    )
    .map_err(|e| KugelAudioError::Model(format!("Failed to load prediction_head: {e}")))?;

    // Acoustic decoder.
    let acoustic_decoder = AcousticDecoder::load(
        &config.acoustic_tokenizer_config,
        vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
    )
    .map_err(|e| KugelAudioError::Model(format!("Failed to load acoustic_decoder: {e}")))?;

    // Scalar buffers stored as 0-d tensors in the checkpoint.
    let speech_scaling_factor = vb
        .get(&[], "model.speech_scaling_factor")
        .map_err(|e| KugelAudioError::WeightLoading(format!("speech_scaling_factor missing: {e}")))?
        .to_dtype(DType::F32)
        .and_then(|t| t.to_scalar::<f32>())
        .map_err(|e| {
            KugelAudioError::WeightLoading(format!("speech_scaling_factor read error: {e}"))
        })?;

    let speech_bias_factor = vb
        .get(&[], "model.speech_bias_factor")
        .map_err(|e| KugelAudioError::WeightLoading(format!("speech_bias_factor missing: {e}")))?
        .to_dtype(DType::F32)
        .and_then(|t| t.to_scalar::<f32>())
        .map_err(|e| {
            KugelAudioError::WeightLoading(format!("speech_bias_factor read error: {e}"))
        })?;

    // ddpm_inference_steps: prefer top-level config override, fall back to
    // diffusion_head_config.
    let ddpm_inference_steps = config
        .ddpm_inference_steps
        .unwrap_or(config.diffusion_head_config.ddpm_num_inference_steps);

    let vae_dim = config.vae_dim() as usize;
    let diffusion_head_config = config.diffusion_head_config;

    Ok(KugelAudioModel {
        lm,
        acoustic_connector,
        prediction_head,
        acoustic_decoder,
        speech_scaling_factor,
        speech_bias_factor,
        ddpm_inference_steps,
        vae_dim,
        diffusion_head_config,
        device: device.clone(),
        dtype,
    })
}

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

/// Load the complete KugelAudio model from a GGUF file.
///
/// The GGUF file must contain all model tensors:
/// - **LM backbone** (quantized): standard llama.cpp naming convention
///   (`token_embd.weight`, `blk.{i}.*`, `output.weight`, `output_norm.weight`)
/// - **Non-LM components** (dequantized at load time): HuggingFace naming
///   (`model.acoustic_connector.*`, `model.prediction_head.*`,
///   `model.acoustic_tokenizer.decoder.*`, `model.speech_scaling_factor`,
///   `model.speech_bias_factor`)
///
/// # GGUF Metadata Keys
///
/// ## KugelAudio configuration
///
/// | Key                 | Type   | Description                                  |
/// |---------------------|--------|----------------------------------------------|
/// | `kugelaudio.config` | String | Full [`KugelAudioConfig`] serialized as JSON  |
///
/// If `kugelaudio.config` is absent, falls back to `config.json` in the same
/// directory as the GGUF file.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be parsed, configuration is
/// missing, or any required tensor is absent.
pub fn load_model_gguf(path: &Path, device: &Device) -> Result<KugelAudioModel> {
    let path = path.canonicalize().map_err(|e| {
        KugelAudioError::WeightLoading(format!("Cannot resolve GGUF path {}: {e}", path.display()))
    })?;
    let gguf_dir = path.parent().unwrap_or(Path::new("."));

    // Quantized models run in F32: QMatMul dequantizes on the fly and returns
    // F32 tensors. Non-LM components must use the same dtype to avoid matmul
    // mismatches when the pipeline passes LM hidden states to them.
    let dtype = DType::F32;

    // Parse GGUF header and tensor metadata
    let file = std::fs::File::open(&path).map_err(|e| {
        KugelAudioError::WeightLoading(format!("Cannot open {}: {e}", path.display()))
    })?;
    let mut reader = BufReader::new(file);
    let ct = gguf_file::Content::read(&mut reader)
        .map_err(|e| KugelAudioError::WeightLoading(format!("Failed to parse GGUF: {e}")))?;

    // Log quantization info from GGUF metadata
    if let Some(gguf_file::Value::String(name)) = ct.metadata.get("general.name") {
        eprint!("GGUF model: {name}");
    }
    // Detect quantization type from the first LM weight matrix
    let quant_type = ct
        .tensor_infos
        .get("blk.0.attn_q.weight")
        .map(|info| format!("{:?}", info.ggml_dtype));
    if let Some(qt) = &quant_type {
        eprintln!(" (quantization: {qt})");
    } else {
        eprintln!();
    }

    // Load KugelAudio config from GGUF metadata or fallback config.json
    let config = load_kugelaudio_config_from_gguf(&ct, gguf_dir)?;

    // Build Qwen2 config for the TTS backbone subset
    let mut qwen2_cfg = Qwen2Config::from_decoder_config(&config.decoder_config);
    qwen2_cfg.num_hidden_layers = config.tts_layers() as usize;

    // Load quantized LM backbone from GGUF tensors
    let quantized_lm = QuantizedQwen2Model::new(&qwen2_cfg, &ct, &mut reader, device)
        .map_err(|e| KugelAudioError::Model(format!("Failed to load quantized LM: {e}")))?;
    let lm = Lm::Quantized(quantized_lm);

    // Dequantize non-LM tensors for DiffusionHead, AcousticDecoder, SpeechConnector.
    // These tensors use HuggingFace naming (e.g. `model.prediction_head.cond_proj.weight`)
    // and are stored quantized in the GGUF but loaded as full-precision via dequantization.
    let mut non_lm_tensors: HashMap<String, Tensor> = HashMap::new();
    for name in ct.tensor_infos.keys() {
        if !is_lm_tensor(name) {
            let qt = ct.tensor(&mut reader, name, device).map_err(|e| {
                KugelAudioError::WeightLoading(format!("Failed to read GGUF tensor '{name}': {e}"))
            })?;
            let tensor = qt.dequantize(device).map_err(|e| {
                KugelAudioError::WeightLoading(format!("Failed to dequantize tensor '{name}': {e}"))
            })?;
            non_lm_tensors.insert(name.clone(), tensor);
        }
    }

    // Extract scalar factors before handing tensors to VarBuilder
    let speech_scaling_factor =
        extract_scalar_tensor(&non_lm_tensors, "model.speech_scaling_factor")?;
    let speech_bias_factor = extract_scalar_tensor(&non_lm_tensors, "model.speech_bias_factor")?;

    // Build VarBuilder from dequantized non-LM tensors
    let vb = VarBuilder::from_tensors(non_lm_tensors, dtype, device);

    // Speech connector
    let acoustic_connector = SpeechConnector::new(
        config.vae_dim() as usize,
        config.decoder_config.hidden_size as usize,
        1e-6,
        vb.pp("model").pp("acoustic_connector"),
    )
    .map_err(|e| KugelAudioError::Model(format!("Failed to load acoustic_connector: {e}")))?;

    // Diffusion prediction head
    let prediction_head = DiffusionHead::new(
        &config.diffusion_head_config,
        vb.pp("model").pp("prediction_head"),
    )
    .map_err(|e| KugelAudioError::Model(format!("Failed to load prediction_head: {e}")))?;

    // Acoustic decoder
    let acoustic_decoder = AcousticDecoder::load(
        &config.acoustic_tokenizer_config,
        vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
    )
    .map_err(|e| KugelAudioError::Model(format!("Failed to load acoustic_decoder: {e}")))?;

    let ddpm_inference_steps = config
        .ddpm_inference_steps
        .unwrap_or(config.diffusion_head_config.ddpm_num_inference_steps);

    let vae_dim = config.vae_dim() as usize;
    let diffusion_head_config = config.diffusion_head_config;

    Ok(KugelAudioModel {
        lm,
        acoustic_connector,
        prediction_head,
        acoustic_decoder,
        speech_scaling_factor,
        speech_bias_factor,
        ddpm_inference_steps,
        vae_dim,
        diffusion_head_config,
        device: device.clone(),
        dtype,
    })
}

// ---------------------------------------------------------------------------
// Public entry point (auto-detection)
// ---------------------------------------------------------------------------

/// Load the complete KugelAudio model, auto-detecting the format.
///
/// - If `model_path` is a file ending in `.gguf` → [`load_model_gguf`]
/// - If `model_path` is a directory → safetensors loading from sharded files
///
/// The directory variant requires `config.json`, `model.safetensors.index.json`,
/// and the corresponding shard files.
///
/// # Errors
///
/// Returns an error if the path is neither a `.gguf` file nor a safetensors
/// directory, or if model loading fails.
pub fn load_model(model_path: &Path, device: &Device) -> Result<KugelAudioModel> {
    if model_path.is_file()
        && model_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        load_model_gguf(model_path, device)
    } else if model_path.is_dir() {
        load_model_safetensors(model_path, device)
    } else {
        Err(KugelAudioError::WeightLoading(format!(
            "{} is neither a .gguf file nor a model directory",
            model_path.display()
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_lm_tensor() {
        // LM backbone tensors (should return true)
        assert!(is_lm_tensor("token_embd.weight"));
        assert!(is_lm_tensor("output.weight"));
        assert!(is_lm_tensor("output_norm.weight"));
        assert!(is_lm_tensor("blk.0.attn_q.weight"));
        assert!(is_lm_tensor("blk.27.ffn_down.weight"));
        assert!(is_lm_tensor("blk.5.attn_norm.weight"));

        // Non-LM tensors (should return false)
        assert!(!is_lm_tensor("model.acoustic_connector.fc1.weight"));
        assert!(!is_lm_tensor("model.prediction_head.cond_proj.weight"));
        assert!(!is_lm_tensor(
            "model.acoustic_tokenizer.decoder.head.conv.conv.weight"
        ));
        assert!(!is_lm_tensor("model.speech_scaling_factor"));
        assert!(!is_lm_tensor("model.speech_bias_factor"));
    }

    #[test]
    fn test_extract_scalar_tensor_0d() {
        let device = Device::Cpu;
        let mut map = HashMap::new();
        let t = Tensor::new(1.5f32, &device).unwrap();
        map.insert("val".to_string(), t);

        let v = extract_scalar_tensor(&map, "val").unwrap();
        assert!((v - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_extract_scalar_tensor_1d() {
        let device = Device::Cpu;
        let mut map = HashMap::new();
        let t = Tensor::new(&[2.5f32], &device).unwrap();
        map.insert("val".to_string(), t);

        let v = extract_scalar_tensor(&map, "val").unwrap();
        assert!((v - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_extract_scalar_tensor_missing() {
        let map: HashMap<String, Tensor> = HashMap::new();
        assert!(extract_scalar_tensor(&map, "missing").is_err());
    }

    #[test]
    fn test_load_model_nonexistent_path() {
        let device = Device::Cpu;
        let result = load_model(Path::new("/nonexistent/path"), &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_auto_detect() {
        let device = Device::Cpu;

        // Create temp artifacts to exercise real auto-detection branches.
        let tmp_dir = std::env::temp_dir().join("kugelaudio_test_gguf_detect");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let gguf_path = tmp_dir.join("model.gguf");
        std::fs::write(&gguf_path, b"not a real gguf").unwrap();

        // A .gguf file routes to load_model_gguf (fails parsing the invalid content).
        let gguf_result = load_model(&gguf_path, &device);
        assert!(gguf_result.is_err());
        let err_msg = format!("{}", gguf_result.err().expect("should fail"));
        assert!(
            err_msg.contains("Failed to parse GGUF"),
            "Expected GGUF parse error, got: {err_msg}"
        );

        // A directory routes to load_model_safetensors (fails on missing config).
        let dir_result = load_model(&tmp_dir, &device);
        assert!(dir_result.is_err());
        let err_msg = format!("{}", dir_result.err().expect("should fail"));
        assert!(
            err_msg.contains("config.json"),
            "Expected safetensors config error, got: {err_msg}"
        );

        // A nonexistent path hits the fallback branch.
        let bad_result = load_model(Path::new("/nonexistent/path"), &device);
        assert!(bad_result.is_err());
        let err_msg = format!("{}", bad_result.err().expect("should fail"));
        assert!(
            err_msg.contains("neither a .gguf file nor a model directory"),
            "Expected format detection error, got: {err_msg}"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
