use std::collections::HashSet;
use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::config::{DiffusionHeadConfig, KugelAudioConfig};
use crate::error::{KugelAudioError, Result};
use crate::model::acoustic_decoder::AcousticDecoder;
use crate::model::connector::SpeechConnector;
use crate::model::diffusion_head::DiffusionHead;
use crate::model::qwen2::{Qwen2Config, Qwen2Model};

// ---------------------------------------------------------------------------
// KugelAudioModel
// ---------------------------------------------------------------------------

/// Complete KugelAudio model for inference.
pub struct KugelAudioModel {
    /// Qwen2 language model backbone (TTS layers only).
    pub lm: Qwen2Model,
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
// Public entry point
// ---------------------------------------------------------------------------

/// Load the complete KugelAudio model from `model_dir`.
///
/// The directory must contain:
/// - `config.json`
/// - `model.safetensors.index.json`
/// - `model-0000X-of-00004.safetensors` shard files
///
/// # Errors
///
/// Returns an error if any config file is missing or malformed, if any shard
/// file cannot be memory-mapped, or if a required weight tensor is absent.
///
/// # Safety
///
/// Uses `VarBuilder::from_mmaped_safetensors` which memory-maps the shard
/// files. The files must not be modified while this function is running.
pub fn load_model(model_dir: &Path, device: &Device) -> Result<KugelAudioModel> {
    let config = load_config(model_dir)?;
    // BF16 for GPU (Metal/CUDA), F32 for CPU (which lacks BF16 matmul).
    let dtype = match device {
        Device::Cpu => DType::F32,
        _ => DType::BF16,
    };

    // Build VarBuilder over all shards.
    let shard_paths = find_safetensor_shards(model_dir)?;
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
    let lm = Qwen2Model::new(&qwen2_cfg, vb.pp("model").pp("language_model"), lm_head_vb)
        .map_err(|e| KugelAudioError::Model(format!("Failed to load language model: {e}")))?;

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
