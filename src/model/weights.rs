use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::{ModuleParameters, ModuleParametersExt, Param},
    nn,
    quantization::{MaybeQuantized, Quantizable},
    Array,
};
use mlx_sys;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use mlx_rs_core::initialize_rope;
use qwen3_mlx::qwen2;

use crate::config::KugelAudioConfig;
use crate::error::{KugelAudioError, Result};
use crate::model::acoustic_decoder::AcousticDecoder;
use crate::model::connector::SpeechConnector;
use crate::model::diffusion_head::DiffusionHead;

/// Complete KugelAudio model for inference.
pub struct KugelAudioModel {
    /// Qwen2 language model (reused from qwen3-mlx)
    pub lm: qwen2::Model,
    /// Speech connector: acoustic features -> LM hidden space
    pub acoustic_connector: SpeechConnector,
    /// Diffusion prediction head
    pub prediction_head: DiffusionHead,
    /// Acoustic tokenizer decoder (latent -> audio)
    pub acoustic_decoder: AcousticDecoder,
    /// Scaling factor for speech latents
    pub speech_scaling_factor: f32,
    /// Bias factor for speech latents
    pub speech_bias_factor: f32,
    /// Number of diffusion inference steps
    pub ddpm_inference_steps: i32,
}

impl KugelAudioModel {
    /// Quantize the LM backbone to 4-bit (group_size=64).
    /// Reduces memory ~4x and speeds up bandwidth-bound inference.
    pub fn quantize_lm(&mut self, group_size: i32, bits: i32) -> Result<()> {
        eprintln!("Quantizing LM to {bits}-bit (group_size={group_size})...");
        // try_into_quantized consumes and returns Self (MaybeQuantized fields switch internally)
        let args_clone = self.lm.args.clone();
        let lm = std::mem::replace(
            &mut self.lm,
            qwen2::Model::new(args_clone).map_err(|e| KugelAudioError::Model(format!("{e}")))?,
        );
        self.lm = lm
            .try_into_quantized(group_size, bits)
            .map_err(|e| KugelAudioError::Model(format!("Quantization failed: {e}")))?;
        self.lm
            .eval()
            .map_err(|e| KugelAudioError::Model(format!("Failed to eval quantized LM: {e}")))?;
        // Release original bf16 arrays from MLX's memory pool back to the OS.
        // Without this, mlx holds the freed bf16 weights in its pool and memory stays high.
        unsafe { mlx_sys::mlx_clear_cache() };
        eprintln!("LM quantized.");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pre-quantized model save / load
// ---------------------------------------------------------------------------

/// Metadata for a pre-quantized model directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedMeta {
    pub group_size: i32,
    pub bits: i32,
    pub speech_scaling_factor: f32,
    pub speech_bias_factor: f32,
    pub ddpm_inference_steps: i32,
}

const QUANTIZED_META_FILE: &str = "quantized_config.json";

/// Save a (quantized) model to a directory with per-component safetensors.
///
/// Output layout:
///   lm.safetensors, connector.safetensors, prediction_head.safetensors,
///   decoder.safetensors, quantized_config.json
pub fn save_quantized_model(
    model: &KugelAudioModel,
    output_dir: impl AsRef<Path>,
    group_size: i32,
    bits: i32,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    eprintln!("Saving LM weights...");
    save_lm_safetensors(&model.lm, &output_dir.join("lm.safetensors"))?;

    eprintln!("Saving connector weights...");
    model
        .acoustic_connector
        .save_safetensors(output_dir.join("connector.safetensors"))
        .map_err(|e| KugelAudioError::WeightLoading(format!("Save connector: {e}")))?;

    eprintln!("Saving prediction_head weights...");
    model
        .prediction_head
        .save_safetensors(output_dir.join("prediction_head.safetensors"))
        .map_err(|e| KugelAudioError::WeightLoading(format!("Save prediction_head: {e}")))?;

    eprintln!("Saving acoustic_decoder weights...");
    model
        .acoustic_decoder
        .save_safetensors(output_dir.join("decoder.safetensors"))
        .map_err(|e| KugelAudioError::WeightLoading(format!("Save decoder: {e}")))?;

    let meta = QuantizedMeta {
        group_size,
        bits,
        speech_scaling_factor: model.speech_scaling_factor,
        speech_bias_factor: model.speech_bias_factor,
        ddpm_inference_steps: model.ddpm_inference_steps,
    };
    std::fs::write(
        output_dir.join(QUANTIZED_META_FILE),
        serde_json::to_string_pretty(&meta)?,
    )?;

    Ok(())
}

/// Load a pre-quantized model from a directory created by `save_quantized_model`.
///
/// Peak memory ≈ quantized model size (~4-6 GB for 7B 4-bit) instead of
/// the full bf16 weight set (~14 GB).
fn load_quantized_model(model_dir: &Path) -> Result<KugelAudioModel> {
    let meta: QuantizedMeta = {
        let path = model_dir.join(QUANTIZED_META_FILE);
        let file = std::fs::File::open(&path)?;
        serde_json::from_reader(file)?
    };

    // We still need the original config for model dimensions.
    let config: KugelAudioConfig = {
        let path = model_dir.join("config.json");
        let file = std::fs::File::open(&path)?;
        serde_json::from_reader(file)?
    };

    eprintln!(
        "Loading pre-quantized model ({}-bit, group_size={})...",
        meta.bits, meta.group_size
    );

    // Build quantized LM skeleton with dummy arrays, then load real weights.
    let mut lm = build_quantized_lm_skeleton(&config, meta.group_size, meta.bits)?;
    load_lm_safetensors(&mut lm, &model_dir.join("lm.safetensors"))?;
    eprintln!("  LM loaded.");

    // Non-LM components: build with default weights, then overwrite from safetensors.
    let mut acoustic_connector =
        SpeechConnector::new(config.vae_dim(), config.decoder_config.hidden_size, 1e-6).map_err(
            |e| KugelAudioError::Model(format!("Failed to create SpeechConnector: {e}")),
        )?;
    acoustic_connector
        .load_safetensors(model_dir.join("connector.safetensors"))
        .map_err(|e| KugelAudioError::WeightLoading(format!("Load connector: {e}")))?;
    eprintln!("  Connector loaded.");

    let mut prediction_head = DiffusionHead::new(&config.diffusion_head_config)
        .map_err(|e| KugelAudioError::Model(format!("Failed to create DiffusionHead: {e}")))?;
    prediction_head
        .load_safetensors(model_dir.join("prediction_head.safetensors"))
        .map_err(|e| KugelAudioError::WeightLoading(format!("Load prediction_head: {e}")))?;
    eprintln!("  Prediction head loaded.");

    let mut acoustic_decoder = AcousticDecoder::new(&config.acoustic_tokenizer_config)
        .map_err(|e| KugelAudioError::Model(format!("Failed to create AcousticDecoder: {e}")))?;
    acoustic_decoder
        .load_safetensors(model_dir.join("decoder.safetensors"))
        .map_err(|e| KugelAudioError::WeightLoading(format!("Load decoder: {e}")))?;
    eprintln!("  Decoder loaded.");

    eprintln!("Pre-quantized model loaded.");
    Ok(KugelAudioModel {
        lm,
        acoustic_connector,
        prediction_head,
        acoustic_decoder,
        speech_scaling_factor: meta.speech_scaling_factor,
        speech_bias_factor: meta.speech_bias_factor,
        ddpm_inference_steps: meta.ddpm_inference_steps,
    })
}

/// Build a qwen2::Model with QuantizedLinear/QuantizedEmbedding layers
/// and zero-weight placeholders. The caller must `load_safetensors` to
/// fill in real weights. This avoids materializing 14 GB of random bf16.
fn build_quantized_lm_skeleton(
    config: &KugelAudioConfig,
    group_size: i32,
    bits: i32,
) -> Result<qwen2::Model> {
    let tts_layers = config.tts_layers();
    let args = qwen2::ModelArgs {
        model_type: config.decoder_config.model_type.clone(),
        hidden_size: config.decoder_config.hidden_size,
        num_hidden_layers: tts_layers,
        intermediate_size: config.decoder_config.intermediate_size,
        num_attention_heads: config.decoder_config.num_attention_heads,
        rms_norm_eps: config.decoder_config.rms_norm_eps,
        vocab_size: config.decoder_config.vocab_size,
        num_key_value_heads: config.decoder_config.num_key_value_heads,
        max_position_embeddings: config.decoder_config.max_position_embeddings,
        rope_theta: config.decoder_config.rope_theta,
        rope_traditional: false,
        rope_scaling: None,
        tie_word_embeddings: config.decoder_config.tie_word_embeddings,
        quantization: Some(qwen2::QuantizationConfig { group_size, bits }),
    };

    let head_dim = args.head_dim();

    let mut layers = Vec::with_capacity(tts_layers as usize);
    for _ in 0..tts_layers {
        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
            &args.rope_scaling,
            args.max_position_embeddings,
        )
        .map_err(|e| KugelAudioError::Model(format!("rope init: {e}")))?;

        let attention = qwen2::Attention {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            scale: (head_dim as f32).sqrt().recip(),
            // Qwen2: q/k/v have bias=true, o_proj has bias=false
            q_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, true)),
            k_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, true)),
            v_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, true)),
            o_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, false)),
            rope,
        };

        let mlp = qwen2::Mlp {
            gate_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, false)),
            down_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, false)),
            up_proj: MaybeQuantized::Quantized(dummy_quantized_linear(group_size, bits, false)),
        };

        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()
            .map_err(|e| KugelAudioError::Model(format!("layernorm: {e}")))?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()
            .map_err(|e| KugelAudioError::Model(format!("layernorm: {e}")))?;

        layers.push(qwen2::TransformerBlock {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn: attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        });
    }

    let qwen2_model = qwen2::Qwen2Model {
        vocab_size: args.vocab_size,
        num_hidden_layers: tts_layers,
        embed_tokens: MaybeQuantized::Quantized(dummy_quantized_embedding(group_size, bits)),
        layers,
        norm: nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()
            .map_err(|e| KugelAudioError::Model(format!("final norm: {e}")))?,
    };

    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(dummy_quantized_linear(
            group_size, bits, false,
        )))
    } else {
        None
    };

    Ok(qwen2::Model {
        args,
        model: qwen2_model,
        lm_head,
    })
}

/// QuantizedLinear with zero-scalar placeholders. `load_safetensors` replaces
/// every Array, so shapes don't matter. `with_bias` must match the original
/// layer (Qwen2 q/k/v have bias=true, all others bias=false).
fn dummy_quantized_linear(group_size: i32, bits: i32, with_bias: bool) -> nn::QuantizedLinear {
    let z = array!(0.0f32);
    let bias = if with_bias { Some(z.clone()) } else { None };
    nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(z.clone()),
        biases: Param::new(z.clone()),
        inner: nn::Linear {
            weight: Param::new(z),
            bias: Param::new(bias),
        },
    }
}

/// QuantizedEmbedding with zero-scalar placeholders.
fn dummy_quantized_embedding(group_size: i32, bits: i32) -> nn::QuantizedEmbedding {
    let z = array!(0.0f32);
    nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(z.clone()),
        biases: Param::new(z.clone()),
        inner: nn::Embedding {
            weight: Param::new(z),
        },
    }
}

/// Save quantized LM weights to safetensors.
///
/// Workaround: upstream `QuantizedEmbedding` is missing `#[param]` attributes,
/// so `ModuleParametersExt::save_safetensors` silently skips embed_tokens weights.
/// We collect them manually and include in the save.
fn save_lm_safetensors(lm: &qwen2::Model, path: &Path) -> Result<()> {
    // Collect standard params (all QuantizedLinear layers, norms, etc.)
    let mut params: HashMap<String, Array> = HashMap::new();
    for (key, array) in lm.parameters().flatten() {
        params.insert(key.to_string(), array.clone());
    }

    // Manually add QuantizedEmbedding fields (missing #[param] upstream)
    match &lm.model.embed_tokens {
        MaybeQuantized::Quantized(qe) => {
            params.insert(
                "model.embed_tokens.inner.weight".into(),
                qe.inner.weight.value.clone(),
            );
            params.insert("model.embed_tokens.scales".into(), qe.scales.value.clone());
            params.insert("model.embed_tokens.biases".into(), qe.biases.value.clone());
        }
        MaybeQuantized::Original(e) => {
            params.insert("model.embed_tokens.weight".into(), e.weight.value.clone());
        }
    }

    Array::save_safetensors(&params, None, path)
        .map_err(|e| KugelAudioError::WeightLoading(format!("Save LM: {e}")))?;
    Ok(())
}

/// Load quantized LM weights from safetensors into a pre-built skeleton.
///
/// Same workaround as `save_lm_safetensors`: QuantizedEmbedding fields must
/// be set manually since they're invisible to ModuleParameters.
fn load_lm_safetensors(lm: &mut qwen2::Model, path: &Path) -> Result<()> {
    let loaded = Array::load_safetensors(path)
        .map_err(|e| KugelAudioError::WeightLoading(format!("Load LM safetensors: {e}")))?;

    let total_loaded = loaded.len();

    // Phase 1: standard params via ModuleParameters
    let mut matched = 0usize;
    {
        let mut params = lm.parameters_mut().flatten();
        for (key, value) in &loaded {
            if let Some(param) = params.get_mut(&**key) {
                **param = value.clone();
                matched += 1;
            }
        }
    } // drop flattened map to release mutable borrow

    // Phase 2: manually set QuantizedEmbedding fields (missing #[param] upstream)
    if let MaybeQuantized::Quantized(ref mut qe) = lm.model.embed_tokens {
        if let Some(w) = loaded.get("model.embed_tokens.inner.weight") {
            qe.inner.weight.value = w.clone();
            matched += 1;
        }
        if let Some(s) = loaded.get("model.embed_tokens.scales") {
            qe.scales.value = s.clone();
            matched += 1;
        }
        if let Some(b) = loaded.get("model.embed_tokens.biases") {
            qe.biases.value = b.clone();
            matched += 1;
        }
    }

    eprintln!("    LM: {matched}/{total_loaded} weights loaded",);

    lm.eval()
        .map_err(|e| KugelAudioError::WeightLoading(format!("Eval quantized LM: {e}")))?;
    Ok(())
}

/// Weight map from safetensors index file.
#[derive(Debug, Clone, Deserialize)]
struct SafetensorsIndex {
    #[allow(dead_code)]
    metadata: HashMap<String, Value>,
    weight_map: HashMap<String, String>,
}

/// Load the complete KugelAudio model from a model directory.
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<KugelAudioModel> {
    let model_dir = model_dir.as_ref();

    // Detect pre-quantized format
    if model_dir.join(QUANTIZED_META_FILE).exists() {
        return load_quantized_model(model_dir);
    }

    // Load config
    let config: KugelAudioConfig = {
        let config_path = model_dir.join("config.json");
        let file = std::fs::File::open(&config_path)?;
        serde_json::from_reader(file)?
    };

    // Build Qwen2 model args for the LM backbone
    let tts_layers = config.tts_layers();
    let lm_args = qwen2::ModelArgs {
        model_type: config.decoder_config.model_type.clone(),
        hidden_size: config.decoder_config.hidden_size,
        num_hidden_layers: tts_layers,
        intermediate_size: config.decoder_config.intermediate_size,
        num_attention_heads: config.decoder_config.num_attention_heads,
        rms_norm_eps: config.decoder_config.rms_norm_eps,
        vocab_size: config.decoder_config.vocab_size,
        num_key_value_heads: config.decoder_config.num_key_value_heads,
        max_position_embeddings: config.decoder_config.max_position_embeddings,
        rope_theta: config.decoder_config.rope_theta,
        rope_traditional: false,
        rope_scaling: None,
        tie_word_embeddings: config.decoder_config.tie_word_embeddings,
        quantization: None,
    };

    // Build all model components
    let lm = qwen2::Model::new(lm_args)
        .map_err(|e| KugelAudioError::Model(format!("Failed to create Qwen2: {}", e)))?;

    let acoustic_connector =
        SpeechConnector::new(config.vae_dim(), config.decoder_config.hidden_size, 1e-6).map_err(
            |e| KugelAudioError::Model(format!("Failed to create SpeechConnector: {}", e)),
        )?;

    let prediction_head = DiffusionHead::new(&config.diffusion_head_config)
        .map_err(|e| KugelAudioError::Model(format!("Failed to create DiffusionHead: {}", e)))?;

    let acoustic_decoder = AcousticDecoder::new(&config.acoustic_tokenizer_config)
        .map_err(|e| KugelAudioError::Model(format!("Failed to create AcousticDecoder: {}", e)))?;

    let mut model = KugelAudioModel {
        lm,
        acoustic_connector,
        prediction_head,
        acoustic_decoder,
        speech_scaling_factor: f32::NAN,
        speech_bias_factor: f32::NAN,
        ddpm_inference_steps: config.diffusion_head_config.ddpm_num_inference_steps,
    };

    // Load weights from safetensors
    load_weights(&mut model, model_dir)?;

    Ok(model)
}

// ---------------------------------------------------------------------------
// Key remapping rules
// ---------------------------------------------------------------------------
//
// KugelAudio checkpoint key         →  Rust component + expected key
// ─────────────────────────────────────────────────────────────────────
// model.language_model.*             →  lm  (qwen2::Model)
//   model.language_model.embed_tokens.weight    → model.embed_tokens.weight
//   model.language_model.layers.N.*             → model.layers.N.*
//   model.language_model.norm.weight            → model.norm.weight
//
// lm_head.weight                     →  lm  (qwen2::Model)
//   lm_head.weight                              → lm_head.weight
//
// model.acoustic_connector.*         →  acoustic_connector (SpeechConnector)
//   model.acoustic_connector.fc1.weight         → fc1.weight
//   model.acoustic_connector.norm.weight        → norm.weight
//   model.acoustic_connector.fc2.weight         → fc2.weight
//
// model.prediction_head.*            →  prediction_head (DiffusionHead)
//   model.prediction_head.noisy_images_proj.weight → noisy_images_proj.weight
//   model.prediction_head.cond_proj.weight         → cond_proj.weight
//   model.prediction_head.t_embedder.mlp.0.weight  → t_embedder.linear1.weight
//   model.prediction_head.t_embedder.mlp.2.weight  → t_embedder.linear2.weight
//   model.prediction_head.layers.N.ffn.*            → layers.N.ffn.*
//   model.prediction_head.layers.N.norm.weight      → layers.N.norm.weight
//   model.prediction_head.layers.N.adaLN_modulation.1.weight → layers.N.ada_ln_linear.weight
//   model.prediction_head.final_layer.*             → final_layer.*
//
// model.acoustic_tokenizer.decoder.* →  acoustic_decoder (AcousticDecoder)
//   (complex hierarchical conv structure)
//
// model.speech_scaling_factor        →  scalar buffer
// model.speech_bias_factor           →  scalar buffer
// ---------------------------------------------------------------------------

/// Remap a KugelAudio checkpoint key to the target component and local key.
///
/// Returns (component, local_key) where component is one of:
/// "lm", "acoustic_connector", "prediction_head", "acoustic_decoder", "scalar", "skip"
fn remap_key(checkpoint_key: &str) -> (&'static str, String) {
    // Language model
    if let Some(rest) = checkpoint_key.strip_prefix("model.language_model.") {
        return ("lm", format!("model.{rest}"));
    }

    // LM head (top-level, not under model.)
    if checkpoint_key.starts_with("lm_head.") {
        return ("lm", checkpoint_key.to_string());
    }

    // Acoustic connector
    if let Some(rest) = checkpoint_key.strip_prefix("model.acoustic_connector.") {
        return ("acoustic_connector", rest.to_string());
    }

    // Prediction head — needs sub-remapping for nn.Sequential indices
    if let Some(rest) = checkpoint_key.strip_prefix("model.prediction_head.") {
        let local = remap_prediction_head_key(rest);
        return ("prediction_head", local);
    }

    // Acoustic tokenizer decoder
    if let Some(rest) = checkpoint_key.strip_prefix("model.acoustic_tokenizer.decoder.") {
        let local = remap_acoustic_decoder_key(rest);
        return ("acoustic_decoder", local);
    }

    // Scalar buffers
    if checkpoint_key == "model.speech_scaling_factor"
        || checkpoint_key == "model.speech_bias_factor"
    {
        return ("scalar", checkpoint_key.to_string());
    }

    // Encoder weights (not needed for inference), semantic tokenizer, etc.
    ("skip", checkpoint_key.to_string())
}

/// Remap prediction head keys from PyTorch nn.Sequential indices to named fields.
///
/// PyTorch:  t_embedder.mlp.0.weight  →  Rust: t_embedder.linear1.weight
/// PyTorch:  t_embedder.mlp.2.weight  →  Rust: t_embedder.linear2.weight
/// PyTorch:  layers.N.adaLN_modulation.1.weight  →  Rust: layers.N.ada_ln_linear.weight
/// PyTorch:  layers.N.ffn.gate_proj.weight →  Rust: layers.N.ffn.gate_proj.weight (same)
/// PyTorch:  final_layer.adaLN_modulation.1.weight → Rust: final_layer.ada_ln_linear.weight
fn remap_prediction_head_key(key: &str) -> String {
    // PyTorch nn.Sequential wraps SiLU at index 0 (no weights) and Linear at index 1
    key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        .replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        .replace("adaLN_modulation.1.", "ada_ln_linear.")
}

/// Remap acoustic decoder keys from PyTorch structure to Rust struct hierarchy.
///
/// PyTorch has extra wrappers (NormConv1d, Convlayer) that add nesting:
///   SConv1d contains NormConv1d.conv (nn.Conv1d)  → path: sconv.conv.conv.*
///   SConvTranspose1d contains NormConvTranspose1d.convtr (nn.ConvTranspose1d) → path: sconvtr.convtr.convtr.*
///   Convlayer contains SConv1d → path: mixer.conv.sconv.normconv.conv.*
///
/// Rust SConv1d directly contains nn::Conv1d as `conv`, so:
///   PyTorch .conv.conv.weight → Rust .conv.weight
///   PyTorch .convtr.convtr.weight → Rust .conv_tr.convtr.weight
fn remap_acoustic_decoder_key(key: &str) -> String {
    // First: collapse double conv wrappers from NormConv1d
    // PyTorch: X.conv.conv.Y → Rust: X.conv.Y (SConv1d.conv = nn::Conv1d)
    // PyTorch: X.convtr.convtr.Y → Rust: X.conv_tr.convtr.Y (SConvTranspose1d.conv_tr = nn::ConvTranspose1d)
    let key = key
        .replace(".conv.conv.conv.", ".conv.") // Convlayer → SConv1d → NormConv1d → Conv1d
        .replace(".conv.conv.", ".conv.") // SConv1d → NormConv1d → Conv1d
        .replace(".convtr.convtr.", ".conv_tr."); // SConvTranspose1d → NormConvTranspose1d → ConvTranspose1d

    // Stem: upsample_layers.0.0.* → stem.*
    if let Some(rest) = key.strip_prefix("upsample_layers.0.0.") {
        return format!("stem.{rest}");
    }

    // Upsample convs: upsample_layers.{i+1}.0.* → upsample_convs.{i-1}.*
    if let Some(after) = key.strip_prefix("upsample_layers.") {
        if let Some(dot_pos) = after.find('.') {
            if let Ok(idx) = after[..dot_pos].parse::<usize>() {
                if idx > 0 {
                    let rest = &after[dot_pos + 1..];
                    let rest = rest.strip_prefix("0.").unwrap_or(rest);
                    let rust_idx = idx - 1;
                    return format!("upsample_convs.{rust_idx}.{rest}");
                }
            }
        }
    }

    // Stage blocks
    if key.starts_with("stages.") {
        let remapped = key.replace(".mixer.", ".mixer_conv.");
        return remap_conv_rms_norm_keys(&remapped);
    }

    // Final norm: norm.weight → norm.norm.weight (ConvRmsNorm wrapper)
    if key.starts_with("norm.") {
        return format!("norm.{key}");
    }

    // Head: head.conv.* → head.conv.* (already collapsed above)
    key
}

/// Insert the `.norm.` wrapper for ConvRmsNorm fields within stage blocks.
///
/// stages.X.Y.norm.weight → stages.X.Y.norm.norm.weight
/// stages.X.Y.ffn_norm.weight → stages.X.Y.ffn_norm.norm.weight
fn remap_conv_rms_norm_keys(key: &str) -> String {
    // Match patterns like stages.X.Y.norm.weight or stages.X.Y.ffn_norm.weight
    // These need .norm.norm.weight and .ffn_norm.norm.weight respectively
    // because ConvRmsNorm { norm: nn::RmsNorm } wraps the actual norm

    // Find the block-level pattern: stages.X.Y.FIELD.PARAM
    // Split at the block identifier
    if let Some(stages_rest) = key.strip_prefix("stages.") {
        // Parse stages.X.Y.rest
        let parts: Vec<&str> = stages_rest.splitn(4, '.').collect();
        if parts.len() == 4 {
            let (stage_idx, block_idx, field, param) = (parts[0], parts[1], parts[2], parts[3]);

            // Only wrap norm and ffn_norm fields (not ffn, gamma, etc.)
            if field == "norm" || field == "ffn_norm" {
                return format!("stages.{stage_idx}.{block_idx}.{field}.norm.{param}");
            }
        }
    }

    key.to_string()
}

/// Check if a weight tensor is a convolution kernel (3D) that needs transposition.
///
/// PyTorch conv1d weight: [out_channels, in_channels/groups, kernel_size]
/// MLX conv1d weight:     [out_channels, kernel_size, in_channels/groups]
///
/// We detect by looking at the original checkpoint keys (before remapping).
fn needs_conv_transpose(key: &str, array: &Array) -> bool {
    if array.ndim() != 3 {
        return false;
    }
    // Match any key that contains conv weight patterns
    key.contains("conv.weight") || key.contains("convtr.weight")
}

/// Transpose a 3D conv weight from PyTorch to MLX layout.
///
/// Conv1d:          PyTorch [O, I/G, K] → MLX [O, K, I/G]  = swap(1,2)
/// ConvTranspose1d: PyTorch [I, O, K]   → MLX [O, K, I]    = transpose(1,2,0)
fn transpose_conv_weight(key: &str, array: &Array) -> std::result::Result<Array, Exception> {
    if key.contains("convtr") {
        // ConvTranspose1d: [I, O, K] → [O, K, I]
        array.transpose_axes(&[1, 2, 0])
    } else {
        // Conv1d: [O, I/G, K] → [O, K, I/G]
        array.swap_axes(1, 2)
    }
}

/// Load and assign weights into a module using remapped keys.
fn load_remapped_weights<M: ModuleParameters>(
    module: &mut M,
    weights: &HashMap<String, Array>,
    component_name: &str,
) -> Result<()> {
    let mut params = module.parameters_mut().flatten();
    let total_params = params.len();

    let mut loaded = 0;
    let mut skipped_keys: Vec<String> = Vec::new();

    for (key, value) in weights {
        if let Some(param) = params.get_mut(&**key) {
            **param = value.clone();
            loaded += 1;
        } else {
            skipped_keys.push(key.clone());
        }
    }

    // Find unloaded params (in model but not in checkpoint)
    let loaded_keys: std::collections::HashSet<&str> = weights.keys().map(|k| k.as_str()).collect();
    let mut unloaded: Vec<&Rc<str>> = Vec::new();
    for key in params.keys() {
        if !loaded_keys.contains(&**key) {
            unloaded.push(key);
        }
    }

    eprintln!(
        "  [{component_name}] {loaded}/{} params loaded, {} checkpoint keys skipped, {} model params unloaded",
        total_params,
        skipped_keys.len(),
        unloaded.len(),
    );

    if !skipped_keys.is_empty() && skipped_keys.len() <= 10 {
        for k in &skipped_keys {
            eprintln!("    SKIP: {k}");
        }
    } else if !skipped_keys.is_empty() {
        for k in skipped_keys.iter().take(5) {
            eprintln!("    SKIP: {k}");
        }
        eprintln!("    ... and {} more", skipped_keys.len() - 5);
    }

    if !unloaded.is_empty() && unloaded.len() <= 10 {
        for k in &unloaded {
            eprintln!("    MISSING: {k}");
        }
    } else if !unloaded.is_empty() {
        for k in unloaded.iter().take(5) {
            eprintln!("    MISSING: {k}");
        }
        eprintln!("    ... and {} more", unloaded.len() - 5);
    }

    if loaded == 0 && !weights.is_empty() {
        let expected: Vec<&Rc<str>> = params.keys().take(5).collect();
        let actual: Vec<&String> = weights.keys().take(5).collect();
        return Err(KugelAudioError::WeightLoading(format!(
            "[{component_name}] No weights matched! Expected keys like {expected:?}, got {actual:?}"
        )));
    }

    Ok(())
}

/// Load safetensors weights into the model with full key remapping.
fn load_weights(model: &mut KugelAudioModel, model_dir: &Path) -> Result<()> {
    // Read the weight index
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_json = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_json)?;

    // Determine unique shard files
    let shard_files: HashSet<&String> = index.weight_map.values().collect();

    // Load all weights from all shards
    let mut all_weights: HashMap<String, Array> = HashMap::new();
    for shard_file in &shard_files {
        let shard_path = model_dir.join(shard_file);
        eprintln!("Loading shard: {shard_file}");
        let loaded = Array::load_safetensors(&shard_path).map_err(|e| {
            KugelAudioError::WeightLoading(format!("Failed to load {shard_file}: {e}"))
        })?;
        all_weights.extend(loaded);
    }

    eprintln!("Loaded {} weight tensors total", all_weights.len());

    // Process weights: keep bf16 (MLX supports it natively), apply conv transpositions
    let mut all_weights_processed: HashMap<String, Array> = HashMap::new();
    for (key, mut array) in all_weights {
        // Keep bfloat16 as-is — MLX supports bf16 natively

        // Conv weight transposition
        if needs_conv_transpose(&key, &array) {
            array = transpose_conv_weight(&key, &array).map_err(|e| {
                KugelAudioError::WeightLoading(format!("Conv transpose failed for {key}: {e}"))
            })?;
        }

        all_weights_processed.insert(key, array);
    }

    // Route weights to components via key remapping
    let mut lm_weights: HashMap<String, Array> = HashMap::new();
    let mut connector_weights: HashMap<String, Array> = HashMap::new();
    let mut head_weights: HashMap<String, Array> = HashMap::new();
    let mut decoder_weights: HashMap<String, Array> = HashMap::new();

    let tts_max_layer = model.lm.model.num_hidden_layers;

    for (checkpoint_key, array) in all_weights_processed {
        let (component, local_key) = remap_key(&checkpoint_key);

        match component {
            "lm" => {
                // Skip layers beyond tts_backbone_num_hidden_layers
                if let Some(after) = local_key.strip_prefix("model.layers.") {
                    if let Some(dot_pos) = after.find('.') {
                        if let Ok(idx) = after[..dot_pos].parse::<i32>() {
                            if idx >= tts_max_layer {
                                continue; // Skip layers 20..27
                            }
                        }
                    }
                }
                lm_weights.insert(local_key, array);
            }
            "acoustic_connector" => {
                connector_weights.insert(local_key, array);
            }
            "prediction_head" => {
                head_weights.insert(local_key, array);
            }
            "acoustic_decoder" => {
                decoder_weights.insert(local_key, array);
            }
            "scalar" => {
                if checkpoint_key == "model.speech_scaling_factor" {
                    model.speech_scaling_factor = array.item::<f32>();
                    eprintln!("speech_scaling_factor = {}", model.speech_scaling_factor);
                } else if checkpoint_key == "model.speech_bias_factor" {
                    model.speech_bias_factor = array.item::<f32>();
                    eprintln!("speech_bias_factor = {}", model.speech_bias_factor);
                }
            }
            _ => {} // skip encoder weights etc.
        }
    }

    // Load into each component
    eprintln!(
        "Loading LM weights ({} tensors, {} layers)...",
        lm_weights.len(),
        tts_max_layer
    );
    load_remapped_weights(&mut model.lm, &lm_weights, "LM")?;

    eprintln!(
        "Loading acoustic_connector weights ({} tensors)...",
        connector_weights.len()
    );
    load_remapped_weights(
        &mut model.acoustic_connector,
        &connector_weights,
        "connector",
    )?;

    eprintln!(
        "Loading prediction_head weights ({} tensors)...",
        head_weights.len()
    );
    load_remapped_weights(&mut model.prediction_head, &head_weights, "pred_head")?;

    eprintln!(
        "Loading acoustic_decoder weights ({} tensors)...",
        decoder_weights.len()
    );
    load_remapped_weights(&mut model.acoustic_decoder, &decoder_weights, "decoder")?;

    // Evaluate to materialize on GPU
    model
        .lm
        .eval()
        .map_err(|e| KugelAudioError::WeightLoading(format!("Failed to eval LM: {e}")))?;
    model
        .acoustic_connector
        .eval()
        .map_err(|e| KugelAudioError::WeightLoading(format!("Failed to eval connector: {e}")))?;
    model.prediction_head.eval().map_err(|e| {
        KugelAudioError::WeightLoading(format!("Failed to eval prediction head: {e}"))
    })?;
    model.acoustic_decoder.eval().map_err(|e| {
        KugelAudioError::WeightLoading(format!("Failed to eval acoustic decoder: {e}"))
    })?;

    eprintln!("All weights loaded successfully.");
    Ok(())
}
