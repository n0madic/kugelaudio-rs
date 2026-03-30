use serde::Deserialize;

/// Top-level KugelAudio model configuration.
///
/// HuggingFace config.json may have `acostic_vae_dim` (typo) alongside
/// `acoustic_vae_dim`. We accept both via serde alias.
#[derive(Debug, Clone, Deserialize)]
pub struct KugelAudioConfig {
    pub model_type: String,

    /// Acoustic VAE dimension (64).
    #[serde(default)]
    pub acoustic_vae_dim: Option<i32>,

    /// Typo variant present in some HF configs.
    #[serde(default)]
    pub acostic_vae_dim: Option<i32>,

    /// Number of transformer layers used for TTS backbone.
    /// Default = decoder_config.num_hidden_layers (use all layers).
    /// Local configs set this to 20 for the 7B model.
    #[serde(default)]
    pub tts_backbone_num_hidden_layers: Option<i32>,

    pub decoder_config: DecoderConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,

    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Top-level override for ddpm_inference_steps (present in some HF configs)
    #[serde(default)]
    pub ddpm_inference_steps: Option<i32>,

    #[serde(default)]
    pub semantic_tokenizer_config: Option<serde_json::Value>,
    #[serde(default)]
    pub semantic_vae_dim: Option<i32>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

impl KugelAudioConfig {
    /// Effective number of TTS backbone layers.
    pub fn tts_layers(&self) -> i32 {
        self.tts_backbone_num_hidden_layers
            .unwrap_or(self.decoder_config.num_hidden_layers)
    }

    /// Acoustic VAE dimension, handling the typo in HF configs.
    pub fn vae_dim(&self) -> i32 {
        self.acoustic_vae_dim.or(self.acostic_vae_dim).unwrap_or(64)
    }
}

/// Qwen2 decoder (language model) configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub intermediate_size: i32,
    pub hidden_act: String,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

/// Diffusion prediction head configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub latent_size: i32,
    pub head_layers: i32,
    pub head_ffn_ratio: f32,
    pub rms_norm_eps: f32,
    pub speech_vae_dim: i32,
    pub diffusion_type: String,
    pub prediction_type: String,
    pub ddpm_num_steps: i32,
    pub ddpm_num_inference_steps: i32,
    pub ddpm_beta_schedule: String,
    #[serde(default = "default_ddpm_batch_mul")]
    pub ddpm_batch_mul: i32,
}

fn default_ddpm_batch_mul() -> i32 {
    4
}

/// Acoustic tokenizer (codec) configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTokenizerConfig {
    pub model_type: String,
    pub vae_dim: i32,
    pub channels: i32,
    pub causal: bool,
    pub encoder_ratios: Vec<i32>,
    pub decoder_ratios: Vec<i32>,
    pub encoder_n_filters: i32,
    pub decoder_n_filters: i32,
    pub encoder_depths: String,
    pub conv_bias: bool,
    pub conv_norm: String,
    pub pad_mode: String,
    pub layernorm: String,
    pub layernorm_eps: f32,
    pub layernorm_elementwise_affine: bool,
    pub mixer_layer: String,
    pub fix_std: f32,
    pub std_dist_type: String,
    #[serde(default)]
    pub decoder_depths: Option<String>,
    #[serde(default)]
    pub disable_last_norm: bool,
    #[serde(default)]
    pub layer_scale_init_value: f64,
    #[serde(default)]
    pub weight_init_value: f64,
}

/// DPM-Solver++ scheduler configuration (derived from DiffusionHeadConfig).
#[derive(Debug, Clone)]
pub struct DpmSolverConfig {
    pub num_train_timesteps: i32,
    pub beta_schedule: String,
    pub prediction_type: String,
    pub algorithm_type: String,
    pub solver_order: i32,
    pub num_inference_steps: i32,
}

impl From<&DiffusionHeadConfig> for DpmSolverConfig {
    fn from(cfg: &DiffusionHeadConfig) -> Self {
        Self {
            num_train_timesteps: cfg.ddpm_num_steps,
            beta_schedule: cfg.ddpm_beta_schedule.clone(),
            prediction_type: cfg.prediction_type.clone(),
            algorithm_type: "sde-dpmsolver++".to_string(),
            solver_order: 2,
            num_inference_steps: cfg.ddpm_num_inference_steps,
        }
    }
}

/// Special token IDs used during generation.
pub struct SpecialTokens {
    pub speech_start_id: u32,
    pub speech_end_id: u32,
    pub speech_diffusion_id: u32,
    pub eos_token_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            speech_start_id: 151652,
            speech_end_id: 151653,
            speech_diffusion_id: 151654,
            eos_token_id: 151643,
        }
    }
}

/// Audio output parameters.
pub const SAMPLE_RATE: u32 = 24000;
pub const AUDIO_CHANNELS: u16 = 1;
