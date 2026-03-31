//! Diffusion prediction head for KugelAudio.
//!
//! Predicts the speech latent noise from noisy input, LM conditioning, and a
//! diffusion timestep. The architecture is a small feed-forward DiT with
//! Adaptive Layer Norm (AdaLN) modulation.
//!
//! # Architecture
//!
//! 1. **Projections**: noisy latent → hidden_size, condition → hidden_size.
//! 2. **TimestepEmbedder**: sinusoidal encoding → Linear → SiLU → Linear.
//! 3. **HeadLayer ×N**: AdaLN modulation + SwiGLU FFN with residual connection.
//! 4. **FinalLayer**: AdaLN modulation → linear projection to output_dim.
//!
//! # Weight key prefix
//!
//! The parent [`VarBuilder`] must be scoped to `model.prediction_head.`.
//!
//! # Example
//!
//! ```ignore
//! let vb = vb.pp("model").pp("prediction_head");
//! let head = DiffusionHead::new(&cfg.diffusion_head_config, vb)?;
//! let output = head.forward(&noisy, &condition, &timesteps)?;
//! ```

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};

use crate::config::DiffusionHeadConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// RmsNorm without learnable affine parameters (affine=False).
///
/// Normalises over the last dimension using RMS, scaled by `eps` for stability.
fn rms_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(x_normed)
}

/// Adaptive layer norm modulation: `x * (1 + scale) + shift`.
///
/// `shift` and `scale` are broadcast over the sequence dimension.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// Sinusoidal timestep embedding.
///
/// Maps a batch of scalar timesteps to `[batch, dim]` frequency features.
///
/// # Arguments
///
/// * `t`          – `[batch]` timestep tensor (f32)
/// * `dim`        – embedding dimension
/// * `max_period` – controls the range of frequencies (default 10_000.0)
fn timestep_embedding(t: &Tensor, dim: usize, max_period: f64) -> Result<Tensor> {
    let half = dim / 2;
    let device = t.device();
    let orig_dtype = t.dtype();

    // Compute in F32 for precision
    let t_f32 = t.to_dtype(DType::F32)?;

    // freqs[i] = exp(-log(max_period) * i / half)
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(max_period.ln()) * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?;

    // args: [batch, half] = t[:, None] * freqs[None, :]
    let t_f32 = t_f32.unsqueeze(1)?;
    let args = t_f32.broadcast_mul(&freqs)?;

    let cos_part = args.cos()?;
    let sin_part = args.sin()?;

    // Concatenate along last dim → [batch, dim] (or [batch, dim+1] if odd)
    let embedding = Tensor::cat(&[cos_part, sin_part], D::Minus1)?;

    let embedding = if dim % 2 != 0 {
        let (b, _) = embedding.dims2()?;
        let pad = Tensor::zeros((b, 1), embedding.dtype(), device)?;
        Tensor::cat(&[embedding, pad], D::Minus1)?
    } else {
        embedding
    };
    embedding.to_dtype(orig_dtype)
}

// ---------------------------------------------------------------------------
// TimestepEmbedder
// ---------------------------------------------------------------------------

/// Two-layer MLP that turns a scalar timestep into a hidden-size embedding.
///
/// Architecture: sinusoidal(t, freq_size) → Linear → SiLU → Linear
///
/// Weight keys (relative to this module's VarBuilder prefix):
/// - `mlp.0.weight`, `mlp.0.bias`
/// - `mlp.2.weight`, `mlp.2.bias`
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    frequency_embedding_size: usize,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
}

impl TimestepEmbedder {
    /// `vb` must be scoped to `t_embedder`.
    pub fn new(
        hidden_size: usize,
        frequency_embedding_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_mlp = vb.pp("mlp");
        // PyTorch nn.Sequential uses numeric string keys: 0, 1 (activation), 2
        let mlp_linear1 = linear_no_bias(frequency_embedding_size, hidden_size, vb_mlp.pp(0))?;
        let mlp_linear2 = linear_no_bias(hidden_size, hidden_size, vb_mlp.pp(2))?;

        Ok(Self {
            frequency_embedding_size,
            mlp_linear1,
            mlp_linear2,
        })
    }

    /// # Arguments
    ///
    /// * `t` – `[batch]` timesteps as f32
    ///
    /// # Returns
    ///
    /// `[batch, hidden_size]` timestep embedding.
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let freq = timestep_embedding(t, self.frequency_embedding_size, 10_000.0)?;
        let x = self.mlp_linear1.forward(&freq)?;
        let x = x.apply(&candle_nn::Activation::Silu)?;
        self.mlp_linear2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// FeedForwardNetwork (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward network.
///
/// `down_proj( silu(gate_proj(x)) * up_proj(x) )`
///
/// All projections have no bias.
#[derive(Debug, Clone)]
struct FeedForwardNetwork {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForwardNetwork {
    fn new(embed_dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(embed_dim, ffn_dim, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(embed_dim, ffn_dim, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(ffn_dim, embed_dim, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(x)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(x)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// HeadLayer
// ---------------------------------------------------------------------------

/// Single diffusion head layer with AdaLN modulation and SwiGLU FFN.
///
/// The conditioning signal `c` modulates the normalised hidden state via a
/// learned linear projection that produces `(shift, scale, gate)` triplets.
/// The residual update is: `x + gate * ffn(modulate(norm(x), shift, scale))`.
///
/// Weight keys (relative to `layers.{i}`):
/// - `norm.weight`
/// - `adaLN_modulation.1.weight`, `adaLN_modulation.1.bias`
/// - `ffn.gate_proj.weight`, `ffn.up_proj.weight`, `ffn.down_proj.weight`
#[derive(Debug, Clone)]
pub struct HeadLayer {
    norm: RmsNorm,
    /// Linear(cond_dim → 3 * embed_dim) after the SiLU activation.
    /// Checkpoint key suffix: `adaLN_modulation.1.*` (index 1 in nn.Sequential).
    ada_ln_linear: Linear,
    ffn: FeedForwardNetwork,
}

impl HeadLayer {
    fn new(
        embed_dim: usize,
        ffn_dim: usize,
        cond_dim: usize,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm: rms_norm(embed_dim, norm_eps, vb.pp("norm"))?,
            // nn.Sequential(nn.SiLU(), nn.Linear(...)) — index 1 is the Linear
            ada_ln_linear: linear_no_bias(
                cond_dim,
                3 * embed_dim,
                vb.pp("adaLN_modulation").pp(1),
            )?,
            ffn: FeedForwardNetwork::new(embed_dim, ffn_dim, vb.pp("ffn"))?,
        })
    }

    /// # Arguments
    ///
    /// * `x` – `[B, T, embed_dim]` hidden states
    /// * `c` – `[B, T, cond_dim]` condition (already summed with timestep embed)
    ///
    /// # Returns
    ///
    /// `[B, T, embed_dim]` updated hidden states.
    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // AdaLN: SiLU(c) → linear → split into (shift, scale, gate)
        let modulation = self
            .ada_ln_linear
            .forward(&c.apply(&candle_nn::Activation::Silu)?)?;
        let chunks = modulation.chunk(3, D::Minus1)?;
        let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);

        // Residual: x + gate * ffn(modulate(norm(x), shift, scale))
        let normed = self.norm.forward(x)?;
        let modulated = modulate(&normed, shift, scale)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        x + gate.broadcast_mul(&ffn_out)?
    }
}

// ---------------------------------------------------------------------------
// FinalLayer
// ---------------------------------------------------------------------------

/// Output projection layer with AdaLN modulation (no learnable norm scale).
///
/// Uses a parameter-free RmsNorm followed by modulation with `(shift, scale)`
/// and a final linear projection to `output_dim`.
///
/// Weight keys (relative to `final_layer`):
/// - `norm.weight`  (unused at runtime — non-affine norm; loaded but ignored)
/// - `adaLN_modulation.1.weight`, `adaLN_modulation.1.bias`
/// - `linear.weight`, `linear.bias`
#[derive(Debug, Clone)]
pub struct FinalLayer {
    norm_eps: f64,
    ada_ln_linear: Linear,
    linear: Linear,
}

impl FinalLayer {
    fn new(
        hidden_size: usize,
        output_dim: usize,
        cond_size: usize,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm_eps,
            ada_ln_linear: linear_no_bias(
                cond_size,
                2 * hidden_size,
                vb.pp("adaLN_modulation").pp(1),
            )?,
            linear: linear_no_bias(hidden_size, output_dim, vb.pp("linear"))?,
        })
    }

    /// # Arguments
    ///
    /// * `x` – `[B, T, hidden_size]`
    /// * `c` – `[B, T, cond_size]`
    ///
    /// # Returns
    ///
    /// `[B, T, output_dim]`
    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // AdaLN: SiLU(c) → linear → split into (shift, scale)
        let modulation = self
            .ada_ln_linear
            .forward(&c.apply(&candle_nn::Activation::Silu)?)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);

        // Non-affine RmsNorm + modulation + projection
        let normed = rms_norm_no_affine(x, self.norm_eps)?;
        let modulated = modulate(&normed, shift, scale)?;
        self.linear.forward(&modulated)
    }
}

// ---------------------------------------------------------------------------
// DiffusionHead
// ---------------------------------------------------------------------------

/// KugelAudio diffusion prediction head.
///
/// Maps `(noisy_speech, condition, timestep)` → predicted noise (or velocity),
/// operating entirely in `[B, T, D]` space.
///
/// Weight key prefix: `model.prediction_head.`
///
/// # Example
///
/// ```ignore
/// let vb = vb.pp("model").pp("prediction_head");
/// let head = DiffusionHead::new(&cfg.diffusion_head_config, vb)?;
///
/// // noisy_speech: [B, T, input_dim],  condition: [B, T, hidden_size]
/// // timesteps:    [B] f32
/// let predicted = head.forward(&noisy_speech, &condition, &timesteps)?;
/// ```
#[derive(Debug, Clone)]
pub struct DiffusionHead {
    /// Linear(input_dim → hidden_size), no bias.
    noisy_images_proj: Linear,
    /// Linear(hidden_size → hidden_size), no bias.
    cond_proj: Linear,
    t_embedder: TimestepEmbedder,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
}

impl DiffusionHead {
    /// Load a [`DiffusionHead`] from `vb`.
    ///
    /// `vb` must be scoped to `model.prediction_head.`.
    pub fn new(cfg: &DiffusionHeadConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size as usize;
        let input_dim = cfg.speech_vae_dim as usize;
        let ffn_dim = (cfg.hidden_size as f32 * cfg.head_ffn_ratio) as usize;
        let num_layers = cfg.head_layers as usize;
        let norm_eps = cfg.rms_norm_eps as f64;
        // Sinusoidal frequency embedding size (256 is the standard DiT value).
        const FREQ_EMBED_SIZE: usize = 256;

        let noisy_images_proj = linear_no_bias(input_dim, hidden_size, vb.pp("noisy_images_proj"))?;
        let cond_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("cond_proj"))?;
        let t_embedder = TimestepEmbedder::new(hidden_size, FREQ_EMBED_SIZE, vb.pp("t_embedder"))?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(HeadLayer::new(
                hidden_size,
                ffn_dim,
                hidden_size,
                norm_eps,
                vb.pp("layers").pp(i),
            )?);
        }

        let final_layer = FinalLayer::new(
            hidden_size,
            input_dim,
            hidden_size,
            norm_eps,
            vb.pp("final_layer"),
        )?;

        Ok(Self {
            noisy_images_proj,
            cond_proj,
            t_embedder,
            layers,
            final_layer,
        })
    }

    /// Run the diffusion head forward pass.
    ///
    /// # Arguments
    ///
    /// * `noisy_speech` – `[B, T, input_dim]` noisy speech latents
    /// * `condition`    – `[B, T, hidden_size]` LM hidden states
    /// * `timesteps`    – `[B]` diffusion timesteps (f32)
    ///
    /// # Returns
    ///
    /// `[B, T, input_dim]` predicted noise (or velocity, depending on scheduler).
    pub fn forward(
        &self,
        noisy_speech: &Tensor,
        condition: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Project noisy latent into hidden space: [B, T, hidden_size]
        let mut x = self.noisy_images_proj.forward(noisy_speech)?;

        // Timestep embedding: [B, hidden_size] → broadcast to [B, 1, hidden_size]
        let t_emb = self.t_embedder.forward(timesteps)?.unsqueeze(1)?;

        // Project condition into hidden space: [B, T, hidden_size]
        let cond = self.cond_proj.forward(condition)?;

        // Sum condition and timestep: [B, T, hidden_size]
        let c = cond.broadcast_add(&t_emb)?;

        // Run through head layers
        for layer in &self.layers {
            x = layer.forward(&x, &c)?;
        }

        self.final_layer.forward(&x, &c)
    }
}
