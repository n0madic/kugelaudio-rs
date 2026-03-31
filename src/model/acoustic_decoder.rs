use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, activation::Activation, linear_b, rms_norm};

use super::causal_conv::{SConv1d, SConvTranspose1d};
use crate::config::AcousticTokenizerConfig;

// ---- ConvRmsNorm: RmsNorm applied on the channel dimension in NCL format ----

/// RmsNorm that operates on the channel (dim=1) of an NCL tensor.
///
/// Candle's `RmsNorm` normalizes the last dimension. For NCL `[B, C, L]` input,
/// this type transposes to `[B, L, C]`, applies norm (over C), and transposes back.
#[derive(Debug)]
pub struct ConvRmsNorm {
    norm: RmsNorm,
}

impl ConvRmsNorm {
    /// Load from checkpoint. Expects `weight` tensor at the current VarBuilder prefix.
    pub fn load(channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm = rms_norm(channels, eps, vb)?;
        Ok(Self { norm })
    }

    /// Apply norm to an NCL tensor `[B, C, L]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, C, L] -> [B, L, C]
        let x_t = x.transpose(1, 2)?;
        // RmsNorm normalizes over last dim (C)
        let normed = self.norm.forward(&x_t)?;
        // [B, L, C] -> [B, C, L]
        normed.transpose(1, 2)
    }
}

// ---- Ffn: channel-wise feedforward network ----

/// Two-layer feedforward network with GELU activation, applied channel-wise on NCL input.
///
/// Input `[B, C, L]` is transposed to `[B, L, C]`, passed through
/// `fc1 -> GELU -> fc2`, then transposed back to `[B, C, L]`.
#[derive(Debug)]
pub struct Ffn {
    fc1: Linear,
    fc2: Linear,
}

impl Ffn {
    /// Load from checkpoint. Expects `linear1.*`, `linear2.*` keys.
    pub fn load(embed_dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_b(embed_dim, ffn_dim, true, vb.pp("linear1"))?;
        let fc2 = linear_b(ffn_dim, embed_dim, true, vb.pp("linear2"))?;
        Ok(Self { fc1, fc2 })
    }

    /// Apply FFN to an NCL tensor `[B, C, L]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, C, L] -> [B, L, C]
        let x_t = x.transpose(1, 2)?;
        // fc1 -> GELU -> fc2
        let h = self.fc1.forward(&x_t)?;
        let h = Activation::Gelu.forward(&h)?;
        let out = self.fc2.forward(&h)?;
        // [B, L, C] -> [B, C, L]
        out.transpose(1, 2)
    }
}

// ---- Block1d: residual block with depthwise conv mixer and FFN ----

/// Residual block used in each decoder stage.
///
/// Forward pass (NCL throughout):
/// 1. norm → depthwise SConv1d → gamma scale → residual add
/// 2. ffn_norm → Ffn → ffn_gamma scale → residual add
#[derive(Debug)]
pub struct Block1d {
    norm: ConvRmsNorm,
    mixer_conv: SConv1d,
    gamma: Option<Tensor>,
    ffn_norm: ConvRmsNorm,
    ffn: Ffn,
    ffn_gamma: Option<Tensor>,
}

impl Block1d {
    /// Load a single residual block from the VarBuilder rooted at this block's prefix.
    ///
    /// Expected checkpoint keys relative to `vb`:
    /// - `norm.weight` — block RmsNorm
    /// - `mixer.conv.conv.conv.weight` — depthwise causal conv
    ///   (`SConv1d::new` appends `.conv` internally, so we pass `mixer.conv.conv`)
    /// - `gamma` — `[channels]` layer scale (optional)
    /// - `ffn_norm.weight` — FFN RmsNorm
    /// - `ffn.fc1.{weight,bias}`, `ffn.fc2.{weight,bias}`
    /// - `ffn_gamma` — `[channels]` FFN layer scale (optional)
    pub fn load(
        channels: usize,
        kernel_size: usize,
        eps: f64,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = ConvRmsNorm::load(channels, eps, vb.pp("norm"))?;

        // Depthwise conv: groups == channels.
        // Checkpoint path: mixer.conv.conv.conv.{weight,bias}.
        // SConv1d::new internally appends pp("conv"), so we pass up to the second .conv level.
        let mixer_conv = SConv1d::new(
            channels,
            channels,
            kernel_size,
            1,
            1,
            channels,
            true, // checkpoint has bias for all convs
            causal,
            vb.pp("mixer").pp("conv").pp("conv"),
        )?;

        // gamma: [channels] — present when layer_scale_init_value > 0
        let gamma = if vb.contains_tensor("gamma") {
            Some(vb.get(channels, "gamma")?)
        } else {
            None
        };

        let ffn_norm = ConvRmsNorm::load(channels, eps, vb.pp("ffn_norm"))?;
        let ffn = Ffn::load(channels, 4 * channels, vb.pp("ffn"))?;

        let ffn_gamma = if vb.contains_tensor("ffn_gamma") {
            Some(vb.get(channels, "ffn_gamma")?)
        } else {
            None
        };

        Ok(Self {
            norm,
            mixer_conv,
            gamma,
            ffn_norm,
            ffn,
            ffn_gamma,
        })
    }

    /// Apply the residual block to an NCL tensor `[B, C, L]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mixer branch
        let residual = x.clone();
        let h = self.norm.forward(x)?;
        let h = self.mixer_conv.forward(&h)?;
        let h = match &self.gamma {
            // gamma is [C]; broadcast over [B, C, L] by reshaping to [1, C, 1]
            Some(g) => {
                let g_bcl = g.unsqueeze(0)?.unsqueeze(2)?;
                h.broadcast_mul(&g_bcl)?
            }
            None => h,
        };
        let x = (residual + h)?;

        // FFN branch
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        let h = match &self.ffn_gamma {
            Some(g) => {
                let g_bcl = g.unsqueeze(0)?.unsqueeze(2)?;
                h.broadcast_mul(&g_bcl)?
            }
            None => h,
        };
        let x = (residual + h)?;

        Ok(x)
    }
}

// ---- AcousticDecoder ----

/// Decoder that transforms speech latents `[B, D, T]` (NCL) to audio samples `[B, 1, samples]`.
///
/// Architecture (loaded from `acoustic_tokenizer_config`):
/// - **stem**: `SConv1d(input_dim → first_stage_channels, kernel=7)`
/// - **stages**: `Vec<Vec<Block1d>>` — residual blocks per stage (depths reversed)
/// - **upsample_convs**: `Vec<SConvTranspose1d>` — transposed conv between stages
/// - **norm**: optional channel-wise `ConvRmsNorm`
/// - **head**: `SConv1d(last_channels → 1, kernel=7)`
///
/// The Python checkpoint uses upsample_layers[0] as the stem and
/// upsample_layers[i+1] as the i-th upsample conv.
#[derive(Debug)]
pub struct AcousticDecoder {
    stem: SConv1d,
    upsample_convs: Vec<SConvTranspose1d>,
    stages: Vec<Vec<Block1d>>,
    norm: Option<ConvRmsNorm>,
    head: SConv1d,
}

impl AcousticDecoder {
    /// Load the acoustic decoder from a VarBuilder.
    ///
    /// `vb` should be rooted at `model.acoustic_tokenizer.decoder.` in the checkpoint.
    pub fn load(config: &AcousticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let eps = config.layernorm_eps as f64;
        let causal = config.causal;
        // Per-architecture constant: all conv layers use kernel_size=7.
        let kernel_size: usize = 7;

        // Parse and reverse depths: "3-3-3-3-3-3-8" -> reversed [8,3,3,3,3,3,3]
        let depths: Vec<usize> = config
            .encoder_depths
            .split('-')
            .map(|s| {
                s.parse::<usize>().map_err(|e| {
                    candle_core::Error::Msg(format!("Invalid depth '{s}' in encoder_depths: {e}"))
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?
            .into_iter()
            .rev()
            .collect();

        let n_stages = depths.len(); // 7 for the default config

        // Stage channel counts: stage 0 has the most channels (before any upsampling),
        // each subsequent stage halves the channel count as the temporal resolution grows.
        // decoder_n_filters is the base (smallest) channel count (at the output stage).
        //
        // For n_stages=7 with decoder_n_filters=64:
        //   stage_channels = [4096, 2048, 1024, 512, 256, 128, 64]
        //   (i.e. 64 << (6-i) for i in 0..7)
        let n_filters = config.decoder_n_filters as usize;
        let stage_channels: Vec<usize> = (0..n_stages)
            .map(|i| n_filters << (n_stages - 1 - i))
            .collect();

        let upsample_vb = vb.pp("upsample_layers");

        // Stem: checkpoint key = upsample_layers.0.0.conv.conv.{weight,bias}.
        // SConv1d::new appends pp("conv") internally, so we pass up to the first ".conv" level.
        // input_dim is the VAE latent dimension (same as vae_dim in the config).
        let stem_in = config.vae_dim as usize;
        let stem_out = stage_channels[0];
        let stem = SConv1d::new(
            stem_in,
            stem_out,
            kernel_size,
            1,
            1,
            1,
            true,
            causal,
            upsample_vb.pp(0usize).pp(0usize).pp("conv"),
        )?;

        // Upsample convs: checkpoint key = upsample_layers.{i+1}.0.convtr.convtr.weight.
        // SConvTranspose1d::new appends pp("convtr") internally, so we pass up to the first ".convtr".
        // decoder_ratios are the per-stage upsample factors (upsample_strides in Python).
        let ratios = &config.decoder_ratios;
        let mut upsample_convs = Vec::with_capacity(ratios.len());
        for (i, &stride) in ratios.iter().enumerate() {
            let in_ch = stage_channels[i];
            let out_ch = stage_channels[i + 1];
            let stride = stride as usize;
            // Kernel = stride * 2 matches the Python AudioCraft convention.
            let conv_kernel = stride * 2;
            let conv_vb = upsample_vb.pp(i + 1).pp(0usize).pp("convtr");
            let upsample =
                SConvTranspose1d::new(in_ch, out_ch, conv_kernel, stride, true, causal, conv_vb)?;
            upsample_convs.push(upsample);
        }

        // Stage blocks: stages.{si}.{bi}.*
        let stages_vb = vb.pp("stages");
        let mut stages: Vec<Vec<Block1d>> = Vec::with_capacity(n_stages);
        for si in 0..n_stages {
            let ch = stage_channels[si];
            let n_blocks = depths[si];
            let stage_vb = stages_vb.pp(si);
            let mut blocks = Vec::with_capacity(n_blocks);
            for bi in 0..n_blocks {
                let block = Block1d::load(ch, kernel_size, eps, causal, stage_vb.pp(bi))?;
                blocks.push(block);
            }
            stages.push(blocks);
        }

        // Final norm: norm.norm.weight (ConvRmsNorm wraps nn::RmsNorm as "norm")
        let norm = if !config.disable_last_norm {
            let last_ch = *stage_channels.last().unwrap();
            Some(ConvRmsNorm::load(last_ch, eps, vb.pp("norm").pp("norm"))?)
        } else {
            None
        };

        // Head: checkpoint key = head.conv.conv.{weight,bias}.
        // SConv1d::new appends pp("conv") internally, so we pass up to the first ".conv".
        let last_ch = *stage_channels.last().unwrap();
        let head = SConv1d::new(
            last_ch,
            1,
            kernel_size,
            1,
            1,
            1,
            true,
            causal,
            vb.pp("head").pp("conv"),
        )?;

        Ok(Self {
            stem,
            upsample_convs,
            stages,
            norm,
            head,
        })
    }

    /// Decode latents `[B, D, T]` (NCL) to audio `[B, 1, samples]` (NCL).
    ///
    /// Execution order mirrors the Python TokenizerDecoder:
    /// ```text
    /// x = stem(latents)
    /// for i in 0..n_stages:
    ///     for block in stages[i]: x = block(x)
    ///     if i < n_upsample: x = upsample_convs[i](x)
    /// x = norm(x)   // optional
    /// x = head(x)
    /// ```
    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let mut x = self.stem.forward(latents)?;

        for i in 0..self.stages.len() {
            for block in &self.stages[i] {
                x = block.forward(&x)?;
            }
            if i < self.upsample_convs.len() {
                x = self.upsample_convs[i].forward(&x)?;
            }
        }

        if let Some(norm) = &self.norm {
            x = norm.forward(&x)?;
        }

        self.head.forward(&x)
    }
}
