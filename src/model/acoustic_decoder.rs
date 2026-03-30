use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn, Array,
};

use super::causal_conv::{SConv1d, SConvTranspose1d};
use crate::config::AcousticTokenizerConfig;

// ---- Convolution-friendly RMSNorm (operates on NLC, norm over C) ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvRmsNorm {
    #[param]
    pub norm: nn::RmsNorm,
}

impl ConvRmsNorm {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        let norm = nn::RmsNormBuilder::new(dim).eps(eps).build()?;
        Ok(Self { norm })
    }
}

impl Module<&Array> for ConvRmsNorm {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        self.norm.forward(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
    }
}

// ---- FFN (GELU-based, used in Block1D) ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct Ffn {
    #[param]
    pub linear1: nn::Linear,
    #[param]
    pub linear2: nn::Linear,
}

impl Ffn {
    pub fn new(embed_dim: i32, ffn_dim: i32, bias: bool) -> Result<Self, Exception> {
        let linear1 = nn::LinearBuilder::new(embed_dim, ffn_dim)
            .bias(bias)
            .build()?;
        let linear2 = nn::LinearBuilder::new(ffn_dim, embed_dim)
            .bias(bias)
            .build()?;
        Ok(Self { linear1, linear2 })
    }
}

impl Module<&Array> for Ffn {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let x = self.linear1.forward(x)?;
        let x = nn::gelu(&x)?;
        self.linear2.forward(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear1.training_mode(mode);
        self.linear2.training_mode(mode);
    }
}

// ---- Block1D: depthwise conv mixer + FFN with layer scale ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct Block1d {
    #[param]
    pub norm: ConvRmsNorm,
    #[param]
    pub mixer_conv: SConv1d,
    #[param]
    pub ffn_norm: ConvRmsNorm,
    #[param]
    pub ffn: Ffn,
    #[param]
    pub gamma: Param<Option<Array>>,
    #[param]
    pub ffn_gamma: Param<Option<Array>>,
}

impl Block1d {
    pub fn new(
        dim: i32,
        kernel_size: i32,
        groups: i32,
        causal: bool,
        bias: bool,
        eps: f32,
        layer_scale_init_value: f64,
    ) -> Result<Self, Exception> {
        let norm = ConvRmsNorm::new(dim, eps)?;
        let mixer_conv = SConv1d::new(dim, dim, kernel_size, 1, 1, groups, bias, causal)?;
        let ffn_norm = ConvRmsNorm::new(dim, eps)?;
        // FFN has bias in the checkpoint
        let ffn = Ffn::new(dim, 4 * dim, true)?;

        let (gamma, ffn_gamma) = if layer_scale_init_value > 0.0 {
            let g = Array::from_slice(&vec![layer_scale_init_value as f32; dim as usize], &[dim]);
            let fg = Array::from_slice(&vec![layer_scale_init_value as f32; dim as usize], &[dim]);
            (Param::new(Some(g)), Param::new(Some(fg)))
        } else {
            (Param::new(None), Param::new(None))
        };

        Ok(Self {
            norm,
            mixer_conv,
            ffn_norm,
            ffn,
            gamma,
            ffn_gamma,
        })
    }
}

impl Module<&Array> for Block1d {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // Mixer branch
        let residual = x.clone();
        let mut h = self.norm.forward(x)?;
        h = self.mixer_conv.forward(&h)?;
        if let Some(gamma) = self.gamma.as_ref() {
            h = h.multiply(gamma)?;
        }
        let mut x = residual.add(h)?;

        // FFN branch
        let residual = x.clone();
        let mut h = self.ffn_norm.forward(&x)?;
        h = self.ffn.forward(&h)?;
        if let Some(ffn_gamma) = self.ffn_gamma.as_ref() {
            h = h.multiply(ffn_gamma)?;
        }
        x = residual.add(h)?;

        Ok(x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        self.mixer_conv.training_mode(mode);
        self.ffn_norm.training_mode(mode);
        self.ffn.training_mode(mode);
    }
}

// ---- Acoustic Decoder ----

/// Decoder that converts latent vectors (64-dim) to 24kHz audio waveforms.
///
/// Python structure (TokenizerDecoder):
///   upsample_layers[0] = stem SConv1d (no upsampling)
///   upsample_layers[1..N] = SConvTranspose1d for upsampling
///   stages[0..N] = Block1D stages (one per depth level)
///   norm = final ConvRMSNorm (optional)
///   head = SConv1d output projection
///
/// Forward: for each i: upsample[i](x) → stages[i](x) → norm → head
#[derive(Debug, Clone, ModuleParameters)]
pub struct AcousticDecoder {
    #[param]
    pub stem: SConv1d,
    #[param]
    pub upsample_convs: Vec<SConvTranspose1d>,
    #[param]
    pub stages: Vec<Vec<Block1d>>,
    #[param]
    pub norm: Option<ConvRmsNorm>,
    #[param]
    pub head: SConv1d,
}

impl AcousticDecoder {
    pub fn new(config: &AcousticTokenizerConfig) -> Result<Self, Exception> {
        let vae_dim = config.vae_dim;
        let n_filters = config.decoder_n_filters;
        let ratios = &config.decoder_ratios;
        let causal = config.causal;
        let bias = config.conv_bias;
        let eps = config.layernorm_eps;
        let layer_scale_init_value = config.layer_scale_init_value;
        let is_depthwise = config.mixer_layer == "depthwise_conv";

        // Parse depths: "3-3-3-3-3-3-8" reversed = [8,3,3,3,3,3,3]
        let depths: Vec<i32> = config
            .encoder_depths
            .split('-')
            .map(|s| s.parse::<i32>().unwrap_or(3))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let n_depth_stages = depths.len(); // 7

        // Stem: vae_dim → n_filters * 2^(n_depth_stages - 1)
        let stem_out_ch = n_filters * (1 << (n_depth_stages - 1));
        let stem = SConv1d::new(vae_dim, stem_out_ch, 7, 1, 1, 1, bias, causal)?;

        // Stage 0 operates at stem resolution (stem_out_ch)
        let groups_0 = if is_depthwise { stem_out_ch } else { 1 };
        let stage_0: Vec<Block1d> = (0..depths[0])
            .map(|_| {
                Block1d::new(
                    stem_out_ch,
                    7,
                    groups_0,
                    causal,
                    bias,
                    eps,
                    layer_scale_init_value,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut all_stages = vec![stage_0];
        let mut upsample_convs = Vec::with_capacity(ratios.len());

        // Stages 1..n_depth_stages: each has an upsample conv then blocks
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_ch = n_filters * (1 << (n_depth_stages - 1 - i));
            let out_ch = if i + 1 < n_depth_stages - 1 {
                n_filters * (1 << (n_depth_stages - 2 - i))
            } else {
                n_filters
            };

            let kernel_size = ratio * 2;
            let upsample = SConvTranspose1d::new(in_ch, out_ch, kernel_size, ratio, bias, causal)?;
            upsample_convs.push(upsample);

            // Stage blocks at the output resolution of this upsample
            let stage_idx = i + 1;
            if stage_idx < n_depth_stages {
                let groups = if is_depthwise { out_ch } else { 1 };
                let stage_blocks: Vec<Block1d> = (0..depths[stage_idx])
                    .map(|_| {
                        Block1d::new(out_ch, 7, groups, causal, bias, eps, layer_scale_init_value)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                all_stages.push(stage_blocks);
            }
        }

        // Final channel count
        let final_ch = n_filters;
        let norm = if !config.disable_last_norm {
            Some(ConvRmsNorm::new(final_ch, eps)?)
        } else {
            None
        };
        let head = SConv1d::new(final_ch, config.channels, 7, 1, 1, 1, bias, causal)?;

        Ok(Self {
            stem,
            upsample_convs,
            stages: all_stages,
            norm,
            head,
        })
    }
}

impl Module<&Array> for AcousticDecoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, latents: &Array) -> Result<Self::Output, Self::Error> {
        // latents: [B, T, vae_dim] (NLC)

        // Stem conv (no upsampling)
        let mut x = self.stem.forward(latents)?;

        // Matches Python mlx-audio decoder:
        //   for i in range(n_stages):
        //       for block in stages[i]: x = block(x)   # blocks first
        //       if i+1 < len(upsample_layers): x = upsample[i+1](x)  # then upsample
        for i in 0..self.stages.len() {
            // Apply stage blocks first (at current resolution)
            for block in &mut self.stages[i] {
                x = block.forward(&x)?;
            }
            // Then upsample (stages[0] blocks run before upsample[0])
            if i < self.upsample_convs.len() {
                x = self.upsample_convs[i].forward(&x)?;
            }
        }

        // Final norm
        if let Some(norm) = &mut self.norm {
            x = norm.forward(&x)?;
        }

        // Head: project to mono audio
        self.head.forward(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.stem.training_mode(mode);
        for conv in &mut self.upsample_convs {
            conv.training_mode(mode);
        }
        for stage in &mut self.stages {
            for block in stage {
                block.training_mode(mode);
            }
        }
        if let Some(norm) = &mut self.norm {
            norm.training_mode(mode);
        }
        self.head.training_mode(mode);
    }
}
