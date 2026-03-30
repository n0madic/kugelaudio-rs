use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{
        concatenate_axis, cos, exp,
        indexing::{IndexOp, NewAxis},
        multiply, rsqrt, sin,
    },
    Array,
};

use crate::config::DiffusionHeadConfig;

// ---- RMSNorm without learnable weight (affine=False) ----

fn rms_norm_no_affine(x: &Array, eps: f32) -> Result<Array, Exception> {
    let x_f32 = x.as_type::<f32>()?;
    let sq = x_f32.power(&array!(2.0f32))?;
    let mean_sq = sq.mean_axes(&[-1], true)?;
    let norm = rsqrt(&mean_sq.add(array!(eps))?)?;
    let out = x_f32.multiply(&norm)?;
    out.as_dtype(x.dtype())
}

// ---- Modulation: x * (1 + scale) + shift ----

fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let scaled = x.multiply(&scale.add(array!(1.0f32))?)?;
    scaled.add(shift)
}

// ---- Sinusoidal timestep embeddings ----

fn timestep_embedding(t: &Array, dim: i32, max_period: f32) -> Result<Array, Exception> {
    let half = dim / 2;

    // freqs = exp(-log(max_period) * arange(half) / half)
    let arange = Array::from_iter(0..half, &[half]).as_type::<f32>()?;
    let log_period = array!(-f32::ln(max_period));
    let freqs = exp(&multiply(
        &log_period,
        &arange.divide(&array!(half as f32))?,
    )?)?;

    // args = t[:, None] * freqs[None, :]
    let t_expanded = t.index((.., NewAxis)).as_type::<f32>()?;
    let freqs_expanded = freqs.index((NewAxis,));
    let args = t_expanded.multiply(&freqs_expanded)?;

    let cos_part = cos(&args)?;
    let sin_part = sin(&args)?;
    let embedding = concatenate_axis(&[&cos_part, &sin_part], -1)?;

    if dim % 2 != 0 {
        let batch = embedding.dim(0);
        let pad = Array::zeros::<f32>(&[batch, 1])?;
        concatenate_axis(&[&embedding, &pad], -1)
    } else {
        Ok(embedding)
    }
}

// ---- TimestepEmbedder ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct TimestepEmbedder {
    pub frequency_embedding_size: i32,
    #[param]
    pub linear1: nn::Linear,
    #[param]
    pub linear2: nn::Linear,
}

impl TimestepEmbedder {
    pub fn new(hidden_size: i32, frequency_embedding_size: i32) -> Result<Self, Exception> {
        let linear1 = nn::LinearBuilder::new(frequency_embedding_size, hidden_size)
            .bias(false)
            .build()?;
        let linear2 = nn::LinearBuilder::new(hidden_size, hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            frequency_embedding_size,
            linear1,
            linear2,
        })
    }
}

impl Module<&Array> for TimestepEmbedder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, t: &Array) -> Result<Self::Output, Self::Error> {
        let t_freq = timestep_embedding(t, self.frequency_embedding_size, 10000.0)?;
        let x = self.linear1.forward(&t_freq)?;
        let x = nn::silu(&x)?;
        self.linear2.forward(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear1.training_mode(mode);
        self.linear2.training_mode(mode);
    }
}

// ---- FeedForwardNetwork (SwiGLU) ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct FeedForwardNetwork {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl FeedForwardNetwork {
    pub fn new(embed_dim: i32, ffn_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(embed_dim, ffn_dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(embed_dim, ffn_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(ffn_dim, embed_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module<&Array> for FeedForwardNetwork {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = nn::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

// ---- HeadLayer (with AdaLN modulation) ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct HeadLayer {
    #[param]
    pub ffn: FeedForwardNetwork,
    #[param]
    pub norm: nn::RmsNorm,
    /// AdaLN modulation: SiLU -> Linear(cond_dim -> 3 * embed_dim)
    #[param]
    pub ada_ln_linear: nn::Linear,
}

impl HeadLayer {
    pub fn new(
        embed_dim: i32,
        ffn_dim: i32,
        cond_dim: i32,
        norm_eps: f32,
    ) -> Result<Self, Exception> {
        let ffn = FeedForwardNetwork::new(embed_dim, ffn_dim)?;
        let norm = nn::RmsNormBuilder::new(embed_dim).eps(norm_eps).build()?;
        let ada_ln_linear = nn::LinearBuilder::new(cond_dim, 3 * embed_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            ffn,
            norm,
            ada_ln_linear,
        })
    }
}

/// Input for HeadLayer: (x, condition)
pub struct HeadLayerInput<'a> {
    pub x: &'a Array,
    pub c: &'a Array,
}

impl Module<HeadLayerInput<'_>> for HeadLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: HeadLayerInput<'_>) -> Result<Self::Output, Self::Error> {
        let HeadLayerInput { x, c } = input;

        // AdaLN modulation: SiLU(c) -> Linear -> split into 3 chunks
        let modulation = self.ada_ln_linear.forward(&nn::silu(c)?)?;
        let chunks = modulation.split(3, -1)?;
        let (shift_ffn, scale_ffn, gate_ffn) = (&chunks[0], &chunks[1], &chunks[2]);

        // x = x + gate * ffn(modulate(norm(x), shift, scale))
        let normed = self.norm.forward(x)?;
        let modulated = modulate(&normed, shift_ffn, scale_ffn)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        let gated = gate_ffn.multiply(&ffn_out)?;
        x.add(gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.ffn.training_mode(mode);
        self.norm.training_mode(mode);
        self.ada_ln_linear.training_mode(mode);
    }
}

// ---- FinalLayer ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct FinalLayer {
    /// RMSNorm without learnable weight (affine=False)
    pub norm_eps: f32,
    #[param]
    pub linear: nn::Linear,
    /// AdaLN modulation: SiLU -> Linear(cond_size -> 2 * hidden_size)
    #[param]
    pub ada_ln_linear: nn::Linear,
}

impl FinalLayer {
    pub fn new(
        hidden_size: i32,
        output_size: i32,
        cond_size: i32,
        norm_eps: f32,
    ) -> Result<Self, Exception> {
        let linear = nn::LinearBuilder::new(hidden_size, output_size)
            .bias(false)
            .build()?;
        let ada_ln_linear = nn::LinearBuilder::new(cond_size, 2 * hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            norm_eps,
            linear,
            ada_ln_linear,
        })
    }
}

impl Module<HeadLayerInput<'_>> for FinalLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: HeadLayerInput<'_>) -> Result<Self::Output, Self::Error> {
        let HeadLayerInput { x, c } = input;

        // AdaLN modulation: SiLU(c) -> Linear -> split into 2 chunks
        let modulation = self.ada_ln_linear.forward(&nn::silu(c)?)?;
        let chunks = modulation.split(2, -1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);

        // Apply non-affine RMSNorm + modulation + linear
        let normed = rms_norm_no_affine(x, self.norm_eps)?;
        let modulated = modulate(&normed, shift, scale)?;
        self.linear.forward(&modulated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear.training_mode(mode);
        self.ada_ln_linear.training_mode(mode);
    }
}

// ---- KugelAudioDiffusionHead ----

#[derive(Debug, Clone, ModuleParameters)]
pub struct DiffusionHead {
    #[param]
    pub noisy_images_proj: nn::Linear,
    #[param]
    pub cond_proj: nn::Linear,
    #[param]
    pub t_embedder: TimestepEmbedder,
    #[param]
    pub layers: Vec<HeadLayer>,
    #[param]
    pub final_layer: FinalLayer,
}

/// Input for DiffusionHead: noisy latents, timesteps, conditioning
pub struct DiffusionHeadInput<'a> {
    pub noisy_images: &'a Array,
    pub timesteps: &'a Array,
    pub condition: &'a Array,
}

impl DiffusionHead {
    pub fn new(config: &DiffusionHeadConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let latent_size = config.latent_size;
        let ffn_dim = (hidden_size as f32 * config.head_ffn_ratio) as i32;

        let noisy_images_proj = nn::LinearBuilder::new(latent_size, hidden_size)
            .bias(false)
            .build()?;
        let cond_proj = nn::LinearBuilder::new(hidden_size, hidden_size)
            .bias(false)
            .build()?;
        let t_embedder = TimestepEmbedder::new(hidden_size, 256)?;

        let layers = (0..config.head_layers)
            .map(|_| HeadLayer::new(hidden_size, ffn_dim, hidden_size, config.rms_norm_eps))
            .collect::<Result<Vec<_>, _>>()?;

        let final_layer =
            FinalLayer::new(hidden_size, latent_size, hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            noisy_images_proj,
            cond_proj,
            t_embedder,
            layers,
            final_layer,
        })
    }
}

impl Module<DiffusionHeadInput<'_>> for DiffusionHead {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: DiffusionHeadInput<'_>) -> Result<Self::Output, Self::Error> {
        let DiffusionHeadInput {
            noisy_images,
            timesteps,
            condition,
        } = input;

        let mut x = self.noisy_images_proj.forward(noisy_images)?;
        let t = self.t_embedder.forward(timesteps)?;
        let condition = self.cond_proj.forward(condition)?;
        let c = condition.add(t)?;

        for layer in &mut self.layers {
            x = layer.forward(HeadLayerInput { x: &x, c: &c })?;
        }

        self.final_layer.forward(HeadLayerInput { x: &x, c: &c })
    }

    fn training_mode(&mut self, mode: bool) {
        self.noisy_images_proj.training_mode(mode);
        self.cond_proj.training_mode(mode);
        self.t_embedder.training_mode(mode);
        for layer in &mut self.layers {
            <HeadLayer as Module<HeadLayerInput>>::training_mode(layer, mode);
        }
        <FinalLayer as Module<HeadLayerInput>>::training_mode(&mut self.final_layer, mode);
    }
}
