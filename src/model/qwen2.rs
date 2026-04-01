//! Qwen2 transformer implementation for KugelAudio.
//!
//! This is a standalone candle implementation with **external KV cache** — the
//! cache is passed in per-layer rather than stored inside the attention module.
//! This is required for Classifier-Free Guidance (CFG), which runs two forward
//! passes (conditioned and unconditioned) through the **same** model weights
//! while maintaining **two separate** KV caches.
//!
//! # Architecture
//!
//! - Grouped Query Attention (GQA): 28 heads, 4 KV heads (7 groups)
//! - Q/K/V projections have bias; O projection does not
//! - Gate/Up/Down MLP projections have no bias; activation is SiLU
//! - RoPE with theta=1_000_000.0, no scaling
//! - RmsNorm eps=1e-6
//! - Tied word embeddings (lm_head == embed_tokens.T)
//!
//! # KV Cache
//!
//! Each layer uses one `Option<(Tensor, Tensor)>` entry in a `KvCache` vec.
//! `None` means the cache is empty (prefill from scratch). On each forward
//! pass the caller supplies the vec and each layer appends to its own slot.
//!
//! # Usage
//!
//! Build a `Qwen2Config` from `DecoderConfig`, construct with a `VarBuilder`,
//! then call `forward` for prefill and single-step decode. Use
//! `forward_from_embeds` to inject acoustic embeddings directly.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_no_bias, rms_norm,
};
use std::sync::Arc;

use crate::config::DecoderConfig;

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

/// Per-layer KV cache. Each element is `None` (empty) or `Some((K, V))`.
///
/// Shape of each tensor: `[batch, num_kv_heads, seq_len_cached, head_dim]`.
pub type KvCache = Vec<Option<(Tensor, Tensor)>>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Qwen2 model configuration derived from [`DecoderConfig`].
#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Qwen2Config {
    /// Build a [`Qwen2Config`] from the project's [`DecoderConfig`].
    pub fn from_decoder_config(cfg: &DecoderConfig) -> Self {
        let hidden_size = cfg.hidden_size as usize;
        let num_attention_heads = cfg.num_attention_heads as usize;
        Self {
            vocab_size: cfg.vocab_size as usize,
            hidden_size,
            intermediate_size: cfg.intermediate_size as usize,
            num_hidden_layers: cfg.num_hidden_layers as usize,
            num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads as usize,
            head_dim: hidden_size / num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings as usize,
            rope_theta: cfg.rope_theta as f64,
            rms_norm_eps: cfg.rms_norm_eps as f64,
            tie_word_embeddings: cfg.tie_word_embeddings,
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary Embeddings
// ---------------------------------------------------------------------------

/// Pre-computed RoPE sin/cos tables.
///
/// Stored as `[max_seq_len, head_dim/2]` so we can cheap-slice by position.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    /// `[max_seq_len, head_dim/2]`
    sin: Tensor,
    /// `[max_seq_len, head_dim/2]`
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen2Config, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let half_dim = head_dim / 2;
        // inv_freq[i] = 1 / theta^(2i / head_dim)
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0_f32 / (cfg.rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?.to_dtype(dtype)?;

        // positions: [max_seq_len, 1]
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(dtype)?
            .reshape((cfg.max_position_embeddings, 1))?;

        // freqs: [max_seq_len, half_dim]
        let freqs = t.matmul(&inv_freq)?;

        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Apply RoPE to query and key tensors.
    ///
    /// `q` and `k` must have shape `[batch, num_heads, seq_len, head_dim]`.
    /// `seqlen_offset` is the number of tokens already in the KV cache.
    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_rot = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// SwiGLU MLP: down_proj( silu(gate_proj(x)) * up_proj(x) )
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(xs)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Attention (stateless — external KV cache)
// ---------------------------------------------------------------------------

/// Multi-head grouped-query self-attention.
///
/// The KV cache is **not** stored here; callers pass `&mut Option<(Tensor, Tensor)>`
/// so that CFG can maintain two independent caches over the same weights.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // Q/K/V have bias=true; O has bias=false
        let q_proj = linear(h, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(h, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(h, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, h, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: h,
            rotary_emb,
        })
    }

    /// Run self-attention.
    ///
    /// # Arguments
    ///
    /// * `xs`            – `[batch, seq_len, hidden_size]`
    /// * `mask`          – optional additive causal mask `[batch, 1, seq_len, full_len]`
    /// * `seqlen_offset` – tokens already stored in `kv_cache`
    /// * `kv_cache`      – mutable slot: `None` = empty, updated in-place
    ///
    /// # Returns
    ///
    /// Output tensor `[batch, seq_len, hidden_size]`.
    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;

        // Linear projections
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Update KV cache: concatenate along seq_len dim (dim 2)
        let (k, v) = match kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k as &Tensor, &k], 2)?;
                let v = Tensor::cat(&[prev_v as &Tensor, &v], 2)?;
                (k, v)
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV heads for GQA: [batch, num_heads, full_seq, head_dim]
        // repeat_kv's expand→reshape already produces contiguous output.
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        // Scaled dot-product attention
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match mask {
            None => attn_weights,
            Some(m) => attn_weights.broadcast_add(m)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v)?;

        // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_out
            .transpose(1, 2)?
            .reshape((b, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer (stateless)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Qwen2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // Pre-norm self-attention with residual
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self
            .self_attn
            .forward(&normed, mask, seqlen_offset, kv_cache)?;
        let xs = (attn_out + residual)?;

        // Pre-norm MLP with residual
        let residual = &xs;
        let mlp_out = self
            .post_attention_layernorm
            .forward(&xs)?
            .apply(&self.mlp)?;
        residual + mlp_out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Repeat KV heads `n_rep` times along the head dimension for GQA.
///
/// Input:  `[batch, num_kv_heads, seq_len, head_dim]`
/// Output: `[batch, num_kv_heads * n_rep, seq_len, head_dim]`
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, n_kv, s, d) = xs.dims4()?;
    // [b, n_kv, s, d] → [b, n_kv, 1, s, d] → expand → [b, n_kv, n_rep, s, d] → reshape → [b, n_kv*n_rep, s, d]
    // This produces contiguous repetition: head 0 repeated n_rep times, then head 1, etc.
    xs.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))
}

/// Build an additive causal mask for prefill.
///
/// Returns `[batch, 1, tgt_len, tgt_len + seqlen_offset]` with 0 for
/// attended positions and `-inf` for masked-out (future) positions.
fn make_causal_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let full_len = tgt_len + seqlen_offset;
    // Upper-triangular (future) positions in the tgt_len × tgt_len block
    let mask_vals: Vec<f32> = (0..tgt_len)
        .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0_f32 }))
        .collect();
    let mask = Tensor::from_slice(&mask_vals, (tgt_len, tgt_len), device)?;

    // Prepend zeros for the already-cached portion
    let mask = if seqlen_offset > 0 {
        let prefix = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&prefix, &mask], D::Minus1)?
    } else {
        mask
    };

    mask.to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand((b_size, 1, tgt_len, full_len))
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// Qwen2 transformer base model with external KV cache.
///
/// Weight key prefix: `model.*`
#[derive(Debug, Clone)]
pub struct Qwen2Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// If `tie_word_embeddings`, this is `None` and we use `embed_tokens`.
    lm_head: Option<Linear>,
    dtype: DType,
    device: Device,
    pub cfg: Qwen2Config,
}

impl Qwen2Model {
    /// Load model weights from a [`VarBuilder`].
    ///
    /// Expected weight key structure:
    /// - `model.embed_tokens.weight`
    /// - `embed_tokens.weight`
    /// - `layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}`
    /// - `layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `layers.{i}.{input,post_attention}_layernorm.weight`
    /// - `norm.weight`
    ///
    /// The caller is responsible for scoping `vb` to the right prefix
    /// (e.g. `vb.pp("model").pp("language_model")`).
    /// `lm_head` weight, if needed, is resolved at the *parent* VarBuilder
    /// level via the separate `lm_head_vb` parameter.
    pub fn new(cfg: &Qwen2Config, vb: VarBuilder, lm_head_vb: Option<VarBuilder>) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            None
        } else if let Some(lm_vb) = lm_head_vb {
            Some(linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                lm_vb.pp("lm_head"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            cfg: cfg.clone(),
        })
    }

    /// Run the transformer starting from pre-computed embeddings.
    ///
    /// Used by KugelAudio's speech connector, which injects acoustic features
    /// into the embedding stream before the first transformer layer.
    ///
    /// # Arguments
    ///
    /// * `embeds`        – `[batch, seq_len, hidden_size]`, already in model dtype
    /// * `seqlen_offset` – length of the prefix already stored in `kv_cache`
    /// * `kv_cache`      – one slot per layer; updated in-place
    ///
    /// # Returns
    ///
    /// Normalized hidden states `[batch, seq_len, hidden_size]`.
    pub fn forward_from_embeds(
        &self,
        embeds: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = embeds.dims3()?;

        // Build causal mask only during prefill (seq_len > 1).
        let mask = if seq_len > 1 {
            Some(make_causal_mask(
                b_size,
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        } else {
            None
        };

        let mut xs = embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset, &mut kv_cache[i])?;
        }
        self.norm.forward(&xs)
    }

    /// Embed token IDs then run the transformer.
    ///
    /// # Arguments
    ///
    /// * `input_ids`     – `[batch, seq_len]` token IDs (i64)
    /// * `seqlen_offset` – tokens already in the KV cache
    /// * `kv_cache`      – one slot per layer; updated in-place
    ///
    /// # Returns
    ///
    /// Normalized hidden states `[batch, seq_len, hidden_size]`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let embeds = self.embed_tokens.forward(input_ids)?;
        self.forward_from_embeds(&embeds, seqlen_offset, kv_cache)
    }

    /// Project hidden states to vocabulary logits.
    ///
    /// When `tie_word_embeddings = true` the embedding matrix is used as the
    /// projection weight (no separate `lm_head` is loaded).
    ///
    /// # Arguments
    ///
    /// * `hidden` – `[batch, seq_len, hidden_size]`
    ///
    /// # Returns
    ///
    /// Logit tensor `[batch, seq_len, vocab_size]`.
    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(hidden),
            None => {
                // Tied embeddings: project with embed_tokens weight transposed.
                // embed_tokens weight shape: [vocab_size, hidden_size]
                // We want: hidden @ weight.T  => [batch, seq_len, vocab_size]
                let w = self.embed_tokens.embeddings();
                hidden.matmul(&w.t()?)
            }
        }
    }

    /// Allocate a fresh, empty KV cache with one slot per layer.
    pub fn new_kv_cache(&self) -> KvCache {
        vec![None; self.layers.len()]
    }
}
