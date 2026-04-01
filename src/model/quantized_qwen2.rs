//! Quantized Qwen2 transformer using GGUF weights.
//!
//! Drop-in replacement for [`super::qwen2::Qwen2Model`] with the same external
//! KV-cache API.  All linear projections use [`QMatMul`] for reduced memory
//! footprint and faster inference on quantized weights.
//!
//! # Weight loading
//!
//! Weights are loaded from a GGUF file via [`gguf_file::Content`].  The
//! standard llama.cpp naming convention for Qwen2 is expected:
//!
//! | Purpose          | GGUF key pattern                    |
//! |------------------|-------------------------------------|
//! | Embedding        | `token_embd.weight`                 |
//! | LM head          | `output.weight` (or tied to embed)  |
//! | Final norm       | `output_norm.weight`                |
//! | Q/K/V weight     | `blk.{i}.attn_q/k/v.weight`        |
//! | Q/K/V bias       | `blk.{i}.attn_q/k/v.bias`          |
//! | O projection     | `blk.{i}.attn_output.weight`        |
//! | Gate/Up/Down MLP | `blk.{i}.ffn_gate/up/down.weight`   |
//! | Layer norms      | `blk.{i}.attn_norm/ffn_norm.weight` |
//!
//! # Differences from the non-quantized model
//!
//! - All `candle_nn::Linear` layers are replaced with [`QMatMul`].
//! - Biased projections (Q/K/V) store the bias as a dequantized [`Tensor`]
//!   and add it after the quantized matmul.
//! - [`RmsNorm`] and [`Embedding`] weights are dequantized to f32 tensors.
//! - RoPE tables are shared via [`Arc`] and computed from config, not loaded.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, RmsNorm};

use super::qwen2::{KvCache, Qwen2Config, RotaryEmbedding, make_causal_mask, repeat_kv};

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// Quantized SwiGLU MLP: down_proj( silu(gate_proj(x)) * up_proj(x) )
struct QuantizedMlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl QuantizedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(xs)?
            .apply(&candle_nn::Activation::Silu)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// Attention (stateless — external KV cache)
// ---------------------------------------------------------------------------

/// Quantized multi-head grouped-query self-attention.
///
/// Like the non-quantized [`super::qwen2`] attention, the KV cache is
/// **external** — callers pass `&mut Option<(Tensor, Tensor)>` per layer.
struct QuantizedAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    q_bias: Tensor,
    k_bias: Tensor,
    v_bias: Tensor,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl QuantizedAttention {
    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;

        // Quantized projections + dequantized bias for Q/K/V
        let q = self.q_proj.forward(xs)?.broadcast_add(&self.q_bias)?;
        let k = self.k_proj.forward(xs)?.broadcast_add(&self.k_bias)?;
        let v = self.v_proj.forward(xs)?.broadcast_add(&self.v_bias)?;

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

        // Repeat KV heads for GQA
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
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((b, q_len, self.hidden_size))?;
        self.o_proj.forward(&attn_out)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

struct QuantizedDecoderLayer {
    self_attn: QuantizedAttention,
    mlp: QuantizedMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedDecoderLayer {
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
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + mlp_out
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// Quantized Qwen2 transformer with external KV cache.
///
/// Provides the same API as [`super::qwen2::Qwen2Model`] so both can be
/// used interchangeably via the [`super::lm::Lm`] enum.
pub struct QuantizedQwen2Model {
    embed_tokens: Embedding,
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    dtype: DType,
    device: Device,
    pub cfg: Qwen2Config,
}

impl QuantizedQwen2Model {
    /// Load a quantized Qwen2 model from a GGUF file.
    ///
    /// # Arguments
    ///
    /// * `cfg`    – model configuration (typically from `config.json`)
    /// * `ct`     – parsed GGUF header (from [`gguf_file::Content::read`])
    /// * `reader` – seekable reader positioned at the GGUF file
    /// * `device` – target compute device
    pub fn new<R: std::io::Seek + std::io::Read>(
        cfg: &Qwen2Config,
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Quantized models always run in F32: QMatMul dequantizes on the fly
        // and returns F32 tensors regardless of device. RoPE tables and causal
        // masks must match this dtype.
        let dtype = DType::F32;

        // Shared rotary embeddings (computed from config, not from weights)
        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, cfg, device)?);

        // Embedding: dequantize QTensor → f32 Tensor for lookup table
        let tok_embd = ct.tensor(reader, "token_embd.weight", device)?;
        let embed_tensor = tok_embd.dequantize(device)?;
        let embed_tokens = Embedding::new(embed_tensor, cfg.hidden_size);

        // LM head: prefer separate output.weight, fall back to tied embeddings
        let lm_head = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => QMatMul::from_qtensor(tok_embd)?,
        };

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{i}");

            // Attention projections — Q/K/V have bias, O does not
            let q_proj = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{prefix}.attn_q.weight"),
                device,
            )?)?;
            let k_proj = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{prefix}.attn_k.weight"),
                device,
            )?)?;
            let v_proj = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{prefix}.attn_v.weight"),
                device,
            )?)?;
            let q_bias = ct
                .tensor(reader, &format!("{prefix}.attn_q.bias"), device)?
                .dequantize(device)?;
            let k_bias = ct
                .tensor(reader, &format!("{prefix}.attn_k.bias"), device)?
                .dequantize(device)?;
            let v_bias = ct
                .tensor(reader, &format!("{prefix}.attn_v.bias"), device)?
                .dequantize(device)?;
            let o_proj = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{prefix}.attn_output.weight"),
                device,
            )?)?;

            let self_attn = QuantizedAttention {
                q_proj,
                k_proj,
                v_proj,
                q_bias,
                k_bias,
                v_bias,
                o_proj,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                hidden_size: cfg.hidden_size,
                rotary_emb: rotary_emb.clone(),
            };

            // MLP projections — no bias
            let mlp = QuantizedMlp {
                gate_proj: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_gate.weight"),
                    device,
                )?)?,
                up_proj: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_up.weight"),
                    device,
                )?)?,
                down_proj: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{prefix}.ffn_down.weight"),
                    device,
                )?)?,
            };

            // Layer norms — dequantize QTensor → Tensor → RmsNorm
            let input_layernorm = RmsNorm::new(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?
                    .dequantize(device)?,
                cfg.rms_norm_eps,
            );
            let post_attention_layernorm = RmsNorm::new(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?
                    .dequantize(device)?,
                cfg.rms_norm_eps,
            );

            layers.push(QuantizedDecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final norm
        let norm = RmsNorm::new(
            ct.tensor(reader, "output_norm.weight", device)?
                .dequantize(device)?,
            cfg.rms_norm_eps,
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype,
            device: device.clone(),
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
    /// * `embeds`        – `[batch, seq_len, hidden_size]`
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

        // Causal mask only needed during prefill (seq_len > 1)
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

    /// Project hidden states to vocabulary logits via the quantized lm_head.
    ///
    /// # Arguments
    ///
    /// * `hidden` – `[batch, seq_len, hidden_size]`
    ///
    /// # Returns
    ///
    /// Logit tensor `[batch, seq_len, vocab_size]`.
    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden)
    }

    /// Allocate a fresh, empty KV cache with one slot per layer.
    pub fn new_kv_cache(&self) -> KvCache {
        vec![None; self.layers.len()]
    }
}
