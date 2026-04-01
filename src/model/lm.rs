//! Unified language model wrapper for KugelAudio.
//!
//! The [`Lm`] enum dispatches to either the full-precision [`Qwen2Model`] or
//! the quantized [`QuantizedQwen2Model`], providing a single interface for
//! the generation pipeline.

use candle_core::{Result, Tensor};

use super::quantized_qwen2::QuantizedQwen2Model;
use super::qwen2::{KvCache, Qwen2Model};

/// Unified language model wrapping both full-precision and quantized variants.
pub enum Lm {
    /// Full-precision model loaded from safetensors.
    Full(Qwen2Model),
    /// Quantized model loaded from GGUF.
    Quantized(QuantizedQwen2Model),
}

impl Lm {
    /// Run the transformer on token IDs.
    ///
    /// See [`Qwen2Model::forward`] for argument details.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        match self {
            Self::Full(m) => m.forward(input_ids, seqlen_offset, kv_cache),
            Self::Quantized(m) => m.forward(input_ids, seqlen_offset, kv_cache),
        }
    }

    /// Run the transformer from pre-computed embeddings.
    ///
    /// See [`Qwen2Model::forward_from_embeds`] for argument details.
    pub fn forward_from_embeds(
        &self,
        embeds: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        match self {
            Self::Full(m) => m.forward_from_embeds(embeds, seqlen_offset, kv_cache),
            Self::Quantized(m) => m.forward_from_embeds(embeds, seqlen_offset, kv_cache),
        }
    }

    /// Project hidden states to vocabulary logits.
    ///
    /// See [`Qwen2Model::forward_lm_head`] for argument details.
    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Full(m) => m.forward_lm_head(hidden),
            Self::Quantized(m) => m.forward_lm_head(hidden),
        }
    }

    /// Allocate a fresh, empty KV cache with one slot per layer.
    pub fn new_kv_cache(&self) -> KvCache {
        match self {
            Self::Full(m) => m.new_kv_cache(),
            Self::Quantized(m) => m.new_kv_cache(),
        }
    }

    /// Number of hidden layers in the model.
    pub fn num_hidden_layers(&self) -> usize {
        match self {
            Self::Full(m) => m.cfg.num_hidden_layers,
            Self::Quantized(m) => m.cfg.num_hidden_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use super::super::qwen2::Qwen2Config;

    /// Build a tiny Qwen2 config for unit tests (2 layers, 4 heads, 2 kv heads).
    fn tiny_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            // Use untied embeddings so we get a proper Linear lm_head.
            // The tied path has a known candle matmul broadcasting limitation
            // with 3D input (not relevant in production where tie=false).
            tie_word_embeddings: false,
        }
    }

    /// Build a zero-initialized Qwen2Model wrapped in Lm::Full for testing.
    fn make_lm() -> Lm {
        let device = Device::Cpu;
        let vmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vmap, DType::F32, &device);
        let cfg = tiny_cfg();
        let model = super::super::qwen2::Qwen2Model::new(&cfg, vb.clone(), Some(vb)).unwrap();
        Lm::Full(model)
    }

    #[test]
    fn test_new_kv_cache_length() {
        let lm = make_lm();
        let cache = lm.new_kv_cache();
        assert_eq!(cache.len(), 2);
        assert!(cache.iter().all(|slot| slot.is_none()));
    }

    #[test]
    fn test_num_hidden_layers() {
        let lm = make_lm();
        assert_eq!(lm.num_hidden_layers(), 2);
    }

    #[test]
    fn test_forward_shape() {
        let device = Device::Cpu;
        let lm = make_lm();
        let mut cache = lm.new_kv_cache();

        // [batch=1, seq_len=3]
        let input_ids = Tensor::new(&[1i64, 2, 3], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hidden = lm.forward(&input_ids, 0, &mut cache).unwrap();

        // Output: [1, 3, hidden_size=16]
        assert_eq!(hidden.dims(), &[1, 3, 16]);

        // KV cache should now be populated
        assert!(cache.iter().all(|slot| slot.is_some()));
    }

    #[test]
    fn test_forward_from_embeds_shape() {
        let device = Device::Cpu;
        let lm = make_lm();
        let mut cache = lm.new_kv_cache();

        // [batch=1, seq_len=2, hidden_size=16]
        let embeds = Tensor::zeros(&[1, 2, 16], DType::F32, &device).unwrap();
        let hidden = lm.forward_from_embeds(&embeds, 0, &mut cache).unwrap();

        assert_eq!(hidden.dims(), &[1, 2, 16]);
    }

    #[test]
    fn test_forward_lm_head_shape() {
        let device = Device::Cpu;
        let lm = make_lm();

        // [batch=1, seq_len=1, hidden_size=16]
        let hidden = Tensor::zeros(&[1, 1, 16], DType::F32, &device).unwrap();
        let logits = lm.forward_lm_head(&hidden).unwrap();

        // Output: [1, 1, vocab_size=32]
        assert_eq!(logits.dims(), &[1, 1, 32]);
    }

    #[test]
    fn test_forward_incremental_decode() {
        let device = Device::Cpu;
        let lm = make_lm();
        let mut cache = lm.new_kv_cache();

        // Prefill with 3 tokens
        let input_ids = Tensor::new(&[1i64, 2, 3], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let _ = lm.forward(&input_ids, 0, &mut cache).unwrap();

        // Single-token decode step (seqlen_offset = 3)
        let next_token = Tensor::new(&[4i64], &device).unwrap().unsqueeze(0).unwrap();
        let hidden = lm.forward(&next_token, 3, &mut cache).unwrap();

        // Output should be [1, 1, 16]
        assert_eq!(hidden.dims(), &[1, 1, 16]);

        // KV cache should have accumulated 4 tokens total
        if let Some((k, _v)) = &cache[0] {
            assert_eq!(k.dim(2).unwrap(), 4); // seq_len dimension
        }
    }
}
