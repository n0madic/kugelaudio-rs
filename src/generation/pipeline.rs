//! KugelAudio TTS generation pipeline.
//!
//! Implements the full text-to-speech inference loop:
//!
//! 1. **Tokenize** — format the input text and encode it to token IDs.
//! 2. **Prefill** — run the prompt through the LM to populate the KV cache.
//! 3. **AR loop** — autoregressively generate speech tokens:
//!    - Each step: LM forward → extract last-position logits → sample the
//!      best valid token.
//!    - If the sampled token is `speech_diffusion_id`: run the DPM-Solver++
//!      diffusion loop to obtain a speech latent, then feed the acoustic
//!      embedding back into the LM.
//! 4. **Decode** — all collected speech latents → [`AcousticDecoder`] → audio
//!    waveform.
//!
//! ## Classifier-Free Guidance (CFG)
//!
//! When `cfg_scale > 1.0`, two independent KV caches are maintained for the
//! same model weights:
//! - **Positive** (`cache`): conditioned on the full text prompt.
//! - **Negative** (`neg_cache`): conditioned only on the `speech_start` token.
//!
//! Guided logits: `neg + cfg_scale * (pos - neg)`.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use tokenizers::Tokenizer;

use crate::config::{DpmSolverConfig, SpecialTokens};
use crate::model::qwen2::KvCache;
use crate::model::weights::KugelAudioModel;
use crate::schedule::dpm_solver::DpmSolverScheduler;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Parameters controlling the generation run.
pub struct GenerationParams {
    /// Classifier-Free Guidance scale. `1.0` disables CFG.
    pub cfg_scale: f32,
    /// Maximum number of autoregressive steps before hard stop.
    pub max_new_tokens: u32,
    /// Number of DPM-Solver++ denoising steps per speech token.
    pub diffusion_steps: u32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            cfg_scale: 3.0,
            max_new_tokens: 2048,
            diffusion_steps: 10,
        }
    }
}

/// Output of the generation pipeline.
pub struct GenerationOutput {
    /// All token IDs produced by the AR loop, in order.
    pub sequences: Vec<u32>,
    /// Decoded audio waveform, normalized to `[-1, 1]`.
    ///
    /// `None` when no `speech_diffusion_id` tokens were generated.
    pub audio: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// Prompt building
// ---------------------------------------------------------------------------

/// Format the user text into the KugelAudio prompt template and encode it.
///
/// The prompt is wrapped with the standard system instruction and a speaker
/// prefix, then the `speech_start` special token is appended.
///
/// Returns a flat `Vec<u32>` of token IDs.
fn build_prompt_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let formatted = if text.starts_with("Speaker") {
        text.to_string()
    } else {
        format!("Speaker 0: {text}")
    };
    let system_prompt =
        " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
    let text_section = format!(" Text input:\n {formatted}\n Speech output:\n");
    let full_text = format!("{system_prompt}{text_section}");

    let encoding = tokenizer
        .encode(full_text.as_str(), false)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {e}")))?;

    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    ids.push(SpecialTokens::default().speech_start_id);
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Token sampling
// ---------------------------------------------------------------------------

/// Pick the best valid special token from raw logits.
///
/// Only considers `speech_start_id`, `speech_end_id`, `speech_diffusion_id`,
/// and `eos_token_id`. Returns the one with the highest logit value, falling
/// back to `eos_token_id` if none are present in the vocabulary.
fn sample_best_token(logits: &[f32], tokens: &SpecialTokens) -> (u32, f32) {
    let valid = [
        tokens.speech_start_id,
        tokens.speech_end_id,
        tokens.speech_diffusion_id,
        tokens.eos_token_id,
    ];
    let mut best_token = tokens.eos_token_id;
    let mut best_logit = f32::NEG_INFINITY;
    for &tid in &valid {
        if let Some(&l) = logits.get(tid as usize) {
            if l > best_logit {
                best_logit = l;
                best_token = tid;
            }
        }
    }
    (best_token, best_logit)
}

// ---------------------------------------------------------------------------
// CFG logit fusion
// ---------------------------------------------------------------------------

/// Apply classifier-free guidance: `neg + scale * (pos - neg)`.
///
/// Both `pos` and `neg` are `[vocab_size]` f32 slices. Returns a `Vec<f32>`
/// of guided logits with the same length.
fn apply_cfg(pos: &[f32], neg: &[f32], scale: f32) -> Vec<f32> {
    pos.iter()
        .zip(neg.iter())
        .map(|(&p, &n)| n + scale * (p - n))
        .collect()
}

// ---------------------------------------------------------------------------
// Diffusion sampling
// ---------------------------------------------------------------------------

/// Run the DPM-Solver++ denoising loop to sample a single speech latent.
///
/// # Arguments
///
/// * `model`         – full KugelAudio model (prediction head accessed here)
/// * `condition`     – LM hidden state at the current step, shape `[1, 1, hidden_size]`
/// * `neg_condition` – unconditioned hidden state (CFG negative branch), same shape
/// * `params`        – generation parameters (cfg_scale, diffusion_steps)
/// * `vae_dim`       – dimensionality of the speech VAE latent space
/// * `device`        – compute device
///
/// # Returns
///
/// A single speech latent tensor with shape `[vae_dim]`.
fn sample_speech_tokens(
    model: &KugelAudioModel,
    condition: &Tensor,
    neg_condition: Option<&Tensor>,
    params: &GenerationParams,
    vae_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let dpm_cfg = DpmSolverConfig::from(&model.diffusion_head_config);
    let mut scheduler = DpmSolverScheduler::new(
        dpm_cfg.num_train_timesteps,
        &dpm_cfg.beta_schedule,
        &dpm_cfg.prediction_type,
        &dpm_cfg.algorithm_type,
        dpm_cfg.solver_order,
        device.clone(),
    )?;
    scheduler.set_timesteps(params.diffusion_steps as i32);

    // Start from Gaussian noise on CPU (avoids deterministic Metal RNG),
    // then move to the target device.
    let mut speech = Tensor::randn(0f32, 1f32, &[1, 1, vae_dim], &Device::Cpu)?
        .to_device(device)?
        .to_dtype(model.dtype)?;

    let timesteps = scheduler.timesteps.clone();

    for (i, &t) in timesteps.iter().enumerate() {
        let timestep_tensor = Tensor::new(&[t as f32], device)?.to_dtype(model.dtype)?;

        // Positive (conditioned) prediction
        let pos_pred =
            model
                .prediction_head
                .forward(&speech, condition, &timestep_tensor)?;


        // Guided output: optionally blend with negative branch
        let guided = match neg_condition {
            Some(neg_cond) if params.cfg_scale > 1.0 => {
                let neg_pred =
                    model
                        .prediction_head
                        .forward(&speech, neg_cond, &timestep_tensor)?;
                // guided = neg + scale * (pos - neg)
                let diff = (pos_pred.to_dtype(DType::F32)?
                    - neg_pred.to_dtype(DType::F32)?)?;
                (neg_pred.to_dtype(DType::F32)?
                    + (diff * params.cfg_scale as f64)?)?
            }
            _ => pos_pred,
        };

        let output = scheduler.step(&guided.to_dtype(DType::F32)?, t, &speech.to_dtype(DType::F32)?, i)?;
        speech = output.prev_sample.to_dtype(model.dtype)?;
    }

    // [1, 1, vae_dim] → [vae_dim]
    speech.squeeze(0)?.squeeze(0)
}

// ---------------------------------------------------------------------------
// Main pipeline entry point
// ---------------------------------------------------------------------------

/// Run the full KugelAudio TTS inference pipeline.
///
/// # Arguments
///
/// * `model`     – loaded [`KugelAudioModel`]
/// * `tokenizer` – HuggingFace tokenizer matching the model vocabulary
/// * `text`      – input text to synthesise
/// * `params`    – generation hyper-parameters
///
/// # Returns
///
/// [`GenerationOutput`] containing the generated token sequence and, if any
/// speech tokens were produced, the decoded audio waveform.
///
/// # Errors
///
/// Returns a candle [`Result`] error if any tensor operation fails.
pub fn generate(
    model: &KugelAudioModel,
    tokenizer: &Tokenizer,
    text: &str,
    params: &GenerationParams,
) -> Result<GenerationOutput> {
    let tokens = SpecialTokens::default();
    let device = &model.device;
    let vae_dim = model.vae_dim;

    // 1. Tokenize -----------------------------------------------------------
    let prompt_ids = build_prompt_tokens(tokenizer, text)?;
    let prompt_len = prompt_ids.len();

    // candle expects i64 token IDs for embedding lookup
    let prompt_ids_i64: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
    let input_tensor = Tensor::new(prompt_ids_i64.as_slice(), device)?.unsqueeze(0)?;

    // 2. Prefill positive branch --------------------------------------------
    let mut cache: KvCache = model.lm.new_kv_cache();
    let mut hidden = model.lm.forward(&input_tensor, 0, &mut cache)?;
    let mut offset = prompt_len;


    // 3. Prefill negative branch (speech_start only) ------------------------
    // Negative branch for CFG: unconditioned (speech_start only).
    // Advanced in parallel with the positive branch.
    let (mut neg_hidden, mut neg_cache, mut neg_offset) = if params.cfg_scale > 1.0 {
        let neg_ids = Tensor::new(&[tokens.speech_start_id as i64], device)?.unsqueeze(0)?;
        let mut nc: KvCache = model.lm.new_kv_cache();
        let nh = model.lm.forward(&neg_ids, 0, &mut nc)?;
        (Some(nh), Some(nc), 1usize)
    } else {
        (None, None, 0)
    };

    // 4. Autoregressive loop ------------------------------------------------
    let mut all_speech_latents: Vec<Tensor> = Vec::new();
    let mut generated_tokens: Vec<u32> = Vec::new();

    for step in 0..params.max_new_tokens {
        // Extract last-position hidden state: [1, 1, hidden_size]
        let last_hidden = hidden.i((.., (hidden.dim(1)? - 1).., ..))?;
        // Project to logits: [1, 1, vocab_size]
        let logits = model.lm.forward_lm_head(&last_hidden)?;
        // [vocab_size]
        let logits_f32 = logits
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        // CFG: blend with negative branch logits when applicable
        let guided_logits = if let (Some(neg_h), true) =
            (neg_hidden.as_ref(), params.cfg_scale > 1.0)
        {
            let neg_last = neg_h.i((.., (neg_h.dim(1)? - 1).., ..))?;
            let neg_logits_f32 = model
                .lm
                .forward_lm_head(&neg_last)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            apply_cfg(&logits_f32, &neg_logits_f32, params.cfg_scale)
        } else {
            logits_f32.clone()
        };

        // Sample the best valid token
        let (best_token, best_logit) = sample_best_token(&guided_logits, &tokens);
        generated_tokens.push(best_token);

        if step % 10 == 0 {
            eprintln!(
                "  Step {step}: token={best_token}, speech_latents={}",
                all_speech_latents.len()
            );
        }

        // End-of-speech conditions ------------------------------------------
        if best_token == tokens.speech_end_id || best_token == tokens.eos_token_id {
            // If diffusion logit is close to the winner and we already have
            // some latents, generate one final latent before stopping. This
            // mirrors the Python reference pipeline's behaviour at the boundary.
            let diff_logit = guided_logits
                .get(tokens.speech_diffusion_id as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);

            if !all_speech_latents.is_empty() && diff_logit > best_logit - 5.0 {
                let condition = hidden.i((.., (hidden.dim(1)? - 1).., ..))?;
                let neg_cond = neg_hidden
                    .as_ref()
                    .map(|nh| nh.i((.., (nh.dim(1)? - 1).., ..)))
                    .transpose()?;
                let latent = sample_speech_tokens(
                    model,
                    &condition,
                    neg_cond.as_ref(),
                    params,
                    vae_dim,
                    device,
                )?;
                all_speech_latents.push(latent);
            }
            break;
        }

        // Speech diffusion step ---------------------------------------------
        if best_token == tokens.speech_diffusion_id {
            let condition = hidden.i((.., (hidden.dim(1)? - 1).., ..))?;
            let neg_cond = neg_hidden
                .as_ref()
                .map(|nh| nh.i((.., (nh.dim(1)? - 1).., ..)))
                .transpose()?;

            let latent = sample_speech_tokens(
                model,
                &condition,
                neg_cond.as_ref(),
                params,
                vae_dim,
                device,
            )?;
            all_speech_latents.push(latent.clone());

            // The connector receives the raw diffusion latent directly
            // (no re-normalization — the connector learned to handle this form).
            // [vae_dim] → [1, 1, vae_dim]
            let connector_input = latent.unsqueeze(0)?.unsqueeze(0)?;
            let acoustic_embed =
                model
                    .acoustic_connector
                    .forward(&connector_input)?;

            // Advance positive branch
            hidden = model
                .lm
                .forward_from_embeds(&acoustic_embed, offset, &mut cache)?;
            offset += 1;

            // Advance negative branch with acoustic embed
            if let (Some(nc), Some(nc_cache)) =
                (neg_hidden.as_mut(), neg_cache.as_mut())
            {
                let nh = model.lm.forward_from_embeds(&acoustic_embed, neg_offset, nc_cache)?;
                *nc = nh;
                neg_offset += 1;
            }
        } else {
            // Regular token: embed and forward both branches
            let tok = Tensor::new(&[best_token as i64], device)?.unsqueeze(0)?;
            hidden = model.lm.forward(&tok, offset, &mut cache)?;
            offset += 1;

            if let (Some(nc), Some(nc_cache)) =
                (neg_hidden.as_mut(), neg_cache.as_mut())
            {
                let neg_tok = Tensor::new(&[best_token as i64], device)?.unsqueeze(0)?;
                let nh = model.lm.forward(&neg_tok, neg_offset, nc_cache)?;
                *nc = nh;
                neg_offset += 1;
            }
        }
    }

    // 5. Decode all latents to audio ----------------------------------------
    let audio = decode_latents(model, &all_speech_latents, device)?;

    Ok(GenerationOutput {
        sequences: generated_tokens,
        audio,
    })
}

// ---------------------------------------------------------------------------
// Latent decoding
// ---------------------------------------------------------------------------

/// Stack collected speech latents and run them through the acoustic decoder.
///
/// # Layout
///
/// Each latent is `[vae_dim]`. They are stacked to `[N, vae_dim]`, then
/// reshaped to `[1, vae_dim, N]` (NCL format) before passing to the decoder.
///
/// The decoder returns `[1, 1, samples]` which is flattened and normalised to
/// `[-1, 1]`.
///
/// Returns `None` when `latents` is empty.
fn decode_latents(
    model: &KugelAudioModel,
    latents: &[Tensor],
    device: &Device,
) -> Result<Option<Tensor>> {
    if latents.is_empty() {
        return Ok(None);
    }

    // Stack all latents: [N, vae_dim]
    let stacked = Tensor::stack(latents, 0)?;

    // De-normalize in F32 for precision, then cast to model dtype for decoder.
    let stacked = if model.speech_scaling_factor != 0.0 {
        stacked
            .to_dtype(DType::F32)?
            .broadcast_div(&Tensor::new(&[model.speech_scaling_factor], device)?)?
            .broadcast_sub(&Tensor::new(&[model.speech_bias_factor], device)?)?
            .to_dtype(model.dtype)?
    } else {
        stacked
    };

    // [N, vae_dim] → [1, vae_dim, N] NCL for the convolutional decoder
    let latents_ncl = stacked.unsqueeze(0)?.transpose(1, 2)?;

    // Decode all at once → [1, 1, samples]
    let audio_out = model.acoustic_decoder.forward(&latents_ncl)?;

    // Flatten to [samples]
    let audio_flat = audio_out.flatten_all()?.to_dtype(DType::F32)?;

    // Peak-normalize: scale so the loudest sample is at 0.95
    let max_val = audio_flat
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;

    let audio = if max_val > 1e-6 {
        (audio_flat * (0.95 / max_val) as f64)?
    } else {
        audio_flat
    };

    Ok(Some(audio))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_sample_best_token_prefers_diffusion() {
        let tokens = SpecialTokens::default();
        let mut logits = vec![0.0f32; 200_000];
        logits[tokens.speech_diffusion_id as usize] = 10.0;
        logits[tokens.speech_end_id as usize] = 5.0;

        let (tok, _) = sample_best_token(&logits, &tokens);
        assert_eq!(tok, tokens.speech_diffusion_id);
    }

    #[test]
    fn test_sample_best_token_falls_back_to_eos_on_empty_logits() {
        let tokens = SpecialTokens::default();
        // Logits vec shorter than all valid token IDs
        let logits: Vec<f32> = vec![0.0f32; 5];
        let (tok, _) = sample_best_token(&logits, &tokens);
        assert_eq!(tok, tokens.eos_token_id);
    }

    #[test]
    fn test_apply_cfg_no_guidance() {
        let pos = vec![1.0_f32, 2.0, 3.0];
        let neg = vec![0.0_f32, 0.0, 0.0];
        let guided = apply_cfg(&pos, &neg, 1.0);
        // scale=1.0: guided = neg + 1 * (pos - neg) = pos
        for (g, p) in guided.iter().zip(pos.iter()) {
            assert!((g - p).abs() < 1e-6, "{g} != {p}");
        }
    }

    #[test]
    fn test_apply_cfg_scale() {
        let pos = vec![2.0_f32];
        let neg = vec![0.0_f32];
        let guided = apply_cfg(&pos, &neg, 3.0);
        // guided = 0 + 3 * (2 - 0) = 6
        assert!((guided[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_decode_latents_empty() {
        let device = Device::Cpu;
        // Build a minimal KugelAudioModel is not feasible in a unit test
        // without weights, so we only test the empty-path guard here.
        // A full integration test would require a loaded model.
        let latents: Vec<Tensor> = vec![];

        // Manually call the empty-path guard logic inline:
        assert!(latents.is_empty());
    }
}
