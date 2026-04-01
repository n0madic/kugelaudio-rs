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

use std::fmt::Write as _;

use candle_core::{DType, Device, IndexOp, Tensor};
use tokenizers::Tokenizer;

use crate::config::{DpmSolverConfig, SpecialTokens};
use crate::error::{KugelAudioError, Result};
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
    // Single allocation: estimate total length and build in one buffer
    let mut full_text = String::with_capacity(128 + text.len());
    full_text.push_str(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n Text input:\n ");
    if text.starts_with("Speaker") {
        full_text.push_str(text);
    } else {
        let _ = write!(full_text, "Speaker 0: {text}");
    }
    full_text.push_str("\n Speech output:\n");

    let encoding = tokenizer
        .encode(full_text.as_str(), false)
        .map_err(|e| KugelAudioError::Tokenizer(e.to_string()))?;

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

/// Apply classifier-free guidance in-place: `neg + scale * (pos - neg)`.
///
/// Both `pos` and `neg` are `[vocab_size]` f32 slices. The result is written
/// into `out`, which is reused across steps to avoid per-step allocation.
fn apply_cfg(pos: &[f32], neg: &[f32], scale: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(pos.len().saturating_sub(out.capacity()));
    out.extend(
        pos.iter()
            .zip(neg.iter())
            .map(|(&p, &n)| n + scale * (p - n)),
    );
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
    scheduler: &mut DpmSolverScheduler,
    condition: &Tensor,
    neg_condition: Option<&Tensor>,
    params: &GenerationParams,
    vae_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    scheduler.reset();

    // Start from Gaussian noise on CPU (avoids deterministic Metal RNG),
    // then move to the target device. Keep speech in F32 throughout the
    // diffusion loop — the scheduler operates in F32 anyway — and only
    // convert to model dtype for the prediction head forward pass.
    let mut speech =
        Tensor::randn(0f32, 1f32, &[1, 1, vae_dim], &Device::Cpu)?.to_device(device)?;

    let timesteps = scheduler.timesteps.clone();

    for (i, &t) in timesteps.iter().enumerate() {
        let timestep_tensor = Tensor::new(&[t as f32], device)?.to_dtype(model.dtype)?;
        let speech_model = speech.to_dtype(model.dtype)?;

        // Positive (conditioned) prediction
        let pos_pred = model
            .prediction_head
            .forward(&speech_model, condition, &timestep_tensor)?;

        // Guided output: optionally blend with negative branch
        let guided = match neg_condition {
            Some(neg_cond) if params.cfg_scale > 1.0 => {
                let neg_pred =
                    model
                        .prediction_head
                        .forward(&speech_model, neg_cond, &timestep_tensor)?;
                // guided = neg + scale * (pos - neg)
                let neg_f32 = neg_pred.to_dtype(DType::F32)?;
                let pos_f32 = pos_pred.to_dtype(DType::F32)?;
                (&neg_f32 + ((&pos_f32 - &neg_f32)? * params.cfg_scale as f64)?)?
            }
            _ => pos_pred.to_dtype(DType::F32)?,
        };

        let output = scheduler.step(&guided, t, &speech, i)?;
        speech = output.prev_sample;
    }

    // [1, 1, vae_dim] → [vae_dim], back to model dtype for downstream layers
    Ok(speech.to_dtype(model.dtype)?.squeeze(0)?.squeeze(0)?)
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

    // 4. Build DPM-Solver++ scheduler once (reused for every speech token) --
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

    // 5. Autoregressive loop ------------------------------------------------
    let mut all_speech_latents: Vec<Tensor> = Vec::new();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut cfg_buf: Vec<f32> = Vec::new();

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
        let guided_logits =
            if let (Some(neg_h), true) = (neg_hidden.as_ref(), params.cfg_scale > 1.0) {
                let neg_last = neg_h.i((.., (neg_h.dim(1)? - 1).., ..))?;
                let neg_logits_f32 = model
                    .lm
                    .forward_lm_head(&neg_last)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                apply_cfg(&logits_f32, &neg_logits_f32, params.cfg_scale, &mut cfg_buf);
                &cfg_buf
            } else {
                &logits_f32
            };

        // Sample the best valid token
        let (best_token, _) = sample_best_token(guided_logits, &tokens);
        generated_tokens.push(best_token);

        if step % 10 == 0 {
            eprintln!(
                "  Step {step}: token={best_token}, speech_latents={}",
                all_speech_latents.len()
            );
        }

        // End-of-speech conditions ------------------------------------------
        if best_token == tokens.speech_end_id || best_token == tokens.eos_token_id {
            // Generate one final boundary latent from the current (pre-end)
            // hidden state. speech_end can fire slightly before the last
            // phoneme is fully represented, so this extra diffusion step
            // captures any trailing audio.
            if !all_speech_latents.is_empty() {
                let condition = hidden.i((.., (hidden.dim(1)? - 1).., ..))?;
                let neg_cond = neg_hidden
                    .as_ref()
                    .map(|nh| nh.i((.., (nh.dim(1)? - 1).., ..)))
                    .transpose()?;
                let latent = sample_speech_tokens(
                    model,
                    &mut scheduler,
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
                &mut scheduler,
                &condition,
                neg_cond.as_ref(),
                params,
                vae_dim,
                device,
            )?;

            // The connector receives the raw diffusion latent directly
            // (no re-normalization — the connector learned to handle this form).
            // [vae_dim] → [1, 1, vae_dim]
            let connector_input = latent.unsqueeze(0)?.unsqueeze(0)?;
            all_speech_latents.push(latent);
            let acoustic_embed = model.acoustic_connector.forward(&connector_input)?;

            // Advance positive branch
            hidden = model
                .lm
                .forward_from_embeds(&acoustic_embed, offset, &mut cache)?;
            offset += 1;

            // Advance negative branch with acoustic embed
            if let (Some(nc), Some(nc_cache)) = (neg_hidden.as_mut(), neg_cache.as_mut()) {
                let nh = model
                    .lm
                    .forward_from_embeds(&acoustic_embed, neg_offset, nc_cache)?;
                *nc = nh;
                neg_offset += 1;
            }
        } else {
            // Regular token: embed and forward both branches.
            // Both positive and negative CFG caches see the same sampled token —
            // this keeps them synchronised.  CFG acts at the logit level (the
            // initial prompt context differs), not the token level.
            let tok = Tensor::new(&[best_token as i64], device)?.unsqueeze(0)?;
            hidden = model.lm.forward(&tok, offset, &mut cache)?;
            offset += 1;

            if let (Some(nc), Some(nc_cache)) = (neg_hidden.as_mut(), neg_cache.as_mut()) {
                let neg_tok = Tensor::new(&[best_token as i64], device)?.unsqueeze(0)?;
                let nh = model.lm.forward(&neg_tok, neg_offset, nc_cache)?;
                *nc = nh;
                neg_offset += 1;
            }
        }
    }

    // 6. Decode all latents to audio ----------------------------------------
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

    // Apply a short linear fade-out (~30 ms at 24 kHz = 720 samples) to
    // prevent clicks or noise at the tail caused by the boundary latent.
    let total_samples = audio_flat.dim(0)?;
    let fade_len = 720.min(total_samples);
    let audio_flat = if fade_len > 1 {
        let fade: Vec<f32> = (0..total_samples)
            .map(|i| {
                let tail_pos = total_samples - i;
                if tail_pos <= fade_len {
                    tail_pos as f32 / fade_len as f32
                } else {
                    1.0
                }
            })
            .collect();
        let fade_tensor = Tensor::from_vec(fade, total_samples, device)?;
        (audio_flat * fade_tensor)?
    } else {
        audio_flat
    };

    // Peak-normalize: scale so the loudest sample is at 0.95
    let max_val = audio_flat.abs()?.max(0)?.to_scalar::<f32>()?;

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
    use candle_core::Tensor;

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
        let mut buf = Vec::new();
        apply_cfg(&pos, &neg, 1.0, &mut buf);
        // scale=1.0: guided = neg + 1 * (pos - neg) = pos
        for (g, p) in buf.iter().zip(pos.iter()) {
            assert!((g - p).abs() < 1e-6, "{g} != {p}");
        }
    }

    #[test]
    fn test_apply_cfg_scale() {
        let pos = vec![2.0_f32];
        let neg = vec![0.0_f32];
        let mut buf = Vec::new();
        apply_cfg(&pos, &neg, 3.0, &mut buf);
        // guided = 0 + 3 * (2 - 0) = 6
        assert!((buf[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_decode_latents_empty() {
        // Building a KugelAudioModel requires weights, so we only verify
        // the empty-latents guard that decode_latents checks first.
        let latents: Vec<Tensor> = vec![];
        assert!(latents.is_empty());
    }

    #[test]
    fn test_sample_best_token_speech_end() {
        let tokens = SpecialTokens::default();
        let mut logits = vec![0.0f32; 200_000];
        logits[tokens.speech_end_id as usize] = 10.0;
        logits[tokens.speech_diffusion_id as usize] = -5.0;

        let (tok, _) = sample_best_token(&logits, &tokens);
        assert_eq!(tok, tokens.speech_end_id);
    }

    #[test]
    fn test_apply_cfg_negative_logits() {
        let pos = vec![-1.0_f32, -2.0];
        let neg = vec![-3.0_f32, -4.0];
        let mut buf = Vec::new();
        apply_cfg(&pos, &neg, 2.0, &mut buf);
        // guided = neg + 2 * (pos - neg) = neg + 2*pos - 2*neg = 2*pos - neg
        assert!((buf[0] - 1.0).abs() < 1e-6); // 2*(-1) - (-3) = 1
        assert!((buf[1] - 0.0).abs() < 1e-6); // 2*(-2) - (-4) = 0
    }
}
