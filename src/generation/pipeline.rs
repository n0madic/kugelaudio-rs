use mlx_rs::{
    array,
    error::Exception,
    module::Module,
    ops::{
        concatenate_axis,
        indexing::{IndexOp, NewAxis},
    },
    random,
    transforms::eval,
    Array,
};
use mlx_rs_core::{create_attention_mask, AttentionMask, ConcatKeyValueCache};
use tokenizers::Tokenizer;

use qwen3_mlx::qwen2::{self, AttentionInput};

use crate::config::SpecialTokens;
use crate::model::diffusion_head::DiffusionHeadInput;
use crate::model::weights::KugelAudioModel;
use crate::schedule::dpm_solver::DpmSolverScheduler;

pub struct GenerationParams {
    pub cfg_scale: f32,
    pub max_new_tokens: u32,
    pub diffusion_steps: u32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            cfg_scale: 3.0,
            max_new_tokens: 2048,
            diffusion_steps: 20,
        }
    }
}

pub struct GenerationOutput {
    pub sequences: Vec<u32>,
    pub audio: Option<Array>,
}

/// Forward through Qwen2 layers bypassing embedding lookup.
fn forward_from_embeds(
    qwen2: &mut qwen2::Qwen2Model,
    h: &Array,
    cache: &mut Vec<Option<ConcatKeyValueCache>>,
) -> Result<Array, Exception> {
    let mask = match create_attention_mask(h, cache, Some(true))? {
        Some(AttentionMask::Array(a)) => Some(a),
        _ => None,
    };
    if cache.is_empty() {
        *cache = (0..qwen2.layers.len())
            .map(|_| Some(ConcatKeyValueCache::default()))
            .collect();
    }
    let mut h = h.clone();
    for (layer, c) in qwen2.layers.iter_mut().zip(cache.iter_mut()) {
        h = layer.forward(AttentionInput {
            x: &h,
            mask: mask.as_ref(),
            cache: c.as_mut(),
        })?;
    }
    qwen2.norm.forward(&h)
}

fn build_prompt_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>, Exception> {
    let formatted = if text.starts_with("Speaker") {
        text.to_string()
    } else {
        format!("Speaker 0: {text}")
    };
    let system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
    let text_section = format!(" Text input:\n {formatted}\n Speech output:\n");
    let full_text = format!("{system_prompt}{text_section}");
    let encoding = tokenizer
        .encode(full_text.as_str(), false)
        .map_err(|e| Exception::custom(format!("Tokenizer error: {e}")))?;
    let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    ids.push(SpecialTokens::default().speech_start_id as i32);
    Ok(ids)
}

/// Run the KugelAudio TTS inference pipeline.
pub fn generate(
    model: &mut KugelAudioModel,
    tokenizer: &Tokenizer,
    text: &str,
    params: &GenerationParams,
) -> Result<GenerationOutput, Exception> {
    let tokens = SpecialTokens::default();
    let vae_dim = 64i32;
    random::seed(0)?;

    let prompt_ids = build_prompt_tokens(tokenizer, text)?;
    let input_ids = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    // Prefill
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let prompt_embeds = model.lm.model.embed_tokens.forward(&input_ids)?;
    let mut hidden_states = forward_from_embeds(&mut model.lm.model, &prompt_embeds, &mut cache)?;
    eval([&hidden_states])?;

    // CFG negative branch
    let mut neg_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut neg_hidden = if params.cfg_scale > 1.0 {
        let neg_ids = Array::from_slice(&[tokens.speech_start_id as i32], &[1, 1]);
        let neg_embeds = model.lm.model.embed_tokens.forward(&neg_ids)?;
        let nh = forward_from_embeds(&mut model.lm.model, &neg_embeds, &mut neg_cache)?;
        eval([&nh])?;
        Some(nh)
    } else {
        None
    };

    // AR generation loop
    let mut all_speech_latents: Vec<Array> = Vec::new();
    let mut generated_tokens: Vec<u32> = Vec::new();

    for step in 0..params.max_new_tokens {
        // LM head → logits
        let last_hidden = hidden_states.index((.., -1.., ..));
        let logits = match model.lm.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&last_hidden)?,
            None => match &mut model.lm.model.embed_tokens {
                mlx_rs::quantization::MaybeQuantized::Original(e) => e.as_linear(&last_hidden)?,
                mlx_rs::quantization::MaybeQuantized::Quantized(q) => q.as_linear(&last_hidden)?,
            },
        };

        // Constrained argmax
        let logits_last = logits.index((.., -1, ..)).as_type::<f32>()?;
        let logits_data = logits_last.as_slice::<f32>();
        let valid = [
            tokens.speech_start_id,
            tokens.speech_end_id,
            tokens.speech_diffusion_id,
            tokens.eos_token_id,
        ];
        let mut best_token = tokens.eos_token_id;
        let mut best_logit = f32::NEG_INFINITY;
        for &tid in &valid {
            if (tid as usize) < logits_data.len() {
                let l = logits_data[tid as usize];
                if l > best_logit {
                    best_logit = l;
                    best_token = tid;
                }
            }
        }
        generated_tokens.push(best_token);

        if step % 10 == 0 {
            eprintln!(
                "  Step {step}: {best_token}, latents={}",
                all_speech_latents.len()
            );
        }

        // End conditions
        if best_token == tokens.speech_end_id || best_token == tokens.eos_token_id {
            let diff_logit = logits_data[tokens.speech_diffusion_id as usize];
            if !all_speech_latents.is_empty() && diff_logit > best_logit - 5.0 {
                let condition = hidden_states.index((.., -1, ..)).as_type::<f32>()?;
                let neg_cond = neg_hidden
                    .as_ref()
                    .map(|h| h.index((.., -1, ..)).as_type::<f32>())
                    .transpose()?;
                let lat = sample_speech_tokens(
                    &mut model.prediction_head,
                    &condition,
                    neg_cond.as_ref(),
                    params,
                    vae_dim,
                )?;
                all_speech_latents.push(lat);
            }
            break;
        }

        if best_token == tokens.speech_diffusion_id {
            let condition = hidden_states.index((.., -1, ..)).as_type::<f32>()?;
            let neg_cond = neg_hidden
                .as_ref()
                .map(|h| h.index((.., -1, ..)).as_type::<f32>())
                .transpose()?;

            let speech_latents = sample_speech_tokens(
                &mut model.prediction_head,
                &condition,
                neg_cond.as_ref(),
                params,
                vae_dim,
            )?;
            all_speech_latents.push(speech_latents.clone());

            // Feedback: latent → connector → LM
            let acoustic_embed =
                model
                    .acoustic_connector
                    .forward(&speech_latents.index((.., NewAxis, ..)))?;
            hidden_states = forward_from_embeds(&mut model.lm.model, &acoustic_embed, &mut cache)?;
            eval([&hidden_states])?;

            if params.cfg_scale > 1.0 {
                let nh = forward_from_embeds(&mut model.lm.model, &acoustic_embed, &mut neg_cache)?;
                eval([&nh])?;
                neg_hidden = Some(nh);
            }
        } else if best_token == tokens.speech_start_id {
            let tok_embed = model
                .lm
                .model
                .embed_tokens
                .forward(&Array::from_slice(&[best_token as i32], &[1, 1]))?;
            hidden_states = forward_from_embeds(&mut model.lm.model, &tok_embed, &mut cache)?;
            eval([&hidden_states])?;
        }
    }

    // Batch-decode all latents
    let audio = if !all_speech_latents.is_empty() {
        let expanded: Vec<Array> = all_speech_latents
            .iter()
            .map(|lat| lat.index((.., NewAxis, ..)))
            .collect();
        let refs: Vec<&Array> = expanded.iter().collect();
        let latent_seq = concatenate_axis(&refs, 1)?;

        let latent_seq = if !model.speech_scaling_factor.is_nan() {
            latent_seq
                .divide(&array!(model.speech_scaling_factor))?
                .subtract(&array!(model.speech_bias_factor))?
        } else {
            latent_seq
        };

        let audio_out = model.acoustic_decoder.forward(&latent_seq)?;
        eval([&audio_out])?;
        let audio = audio_out.squeeze()?;

        let max_val = audio.abs()?.max(None)?.item::<f32>();
        let audio = if max_val > 1.0 {
            audio.multiply(&array!(0.95 / max_val))?
        } else {
            audio
        };
        Some(audio)
    } else {
        None
    };

    Ok(GenerationOutput {
        sequences: generated_tokens,
        audio,
    })
}

fn sample_speech_tokens(
    prediction_head: &mut crate::model::diffusion_head::DiffusionHead,
    condition: &Array,
    neg_condition: Option<&Array>,
    params: &GenerationParams,
    vae_dim: i32,
) -> Result<Array, Exception> {
    let batch_size = condition.dim(0);
    let mut scheduler =
        DpmSolverScheduler::new(1000, "cosine", "v_prediction", "sde-dpmsolver++", 2)?;
    scheduler.set_timesteps(params.diffusion_steps as i32);

    let condition_f32 = condition.as_type::<f32>()?;
    let timesteps = scheduler.timesteps.clone();

    let neg_f32 = match neg_condition {
        Some(nc) if params.cfg_scale > 1.0 => Some(nc.as_type::<f32>()?),
        _ => None,
    };

    if let Some(neg_f32) = neg_f32 {
        let combined_cond = concatenate_axis(&[&condition_f32, &neg_f32], 0)?;
        let n = batch_size;

        let mut speech = random::normal::<f32>(&[batch_size, vae_dim], None, None, None)?;
        for &t in &timesteps {
            let combined_speech = concatenate_axis(&[&speech, &speech], 0)?;
            let t_tensor =
                Array::from_slice(&vec![t; (n * 2) as usize], &[n * 2]).as_type::<f32>()?;
            let eps = prediction_head
                .forward(DiffusionHeadInput {
                    noisy_images: &combined_speech,
                    timesteps: &t_tensor,
                    condition: &combined_cond,
                })?
                .as_type::<f32>()?;

            let cond_eps = eps.index((..n, ..));
            let uncond_eps = eps.index((n.., ..));
            // CFG: guided = uncond + scale * (cond - uncond)
            let guided = uncond_eps.add(
                cond_eps
                    .subtract(&uncond_eps)?
                    .multiply(&array!(params.cfg_scale))?,
            )?;

            let result = scheduler.step(&guided, t, &speech)?;
            speech = result.prev_sample;
            eval([&speech])?;
        }
        Ok(speech)
    } else {
        // No CFG — single forward pass per step
        let mut speech = random::normal::<f32>(&[batch_size, vae_dim], None, None, None)?;
        for &t in &timesteps {
            let t_tensor =
                Array::from_slice(&vec![t; batch_size as usize], &[batch_size]).as_type::<f32>()?;
            let eps = prediction_head
                .forward(DiffusionHeadInput {
                    noisy_images: &speech,
                    timesteps: &t_tensor,
                    condition: &condition_f32,
                })?
                .as_type::<f32>()?;
            let result = scheduler.step(&eps, t, &speech)?;
            speech = result.prev_sample;
            eval([&speech])?;
        }
        Ok(speech)
    }
}
