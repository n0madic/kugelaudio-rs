use std::sync::mpsc;

use candle_core::Device;
use tokenizers::Tokenizer;

use crate::audio::wav;
use crate::config::SpecialTokens;
use crate::generation::pipeline::{self, GenerationParams, MAX_TEXT_CHARS};
use crate::model::weights::KugelAudioModel;

use super::protocol::{GenerateResponse, WorkItem};

/// CLI-supplied fallback values used when a request omits a field.
pub struct ServerDefaults {
    pub cfg_scale: f32,
    pub max_tokens: u32,
    pub diffusion_steps: u32,
}

/// Block forever, processing one request at a time from `work_rx`.
///
/// The model stays on the calling thread for the lifetime of the server.
pub fn run_server_loop(
    model: &KugelAudioModel,
    tokenizer: &Tokenizer,
    defaults: &ServerDefaults,
    work_rx: mpsc::Receiver<WorkItem>,
) {
    let _tokens = SpecialTokens::default();

    for item in work_rx {
        let req = &item.request;

        // Reject oversized text early, before tokenization.
        if req.text.len() > MAX_TEXT_CHARS {
            let _ = item.reply_tx.send(GenerateResponse::Error(format!(
                "Text too long ({} chars, max {MAX_TEXT_CHARS})",
                req.text.len()
            )));
            continue;
        }

        let params = GenerationParams {
            cfg_scale: req.cfg_scale.unwrap_or(defaults.cfg_scale),
            max_new_tokens: req.max_tokens.unwrap_or(defaults.max_tokens),
            diffusion_steps: req.diffusion_steps.unwrap_or(defaults.diffusion_steps),
        }
        .validated();

        let response = match pipeline::generate(model, tokenizer, &req.text, &params) {
            Err(e) => GenerateResponse::Error(format!("Generation failed: {e}")),
            Ok(output) => {
                // Ensure all GPU work has completed before reading tensors
                // back to the host for WAV encoding.
                if let Device::Metal(_) | Device::Cuda(_) = &model.device {
                    if let Err(e) = model.device.synchronize() {
                        eprintln!("[warn] device synchronize failed: {e}");
                    }
                }

                match output.audio {
                    None => GenerateResponse::Error("No speech tokens generated".to_string()),
                    Some(ref audio) => match wav::to_wav_bytes(audio) {
                        Err(e) => GenerateResponse::Error(format!("WAV encoding failed: {e}")),
                        Ok(bytes) => GenerateResponse::Ok(bytes),
                    },
                }
            }
        };

        // If the acceptor thread has already disconnected, discard the response.
        let _ = item.reply_tx.send(response);
    }
}
