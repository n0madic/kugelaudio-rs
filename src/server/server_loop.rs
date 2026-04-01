use std::sync::mpsc;

use base64::Engine as _;
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::audio::wav;
use crate::config::{SAMPLE_RATE, SpecialTokens};
use crate::generation::pipeline::{self, GenerationParams};
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
    let tokens = SpecialTokens::default();

    for item in work_rx {
        let req = &item.request;

        let params = GenerationParams {
            cfg_scale: req.cfg_scale.unwrap_or(defaults.cfg_scale),
            max_new_tokens: req.max_tokens.unwrap_or(defaults.max_tokens),
            diffusion_steps: req.diffusion_steps.unwrap_or(defaults.diffusion_steps),
        };

        let response = match pipeline::generate(model, tokenizer, &req.text, &params) {
            Err(e) => GenerateResponse::Error {
                message: format!("Generation failed: {e}"),
            },
            Ok(output) => {
                // Ensure all GPU work has completed before reading tensors
                // back to the host for WAV encoding.
                if let Device::Metal(_) | Device::Cuda(_) = &model.device {
                    if let Err(e) = model.device.synchronize() {
                        eprintln!("[warn] device synchronize failed: {e}");
                    }
                }

                let speech_tokens = output
                    .sequences
                    .iter()
                    .filter(|&&t| t == tokens.speech_diffusion_id)
                    .count();
                let duration_s = speech_tokens as f32 * 3200.0 / SAMPLE_RATE as f32;

                match output.audio {
                    None => GenerateResponse::Error {
                        message: "No speech tokens generated".to_string(),
                    },
                    Some(ref audio) => match wav::to_wav_bytes(audio) {
                        Err(e) => GenerateResponse::Error {
                            message: format!("WAV encoding failed: {e}"),
                        },
                        Ok(bytes) => {
                            let audio_b64 =
                                base64::engine::general_purpose::STANDARD.encode(&bytes);
                            GenerateResponse::Ok {
                                duration_s,
                                speech_tokens,
                                audio_b64,
                            }
                        }
                    },
                }
            }
        };

        // If the acceptor thread has already disconnected, discard the response.
        let _ = item.reply_tx.send(response);
    }
}
