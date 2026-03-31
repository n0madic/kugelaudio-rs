use std::sync::mpsc;

use serde::{Deserialize, Serialize};

/// Inbound JSON request — identical shape for Unix socket and HTTP.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub text: String,
    /// Overrides the server default cfg_scale.
    pub cfg_scale: Option<f32>,
    /// Overrides the server default diffusion_steps.
    pub diffusion_steps: Option<u32>,
    /// Overrides the server default max_tokens.
    pub max_tokens: Option<u32>,
}

/// Outbound JSON response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum GenerateResponse {
    Ok {
        duration_s: f32,
        speech_tokens: usize,
        /// WAV audio encoded as base64.
        audio_b64: String,
    },
    Error {
        message: String,
    },
}

/// A single unit of work sent from an acceptor thread to the main loop.
pub struct WorkItem {
    pub request: GenerateRequest,
    /// The main loop sends the response here; the acceptor thread blocks on recv().
    pub reply_tx: mpsc::Sender<GenerateResponse>,
}
