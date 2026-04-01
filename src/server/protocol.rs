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

/// JSON error response sent when generation fails.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub status: String,
    pub message: String,
}

impl ErrorResponse {
    pub fn new(message: String) -> Self {
        Self {
            status: "error".to_string(),
            message,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"status":"error","message":"serialization failed"}"#.to_string()
        })
    }
}

/// Server-internal response passed from the generation loop to transport threads.
///
/// Success carries raw WAV bytes — no base64 overhead.
/// Error carries a message that each transport serializes as JSON.
pub enum GenerateResponse {
    /// Raw WAV audio bytes.
    Ok(Vec<u8>),
    /// Human-readable error message.
    Error(String),
}

/// A single unit of work sent from an acceptor thread to the main loop.
pub struct WorkItem {
    pub request: GenerateRequest,
    /// The main loop sends the response here; the acceptor thread blocks on recv().
    pub reply_tx: mpsc::Sender<GenerateResponse>,
}
