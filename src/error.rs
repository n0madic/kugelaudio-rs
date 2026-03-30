use thiserror::Error;

#[derive(Error, Debug)]
pub enum KugelAudioError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Weight loading error: {0}")]
    WeightLoading(String),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, KugelAudioError>;
