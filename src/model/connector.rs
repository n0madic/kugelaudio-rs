use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_b, rms_norm};

/// Speech connector that maps acoustic features (64-dim) into the
/// language model's hidden space (3584-dim for 7B).
///
/// Architecture: Linear(input_dim -> output_dim) -> RMSNorm(output_dim) -> Linear(output_dim -> output_dim)
///
/// Expected VarBuilder prefix: `model.acoustic_connector.`
pub struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    /// Create a `SpeechConnector` by loading weights from `vb`.
    ///
    /// The builder must expose keys `fc1.weight`, `fc1.bias`, `norm.weight`,
    /// `fc2.weight`, and `fc2.bias` at its current prefix.
    pub fn new(input_dim: usize, output_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_b(input_dim, output_dim, true, vb.pp("fc1"))?;
        let norm = rms_norm(output_dim, eps, vb.pp("norm"))?;
        let fc2 = linear_b(output_dim, output_dim, true, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }

    /// Map acoustic feature tensor through the connector.
    ///
    /// Input shape: `[batch, seq_len, input_dim]`
    /// Output shape: `[batch, seq_len, output_dim]`
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(features)?;
        let x = self.norm.forward(&x)?;
        self.fc2.forward(&x)
    }
}
