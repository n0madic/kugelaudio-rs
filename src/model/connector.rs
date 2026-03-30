use mlx_rs::{
    builder::Builder, error::Exception, macros::ModuleParameters, module::Module, nn, Array,
};

/// Speech connector that maps acoustic features (64-dim) into the
/// language model's hidden space (3584-dim for 7B).
///
/// Architecture: Linear(input_dim -> hidden_size) -> RMSNorm -> Linear(hidden_size -> hidden_size)
#[derive(Debug, Clone, ModuleParameters)]
pub struct SpeechConnector {
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub fc2: nn::Linear,
}

impl SpeechConnector {
    pub fn new(input_dim: i32, output_dim: i32, eps: f32) -> Result<Self, Exception> {
        let fc1 = nn::LinearBuilder::new(input_dim, output_dim)
            .bias(true)
            .build()?;
        let norm = nn::RmsNormBuilder::new(output_dim).eps(eps).build()?;
        let fc2 = nn::LinearBuilder::new(output_dim, output_dim)
            .bias(true)
            .build()?;

        Ok(Self { fc1, norm, fc2 })
    }
}

impl Module<&Array> for SpeechConnector {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, features: &Array) -> Result<Self::Output, Self::Error> {
        let x = self.fc1.forward(features)?;
        let x = self.norm.forward(&x)?;
        self.fc2.forward(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.fc1.training_mode(mode);
        self.norm.training_mode(mode);
        self.fc2.training_mode(mode);
    }
}
