use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{indexing::IndexOp, pad},
    Array,
};

/// Causal Conv1d wrapper.
///
/// MLX Conv1d uses NLC format (batch, length, channels).
/// Causal padding: pad left by (kernel_size - 1) * dilation, no right padding.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SConv1d {
    #[param]
    pub conv: nn::Conv1d,
    pub causal: bool,
    pub kernel_size: i32,
    pub dilation: i32,
    pub stride: i32,
}

impl SConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        dilation: i32,
        groups: i32,
        bias: bool,
        causal: bool,
    ) -> Result<Self, Exception> {
        // MLX Conv1d with no padding (we handle it manually for causal)
        let conv = nn::Conv1dBuilder::new(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(0)
            .dilation(dilation)
            .groups(groups)
            .bias(bias)
            .build()?;

        Ok(Self {
            conv,
            causal,
            kernel_size,
            dilation,
            stride,
        })
    }

    /// Calculate the left padding needed for causal convolution.
    fn causal_padding(&self) -> i32 {
        (self.kernel_size - 1) * self.dilation
    }
}

impl Module<&Array> for SConv1d {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // x shape: [batch, length, channels] (NLC in MLX)
        if self.causal {
            let pad_left = self.causal_padding();
            // Pad on the time dimension (axis 1) with zeros on the left
            let widths: &[(i32, i32)] = &[(0, 0), (pad_left, 0), (0, 0)];
            let x_padded = pad(x, widths, None, None)?;
            self.conv.forward(&x_padded)
        } else {
            // Non-causal: symmetric padding
            let total_padding = (self.kernel_size - 1) * self.dilation;
            let pad_left = total_padding / 2;
            let pad_right = total_padding - pad_left;
            let widths: &[(i32, i32)] = &[(0, 0), (pad_left, pad_right), (0, 0)];
            let x_padded = pad(x, widths, None, None)?;
            self.conv.forward(&x_padded)
        }
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv.training_mode(mode);
    }
}

/// Causal ConvTranspose1d wrapper.
///
/// Removes right-side padding to maintain causal property after transposed convolution.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SConvTranspose1d {
    #[param]
    pub conv_tr: nn::ConvTranspose1d,
    pub causal: bool,
    pub kernel_size: i32,
    pub stride: i32,
}

impl SConvTranspose1d {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        bias: bool,
        causal: bool,
    ) -> Result<Self, Exception> {
        let conv_tr = nn::ConvTranspose1dBuilder::new(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(0)
            .bias(bias)
            .build()?;

        Ok(Self {
            conv_tr,
            causal,
            kernel_size,
            stride,
        })
    }
}

impl Module<&Array> for SConvTranspose1d {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // x shape: [batch, length, channels] (NLC)
        let out = self.conv_tr.forward(x)?;

        if self.causal {
            // Remove right padding: trim to expected output length
            // TransposeConv output length = (L - 1) * stride + kernel_size
            // We want: L * stride (the causal upsampled length)
            let trim = self.kernel_size - self.stride;
            if trim > 0 {
                let out_len = out.dim(1);
                let target_len = out_len - trim;
                // Slice: out[:, :target_len, :]
                Ok(out.index((.., ..target_len, ..)))
            } else {
                Ok(out)
            }
        } else {
            Ok(out)
        }
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv_tr.training_mode(mode);
    }
}
