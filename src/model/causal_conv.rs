use candle_core::{Result, Tensor};
use candle_nn::{conv1d, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

/// Causal (or non-causal) Conv1d wrapper.
///
/// Tensor format is NCL `[batch, channels, length]` throughout.
///
/// For causal mode the left padding applied before the convolution is
/// `(kernel_size - 1) * dilation`, ensuring each output position depends
/// only on current and past inputs.  The inner `Conv1d` is always created
/// with `padding = 0`; all padding is handled manually here.
///
/// For non-causal mode symmetric padding is applied:
/// `pad_left = total / 2`, `pad_right = total - pad_left`.
///
/// The inner conv weights are loaded from `vb.pp("conv")` which must
/// expose the keys `weight` and, when `bias` is true, `bias`.
///
/// # Example
///
/// ```rust,ignore
/// let conv = SConv1d::new(64, 128, 4, 1, 1, 1, true, true, vb.pp("encoder.conv"))?;
/// let out = conv.forward(&x)?; // x: [B, 64, L]
/// ```
pub struct SConv1d {
    conv: Conv1d,
    causal: bool,
    kernel_size: usize,
    dilation: usize,
    _stride: usize,
}

impl SConv1d {
    /// Construct a `SConv1d` layer.
    ///
    /// `vb` must be scoped to the parent of the inner conv so that
    /// `vb.pp("conv")` resolves to the correct weight keys.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let vb_conv = vb.pp("conv");
        let conv = if bias {
            conv1d(in_channels, out_channels, kernel_size, cfg, vb_conv)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, vb_conv)?
        };
        Ok(Self {
            conv,
            causal,
            kernel_size,
            dilation,
            _stride: stride,
        })
    }

    /// Left padding required to make the convolution causal.
    fn causal_padding(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    /// Apply the (optionally causal) convolution.
    ///
    /// Input shape: `[batch, channels, length]` (NCL)
    /// Output shape: `[batch, out_channels, out_length]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.causal {
            // Pad left on the length dimension (dim 2 in NCL).
            let pad_left = self.causal_padding();
            let x_padded = x.pad_with_zeros(2, pad_left, 0)?;
            self.conv.forward(&x_padded)
        } else {
            // Symmetric padding: distribute evenly, extra sample goes right.
            let total = (self.kernel_size - 1) * self.dilation;
            let pad_left = total / 2;
            let pad_right = total - pad_left;
            let x_padded = x.pad_with_zeros(2, pad_left, pad_right)?;
            self.conv.forward(&x_padded)
        }
    }
}

/// Causal (or non-causal) ConvTranspose1d wrapper.
///
/// Tensor format is NCL `[batch, channels, length]` throughout.
///
/// For causal mode the right-side artifact produced by the transposed
/// convolution is trimmed: `trim = kernel_size - stride` samples are
/// removed from the end of the output, yielding a causally upsampled
/// sequence of length `input_length * stride`.
///
/// The inner convtr weights are loaded from `vb.pp("convtr")` which must
/// expose the keys `weight` and, when `bias` is true, `bias`.
///
/// # Example
///
/// ```rust,ignore
/// let convtr = SConvTranspose1d::new(128, 64, 8, 4, true, true, vb.pp("decoder.convtr"))?;
/// let out = convtr.forward(&x)?; // x: [B, 128, L] -> [B, 64, L*4]
/// ```
pub struct SConvTranspose1d {
    convtr: ConvTranspose1d,
    causal: bool,
    kernel_size: usize,
    stride: usize,
}

impl SConvTranspose1d {
    /// Construct a `SConvTranspose1d` layer.
    ///
    /// `vb` must be scoped to the parent of the inner convtr so that
    /// `vb.pp("convtr")` resolves to the correct weight keys.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            stride,
            dilation: 1,
            output_padding: 0,
            groups: 1,
        };
        let convtr = if bias {
            conv_transpose1d(in_channels, out_channels, kernel_size, cfg, vb.pp("convtr"))?
        } else {
            candle_nn::conv_transpose1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                cfg,
                vb.pp("convtr"),
            )?
        };
        Ok(Self {
            convtr,
            causal,
            kernel_size,
            stride,
        })
    }

    /// Apply the (optionally causal) transposed convolution.
    ///
    /// Input shape:  `[batch, in_channels, length]` (NCL)
    /// Output shape: `[batch, out_channels, length * stride]` (causal mode)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.convtr.forward(x)?;

        if self.causal {
            // The transposed conv produces (L - 1) * stride + kernel_size samples.
            // Trim the right side to recover the causal length L * stride.
            let trim = self.kernel_size.saturating_sub(self.stride);
            if trim > 0 {
                let out_len = out.dim(2)?;
                out.narrow(2, 0, out_len - trim)
            } else {
                Ok(out)
            }
        } else {
            Ok(out)
        }
    }
}
