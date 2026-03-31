//! SDE-DPM-Solver++ multistep scheduler for diffusion inference.
//!
//! Supports both SDE (stochastic, with noise injection) and ODE (deterministic)
//! variants. Uses a cosine beta schedule, v-prediction, and 2nd-order multistep
//! updates.

use std::f64::consts::PI;

use candle_core::{Device, Result, Tensor};

/// Output from a single scheduler step.
pub struct SchedulerOutput {
    pub prev_sample: Tensor,
    pub x0_pred: Tensor,
}

/// SDE-DPM-Solver++ multistep scheduler.
///
/// The scheduler maintains per-step cached noise-schedule values
/// (`alpha_t`, `sigma_t`, `lambda`) so that the per-step arithmetic is
/// performed in pure Rust (f64), while only the actual sample updates use
/// candle `Tensor` ops.
pub struct DpmSolverScheduler {
    alpha_t_all: Vec<f64>,

    cached_alpha_t: Vec<f64>,
    cached_sigma_t: Vec<f64>,
    cached_lambda: Vec<f64>,

    /// Inference timesteps (descending order).
    pub timesteps: Vec<i32>,

    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: i32,

    num_inference_steps: i32,
    solver_order: i32,
    prediction_type: String,
    is_sde: bool,

    device: Device,
}

/// Threshold below which the last step falls back to first-order to stabilise.
const LOWER_ORDER_FINAL_THRESHOLD: i32 = 15;

impl DpmSolverScheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    ///
    /// * `num_train_timesteps` – number of training diffusion steps (e.g. 1000)
    /// * `beta_schedule`       – `"cosine"`, `"squaredcos_cap_v2"`, or `"scaled_linear"`
    /// * `prediction_type`     – `"v_prediction"` or `"epsilon"`
    /// * `algorithm_type`      – string containing `"sde"` enables stochastic variant
    /// * `solver_order`        – multistep order (1 or 2)
    /// * `device`              – candle device for noise tensors
    pub fn new(
        num_train_timesteps: i32,
        beta_schedule: &str,
        prediction_type: &str,
        algorithm_type: &str,
        solver_order: i32,
        device: Device,
    ) -> Result<Self> {
        let betas = match beta_schedule {
            "cosine" | "squaredcos_cap_v2" | "scaled_linear" => {
                betas_for_alpha_bar_cosine(num_train_timesteps as usize)
            }
            other => {
                candle_core::bail!("Unsupported beta_schedule: {other}")
            }
        };

        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(alphas.len());
        let mut prod = 1.0_f64;
        for &a in &alphas {
            prod *= a;
            alphas_cumprod.push(prod);
        }

        let alpha_t_all: Vec<f64> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();

        Ok(Self {
            alpha_t_all,
            cached_alpha_t: Vec::new(),
            cached_sigma_t: Vec::new(),
            cached_lambda: Vec::new(),
            timesteps: Vec::new(),
            model_outputs: vec![None; solver_order as usize],
            lower_order_nums: 0,
            num_inference_steps: 0,
            solver_order,
            prediction_type: prediction_type.to_string(),
            is_sde: algorithm_type.contains("sde"),
            device,
        })
    }

    /// Reset the per-run state (model output buffer and order counter).
    ///
    /// Called automatically by [`set_timesteps`].
    pub fn reset(&mut self) {
        self.model_outputs = vec![None; self.solver_order as usize];
        self.lower_order_nums = 0;
    }

    /// Compute and cache inference timesteps.
    ///
    /// Must be called once before the denoising loop. Resets internal state.
    pub fn set_timesteps(&mut self, num_inference_steps: i32) {
        self.num_inference_steps = num_inference_steps;
        let n = num_inference_steps;
        let train = self.alpha_t_all.len() as f64;

        let timestep_values: Vec<i32> = (0..n)
            .map(|i| ((train - 1.0) * (1.0 - i as f64 / n as f64)).round() as i32)
            .collect();

        self.cached_alpha_t.clear();
        self.cached_sigma_t.clear();
        self.cached_lambda.clear();

        for &t in &timestep_values {
            // alpha_t_all[t] = sqrt(alphas_cumprod[t])
            let alpha_t_val = self.alpha_t_all[t as usize];
            let sigma_t_val = (1.0 - alpha_t_val * alpha_t_val).sqrt();
            let lambda_val = alpha_t_val.ln() - sigma_t_val.ln();

            self.cached_alpha_t.push(alpha_t_val);
            self.cached_sigma_t.push(sigma_t_val);
            self.cached_lambda.push(lambda_val);
        }

        // Sentinel final entry: alpha=1, sigma=0, lambda=+inf (clean sample target).
        self.cached_alpha_t.push(1.0);
        self.cached_sigma_t.push(0.0);
        self.cached_lambda.push(f64::INFINITY);

        self.timesteps = timestep_values;
        self.reset();
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Convert raw model output to a predicted x0 estimate.
    fn convert_model_output(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        let alpha_t = self.cached_alpha_t[step_idx] as f32;
        let sigma_t = self.cached_sigma_t[step_idx] as f32;

        match self.prediction_type.as_str() {
            "v_prediction" => {
                // x0 = alpha_t * x - sigma_t * v
                (sample * alpha_t as f64)?.broadcast_sub(&(model_output * sigma_t as f64)?)
            }
            "epsilon" => {
                // x0 = (x - sigma_t * eps) / alpha_t
                (sample - (model_output * sigma_t as f64)?)? / alpha_t as f64
            }
            other => candle_core::bail!("Unknown prediction_type: {other}"),
        }
    }

    fn sde_first_order(
        &self,
        x0_pred: &Tensor,
        sample: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        let alpha_s = self.cached_alpha_t[step_idx + 1] as f32;
        let sigma_s = self.cached_sigma_t[step_idx + 1] as f32;
        let sigma_t = self.cached_sigma_t[step_idx] as f32;
        let h = self.cached_lambda[step_idx + 1] - self.cached_lambda[step_idx];

        let sigma_ratio = if sigma_t > 0.0 {
            sigma_s / sigma_t
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp() as f32;
        let one_minus_exp_neg_2h = 1.0_f32 - (-2.0 * h).exp() as f32;

        let noise = Tensor::randn(0f32, 1f32, sample.shape(), &candle_core::Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(sample.dtype())?;

        let term1 = (sample * (sigma_ratio * exp_neg_h) as f64)?;
        let term2 = (x0_pred * (alpha_s * one_minus_exp_neg_2h) as f64)?;
        let term3 = (noise * (sigma_s * one_minus_exp_neg_2h.sqrt()) as f64)?;
        (term1 + term2)? + term3
    }

    fn sde_second_order(
        &self,
        x0_pred: &Tensor,
        prev_x0: &Tensor,
        sample: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        let alpha_s = self.cached_alpha_t[step_idx + 1] as f32;
        let sigma_s = self.cached_sigma_t[step_idx + 1] as f32;
        let sigma_t = self.cached_sigma_t[step_idx] as f32;

        let lambda_s = self.cached_lambda[step_idx + 1];
        let lambda_s0 = self.cached_lambda[step_idx];
        let lambda_s1 = if step_idx > 0 {
            self.cached_lambda[step_idx - 1]
        } else {
            lambda_s0
        };

        let h = lambda_s - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let r0 = if h != 0.0 { h0 / h } else { 1.0 };

        let d1 = if r0 != 0.0 {
            ((x0_pred - prev_x0)? * (1.0 / r0))?
        } else {
            Tensor::zeros_like(x0_pred)?
        };

        let sigma_ratio = if sigma_t > 0.0 {
            sigma_s / sigma_t
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp() as f32;
        let one_minus_exp_neg_2h = 1.0_f32 - (-2.0 * h).exp() as f32;
        let noise = Tensor::randn(0f32, 1f32, sample.shape(), &candle_core::Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(sample.dtype())?;

        let term1 = (sample * (sigma_ratio * exp_neg_h) as f64)?;
        let term2 = (x0_pred * (alpha_s * one_minus_exp_neg_2h) as f64)?;
        let term3 = (d1 * (0.5 * alpha_s * one_minus_exp_neg_2h) as f64)?;
        let term4 = (noise * (sigma_s * one_minus_exp_neg_2h.sqrt()) as f64)?;
        ((term1 + term2)? + term3)? + term4
    }

    fn ode_first_order(
        &self,
        x0_pred: &Tensor,
        sample: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        let alpha_t = self.cached_alpha_t[step_idx + 1] as f32;
        let sigma_next = self.cached_sigma_t[step_idx + 1] as f32;
        let sigma_curr = self.cached_sigma_t[step_idx] as f32;
        let h = self.cached_lambda[step_idx + 1] - self.cached_lambda[step_idx];

        let sigma_ratio = if sigma_curr > 0.0 {
            sigma_next / sigma_curr
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp() as f32;

        let term1 = (sample * sigma_ratio as f64)?;
        let term2 = (x0_pred * (alpha_t * (exp_neg_h - 1.0)) as f64)?;
        term1 - term2
    }

    fn ode_second_order(
        &self,
        x0_pred: &Tensor,
        prev_x0: &Tensor,
        sample: &Tensor,
        step_idx: usize,
    ) -> Result<Tensor> {
        let alpha_t = self.cached_alpha_t[step_idx + 1] as f32;
        let sigma_next = self.cached_sigma_t[step_idx + 1] as f32;
        let sigma_curr = self.cached_sigma_t[step_idx] as f32;

        let lambda_t = self.cached_lambda[step_idx + 1];
        let lambda_s0 = self.cached_lambda[step_idx];
        let lambda_s1 = if step_idx > 0 {
            self.cached_lambda[step_idx - 1]
        } else {
            lambda_s0
        };

        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let r0 = if h != 0.0 { h0 / h } else { 1.0 };

        let d1 = if r0 != 0.0 {
            ((x0_pred - prev_x0)? * (1.0 / r0))?
        } else {
            Tensor::zeros_like(x0_pred)?
        };

        let sigma_ratio = if sigma_curr > 0.0 {
            sigma_next / sigma_curr
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp() as f32;

        let term1 = (sample * sigma_ratio as f64)?;
        let term2 = (x0_pred * (alpha_t * (exp_neg_h - 1.0)) as f64)?;
        let term3 = (d1 * (0.5 * alpha_t * (exp_neg_h - 1.0)) as f64)?;
        (term1 - term2)? - term3
    }

    // -----------------------------------------------------------------------
    // Public step API
    // -----------------------------------------------------------------------

    /// Perform one denoising step.
    ///
    /// # Arguments
    ///
    /// * `model_output` – noise/velocity prediction from the diffusion model
    /// * `_timestep`    – current integer timestep (unused after precomputation)
    /// * `sample`       – current noisy sample tensor
    /// * `step_idx`     – zero-based index into `self.timesteps`
    ///
    /// # Returns
    ///
    /// [`SchedulerOutput`] containing `prev_sample` and `x0_pred`.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        _timestep: i32,
        sample: &Tensor,
        step_idx: usize,
    ) -> Result<SchedulerOutput> {
        let x0_pred = self.convert_model_output(model_output, sample, step_idx)?;

        // Shift the model-output ring buffer.
        for i in (1..self.solver_order as usize).rev() {
            self.model_outputs[i] = self.model_outputs[i - 1].take();
        }
        self.model_outputs[0] = Some(x0_pred.clone());

        let is_final_step = step_idx == self.num_inference_steps as usize - 1;
        let lower_order_final =
            is_final_step && self.num_inference_steps < LOWER_ORDER_FINAL_THRESHOLD;
        let use_first_order = self.lower_order_nums < 1 || lower_order_final;

        let prev_sample = if self.is_sde {
            if use_first_order {
                self.sde_first_order(&x0_pred, sample, step_idx)?
            } else if let Some(prev_x0) = self.model_outputs[1].clone() {
                self.sde_second_order(&x0_pred, &prev_x0, sample, step_idx)?
            } else {
                self.sde_first_order(&x0_pred, sample, step_idx)?
            }
        } else if use_first_order {
            self.ode_first_order(&x0_pred, sample, step_idx)?
        } else if let Some(prev_x0) = self.model_outputs[1].clone() {
            self.ode_second_order(&x0_pred, &prev_x0, sample, step_idx)?
        } else {
            self.ode_first_order(&x0_pred, sample, step_idx)?
        };

        if self.lower_order_nums < self.solver_order - 1 {
            self.lower_order_nums += 1;
        }

        Ok(SchedulerOutput {
            prev_sample,
            x0_pred,
        })
    }
}

// ---------------------------------------------------------------------------
// Beta schedule
// ---------------------------------------------------------------------------

fn betas_for_alpha_bar_cosine(num_timesteps: usize) -> Vec<f64> {
    let alpha_bar = |t: f64| -> f64 { ((t + 0.008) / 1.008 * PI / 2.0).cos().powi(2) };
    (0..num_timesteps)
        .map(|i| {
            let t1 = i as f64 / num_timesteps as f64;
            let t2 = (i + 1) as f64 / num_timesteps as f64;
            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use candle_core::DType;

    use super::*;

    /// Verify that set_timesteps produces the expected number of cached entries.
    #[test]
    fn test_set_timesteps_lengths() {
        let device = Device::Cpu;
        let mut sched =
            DpmSolverScheduler::new(1000, "cosine", "v_prediction", "sde-dpmsolver++", 2, device)
                .unwrap();
        sched.set_timesteps(10);

        assert_eq!(sched.timesteps.len(), 10);
        // cached arrays have n+1 entries (sentinel appended)
        assert_eq!(sched.cached_alpha_t.len(), 11);
        assert_eq!(sched.cached_sigma_t.len(), 11);
        assert_eq!(sched.cached_lambda.len(), 11);
    }

    /// Verify the first timestep is near the maximum training timestep.
    #[test]
    fn test_timesteps_descending() {
        let device = Device::Cpu;
        let mut sched =
            DpmSolverScheduler::new(1000, "cosine", "v_prediction", "sde-dpmsolver++", 2, device)
                .unwrap();
        sched.set_timesteps(10);

        let ts = &sched.timesteps;
        // Should be in descending order.
        for window in ts.windows(2) {
            assert!(window[0] > window[1], "timesteps not descending: {:?}", ts);
        }
        // First value close to 999
        assert!(ts[0] >= 900, "unexpected first timestep: {}", ts[0]);
    }

    /// Smoke-test a full ODE pass with a trivial model (all-zeros prediction).
    #[test]
    fn test_ode_step_smoke() {
        let device = Device::Cpu;
        let mut sched = DpmSolverScheduler::new(
            1000,
            "cosine",
            "v_prediction",
            "ode-dpmsolver++",
            2,
            device.clone(),
        )
        .unwrap();
        sched.set_timesteps(5);

        let mut sample = Tensor::zeros(&[1, 64], DType::F32, &device).unwrap();
        let timesteps = sched.timesteps.clone();
        for (i, &t) in timesteps.iter().enumerate() {
            let pred = Tensor::zeros_like(&sample).unwrap();
            let out = sched.step(&pred, t, &sample, i).unwrap();
            sample = out.prev_sample;
        }
        // Should not panic; just verify shape is preserved.
        assert_eq!(sample.dims(), &[1, 64]);
    }

    /// Smoke-test a full SDE pass with a trivial model (all-zeros prediction).
    #[test]
    fn test_sde_step_smoke() {
        let device = Device::Cpu;
        let mut sched = DpmSolverScheduler::new(
            1000,
            "cosine",
            "v_prediction",
            "sde-dpmsolver++",
            2,
            device.clone(),
        )
        .unwrap();
        sched.set_timesteps(5);

        let mut sample = Tensor::zeros(&[1, 64], DType::F32, &device).unwrap();
        let timesteps = sched.timesteps.clone();
        for (i, &t) in timesteps.iter().enumerate() {
            let pred = Tensor::zeros_like(&sample).unwrap();
            let out = sched.step(&pred, t, &sample, i).unwrap();
            sample = out.prev_sample;
        }
        assert_eq!(sample.dims(), &[1, 64]);
    }

    /// Verify that reset clears model outputs but preserves timesteps.
    #[test]
    fn test_reset_preserves_timesteps() {
        let device = Device::Cpu;
        let mut sched =
            DpmSolverScheduler::new(1000, "cosine", "v_prediction", "sde-dpmsolver++", 2, device)
                .unwrap();
        sched.set_timesteps(10);
        let ts_before = sched.timesteps.clone();

        // Run one step to populate model_outputs.
        let sample = Tensor::zeros(&[1, 64], DType::F32, &sched.device.clone()).unwrap();
        let pred = Tensor::zeros_like(&sample).unwrap();
        let _ = sched.step(&pred, ts_before[0], &sample, 0).unwrap();
        assert!(sched.lower_order_nums > 0);

        // Reset should clear order counter but keep timesteps.
        sched.reset();
        assert_eq!(sched.lower_order_nums, 0);
        assert_eq!(sched.timesteps, ts_before);
    }

    /// Verify epsilon prediction mode doesn't panic.
    #[test]
    fn test_epsilon_prediction() {
        let device = Device::Cpu;
        let mut sched = DpmSolverScheduler::new(
            1000,
            "cosine",
            "epsilon",
            "ode-dpmsolver++",
            2,
            device.clone(),
        )
        .unwrap();
        sched.set_timesteps(5);

        let mut sample = Tensor::ones(&[1, 32], DType::F32, &device).unwrap();
        let timesteps = sched.timesteps.clone();
        for (i, &t) in timesteps.iter().enumerate() {
            let pred = Tensor::zeros_like(&sample).unwrap();
            let out = sched.step(&pred, t, &sample, i).unwrap();
            sample = out.prev_sample;
        }
        assert_eq!(sample.dims(), &[1, 32]);
    }
}
