use std::f64::consts::PI;

use mlx_rs::{array, error::Exception, random, Array};

/// Output from a scheduler step.
pub struct SchedulerOutput {
    pub prev_sample: Array,
    pub x0_pred: Array,
}

/// SDE-DPM-Solver++ multistep scheduler for diffusion inference.
///
/// Supports both SDE (stochastic, with noise injection) and ODE (deterministic) variants.
/// Uses cosine beta schedule, v_prediction, and 2nd-order multistep updates.
pub struct DpmSolverScheduler {
    alpha_t_all: Vec<f64>,

    cached_alpha_t: Vec<f64>,
    cached_sigma_t: Vec<f64>,
    cached_lambda: Vec<f64>,

    /// Timesteps selected for inference (descending).
    pub timesteps: Vec<i32>,

    model_outputs: Vec<Option<Array>>,
    lower_order_nums: i32,
    step_index: Option<usize>,
    num_inference_steps: i32,

    solver_order: i32,
    prediction_type: String,
    is_sde: bool,
}

/// Threshold below which the last step uses first-order to stabilize sampling.
const LOWER_ORDER_FINAL_THRESHOLD: i32 = 15;

impl DpmSolverScheduler {
    /// Create a new scheduler.
    ///
    /// `beta_schedule`: one of `"cosine"`, `"squaredcos_cap_v2"`, `"scaled_linear"`.
    /// `prediction_type`: `"v_prediction"` or `"epsilon"`.
    /// `algorithm_type`: contains `"sde"` for stochastic variant.
    pub fn new(
        num_train_timesteps: i32,
        beta_schedule: &str,
        prediction_type: &str,
        algorithm_type: &str,
        solver_order: i32,
    ) -> Result<Self, Exception> {
        let betas = match beta_schedule {
            "cosine" | "squaredcos_cap_v2" | "scaled_linear" => {
                betas_for_alpha_bar_cosine(num_train_timesteps as usize)
            }
            other => {
                return Err(Exception::custom(format!(
                    "Unsupported beta_schedule: {other}"
                )))
            }
        };

        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(alphas.len());
        let mut prod = 1.0;
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
            step_index: None,
            num_inference_steps: 0,
            solver_order,
            prediction_type: prediction_type.to_string(),
            is_sde: algorithm_type.contains("sde"),
        })
    }

    /// Reset state for a new diffusion run.
    pub fn reset(&mut self) {
        self.model_outputs = vec![None; self.solver_order as usize];
        self.lower_order_nums = 0;
        self.step_index = None;
    }

    /// Set inference timesteps (linspace from max to 0).
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
            let alpha = self.alpha_t_all[t as usize];
            let sigma_from_alpha = ((1.0 - alpha * alpha) / (alpha * alpha)).sqrt();
            let alpha_t_val = 1.0 / (sigma_from_alpha * sigma_from_alpha + 1.0).sqrt();
            let sigma_t_val = sigma_from_alpha * alpha_t_val;
            let lambda_val = alpha_t_val.ln() - sigma_t_val.ln();

            self.cached_alpha_t.push(alpha_t_val);
            self.cached_sigma_t.push(sigma_t_val);
            self.cached_lambda.push(lambda_val);
        }

        // Final step: alpha=1, sigma=0, lambda=inf
        self.cached_alpha_t.push(1.0);
        self.cached_sigma_t.push(0.0);
        self.cached_lambda.push(f64::INFINITY);

        self.timesteps = timestep_values;
        self.reset();
    }

    fn convert_model_output(
        &self,
        model_output: &Array,
        sample: &Array,
        step_idx: usize,
    ) -> Result<Array, Exception> {
        let alpha_t = self.cached_alpha_t[step_idx] as f32;
        let sigma_t = self.cached_sigma_t[step_idx] as f32;

        match self.prediction_type.as_str() {
            "v_prediction" => sample
                .multiply(&array!(alpha_t))?
                .subtract(&model_output.multiply(&array!(sigma_t))?),
            "epsilon" => sample
                .subtract(&model_output.multiply(&array!(sigma_t))?)?
                .divide(&array!(alpha_t)),
            other => Err(Exception::custom(format!(
                "Unknown prediction_type: {other}"
            ))),
        }
    }

    fn sde_first_order(
        &self,
        x0_pred: &Array,
        sample: &Array,
        step_idx: usize,
    ) -> Result<Array, Exception> {
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
        let one_minus_exp_neg_2h = 1.0 - (-2.0 * h).exp() as f32;

        let noise = random::normal::<f32>(sample.shape(), None, None, None)?;

        sample
            .multiply(&array!(sigma_ratio * exp_neg_h))?
            .add(x0_pred.multiply(&array!(alpha_s * one_minus_exp_neg_2h))?)?
            .add(noise.multiply(&array!(sigma_s * one_minus_exp_neg_2h.sqrt()))?)
    }

    fn sde_second_order(
        &self,
        x0_pred: &Array,
        prev_x0: &Array,
        sample: &Array,
        step_idx: usize,
    ) -> Result<Array, Exception> {
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
            x0_pred
                .subtract(prev_x0)?
                .multiply(&array!((1.0 / r0) as f32))?
        } else {
            Array::zeros::<f32>(x0_pred.shape())?
        };

        let sigma_ratio = if sigma_t > 0.0 {
            sigma_s / sigma_t
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp() as f32;
        let one_minus_exp_neg_2h = 1.0 - (-2.0 * h).exp() as f32;
        let noise = random::normal::<f32>(sample.shape(), None, None, None)?;

        sample
            .multiply(&array!(sigma_ratio * exp_neg_h))?
            .add(x0_pred.multiply(&array!(alpha_s * one_minus_exp_neg_2h))?)?
            .add(d1.multiply(&array!(0.5 * alpha_s * one_minus_exp_neg_2h))?)?
            .add(noise.multiply(&array!(sigma_s * one_minus_exp_neg_2h.sqrt()))?)
    }

    fn ode_first_order(
        &self,
        x0_pred: &Array,
        sample: &Array,
        step_idx: usize,
    ) -> Result<Array, Exception> {
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

        sample
            .multiply(&array!(sigma_ratio))?
            .subtract(&x0_pred.multiply(&array!(alpha_t * (exp_neg_h - 1.0)))?)
    }

    fn ode_second_order(
        &self,
        x0_pred: &Array,
        prev_x0: &Array,
        sample: &Array,
        step_idx: usize,
    ) -> Result<Array, Exception> {
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
            x0_pred
                .subtract(prev_x0)?
                .multiply(&array!((1.0 / r0) as f32))?
        } else {
            Array::zeros::<f32>(x0_pred.shape())?
        };

        let sigma_ratio = if sigma_curr > 0.0 {
            sigma_next / sigma_curr
        } else {
            0.0
        };
        let exp_neg_h = (-h).exp() as f32;

        sample
            .multiply(&array!(sigma_ratio))?
            .subtract(&x0_pred.multiply(&array!(alpha_t * (exp_neg_h - 1.0)))?)?
            .subtract(&d1.multiply(&array!(0.5 * alpha_t * (exp_neg_h - 1.0)))?)
    }

    /// Perform one denoising step.
    pub fn step(
        &mut self,
        model_output: &Array,
        _timestep: i32,
        sample: &Array,
    ) -> Result<SchedulerOutput, Exception> {
        let step_idx = self.step_index.unwrap_or(0);

        let x0_pred = self.convert_model_output(model_output, sample, step_idx)?;

        // Shift model_outputs buffer
        for i in (1..self.solver_order as usize).rev() {
            self.model_outputs[i] = self.model_outputs[i - 1].take();
        }
        self.model_outputs[0] = Some(x0_pred.clone());

        let is_final_step = step_idx == self.num_inference_steps as usize - 1;
        let lower_order_final_flag =
            is_final_step && self.num_inference_steps < LOWER_ORDER_FINAL_THRESHOLD;

        let use_first_order = self.lower_order_nums < 1 || lower_order_final_flag;

        let prev_sample = if self.is_sde {
            if use_first_order {
                self.sde_first_order(&x0_pred, sample, step_idx)?
            } else if let Some(prev_x0) = &self.model_outputs[1] {
                self.sde_second_order(&x0_pred, prev_x0, sample, step_idx)?
            } else {
                self.sde_first_order(&x0_pred, sample, step_idx)?
            }
        } else if use_first_order {
            self.ode_first_order(&x0_pred, sample, step_idx)?
        } else if let Some(prev_x0) = &self.model_outputs[1] {
            self.ode_second_order(&x0_pred, prev_x0, sample, step_idx)?
        } else {
            self.ode_first_order(&x0_pred, sample, step_idx)?
        };

        if self.lower_order_nums < self.solver_order - 1 {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(step_idx + 1);

        Ok(SchedulerOutput {
            prev_sample,
            x0_pred,
        })
    }
}

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
