use std::path::Path;

use clap::Parser;
use candle_core::Device;

use kugelaudio_rs::audio::wav;
use kugelaudio_rs::config::{SpecialTokens, SAMPLE_RATE};
use kugelaudio_rs::generation::pipeline::{self, GenerationParams};
use kugelaudio_rs::model::weights;

#[derive(Parser, Debug)]
#[command(name = "kugelaudio-rs")]
#[command(about = "KugelAudio TTS inference via candle (Metal, CUDA, CPU)")]
struct Args {
    /// Path to the model directory (containing config.json, model.safetensors.index.json,
    /// shard files, and tokenizer.json)
    #[arg(long)]
    model_path: String,

    /// Text to synthesise
    #[arg(long)]
    text: String,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: String,

    /// Classifier-free guidance scale (1.0 = no guidance, 3.0 = default)
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f32,

    /// Maximum number of autoregressive tokens to generate
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,

    /// Number of DPM-Solver++ diffusion steps per speech token (fewer = faster)
    #[arg(long, default_value_t = 10)]
    diffusion_steps: u32,

    /// Compute device: "cpu", "metal", or "cuda"
    #[arg(long, default_value = "metal")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("KugelAudio-RS v{}", env!("CARGO_PKG_VERSION"));

    // Select compute device.
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)
            .map_err(|e| anyhow::anyhow!("Failed to open Metal device: {e}"))?,
        "cuda" => Device::new_cuda(0)
            .map_err(|e| anyhow::anyhow!("Failed to open CUDA device: {e}"))?,
        other => anyhow::bail!("Unknown device '{}'. Use cpu, metal, or cuda.", other),
    };

    // Load model weights.
    eprintln!("Loading model from {}...", args.model_path);
    let model_dir = Path::new(&args.model_path);
    let model = weights::load_model(model_dir, &device)
        .map_err(|e| anyhow::anyhow!("Model loading failed: {e}"))?;
    eprintln!("Model loaded.");

    // Load tokenizer.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer loading failed: {e}"))?;

    // Build generation parameters.
    let params = GenerationParams {
        cfg_scale: args.cfg_scale,
        max_new_tokens: args.max_tokens,
        diffusion_steps: args.diffusion_steps,
    };

    // Run the pipeline.
    eprintln!("Generating: \"{}\"", args.text);
    let output = pipeline::generate(&model, &tokenizer, &args.text, &params)
        .map_err(|e| anyhow::anyhow!("Generation failed: {e}"))?;

    // Write output WAV.
    if let Some(audio) = &output.audio {
        wav::write_wav(&args.output, audio)
            .map_err(|e| anyhow::anyhow!("WAV write failed: {e}"))?;

        let tokens = SpecialTokens::default();
        let n_speech = output
            .sequences
            .iter()
            .filter(|&&t| t == tokens.speech_diffusion_id)
            .count();
        // Each speech token covers ~3200 samples at 24 kHz (≈ 133 ms).
        let duration_s = n_speech as f32 * 3200.0 / SAMPLE_RATE as f32;
        eprintln!(
            "Saved {} ({duration_s:.1}s, {n_speech} speech tokens)",
            args.output
        );
    } else {
        eprintln!("No audio generated (no speech_diffusion tokens produced).");
    }

    Ok(())
}
