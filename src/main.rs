use clap::Parser;

use kugelaudio_rs::audio::wav;
use kugelaudio_rs::config::{SpecialTokens, SAMPLE_RATE};
use kugelaudio_rs::generation::pipeline::{self, GenerationParams};
use kugelaudio_rs::model::weights;

#[derive(Parser, Debug)]
#[command(name = "kugelaudio-rs")]
#[command(about = "KugelAudio TTS inference on Apple Silicon with MLX")]
struct Args {
    /// Path to the model directory (containing safetensors, config.json, tokenizer.json)
    #[arg(long)]
    model_path: String,

    /// Text to synthesize
    #[arg(long)]
    text: String,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: String,

    /// Classifier-free guidance scale (1.0 = no guidance, 3.0 = default)
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f32,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,

    /// Number of diffusion inference steps (fewer = faster, default 10)
    #[arg(long, default_value_t = 10)]
    diffusion_steps: u32,

    /// Random seed for reproducibility (omit for non-deterministic)
    #[arg(long)]
    seed: Option<u64>,

    /// Quantize LM at runtime to N bits (4 or 8). Use convert-quantized for pre-quantized weights.
    #[arg(long)]
    quantize: Option<i32>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("KugelAudio-RS v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Loading model from {}...", args.model_path);

    let mut model = weights::load_model(&args.model_path)?;
    if let Some(bits) = args.quantize {
        model.quantize_lm(64, bits)?;
    }
    eprintln!("Model loaded.");

    let tokenizer = qwen3_mlx::qwen2::load_qwen2_tokenizer(&args.model_path)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    eprintln!("Generating: \"{}\"", args.text);
    let params = GenerationParams {
        cfg_scale: args.cfg_scale,
        max_new_tokens: args.max_tokens,
        diffusion_steps: args.diffusion_steps,
        seed: args.seed,
    };

    let output = pipeline::generate(&mut model, &tokenizer, &args.text, &params)?;

    if let Some(audio) = &output.audio {
        wav::write_wav(&args.output, audio)?;
        let tokens = SpecialTokens::default();
        let n_speech = output
            .sequences
            .iter()
            .filter(|&&t| t == tokens.speech_diffusion_id)
            .count();
        let duration = n_speech as f32 * 3200.0 / SAMPLE_RATE as f32;
        eprintln!(
            "Saved {} ({duration:.1}s, {n_speech} speech tokens)",
            args.output
        );
    } else {
        eprintln!("No audio generated.");
    }

    Ok(())
}
