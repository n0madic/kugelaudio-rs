use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::mpsc;

use base64::Engine as _;
use candle_core::Device;
use clap::Parser;

use kugelaudio_rs::audio::wav;
use kugelaudio_rs::config::{SAMPLE_RATE, SpecialTokens};
use kugelaudio_rs::generation::pipeline::{self, GenerationParams};
use kugelaudio_rs::model::weights::{self, KugelAudioModel};
use kugelaudio_rs::server::http::spawn_http_listener;
use kugelaudio_rs::server::protocol::{GenerateRequest, GenerateResponse, WorkItem};
use kugelaudio_rs::server::server_loop::{ServerDefaults, run_server_loop};
use kugelaudio_rs::server::unix::spawn_unix_listener;

#[derive(Parser, Debug)]
#[command(name = "kugelaudio-rs")]
#[command(about = "KugelAudio TTS inference via candle (Metal, CUDA, CPU)")]
#[command(long_about = "Three modes of operation:\n\
      \n\
      CLI mode    — provide --model-path and --text, generates one or more WAV files.\n\
      Server mode — provide --model-path and --serve, loads the model once and listens\n\
                    for requests over a Unix socket and/or HTTP.\n\
      Client mode — omit --model-path, connects to a running server via --socket-path.")]
struct Args {
    /// Path to the model directory.
    /// Required for CLI and server mode; omit to run as a client.
    #[arg(long)]
    model_path: Option<String>,

    /// Text to synthesise (required in CLI and client modes)
    #[arg(long)]
    text: Option<String>,

    /// Output WAV file path
    #[arg(long)]
    output: Option<String>,

    /// Classifier-free guidance scale (1.0 = no guidance, 3.0 = default)
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f32,

    /// Maximum number of autoregressive tokens to generate
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,

    /// Number of DPM-Solver++ diffusion steps per speech token (fewer = faster)
    #[arg(long, default_value_t = 10)]
    diffusion_steps: u32,

    /// Compute device: "metal", "cuda", or "cpu" (default: auto-detect; CLI/server mode only)
    #[arg(long)]
    device: Option<String>,

    /// Number of samples to generate (CLI mode only)
    #[arg(long, default_value_t = 1)]
    count: u32,

    /// Run as a persistent server (requires --model-path)
    #[arg(long)]
    serve: bool,

    /// Unix domain socket path — server listens here; client connects here
    #[arg(long, default_value = "/tmp/kugelaudio.sock")]
    socket_path: String,

    /// HTTP bind address, e.g. 127.0.0.1:8080 or :8000 (server mode only; absent = disabled)
    #[arg(long)]
    http_bind: Option<String>,
}

/// Try GPU backends in order of preference, fall back to CPU.
fn select_device(requested: Option<&str>) -> anyhow::Result<Device> {
    match requested {
        Some("metal") => {
            Device::new_metal(0).map_err(|e| anyhow::anyhow!("Failed to open Metal device: {e}"))
        }
        Some("cuda") => {
            Device::new_cuda(0).map_err(|e| anyhow::anyhow!("Failed to open CUDA device: {e}"))
        }
        Some("cpu") => Ok(Device::Cpu),
        Some(other) => anyhow::bail!("Unknown device '{other}'. Use metal, cuda, or cpu."),
        None => {
            if let Ok(dev) = Device::new_cuda(0) {
                eprintln!("Using CUDA device.");
                return Ok(dev);
            }
            if let Ok(dev) = Device::new_metal(0) {
                eprintln!("Using Metal device.");
                return Ok(dev);
            }
            eprintln!("Using CPU device.");
            Ok(Device::Cpu)
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    anyhow::ensure!(
        args.cfg_scale >= 0.0,
        "--cfg-scale must be non-negative (got {})",
        args.cfg_scale
    );

    match (args.serve, args.model_path.is_some()) {
        // Server mode
        (true, true) => {
            eprintln!("KugelAudio-RS v{}", env!("CARGO_PKG_VERSION"));
            let (model, tokenizer) = load_model_and_tokenizer(&args)?;
            run_serve_mode(&args, &model, &tokenizer)
        }
        // CLI mode
        (false, true) => {
            eprintln!("KugelAudio-RS v{}", env!("CARGO_PKG_VERSION"));
            let (model, tokenizer) = load_model_and_tokenizer(&args)?;
            run_cli_mode(&args, &model, &tokenizer)
        }
        // Client mode
        (false, false) => run_client_mode(&args),
        // Error
        (true, false) => {
            anyhow::bail!("--model-path is required when using --serve")
        }
    }
}

fn load_model_and_tokenizer(
    args: &Args,
) -> anyhow::Result<(KugelAudioModel, tokenizers::Tokenizer)> {
    let model_path = args.model_path.as_deref().unwrap();
    let device = select_device(args.device.as_deref())?;

    eprintln!("Loading model from {model_path}...");
    let model_dir = Path::new(model_path);
    let model = weights::load_model(model_dir, &device)
        .map_err(|e| anyhow::anyhow!("Model loading failed: {e}"))?;
    eprintln!("Model loaded.");

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer loading failed: {e}"))?;

    Ok((model, tokenizer))
}

// ---------------------------------------------------------------------------
// CLI mode
// ---------------------------------------------------------------------------

fn run_cli_mode(
    args: &Args,
    model: &KugelAudioModel,
    tokenizer: &tokenizers::Tokenizer,
) -> anyhow::Result<()> {
    let text = args
        .text
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--text is required in CLI mode"))?;

    let output = args.output.as_deref().unwrap_or("output.wav");

    anyhow::ensure!(args.count >= 1, "--count must be at least 1");

    let params = GenerationParams {
        cfg_scale: args.cfg_scale,
        max_new_tokens: args.max_tokens,
        diffusion_steps: args.diffusion_steps,
    };

    let output_path = Path::new(output);
    let stem = output_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let ext = output_path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let parent = output_path.parent().unwrap_or(Path::new(""));

    for i in 1..=args.count {
        let out_file = if args.count > 1 {
            parent
                .join(format!("{stem}_{i}{ext}"))
                .to_string_lossy()
                .into_owned()
        } else {
            output.to_string()
        };

        eprintln!("Generating sample {}/{}: \"{}\"", i, args.count, text);
        let result = pipeline::generate(model, tokenizer, text, &params)
            .map_err(|e| anyhow::anyhow!("Generation failed (sample {i}): {e}"))?;

        if let Some(audio) = &result.audio {
            wav::write_wav(&out_file, audio)
                .map_err(|e| anyhow::anyhow!("WAV write failed (sample {i}): {e}"))?;

            let tokens = SpecialTokens::default();
            let n_speech = result
                .sequences
                .iter()
                .filter(|&&t| t == tokens.speech_diffusion_id)
                .count();
            let duration_s = n_speech as f32 * 3200.0 / SAMPLE_RATE as f32;
            eprintln!("Saved {out_file} ({duration_s:.1}s, {n_speech} speech tokens)");
        } else {
            eprintln!("No audio generated for sample {i} (no speech_diffusion tokens produced).");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Server mode
// ---------------------------------------------------------------------------

fn run_serve_mode(
    args: &Args,
    model: &KugelAudioModel,
    tokenizer: &tokenizers::Tokenizer,
) -> anyhow::Result<()> {
    let (work_tx, work_rx) = mpsc::channel::<WorkItem>();

    let _unix = spawn_unix_listener(args.socket_path.clone(), work_tx.clone());
    let _http = spawn_http_listener(args.http_bind.clone(), work_tx);

    let defaults = ServerDefaults {
        cfg_scale: args.cfg_scale,
        max_tokens: args.max_tokens,
        diffusion_steps: args.diffusion_steps,
    };

    run_server_loop(model, tokenizer, &defaults, work_rx);

    Ok(())
}

// ---------------------------------------------------------------------------
// Client mode
// ---------------------------------------------------------------------------

fn run_client_mode(args: &Args) -> anyhow::Result<()> {
    let text = args
        .text
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--text is required in client mode"))?;

    let output = args.output.as_deref().unwrap_or("output.wav");

    let req = GenerateRequest {
        text: text.to_string(),
        cfg_scale: Some(args.cfg_scale),
        diffusion_steps: Some(args.diffusion_steps),
        max_tokens: Some(args.max_tokens),
    };

    let json_req = serde_json::to_string(&req)?;

    let stream = UnixStream::connect(&args.socket_path)
        .map_err(|e| anyhow::anyhow!("Cannot connect to {}: {e}", args.socket_path))?;
    let mut write_half = stream.try_clone()?;

    write_half.write_all(json_req.as_bytes())?;
    write_half.write_all(b"\n")?;

    let mut response_line = String::new();
    BufReader::new(stream).read_line(&mut response_line)?;

    let resp: GenerateResponse = serde_json::from_str(response_line.trim())
        .map_err(|e| anyhow::anyhow!("Invalid response from server: {e}"))?;

    match resp {
        GenerateResponse::Error { message } => {
            anyhow::bail!("Server error: {message}");
        }
        GenerateResponse::Ok {
            duration_s,
            speech_tokens,
            audio_b64,
        } => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&audio_b64)
                .map_err(|e| anyhow::anyhow!("base64 decode failed: {e}"))?;
            std::fs::write(output, &bytes)
                .map_err(|e| anyhow::anyhow!("Failed to write {output}: {e}"))?;
            eprintln!("Saved {output} ({duration_s:.1}s, {speech_tokens} speech tokens)");
        }
    }

    Ok(())
}
