use std::path::Path;

use clap::Parser;

use kugelaudio_rs::model::weights;

#[derive(Parser, Debug)]
#[command(name = "convert-quantized")]
#[command(about = "Convert KugelAudio model weights to 4-bit quantized format")]
struct Args {
    /// Path to the original model directory (bf16 safetensors + config.json + tokenizer.json)
    #[arg(long)]
    model_path: String,

    /// Output directory for the quantized model
    #[arg(long)]
    output_path: String,

    /// Quantization group size
    #[arg(long, default_value_t = 64)]
    group_size: i32,

    /// Bits per weight
    #[arg(long, default_value_t = 4)]
    bits: i32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("Loading full-precision model from {}...", args.model_path);
    let mut model = weights::load_model(&args.model_path)?;

    eprintln!(
        "Quantizing LM to {}-bit (group_size={})...",
        args.bits, args.group_size
    );
    model.quantize_lm(args.group_size, args.bits)?;

    eprintln!("Saving quantized model to {}...", args.output_path);
    weights::save_quantized_model(&model, &args.output_path, args.group_size, args.bits)?;

    // Copy config.json and tokenizer.json from original directory
    let src = Path::new(&args.model_path);
    let dst = Path::new(&args.output_path);
    for file in &["config.json", "tokenizer.json"] {
        let src_path = src.join(file);
        let dst_path = dst.join(file);
        if src_path.exists() {
            std::fs::copy(&src_path, &dst_path)?;
            eprintln!("Copied {file}");
        }
    }

    eprintln!(
        "Done! Quantized model saved to {}\n\
         Use with: kugelaudio-rs --model-path {}",
        args.output_path, args.output_path
    );
    Ok(())
}
