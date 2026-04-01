//! Convert KugelAudio safetensors weights to GGUF format.
//!
//! Reads a model directory containing `config.json` and sharded safetensors
//! files, then writes a single `.gguf` file with:
//! - LM backbone tensors quantized to user-selected type (Q4K/Q5K/Q8_0/F16)
//! - Non-LM weight matrices stored as F16
//! - Bias, norm, and scalar tensors stored as F32
//! - Full model configuration embedded as GGUF metadata

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail, ensure};
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

use kugelaudio_rs::config::KugelAudioConfig;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "convert_gguf")]
#[command(about = "Convert KugelAudio safetensors to GGUF format")]
struct Args {
    /// Path to the model directory containing config.json and safetensors shards
    #[arg(long)]
    model_dir: PathBuf,

    /// Output GGUF file path
    #[arg(long)]
    output: PathBuf,

    /// Quantization type for LM backbone weight matrices (q4k, q5k, q8_0, f16)
    #[arg(long, default_value = "q8_0")]
    quant_type: String,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// HF safetensors prefix for the Qwen2 LM backbone.
const LM_PREFIX: &str = "model.language_model.";

/// HF safetensors key for the LM head (at root level, not under language_model).
///
/// In `load_model_safetensors()`, the lm_head VarBuilder is scoped to the root
/// (not under `model.language_model`), so the safetensors key is `lm_head.weight`.
const LM_HEAD_KEY: &str = "lm_head.weight";

// ---------------------------------------------------------------------------
// Quantization type parsing
// ---------------------------------------------------------------------------

fn parse_quantization(s: &str) -> anyhow::Result<GgmlDType> {
    match s.to_lowercase().as_str() {
        "q4k" | "q4_k" => Ok(GgmlDType::Q4K),
        "q5k" | "q5_k" => Ok(GgmlDType::Q5K),
        "q8_0" | "q8" => Ok(GgmlDType::Q8_0),
        "f16" => Ok(GgmlDType::F16),
        other => bail!("Unsupported quantization type '{other}'. Use: q4k, q5k, q8_0, f16"),
    }
}

/// Map quantization type to GGUF file type ID (llama.cpp convention).
fn file_type_id(dtype: GgmlDType) -> u32 {
    match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 | GgmlDType::BF16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8K => 7,
        GgmlDType::Q5_0 => 8,
        GgmlDType::Q5_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 12,
        GgmlDType::Q4K => 15,
        GgmlDType::Q5K => 17,
        GgmlDType::Q6K => 18,
    }
}

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

/// Returns `true` if the GGUF tensor name belongs to the LM backbone.
///
/// Duplicated from `weights.rs` — kept local since it's a small predicate
/// and the binary shouldn't force the library to expose internal helpers.
fn is_lm_tensor(name: &str) -> bool {
    name.starts_with("blk.")
        || name == "token_embd.weight"
        || name == "output.weight"
        || name == "output_norm.weight"
}

/// Decide GGUF storage dtype for a tensor given its GGUF name.
fn decide_storage(gguf_name: &str, quant_dtype: GgmlDType) -> GgmlDType {
    // Scalar tensors → always F32
    if gguf_name == "model.speech_scaling_factor" || gguf_name == "model.speech_bias_factor" {
        return GgmlDType::F32;
    }
    // LM norm/bias (small 1-D tensors, need precision) → F32
    if is_lm_tensor(gguf_name) && (gguf_name.contains("norm") || gguf_name.contains("bias")) {
        return GgmlDType::F32;
    }
    // LM weight matrices → user-selected quantization
    if is_lm_tensor(gguf_name) {
        return quant_dtype;
    }
    // Non-LM norm, bias, gamma → F32
    if gguf_name.contains("norm.weight")
        || gguf_name.contains("bias")
        || gguf_name.contains("gamma")
    {
        return GgmlDType::F32;
    }
    // Non-LM weight matrices → F16
    GgmlDType::F16
}

// ---------------------------------------------------------------------------
// Name remapping (HF safetensors → GGUF)
// ---------------------------------------------------------------------------

/// Remap an HF safetensors tensor name to GGUF convention.
///
/// Returns `Some(gguf_name)` for tensors to include, `None` for those to skip
/// (layers beyond `tts_layers`, `lm_head.weight` when tied, unknown LM tensors).
fn remap_tensor_name(
    hf_name: &str,
    tts_layers: usize,
    tie_word_embeddings: bool,
) -> Option<String> {
    // LM head at root level (not under model.language_model.)
    if hf_name == LM_HEAD_KEY {
        if tie_word_embeddings {
            return None; // Loader falls back to token_embd.weight
        }
        return Some("output.weight".to_string());
    }

    // Non-LM tensors (model.acoustic_connector.*, model.prediction_head.*, etc.)
    if !hf_name.starts_with(LM_PREFIX) {
        return Some(hf_name.to_string());
    }

    // LM backbone tensors: strip the HF prefix and remap
    let suffix = &hf_name[LM_PREFIX.len()..];

    if suffix == "embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }

    if suffix == "norm.weight" {
        return Some("output_norm.weight".to_string());
    }

    // Layer tensors: layers.{i}.{path}
    if let Some(rest) = suffix.strip_prefix("layers.") {
        let dot_pos = rest.find('.')?;
        let layer_idx: usize = rest[..dot_pos].parse().ok()?;
        if layer_idx >= tts_layers {
            return None; // Skip layers beyond TTS backbone
        }
        let layer_suffix = &rest[dot_pos + 1..];

        let gguf_suffix = match layer_suffix {
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.q_proj.bias" => "attn_q.bias",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.k_proj.bias" => "attn_k.bias",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.v_proj.bias" => "attn_v.bias",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            _ => {
                eprintln!("Warning: unknown LM layer tensor '{hf_name}', skipping");
                return None;
            }
        };

        return Some(format!("blk.{layer_idx}.{gguf_suffix}"));
    }

    // Unknown LM tensor (e.g. rotary_emb.inv_freq — computed from config, not needed)
    eprintln!("Warning: unknown LM tensor '{hf_name}', skipping");
    None
}

// ---------------------------------------------------------------------------
// Safetensors shard discovery
// ---------------------------------------------------------------------------

/// Minimal representation of `model.safetensors.index.json`.
#[derive(Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

/// Read `model.safetensors.index.json` and return the sorted list of unique
/// shard file paths.
fn find_safetensor_shards(model_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let file = fs::File::open(&index_path)
        .with_context(|| format!("Cannot open {}", index_path.display()))?;
    let index: SafetensorIndex =
        serde_json::from_reader(file).context("Failed to parse model.safetensors.index.json")?;

    let mut seen = HashSet::new();
    let mut shards: Vec<String> = index
        .weight_map
        .into_values()
        .filter(|name| seen.insert(name.clone()))
        .collect();
    shards.sort();

    ensure!(!shards.is_empty(), "weight_map in index.json is empty");

    Ok(shards
        .into_iter()
        .map(|name| model_dir.join(name))
        .collect())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    ensure!(
        args.model_dir.is_dir(),
        "{} is not a directory",
        args.model_dir.display()
    );

    let quant_dtype = parse_quantization(&args.quant_type)?;
    eprintln!("KugelAudio GGUF converter — quantization: {quant_dtype:?}");

    // ── 1. Load config.json (both parsed struct and raw JSON string) ─────
    let config_path = args.model_dir.join("config.json");
    let config_raw = fs::read_to_string(&config_path)
        .with_context(|| format!("Cannot read {}", config_path.display()))?;
    let config: KugelAudioConfig =
        serde_json::from_str(&config_raw).context("Failed to parse config.json")?;

    let tts_layers = config.tts_layers() as usize;
    let tie_word_embeddings = config.decoder_config.tie_word_embeddings;
    eprintln!(
        "TTS backbone: {tts_layers} of {} layers, tie_word_embeddings: {tie_word_embeddings}",
        config.decoder_config.num_hidden_layers
    );

    // ── 2. Load all safetensors shards into memory (CPU) ─────────────────
    let shard_paths = find_safetensor_shards(&args.model_dir)?;
    eprintln!("Loading {} safetensors shard(s)...", shard_paths.len());

    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();
    for shard_path in &shard_paths {
        eprintln!("  {}", shard_path.display());
        let shard = candle_core::safetensors::load(shard_path, &Device::Cpu)
            .with_context(|| format!("Failed to load {}", shard_path.display()))?;
        all_tensors.extend(shard);
    }
    eprintln!("Loaded {} tensors total.", all_tensors.len());

    // ── 3. Remap names, classify, and quantize each tensor ───────────────
    let pb = ProgressBar::new(all_tensors.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("valid template")
            .progress_chars("=>-"),
    );

    let mut gguf_tensors: Vec<(String, QTensor)> = Vec::new();
    let mut skipped = 0u32;

    // Sort keys for deterministic output
    let mut hf_names: Vec<String> = all_tensors.keys().cloned().collect();
    hf_names.sort();

    for hf_name in &hf_names {
        pb.set_message(hf_name.clone());

        let gguf_name = match remap_tensor_name(hf_name, tts_layers, tie_word_embeddings) {
            Some(name) => name,
            None => {
                skipped += 1;
                pb.inc(1);
                continue;
            }
        };

        let storage_dtype = decide_storage(&gguf_name, quant_dtype);

        let tensor = all_tensors.get(hf_name).expect("key exists");

        // QTensor::quantize requires CPU F32 tensors with non-empty shape.
        // Scalar (0-d) tensors must be reshaped to [1] first.
        let tensor_f32 = tensor
            .to_dtype(DType::F32)
            .with_context(|| format!("Failed to convert '{hf_name}' to F32"))?;
        let tensor_f32 = if tensor_f32.dims().is_empty() {
            tensor_f32
                .reshape(1)
                .with_context(|| format!("Failed to reshape scalar '{hf_name}' to [1]"))?
        } else {
            tensor_f32
        };

        let qtensor = QTensor::quantize(&tensor_f32, storage_dtype)
            .with_context(|| format!("Failed to quantize '{hf_name}' as {storage_dtype:?}"))?;

        gguf_tensors.push((gguf_name, qtensor));
        pb.inc(1);
    }

    pb.finish_with_message("done");
    eprintln!(
        "Converted {} tensors, skipped {skipped}.",
        gguf_tensors.len()
    );

    // ── 4. Embed tokenizer ────────────────────────────────────────────────
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer_val = if tokenizer_path.exists() {
        let raw = fs::read_to_string(&tokenizer_path)
            .with_context(|| format!("Failed to read {}", tokenizer_path.display()))?;
        eprintln!("Embedding tokenizer ({} bytes).", raw.len());
        Some(Value::String(raw))
    } else {
        eprintln!("Warning: tokenizer.json not found, GGUF will not contain tokenizer.");
        None
    };

    // ── 5. Build GGUF metadata ───────────────────────────────────────────
    let arch_val = Value::String("kugelaudio".to_string());
    let name_val = Value::String("KugelAudio-7B-TTS".to_string());
    let file_type_val = Value::U32(file_type_id(quant_dtype));
    let quant_ver_val = Value::U32(2);
    let config_val = Value::String(config_raw);
    let tts_layers_val = Value::U32(tts_layers as u32);
    let block_count_val = Value::U32(tts_layers as u32);
    let embedding_length_val = Value::U32(config.decoder_config.hidden_size as u32);
    let head_count_val = Value::U32(config.decoder_config.num_attention_heads as u32);
    let head_count_kv_val = Value::U32(config.decoder_config.num_key_value_heads as u32);
    let vae_dim_val = Value::U32(config.vae_dim() as u32);

    let mut metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch_val),
        ("general.name", &name_val),
        ("general.file_type", &file_type_val),
        ("general.quantization_version", &quant_ver_val),
        ("kugelaudio.config", &config_val),
        ("kugelaudio.tts_layers", &tts_layers_val),
        ("qwen2.block_count", &block_count_val),
        ("qwen2.embedding_length", &embedding_length_val),
        ("qwen2.attention.head_count", &head_count_val),
        ("qwen2.attention.head_count_kv", &head_count_kv_val),
        ("kugelaudio.vae_dim", &vae_dim_val),
    ];
    if let Some(ref tok_val) = tokenizer_val {
        metadata.push(("tokenizer.huggingface.json", tok_val));
    }

    // ── 5. Write GGUF file ───────────────────────────────────────────────
    let tensor_refs: Vec<(&str, &QTensor)> = gguf_tensors
        .iter()
        .map(|(name, qt)| (name.as_str(), qt))
        .collect();

    eprintln!("Writing GGUF to {}...", args.output.display());
    let file = fs::File::create(&args.output)
        .with_context(|| format!("Cannot create {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    gguf_file::write(&mut writer, &metadata, &tensor_refs).context("Failed to write GGUF file")?;

    // ── 6. Summary ───────────────────────────────────────────────────────
    let file_size = fs::metadata(&args.output)?.len();
    let size_gb = file_size as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("Done! {size_gb:.2} GB written.");
    eprintln!("  Tensors: {} (skipped {skipped})", gguf_tensors.len());
    eprintln!("  LM quantization: {quant_dtype:?}");
    eprintln!("  TTS layers: {tts_layers}");

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_quantization -----------------------------------------------

    #[test]
    fn test_parse_quantization_valid() {
        assert_eq!(parse_quantization("q4k").unwrap(), GgmlDType::Q4K);
        assert_eq!(parse_quantization("Q4K").unwrap(), GgmlDType::Q4K);
        assert_eq!(parse_quantization("q4_k").unwrap(), GgmlDType::Q4K);
        assert_eq!(parse_quantization("q5k").unwrap(), GgmlDType::Q5K);
        assert_eq!(parse_quantization("Q5_K").unwrap(), GgmlDType::Q5K);
        assert_eq!(parse_quantization("q8_0").unwrap(), GgmlDType::Q8_0);
        assert_eq!(parse_quantization("q8").unwrap(), GgmlDType::Q8_0);
        assert_eq!(parse_quantization("f16").unwrap(), GgmlDType::F16);
        assert_eq!(parse_quantization("F16").unwrap(), GgmlDType::F16);
    }

    #[test]
    fn test_parse_quantization_invalid() {
        assert!(parse_quantization("q3k").is_err());
        assert!(parse_quantization("invalid").is_err());
        assert!(parse_quantization("").is_err());
    }

    // -- remap_tensor_name ------------------------------------------------

    #[test]
    fn test_remap_lm_embedding() {
        assert_eq!(
            remap_tensor_name("model.language_model.embed_tokens.weight", 20, false),
            Some("token_embd.weight".to_string())
        );
    }

    #[test]
    fn test_remap_lm_head_untied() {
        assert_eq!(
            remap_tensor_name("lm_head.weight", 20, false),
            Some("output.weight".to_string())
        );
    }

    #[test]
    fn test_remap_lm_head_tied() {
        assert_eq!(remap_tensor_name("lm_head.weight", 20, true), None);
    }

    #[test]
    fn test_remap_lm_norm() {
        assert_eq!(
            remap_tensor_name("model.language_model.norm.weight", 20, false),
            Some("output_norm.weight".to_string())
        );
    }

    #[test]
    fn test_remap_lm_layer_tensors() {
        let cases = [
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.bias",
                "blk.0.attn_q.bias",
            ),
            (
                "model.language_model.layers.5.self_attn.k_proj.weight",
                "blk.5.attn_k.weight",
            ),
            (
                "model.language_model.layers.5.self_attn.k_proj.bias",
                "blk.5.attn_k.bias",
            ),
            (
                "model.language_model.layers.10.self_attn.v_proj.weight",
                "blk.10.attn_v.weight",
            ),
            (
                "model.language_model.layers.10.self_attn.v_proj.bias",
                "blk.10.attn_v.bias",
            ),
            (
                "model.language_model.layers.19.self_attn.o_proj.weight",
                "blk.19.attn_output.weight",
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "blk.0.ffn_gate.weight",
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                "blk.0.ffn_up.weight",
            ),
            (
                "model.language_model.layers.0.mlp.down_proj.weight",
                "blk.0.ffn_down.weight",
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
        ];

        for (hf, expected) in &cases {
            assert_eq!(
                remap_tensor_name(hf, 20, false).as_deref(),
                Some(*expected),
                "Failed for {hf}"
            );
        }
    }

    #[test]
    fn test_remap_layer_filtering() {
        // Layer 19 included (tts_layers = 20)
        assert!(
            remap_tensor_name(
                "model.language_model.layers.19.self_attn.q_proj.weight",
                20,
                false,
            )
            .is_some()
        );

        // Layer 20 excluded
        assert_eq!(
            remap_tensor_name(
                "model.language_model.layers.20.self_attn.q_proj.weight",
                20,
                false,
            ),
            None
        );

        // Layer 27 excluded
        assert_eq!(
            remap_tensor_name(
                "model.language_model.layers.27.self_attn.q_proj.weight",
                20,
                false,
            ),
            None
        );
    }

    #[test]
    fn test_remap_non_lm_tensors_keep_names() {
        let cases = [
            "model.acoustic_connector.fc1.weight",
            "model.acoustic_connector.fc1.bias",
            "model.prediction_head.cond_proj.weight",
            "model.acoustic_tokenizer.decoder.head.conv.conv.weight",
            "model.speech_scaling_factor",
            "model.speech_bias_factor",
        ];

        for hf in &cases {
            assert_eq!(
                remap_tensor_name(hf, 20, false).as_deref(),
                Some(*hf),
                "Non-LM tensor '{hf}' should keep its name"
            );
        }
    }

    #[test]
    fn test_remap_unknown_lm_tensor_skipped() {
        // rotary_emb is computed from config, not loaded — should be skipped
        assert_eq!(
            remap_tensor_name("model.language_model.rotary_emb.inv_freq", 20, false),
            None
        );
    }

    // -- decide_storage ---------------------------------------------------

    #[test]
    fn test_decide_storage_scalars() {
        assert_eq!(
            decide_storage("model.speech_scaling_factor", GgmlDType::Q8_0),
            GgmlDType::F32
        );
        assert_eq!(
            decide_storage("model.speech_bias_factor", GgmlDType::Q4K),
            GgmlDType::F32
        );
    }

    #[test]
    fn test_decide_storage_lm_weights_use_quant() {
        assert_eq!(
            decide_storage("token_embd.weight", GgmlDType::Q8_0),
            GgmlDType::Q8_0
        );
        assert_eq!(
            decide_storage("output.weight", GgmlDType::Q4K),
            GgmlDType::Q4K
        );
        assert_eq!(
            decide_storage("blk.0.attn_q.weight", GgmlDType::Q5K),
            GgmlDType::Q5K
        );
        assert_eq!(
            decide_storage("blk.19.ffn_gate.weight", GgmlDType::F16),
            GgmlDType::F16
        );
    }

    #[test]
    fn test_decide_storage_lm_bias_norm_f32() {
        assert_eq!(
            decide_storage("blk.0.attn_q.bias", GgmlDType::Q8_0),
            GgmlDType::F32
        );
        assert_eq!(
            decide_storage("blk.5.attn_norm.weight", GgmlDType::Q4K),
            GgmlDType::F32
        );
        assert_eq!(
            decide_storage("blk.10.ffn_norm.weight", GgmlDType::Q5K),
            GgmlDType::F32
        );
        assert_eq!(
            decide_storage("output_norm.weight", GgmlDType::Q8_0),
            GgmlDType::F32
        );
    }

    #[test]
    fn test_decide_storage_non_lm_weights_f16() {
        assert_eq!(
            decide_storage("model.acoustic_connector.fc1.weight", GgmlDType::Q8_0),
            GgmlDType::F16
        );
        assert_eq!(
            decide_storage("model.prediction_head.cond_proj.weight", GgmlDType::Q8_0),
            GgmlDType::F16
        );
        assert_eq!(
            decide_storage(
                "model.acoustic_tokenizer.decoder.head.conv.conv.weight",
                GgmlDType::Q8_0
            ),
            GgmlDType::F16
        );
    }

    #[test]
    fn test_decide_storage_non_lm_bias_norm_gamma_f32() {
        assert_eq!(
            decide_storage("model.acoustic_connector.fc1.bias", GgmlDType::Q8_0),
            GgmlDType::F32
        );
        assert_eq!(
            decide_storage(
                "model.prediction_head.layers.0.norm.weight",
                GgmlDType::Q8_0
            ),
            GgmlDType::F32
        );
        assert_eq!(
            decide_storage(
                "model.acoustic_tokenizer.decoder.block.0.gamma",
                GgmlDType::Q8_0
            ),
            GgmlDType::F32
        );
    }

    // -- file_type_id -----------------------------------------------------

    #[test]
    fn test_file_type_id() {
        assert_eq!(file_type_id(GgmlDType::F16), 1);
        assert_eq!(file_type_id(GgmlDType::Q4K), 15);
        assert_eq!(file_type_id(GgmlDType::Q5K), 17);
        assert_eq!(file_type_id(GgmlDType::Q8_0), 7);
        assert_eq!(file_type_id(GgmlDType::F32), 0);
    }

    // -- is_lm_tensor -----------------------------------------------------

    #[test]
    fn test_is_lm_tensor() {
        assert!(is_lm_tensor("token_embd.weight"));
        assert!(is_lm_tensor("output.weight"));
        assert!(is_lm_tensor("output_norm.weight"));
        assert!(is_lm_tensor("blk.0.attn_q.weight"));
        assert!(is_lm_tensor("blk.19.ffn_down.weight"));

        assert!(!is_lm_tensor("model.acoustic_connector.fc1.weight"));
        assert!(!is_lm_tensor("model.prediction_head.cond_proj.weight"));
        assert!(!is_lm_tensor("model.speech_scaling_factor"));
    }
}
