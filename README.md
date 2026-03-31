# kugelaudio-rs

Rust implementation of [KugelAudio](https://huggingface.co/kugelaudio/kugelaudio-0-open) text-to-speech inference, powered by [candle](https://github.com/huggingface/candle).

Runs on **Metal** (Apple Silicon), **CUDA** (NVIDIA GPUs), and **CPU** from a single codebase.

## Architecture

KugelAudio is a hybrid autoregressive + diffusion TTS model:

- **Qwen2-7B LM backbone** (28 layers, GQA) drives autoregressive token generation
- **DPM-Solver++ SDE** diffusion with classifier-free guidance (CFG) produces speech latents
- **Acoustic decoder** (7-stage convolutional upsampler) converts latents to 24 kHz audio
- **Speech connector** maps acoustic features back into the LM hidden space for the feedback loop

```
Text ─→ Tokenizer ─→ Qwen2 LM ─→ AR token loop ───┐
                         ↑                        │
                   acoustic embed            diffusion head
                         ↑                        │
                    connector ←── speech latent ←─┘
                                      │
                               acoustic decoder ─→ WAV
```

## Quick start

### 1. Download the model

```bash
# Requires ~14 GB disk space
huggingface-cli download kugelaudio/kugelaudio-0-open --local-dir /tmp/kugelaudio_model
```

### 2. Build

```bash
# macOS (Apple Silicon)
cargo build --release --features metal

# Linux (NVIDIA GPU)
cargo build --release --features cuda

# CPU only (any platform, slower)
cargo build --release
```

### 3. Run

```bash
./target/release/kugelaudio-rs \
  --model-path /tmp/kugelaudio_model \
  --text "Hello world" \
  --output hello.wav
```

Generate multiple samples at once (outputs `hello_1.wav`, `hello_2.wav`, `hello_3.wav`):

```bash
./target/release/kugelaudio-rs \
  --model-path /tmp/kugelaudio_model \
  --text "Hello world" \
  --output hello.wav \
  --count 3
```

The device is auto-detected (CUDA > Metal > CPU). Override with `--device`:

```bash
./target/release/kugelaudio-rs --device cpu --model-path /tmp/kugelaudio_model --text "Hello"
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *required* | Path to model directory |
| `--text` | *required* | Text to synthesize |
| `--output` | `output.wav` | Output WAV file path |
| `--cfg-scale` | `3.0` | Classifier-free guidance scale (1.0 = disabled) |
| `--max-tokens` | `2048` | Maximum autoregressive tokens |
| `--diffusion-steps` | `10` | DPM-Solver++ steps per speech token |
| `--device` | auto | `metal`, `cuda`, or `cpu` |
| `--count` | `1` | Number of samples to generate; output files are named `<stem>_1.wav`, `<stem>_2.wav`, … |

## Requirements

- **Rust** 1.85+
- **Model weights**: [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) (~14 GB, bfloat16 safetensors)
- **Memory**: ~14 GB GPU/unified memory for bfloat16 inference; ~28 GB for CPU (float32)

### Platform-specific

| Platform | Feature flag | Backend | Notes |
|----------|-------------|---------|-------|
| macOS Apple Silicon | `--features metal` | Metal GPU | Recommended for Mac |
| macOS Apple Silicon | `--features accelerate` | CPU with Accelerate | BLAS-accelerated CPU |
| Linux NVIDIA | `--features cuda` | CUDA | Requires CUDA toolkit |
| Any | *(none)* | CPU | Slowest, works everywhere |

## Project structure

```
src/
├── main.rs                    # CLI entry point
├── lib.rs                     # Library root
├── config.rs                  # Model configuration (serde)
├── error.rs                   # Error types
├── audio/
│   └── wav.rs                 # WAV file writer (24 kHz PCM)
├── model/
│   ├── qwen2.rs               # Qwen2 transformer (external KV cache for CFG)
│   ├── connector.rs           # Speech connector (Linear → RMSNorm → Linear)
│   ├── diffusion_head.rs      # Diffusion prediction head (AdaLN + SwiGLU)
│   ├── acoustic_decoder.rs    # Convolutional audio decoder (7 upsample stages)
│   ├── causal_conv.rs         # Causal Conv1d / ConvTranspose1d wrappers
│   └── weights.rs             # Weight loading via VarBuilder
├── generation/
│   └── pipeline.rs            # End-to-end TTS generation pipeline
└── schedule/
    └── dpm_solver.rs          # DPM-Solver++ SDE/ODE scheduler
```

## Key design decisions

- **Custom Qwen2 with external KV cache** — CFG requires two forward passes (conditioned + unconditioned) through the same model with separate caches. The standard candle-transformers Qwen2 stores the cache internally, making this impossible.
- **NCL tensor format throughout** — Matches PyTorch/candle convention. No NLC transpositions needed; checkpoint weights load directly.
- **VarBuilder weight loading** — No key remapping. Rust struct hierarchy mirrors the checkpoint key structure, and `VarBuilder::pp()` handles namespacing.
- **BF16 inference** on GPU, **F32** on CPU — Auto-selected based on device capabilities.

## License

MIT
