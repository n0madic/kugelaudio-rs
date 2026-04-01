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

### 2. Build (optional: convert to quantized GGUF)

Quantized GGUF models use 2-4x less memory and load faster:

```bash
# Build the conversion tool
cargo build --release --bin convert_gguf

# Convert to GGUF (Q8_0 quantization — best quality/size tradeoff)
./target/release/convert_gguf \
  --model-dir /tmp/kugelaudio_model \
  --output /tmp/kugelaudio_model/kugelaudio-q8_0.gguf \
  --quant-type q8_0

# Other quantization options: q4k (smallest), q5k, q8_0 (default), f16 (no quantization)
```

### 3. Build

```bash
# macOS (Apple Silicon)
cargo build --release --features metal

# Linux (NVIDIA GPU)
cargo build --release --features cuda

# CPU only (any platform, slower)
cargo build --release
```

### 4. Run

#### CLI mode

```bash
# From safetensors directory (full precision, ~14 GB)
./target/release/kugelaudio-rs \
  --model-path /tmp/kugelaudio_model \
  --text "Hello world" \
  --output hello.wav

# From GGUF file (quantized, ~4-8 GB depending on quant type)
./target/release/kugelaudio-rs \
  --model-path /tmp/kugelaudio_model/kugelaudio-q8_0.gguf \
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

#### Server mode

Model loading takes ~28 s. Server mode loads the model **once** and then handles requests
in ~4 s each, eliminating the per-invocation loading cost.

```bash
# Start server — Unix socket only (default)
./target/release/kugelaudio-rs \
  --model-path /tmp/kugelaudio_model \
  --serve

# Start server — Unix socket + HTTP
./target/release/kugelaudio-rs \
  --model-path /tmp/kugelaudio_model \
  --serve \
  --http-bind 127.0.0.1:8080
```

**HTTP** (requires `--http-bind`):

```bash
# Success → raw WAV bytes (Content-Type: audio/wav)
curl -s -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world"}' -o output.wav

# Health check
curl http://127.0.0.1:8080/health
```

**Unix socket** — send one JSON request line, receive raw WAV bytes on success:

```bash
echo '{"text":"Hello world"}' | nc -U /tmp/kugelaudio.sock > output.wav
```

#### Client mode

The same binary connects to a running server when `--model-path` is omitted:

```bash
kugelaudio-rs \
  --socket-path /tmp/kugelaudio.sock \
  --text "Hello world" \
  --output hello.wav

# Override generation parameters for this request only
kugelaudio-rs \
  --socket-path /tmp/kugelaudio.sock \
  --text "Hello world" \
  --cfg-scale 1.0 \
  --diffusion-steps 5 \
  --output hello.wav
```

**External clients (Python)**

```python
# Python — HTTP
import requests

def synthesize(text: str) -> bytes:
    resp = requests.post(
        "http://127.0.0.1:8080/generate",
        json={"text": text},
    )
    if resp.headers["Content-Type"] == "audio/wav":
        return resp.content
    raise RuntimeError(resp.json()["message"])
```

```python
# Python — Unix socket
import socket, json

def synthesize(text: str) -> bytes:
    req = {"text": text}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect("/tmp/kugelaudio.sock")
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while chunk := s.recv(65536):
            buf += chunk
    if buf[:4] == b"RIFF":
        return buf  # raw WAV
    resp = json.loads(buf)
    raise RuntimeError(resp["message"])
```

**Request fields** (both transports):

| Field | Required | Description |
|-------|----------|-------------|
| `text` | yes | Text to synthesize (max 10,000 chars) |
| `cfg_scale` | no | Overrides server default (0.0–10.0) |
| `diffusion_steps` | no | Overrides server default (1–100) |
| `max_tokens` | no | Overrides server default (1–8192) |

**Response protocol**:

| Transport | Success | Error |
|-----------|---------|-------|
| HTTP | Raw WAV bytes, `Content-Type: audio/wav` | JSON, `Content-Type: application/json` |
| Unix socket | Raw WAV bytes (starts with `RIFF`) | JSON line (starts with `{`) |

Error JSON format:

```json
{"status":"error","message":"..."}
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | — | Path to model directory (safetensors) or `.gguf` file. Required for CLI/server mode. **Omit to run as a client.** |
| `--text` | — | Text to synthesize. Required in CLI and client modes. |
| `--output` | `output.wav` | Output WAV file path |
| `--cfg-scale` | `3.0` | Classifier-free guidance scale (1.0 = disabled) |
| `--max-tokens` | `2048` | Maximum autoregressive tokens |
| `--diffusion-steps` | `10` | DPM-Solver++ steps per speech token |
| `--device` | auto | `metal`, `cuda`, or `cpu` (CLI/server mode only) |
| `--count` | `1` | Number of samples to generate; output files named `<stem>_1.wav`, … (CLI mode only) |
| `--serve` | off | Enable server mode (requires `--model-path`) |
| `--socket-path` | `/tmp/kugelaudio.sock` | Unix socket — server listens here; client connects here |
| `--http-bind` | *(disabled)* | HTTP bind address, e.g. `127.0.0.1:8080` or `:8000` (server mode only) |

## Requirements

- **Rust** 1.85+
- **Model weights**: [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open) (~14 GB, bfloat16 safetensors) — or convert to GGUF for smaller size
- **Memory**: ~14 GB GPU/unified memory for bfloat16 inference; ~28 GB for CPU (float32); ~4-8 GB for quantized GGUF

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
├── main.rs                    # CLI entry point and server startup
├── lib.rs                     # Library root
├── config.rs                  # Model configuration (serde)
├── error.rs                   # Error types
├── bin/
│   └── convert_gguf.rs        # Safetensors → GGUF conversion tool
├── audio/
│   └── wav.rs                 # WAV file writer and in-memory encoder (24 kHz PCM)
├── model/
│   ├── qwen2.rs               # Qwen2 transformer (external KV cache for CFG)
│   ├── quantized_qwen2.rs     # Quantized Qwen2 (QMatMul, GGUF weights)
│   ├── lm.rs                  # Unified Lm enum (Full | Quantized)
│   ├── connector.rs           # Speech connector (Linear → RMSNorm → Linear)
│   ├── diffusion_head.rs      # Diffusion prediction head (AdaLN + SwiGLU)
│   ├── acoustic_decoder.rs    # Convolutional audio decoder (7 upsample stages)
│   ├── causal_conv.rs         # Causal Conv1d / ConvTranspose1d wrappers
│   └── weights.rs             # Weight loading (safetensors + GGUF auto-detect)
├── generation/
│   └── pipeline.rs            # End-to-end TTS generation pipeline
├── schedule/
│   └── dpm_solver.rs          # DPM-Solver++ SDE/ODE scheduler
└── server/
    ├── protocol.rs            # Request/response types and WorkItem channel message
    ├── server_loop.rs         # Main-thread processing loop (owns the model)
    ├── unix.rs                # Unix domain socket acceptor
    └── http.rs                # HTTP server (tiny_http, POST /generate, GET /health)
```

## Key design decisions

- **Custom Qwen2 with external KV cache** — CFG requires two forward passes (conditioned + unconditioned) through the same model with separate caches. The standard candle-transformers Qwen2 stores the cache internally, making this impossible.
- **NCL tensor format throughout** — Matches PyTorch/candle convention. No NLC transpositions needed; checkpoint weights load directly.
- **VarBuilder weight loading** — No key remapping. Rust struct hierarchy mirrors the checkpoint key structure, and `VarBuilder::pp()` handles namespacing.
- **BF16 inference** on GPU, **F32** on CPU — Auto-selected based on device capabilities.
- **GGUF quantization** — LM backbone quantized to Q4K/Q5K/Q8_0 via `convert_gguf` binary. Non-LM components (diffusion head, acoustic decoder, connector) stored as F16 for quality. Unified `Lm` enum dispatches both full-precision and quantized models transparently.

## License

MIT
