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

#### CLI mode

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

**Unix socket** — one JSON request line, one JSON response line:

```bash
# Audio always returned as base64 WAV in the response
echo '{"text":"Hello world"}' | nc -U /tmp/kugelaudio.sock
```

**HTTP** (requires `--http-bind`):

```bash
# Receive base64 WAV in response
curl -s -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world"}' | jq -r .audio_b64 | base64 -d > out.wav

# Health check
curl http://127.0.0.1:8080/health
```

#### Client mode

The same binary connects to a running server when `--model-path` is omitted.
The server always returns the audio as base64; the client saves it locally.

```bash
# Save WAV to hello.wav
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

**External clients (shell / Python)**

```bash
# Shell — nc + jq
echo '{"text":"Hello world"}' \
  | nc -U /tmp/kugelaudio.sock \
  | jq -r .audio_b64 \
  | base64 -d > out.wav
```

```python
# Python — Unix socket
import socket, json, base64

def synthesize(text: str) -> bytes:
    req = {"text": text}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect("/tmp/kugelaudio.sock")
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            buf += s.recv(4096)
    resp = json.loads(buf)
    if resp["status"] != "ok":
        raise RuntimeError(resp["message"])
    return base64.b64decode(resp["audio_b64"])
```

```python
# Python — HTTP
import requests, base64

def synthesize(text: str) -> bytes:
    resp = requests.post(
        "http://127.0.0.1:8080/generate",
        json={"text": text},
    ).json()
    if resp["status"] != "ok":
        raise RuntimeError(resp["message"])
    return base64.b64decode(resp["audio_b64"])
```

**Request fields** (both transports):

| Field | Required | Description |
|-------|----------|-------------|
| `text` | yes | Text to synthesize |
| `cfg_scale` | no | Overrides server default |
| `diffusion_steps` | no | Overrides server default |
| `max_tokens` | no | Overrides server default |

**Response** on success:

```json
{"status":"ok","duration_s":4.42,"speech_tokens":32,"audio_b64":"UklGR..."}
```

**Response** on error:

```json
{"status":"error","message":"..."}
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | — | Path to model directory. Required for CLI/server mode. **Omit to run as a client.** |
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
├── main.rs                    # CLI entry point and server startup
├── lib.rs                     # Library root
├── config.rs                  # Model configuration (serde)
├── error.rs                   # Error types
├── audio/
│   └── wav.rs                 # WAV file writer and in-memory encoder (24 kHz PCM)
├── model/
│   ├── qwen2.rs               # Qwen2 transformer (external KV cache for CFG)
│   ├── connector.rs           # Speech connector (Linear → RMSNorm → Linear)
│   ├── diffusion_head.rs      # Diffusion prediction head (AdaLN + SwiGLU)
│   ├── acoustic_decoder.rs    # Convolutional audio decoder (7 upsample stages)
│   ├── causal_conv.rs         # Causal Conv1d / ConvTranspose1d wrappers
│   └── weights.rs             # Weight loading via VarBuilder
├── generation/
│   └── pipeline.rs            # End-to-end TTS generation pipeline
├── schedule/
│   └── dpm_solver.rs          # DPM-Solver++ SDE/ODE scheduler
└── server/
    ├── protocol.rs            # Request/response JSON types and WorkItem channel message
    ├── loop_.rs               # Main-thread processing loop (owns the model)
    ├── unix.rs                # Unix domain socket acceptor
    └── http.rs                # HTTP server (tiny_http, POST /generate, GET /health)
```

## Key design decisions

- **Custom Qwen2 with external KV cache** — CFG requires two forward passes (conditioned + unconditioned) through the same model with separate caches. The standard candle-transformers Qwen2 stores the cache internally, making this impossible.
- **NCL tensor format throughout** — Matches PyTorch/candle convention. No NLC transpositions needed; checkpoint weights load directly.
- **VarBuilder weight loading** — No key remapping. Rust struct hierarchy mirrors the checkpoint key structure, and `VarBuilder::pp()` handles namespacing.
- **BF16 inference** on GPU, **F32** on CPU — Auto-selected based on device capabilities.

## License

MIT
