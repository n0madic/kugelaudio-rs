#!/usr/bin/env python3
"""Convert KugelAudio voice .pt files to .safetensors format for Rust loading."""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_voice(input_path: str, output_path: str | None = None):
    """Convert a .pt voice file to .safetensors."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".safetensors")
    else:
        output_path = Path(output_path)

    print(f"Loading {input_path}...")
    voice = torch.load(input_path, map_location="cpu", weights_only=True)

    if "acoustic_mean" not in voice:
        print(f"Error: {input_path} does not contain 'acoustic_mean'")
        print(f"Available keys: {list(voice.keys())}")
        sys.exit(1)

    # Convert to float16 (MLX doesn't support bfloat16)
    tensors = {
        "acoustic_mean": voice["acoustic_mean"].to(torch.float16),
    }
    if "acoustic_std" in voice:
        tensors["acoustic_std"] = voice["acoustic_std"].to(torch.float16)

    print(f"acoustic_mean shape: {tensors['acoustic_mean'].shape}")
    print(f"Saving to {output_path}...")
    save_file(tensors, str(output_path))
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_voices.py <input.pt> [output.safetensors]")
        print("       python convert_voices.py <voices_dir>  # converts all .pt files")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        # Convert all .pt files in directory
        for pt_file in sorted(path.glob("*.pt")):
            convert_voice(str(pt_file))
    else:
        output = sys.argv[2] if len(sys.argv) > 2 else None
        convert_voice(str(path), output)
