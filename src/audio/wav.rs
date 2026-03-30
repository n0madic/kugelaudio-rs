use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use mlx_rs::{error::Exception, Array};

use crate::config::SAMPLE_RATE;

/// Write audio samples to a WAV file (24kHz, 16-bit PCM, mono).
pub fn write_wav(path: impl AsRef<Path>, audio: &Array) -> Result<(), Exception> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| Exception::custom(format!("Failed to create WAV file: {}", e)))?;

    // Convert MLX array to f32 samples, then to i16
    let audio_f32 = audio.as_type::<f32>()?;
    let samples = audio_f32.as_slice::<f32>();

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| Exception::custom(format!("Failed to write sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| Exception::custom(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}
