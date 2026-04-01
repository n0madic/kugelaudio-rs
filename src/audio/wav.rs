use std::path::Path;

use candle_core::{DType, Tensor};
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::config::SAMPLE_RATE;
use crate::error::Result;

/// Write audio samples to a WAV file (24kHz, 16-bit PCM, mono).
pub fn write_wav(path: impl AsRef<Path>, audio: &Tensor) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| crate::error::KugelAudioError::Audio(format!("Failed to create WAV: {e}")))?;

    let samples = audio
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    for &sample in &samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| crate::error::KugelAudioError::Audio(format!("Write sample: {e}")))?;
    }

    writer
        .finalize()
        .map_err(|e| crate::error::KugelAudioError::Audio(format!("Finalize WAV: {e}")))?;

    Ok(())
}

/// Encode audio samples as a WAV file in memory (24kHz, 16-bit PCM, mono).
///
/// Returns the complete WAV file as raw bytes (including header).
pub fn to_wav_bytes(audio: &Tensor) -> crate::error::Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let samples = audio
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| {
        crate::error::KugelAudioError::Audio(format!("Failed to create WAV writer: {e}"))
    })?;

    for &sample in &samples {
        let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer
            .write_sample(i16_sample)
            .map_err(|e| crate::error::KugelAudioError::Audio(format!("Write sample: {e}")))?;
    }

    writer
        .finalize()
        .map_err(|e| crate::error::KugelAudioError::Audio(format!("Finalize WAV: {e}")))?;

    Ok(cursor.into_inner())
}
