pub mod add_punctuation;
pub mod embedding_manager;
pub mod keyword_spotting;
pub mod language_id;
pub mod online;
pub mod speaker_id;
pub mod transcribe;
pub mod vad;

use eyre::{bail, Result};

#[cfg(feature = "tts")]
pub mod tts;

#[derive(Debug, Clone, Copy)]
pub enum Provider {
    Cpu,
    Cuda,
    Coreml,
    Directml,
}

impl Provider {
    pub fn to_ptr(&self) -> *const u8 {
        match self {
            Provider::Cpu => "cpu\0".as_ptr(),
            Provider::Cuda => "cuda\0".as_ptr(),
            Provider::Coreml => "coreml\0".as_ptr(),
            Provider::Directml => "directml\0".as_ptr(),
        }
    }
}

pub const fn get_default_provider() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(target_os = "macos") {
        "coreml"
    } else if cfg!(feature = "directml") {
        "directml"
    } else {
        "cpu"
    }
}

pub fn read_audio_file(path: &str) -> Result<(i32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let sample_rate = reader.spec().sample_rate as i32;

    // Check if the sample rate is 16000
    if sample_rate != 16000 {
        bail!("The sample rate must be 16000.");
    }

    // Collect samples into a Vec<f32>
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    Ok((sample_rate, samples))
}

#[macro_export]
macro_rules! cstr {
    ($s:expr) => {
        CString::new($s).expect("Failed to create CString")
    };
}

#[macro_export]
macro_rules! cstr_to_string {
    ($ptr:expr) => {
        std::ffi::CStr::from_ptr($ptr).to_string_lossy().to_string()
    };
}
