use std::{ffi::CString, path::Path};

use sherpa_rs_sys::SherpaOnnxOnlineTransducerModelConfig;

#[derive(Debug)]
pub struct Transducer {
    encoder: CString,
    decoder: CString,
    joiner: CString,
}

impl Transducer {
    pub fn new(encoder: &Path, decoder: &Path, joiner: &Path) -> Self {
        Self {
            encoder: CString::new(encoder.to_str().unwrap()).unwrap(),
            decoder: CString::new(decoder.to_str().unwrap()).unwrap(),
            joiner: CString::new(joiner.to_str().unwrap()).unwrap(),
        }
    }

    /// Safety: The caller must ensure that the returned config is not used after the lifetime of self
    pub(crate) unsafe fn as_config(&self) -> SherpaOnnxOnlineTransducerModelConfig {
        SherpaOnnxOnlineTransducerModelConfig {
            encoder: self.encoder.as_ptr(),
            decoder: self.decoder.as_ptr(),
            joiner: self.joiner.as_ptr(),
        }
    }

    pub(crate) fn model_type(&self) -> CString {
        CString::new("transducer").unwrap()
    }
}
