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

    pub(crate) fn as_config(self) -> SherpaOnnxOnlineTransducerModelConfig {
        SherpaOnnxOnlineTransducerModelConfig {
            encoder: self.encoder.into_raw(),
            decoder: self.decoder.into_raw(),
            joiner: self.joiner.into_raw(),
        }
    }

    pub(crate) fn model_type(&self) -> CString {
        CString::new("transducer").unwrap()
    }
}
