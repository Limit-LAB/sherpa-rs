use std::{ffi::CString, path::Path};

use sherpa_rs_sys::SherpaOnnxOnlineParaformerModelConfig;

#[derive(Debug)]
pub struct Paraformer {
    encoder: CString,
    decoder: CString,
}

impl Paraformer {
    pub fn new(encoder: &Path, decoder: &Path) -> Self {
        Self {
            encoder: CString::new(encoder.to_str().unwrap()).unwrap(),
            decoder: CString::new(decoder.to_str().unwrap()).unwrap(),
        }
    }

    pub(crate) fn as_config(self) -> SherpaOnnxOnlineParaformerModelConfig {
        SherpaOnnxOnlineParaformerModelConfig {
            encoder: self.encoder.into_raw(),
            decoder: self.decoder.into_raw(),
        }
    }

    pub(crate) fn model_type(&self) -> CString {
        CString::new("paraformer").unwrap()
    }
}
