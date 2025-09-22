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

    /// Safety: The caller must ensure that the returned config is not used after the lifetime of self
    pub(crate) unsafe fn as_config(&self) -> SherpaOnnxOnlineParaformerModelConfig {
        SherpaOnnxOnlineParaformerModelConfig {
            encoder: self.encoder.as_ptr(),
            decoder: self.decoder.as_ptr(),
        }
    }

    pub(crate) fn model_type(&self) -> CString {
        CString::new("paraformer").unwrap()
    }
}
