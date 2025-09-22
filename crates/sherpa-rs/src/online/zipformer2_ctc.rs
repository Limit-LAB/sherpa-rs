use std::{ffi::CString, path::Path};

use sherpa_rs_sys::SherpaOnnxOnlineZipformer2CtcModelConfig;

#[derive(Debug)]
pub struct Zipformer2Ctc {
    model: CString,
}

impl Zipformer2Ctc {
    pub fn new(model: &Path) -> Self {
        Self {
            model: CString::new(model.to_str().unwrap()).unwrap(),
        }
    }

    /// Safety: The caller must ensure that the returned config is not used after the lifetime of self
    pub(crate) unsafe fn as_config(&self) -> SherpaOnnxOnlineZipformer2CtcModelConfig {
        SherpaOnnxOnlineZipformer2CtcModelConfig {
            model: self.model.as_ptr(),
        }
    }
}
