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

    pub(crate) fn as_config(self) -> SherpaOnnxOnlineZipformer2CtcModelConfig {
        SherpaOnnxOnlineZipformer2CtcModelConfig {
            model: self.model.into_raw(),
        }
    }
}
