use sherpa_rs_sys::{SherpaOnnxOfflinePunctuationConfig, SherpaOnnxOfflinePunctuationModelConfig};

use crate::get_default_provider;
use std::ffi::{CStr, CString};

#[derive(Debug)]
pub struct PunctuationAdder {
    adder: *mut sherpa_rs_sys::SherpaOnnxOfflinePunctuation,
}

impl PunctuationAdder {
    pub fn new(
        provider: Option<&str>,
        model: String,
        num_threads: Option<i32>,
        debug: Option<bool>,
    ) -> Self {
        // SherpaOnnxOfflinePunctuationConfig config;
        let model_c = CString::new(model).unwrap();

        let provider = provider.unwrap_or(get_default_provider());
        let provider_c = CString::new(provider).unwrap();

        let config = SherpaOnnxOfflinePunctuationConfig {
            model: SherpaOnnxOfflinePunctuationModelConfig {
                ct_transformer: model_c.into_raw(),
                num_threads: num_threads.unwrap_or(1),
                debug: debug.unwrap_or(true) as i32,
                provider: provider_c.into_raw(),
            },
        };

        let adder = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflinePunctuation(&config) } as _;
        Self { adder }
    }

    pub fn add_punctuation<T: Into<String>>(&self, text: T) -> String {
        let text_c = CString::new(text.into()).unwrap();
        let text_with_punct = unsafe {
            sherpa_rs_sys::SherpaOfflinePunctuationAddPunct(self.adder, text_c.into_raw())
        };
        let text_with_punct = unsafe { CStr::from_ptr(text_with_punct) };
        let text_with_punct = text_with_punct.to_str().unwrap().to_string();
        text_with_punct
    }
}

impl Drop for PunctuationAdder {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflinePunctuation(self.adder);
        }
    }
}
