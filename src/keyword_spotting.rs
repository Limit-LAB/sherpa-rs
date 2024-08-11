use std::{
    ffi::{CStr, CString},
    path::Path,
};

use sherpa_rs_sys::{
    SherpaOnnxFeatureConfig, SherpaOnnxKeywordSpotterConfig, SherpaOnnxOnlineModelConfig,
    SherpaOnnxOnlineStreamAcceptWaveform,
};

use crate::{
    get_default_provider,
    transcribe::online::{Transducer, Zipformer2Ctc},
};

pub struct KeywordSpotter {
    spotter: *mut sherpa_rs_sys::SherpaOnnxKeywordSpotter,
    stream: *mut sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl Drop for KeywordSpotter {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyKeywordSpotter(self.spotter);
        }
    }
}

impl KeywordSpotter {
    pub fn from_transducer(
        transducer: Transducer,
        provider: Option<&str>,
        tokens: &Path,
        debug: bool,

        file: &Path,
        num_threads: Option<i32>,

        keywords: Option<&str>,
    ) -> Self {
        let tokens_c = CString::new(tokens.to_str().unwrap()).unwrap();
        let provider_c = CString::new(provider.unwrap_or(get_default_provider())).unwrap();
        let model_type = transducer.model_type();
        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        // model_config.model_type = zipformer.model_type().into_raw();
        model_config.transducer = transducer.as_config();
        model_config.tokens = tokens_c.into_raw();
        model_config.num_threads = num_threads.unwrap_or(1);
        model_config.provider = provider_c.into_raw();
        model_config.debug = debug as i32;
        model_config.modeling_unit = CString::new("cjkchar").unwrap().into_raw();
        model_config.model_type = model_type.into_raw();

        Self::new(model_config, file, keywords)
    }

    pub fn new(model: SherpaOnnxOnlineModelConfig, file: &Path, keywords: Option<&str>) -> Self {
        let files_c = CString::new(file.to_str().unwrap()).unwrap();

        let mut config = unsafe { std::mem::zeroed::<SherpaOnnxKeywordSpotterConfig>() };
        config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        config.model_config = model;
        config.num_trailing_blanks = 1;
        config.keywords_score = 1.0;
        config.keywords_threshold = 0.25;
        config.keywords_file = files_c.into_raw();
        let spotter = unsafe { sherpa_rs_sys::SherpaOnnxCreateKeywordSpotter(&config) };

        let stream = if let Some(keywords) = keywords {
            let keywords = std::ffi::CString::new(keywords).unwrap();
            unsafe {
                sherpa_rs_sys::SherpaOnnxCreateKeywordStreamWithKeywords(spotter, keywords.as_ptr())
            }
        } else {
            unsafe { sherpa_rs_sys::SherpaOnnxCreateKeywordStream(spotter) }
        };
        Self { spotter, stream }
    }

    pub fn accept_waveform(&self, sample_rate: i32, samples: Vec<f32>) {
        unsafe {
            SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    pub fn decode_stream(&self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeKeywordStream(self.spotter, self.stream);
        }
    }

    pub fn is_ready(&self) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxIsKeywordStreamReady(self.spotter, self.stream) == 1 }
    }

    pub fn get_result(&self) -> String {
        unsafe {
            let result = sherpa_rs_sys::SherpaOnnxGetKeywordResult(self.spotter, self.stream);
            let result = std::ptr::read(result);
            let keyword = CStr::from_ptr(result.keyword);
            let keyword = keyword.to_str().unwrap().to_string();
            keyword
        }
    }

    // def get_result(self, s: OnlineStream) -> str:
    //     return self.keyword_spotter.get_result(s).keyword.strip()

    // def tokens(self, s: OnlineStream) -> List[str]:
    //     return self.keyword_spotter.get_result(s).tokens

    // def timestamps(self, s: OnlineStream) -> List[float]:
    //     return self.keyword_spotter.get_result(s).timestamps
}
