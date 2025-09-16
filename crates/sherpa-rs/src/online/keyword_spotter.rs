use std::{
    ffi::{CStr, CString},
    path::Path,
    sync::Arc,
};

use eyre::Result;
use sherpa_rs_sys::{
    SherpaOnnxFeatureConfig, SherpaOnnxKeywordSpotterConfig, SherpaOnnxOnlineModelConfig,
};

use crate::{get_default_provider, online::transducer::Transducer};

use super::Stream;

pub struct KeywordSpotter {
    spotter: *const sherpa_rs_sys::SherpaOnnxKeywordSpotter,
}

impl KeywordSpotter {
    pub fn from_transducer(
        transducer: Transducer,
        provider: Option<&str>,
        tokens: &Path,
        debug: bool,

        file: &Path,
        num_threads: Option<i32>,
    ) -> Result<Arc<Self>> {
        let tokens_c = CString::new(tokens.to_str().unwrap_or(&tokens.to_string_lossy())).unwrap();
        let provider_c = CString::new(provider.unwrap_or(&get_default_provider())).unwrap();
        let model_type_c = transducer.model_type();
        let modeling_unit_c = CString::new("cjkchar").unwrap();
        let files_c = CString::new(file.to_str().unwrap_or(&file.to_string_lossy())).unwrap();

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        model_config.transducer = transducer.as_config();
        model_config.tokens = tokens_c.as_ptr();
        model_config.num_threads = num_threads.unwrap_or(1);
        model_config.provider = provider_c.as_ptr();
        model_config.debug = debug as i32;
        model_config.modeling_unit = modeling_unit_c.as_ptr();
        model_config.model_type = model_type_c.as_ptr();

        let mut config = unsafe { std::mem::zeroed::<SherpaOnnxKeywordSpotterConfig>() };
        config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        config.model_config = model_config;
        config.num_trailing_blanks = 1;
        config.keywords_score = 1.0;
        config.keywords_threshold = 0.25;
        config.keywords_file = files_c.as_ptr();
        let spotter = unsafe { sherpa_rs_sys::SherpaOnnxCreateKeywordSpotter(&config) };
        if spotter.is_null() {
            eyre::bail!("Failed to create keyword spotter");
        }

        Ok(Arc::new(Self { spotter }))
    }

    pub fn create_stream(
        self: &Arc<Self>,
        keywords: Option<&str>,
    ) -> Result<KeywordSpottingStream> {
        KeywordSpottingStream::new(self.clone(), keywords)
    }
}

impl Drop for KeywordSpotter {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyKeywordSpotter(self.spotter);
        }
    }
}

pub struct KeywordSpottingStream {
    spotter: Arc<KeywordSpotter>,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl KeywordSpottingStream {
    pub fn new(spotter: Arc<KeywordSpotter>, keywords: Option<&str>) -> Result<Self> {
        let stream = if let Some(keywords) = keywords {
            let keywords = CString::new(keywords).unwrap();
            unsafe {
                sherpa_rs_sys::SherpaOnnxCreateKeywordStreamWithKeywords(
                    spotter.spotter,
                    keywords.as_ptr(),
                )
            }
        } else {
            unsafe { sherpa_rs_sys::SherpaOnnxCreateKeywordStream(spotter.spotter) }
        };

        if stream.is_null() {
            eyre::bail!("Failed to create SherpaOnnxOnlineStream");
        }
        Ok(Self { spotter, stream })
    }
}

impl Drop for KeywordSpottingStream {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
        }
    }
}

impl Stream for KeywordSpottingStream {
    fn accept_waveform(&mut self, sample_rate: i32, samples: impl AsRef<[f32]>) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate,
                samples.as_ref().as_ptr(),
                samples.as_ref().len() as i32,
            );
        }
    }

    fn decode_stream(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeKeywordStream(self.spotter.spotter, self.stream);
        }
    }

    fn is_ready(&mut self) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxIsKeywordStreamReady(self.spotter.spotter, self.stream) == 1
        }
    }

    fn get_result(&mut self) -> String {
        unsafe {
            let raw_result =
                sherpa_rs_sys::SherpaOnnxGetKeywordResult(self.spotter.spotter, self.stream);
            let result = raw_result.read();
            let keyword = CStr::from_ptr(result.keyword).to_string_lossy().to_string();

            sherpa_rs_sys::SherpaOnnxDestroyKeywordResult(raw_result);
            keyword
        }
    }

    fn reset(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxResetKeywordStream(self.spotter.spotter, self.stream);
        }
    }
}
