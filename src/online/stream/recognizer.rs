use std::{
    ffi::{CStr, CString},
    path::Path,
};

use sherpa_rs_sys::{
    SherpaOnnxCreateOnlineRecognizer, SherpaOnnxCreateOnlineStream, SherpaOnnxFeatureConfig,
    SherpaOnnxOnlineCtcFstDecoderConfig, SherpaOnnxOnlineModelConfig,
    SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineStreamIsEndpoint,
    SherpaOnnxOnlineStreamReset,
};

use crate::{
    get_default_provider,
    online::{paraformer::Paraformer, transducer::Transducer, zipformer2_ctc::Zipformer2Ctc},
};

use super::OnlineStream;

#[derive(Debug)]
pub enum Search {
    Greedy,
    Beam,
}

impl Search {
    fn into_raw(self) -> CString {
        match self {
            Self::Greedy => CString::new("greedy_search").unwrap(),
            Self::Beam => CString::new("modified_beam_search").unwrap(),
        }
    }
}

pub struct RecognizerStream {
    recognizer: *mut sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
    stream: *mut sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl RecognizerStream {
    pub fn from_transducer(
        transducer: Transducer,
        provider: Option<&str>,
        tokens: &Path,
        search: Search,
        debug: bool,

        num_threads: Option<i32>,
        hotwords: Option<&Path>,
        hotwords_score: Option<f32>,
    ) -> Self {
        let tokens_c = CString::new(tokens.to_str().unwrap()).unwrap();
        let provider_c = CString::new(provider.unwrap_or(get_default_provider())).unwrap();

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        model_config.transducer = transducer.as_config();
        model_config.tokens = tokens_c.into_raw();
        model_config.num_threads = num_threads.unwrap_or(1);
        model_config.provider = provider_c.into_raw();
        model_config.debug = debug as i32;
        model_config.modeling_unit = CString::new("cjkchar").unwrap().into_raw();

        let mut rec_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineRecognizerConfig>() };
        rec_config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        rec_config.model_config = model_config;
        rec_config.decoding_method = search.into_raw().into_raw();
        rec_config.max_active_paths = 4;
        rec_config.enable_endpoint = 1;
        rec_config.rule1_min_trailing_silence = 2.4;
        rec_config.rule2_min_trailing_silence = 1.2;
        rec_config.rule3_min_utterance_length = 300.0;

        if hotwords.is_some() {
            let hotwords_c =
                hotwords.map(|p| CString::new(p.to_str().unwrap()).unwrap().into_raw());
            rec_config.hotwords_file = hotwords_c.unwrap_or(std::ptr::null_mut());
            rec_config.hotwords_score = hotwords_score.unwrap_or(1.5);
        }

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&rec_config) };
        let stream = unsafe { SherpaOnnxCreateOnlineStream(recognizer) };
        // let display = unsafe { SherpaOnnxCreateDisplay()}
        Self { recognizer, stream }
    }

    pub fn from_paraformer(
        paraformer: Paraformer,
        provider: Option<&str>,
        tokens: &Path,
        search: Search,
        debug: bool,

        num_threads: Option<i32>,
        hotwords: Option<&Path>,
        hotwords_score: Option<f32>,
    ) -> Self {
        let tokens_c = CString::new(tokens.to_str().unwrap()).unwrap();
        let provider_c = CString::new(provider.unwrap_or(get_default_provider())).unwrap();

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        model_config.model_type = paraformer.model_type().into_raw();
        model_config.paraformer = paraformer.as_config();
        model_config.tokens = tokens_c.into_raw();
        model_config.num_threads = num_threads.unwrap_or(1);
        model_config.provider = provider_c.into_raw();
        model_config.debug = debug as i32;
        model_config.modeling_unit = CString::new("cjkchar").unwrap().into_raw();

        let mut rec_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineRecognizerConfig>() };
        rec_config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        rec_config.model_config = model_config;
        rec_config.decoding_method = search.into_raw().into_raw();
        rec_config.max_active_paths = 4;
        rec_config.enable_endpoint = 1;
        rec_config.rule1_min_trailing_silence = 2.4;
        rec_config.rule2_min_trailing_silence = 1.2;
        rec_config.rule3_min_utterance_length = 300.0;

        if hotwords.is_some() {
            let hotwords_c =
                hotwords.map(|p| CString::new(p.to_str().unwrap()).unwrap().into_raw());
            rec_config.hotwords_file = hotwords_c.unwrap_or(std::ptr::null_mut());
            rec_config.hotwords_score = hotwords_score.unwrap_or(1.5);
        }

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&rec_config) };
        let stream = unsafe { SherpaOnnxCreateOnlineStream(recognizer) };
        // let display = unsafe { SherpaOnnxCreateDisplay()}

        println!("recognizer: {:?}, stream: {:?}", recognizer, stream);
        Self { recognizer, stream }
    }

    pub fn from_zipformer(
        zipformer: Zipformer2Ctc,
        provider: Option<&str>,
        tokens: &Path,
        search: Search,
        debug: bool,

        num_threads: Option<i32>,
        graph: Option<&Path>,
        hotwords: Option<&Path>,
        hotwords_score: Option<f32>,
    ) -> Self {
        let tokens_c = CString::new(tokens.to_str().unwrap()).unwrap();
        let provider_c = CString::new(provider.unwrap_or(get_default_provider())).unwrap();
        let graph_c = CString::new(graph.unwrap().to_str().unwrap()).unwrap();

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        // model_config.model_type = zipformer.model_type().into_raw();
        model_config.zipformer2_ctc = zipformer.as_config();
        model_config.tokens = tokens_c.into_raw();
        model_config.num_threads = num_threads.unwrap_or(1);
        model_config.provider = provider_c.into_raw();
        model_config.debug = debug as i32;
        model_config.modeling_unit = CString::new("cjkchar").unwrap().into_raw();

        let mut rec_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineRecognizerConfig>() };
        rec_config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        rec_config.model_config = model_config;
        rec_config.decoding_method = search.into_raw().into_raw();
        rec_config.max_active_paths = 4;
        rec_config.enable_endpoint = 1;
        rec_config.rule1_min_trailing_silence = 2.4;
        rec_config.rule2_min_trailing_silence = 1.2;
        rec_config.rule3_min_utterance_length = 300.0;
        rec_config.ctc_fst_decoder_config = SherpaOnnxOnlineCtcFstDecoderConfig {
            graph: graph_c.into_raw(),
            max_active: 3000,
        };

        if hotwords.is_some() {
            let hotwords_c =
                hotwords.map(|p| CString::new(p.to_str().unwrap()).unwrap().into_raw());
            rec_config.hotwords_file = hotwords_c.unwrap_or(std::ptr::null_mut());
            rec_config.hotwords_score = hotwords_score.unwrap_or(1.5);
        }

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&rec_config) };
        let stream = unsafe { SherpaOnnxCreateOnlineStream(recognizer) };
        // let display = unsafe { SherpaOnnxCreateDisplay()}

        println!("recognizer: {:?}, stream: {:?}", recognizer, stream);
        Self { recognizer, stream }
    }
}

impl Drop for RecognizerStream {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

impl OnlineStream for RecognizerStream {
    fn accept_waveform(&mut self, sample_rate: i32, samples: Vec<f32>) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamAcceptWaveform(
                self.stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
        }
    }

    fn decode_stream(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
        }
    }

    fn is_ready(&mut self) -> bool {
        unsafe { sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) == 1 }
    }

    fn get_result(&mut self) -> String {
        unsafe {
            let result =
                sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            let raw_result = result.read();
            let text = CStr::from_ptr(raw_result.text);
            let text = text.to_str().unwrap().to_string();
            // let timestamps: &[f32] =
            // std::slice::from_raw_parts(raw_result.timestamps, raw_result.count as usize);
            let str = text;
            // Free
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(result);
            str
        }
    }

    fn is_endpoint(&mut self) -> bool {
        unsafe { SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) == 1 }
    }

    fn reset(&mut self) {
        unsafe {
            SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
        }
    }
}
