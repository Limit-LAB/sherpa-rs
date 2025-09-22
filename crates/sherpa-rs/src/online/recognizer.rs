use std::{
    ffi::{CStr, CString},
    path::Path,
};

use eyre::Result;
use sherpa_rs_sys::{
    SherpaOnnxCreateOnlineRecognizer, SherpaOnnxCreateOnlineStream, SherpaOnnxFeatureConfig,
    SherpaOnnxOnlineCtcFstDecoderConfig, SherpaOnnxOnlineModelConfig,
    SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineStreamIsEndpoint,
    SherpaOnnxOnlineStreamReset,
};

use crate::{
    online::{paraformer::Paraformer, transducer::Transducer, zipformer2_ctc::Zipformer2Ctc},
    OnnxConfig,
};

use super::Stream;

use std::sync::Arc;

#[derive(Debug)]
pub enum Search {
    Greedy,
    Beam,
}

impl Search {
    fn to_cstring(&self) -> CString {
        match self {
            Self::Greedy => CString::new("greedy_search").unwrap(),
            Self::Beam => CString::new("modified_beam_search").unwrap(),
        }
    }
}

#[derive(Debug)]
pub struct Recognizer {
    recognizer: *const sherpa_rs_sys::SherpaOnnxOnlineRecognizer,
}

impl Recognizer {
    pub fn from_transducer(
        transducer: Transducer,
        tokens: &Path,
        search: Search,

        onnx_config: OnnxConfig,
        hotwords: Option<&Path>,
        hotwords_score: Option<f32>,
    ) -> Result<Arc<Self>> {
        let tokens_c = CString::new(tokens.to_str().unwrap()).unwrap();
        let provider_c = CString::new(onnx_config.provider).unwrap();
        let decoding_method = search.to_cstring();
        let modeling_unit_c = CString::new("cjkchar").unwrap();
        let hotwords_c =
            hotwords.map(|p| CString::new(p.to_str().unwrap_or(&p.to_string_lossy())).unwrap());

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        model_config.transducer = unsafe { transducer.as_config() };
        model_config.tokens = tokens_c.as_ptr();
        model_config.num_threads = onnx_config.num_threads;
        model_config.provider = provider_c.as_ptr();
        model_config.debug = onnx_config.debug as i32;
        model_config.modeling_unit = modeling_unit_c.as_ptr();

        let mut rec_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineRecognizerConfig>() };
        rec_config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        rec_config.model_config = model_config;
        rec_config.decoding_method = decoding_method.as_ptr();
        rec_config.max_active_paths = 4;
        rec_config.enable_endpoint = 1;
        rec_config.rule1_min_trailing_silence = 2.4;
        rec_config.rule2_min_trailing_silence = 1.2;
        rec_config.rule3_min_utterance_length = 300.0;

        if let Some(ref hw) = hotwords_c {
            rec_config.hotwords_file = hw.as_ptr();
            rec_config.hotwords_score = hotwords_score.unwrap_or(1.5);
        }

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&rec_config) };
        if recognizer.is_null() {
            eyre::bail!("Failed to create recognizer");
        }
        Ok(Arc::new(Self { recognizer }))
    }

    pub fn from_paraformer(
        paraformer: Paraformer,
        tokens: &Path,
        search: Search,

        onnx_config: OnnxConfig,
        hotwords: Option<&Path>,
        hotwords_score: Option<f32>,
    ) -> Result<Arc<Self>> {
        let tokens_c = CString::new(tokens.to_str().unwrap()).unwrap();
        let provider_c = CString::new(onnx_config.provider).unwrap();
        let decoding_method_c = search.to_cstring();
        let model_type_c = paraformer.model_type();
        let modeling_unit_c = CString::new("cjkchar").unwrap();
        let hotwords_c =
            hotwords.map(|p| CString::new(p.to_str().unwrap_or(&p.to_string_lossy())).unwrap());

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        model_config.model_type = model_type_c.as_ptr();
        model_config.paraformer = unsafe { paraformer.as_config() };
        model_config.tokens = tokens_c.as_ptr();
        model_config.num_threads = onnx_config.num_threads;
        model_config.provider = provider_c.as_ptr();
        model_config.debug = onnx_config.debug as i32;
        model_config.modeling_unit = modeling_unit_c.as_ptr();

        let mut rec_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineRecognizerConfig>() };
        rec_config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        rec_config.model_config = model_config;
        rec_config.decoding_method = decoding_method_c.as_ptr();
        rec_config.max_active_paths = 4;
        rec_config.enable_endpoint = 1;
        rec_config.rule1_min_trailing_silence = 2.4;
        rec_config.rule2_min_trailing_silence = 1.2;
        rec_config.rule3_min_utterance_length = 300.0;

        if let Some(ref hw) = hotwords_c {
            rec_config.hotwords_file = hw.as_ptr();
            rec_config.hotwords_score = hotwords_score.unwrap_or(1.5);
        }

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&rec_config) };
        if recognizer.is_null() {
            eyre::bail!("Failed to create recognizer");
        }

        Ok(Arc::new(Self { recognizer }))
    }

    pub fn from_zipformer(
        zipformer: Zipformer2Ctc,
        tokens: &Path,
        graph: &Path,
        search: Search,

        onnx_config: OnnxConfig,
        hotwords: Option<&Path>,
        hotwords_score: Option<f32>,
    ) -> Result<Arc<Self>> {
        let tokens_c = CString::new(tokens.to_str().unwrap_or(&tokens.to_string_lossy())).unwrap();
        let provider_c = CString::new(onnx_config.provider).unwrap();
        let graph_c = CString::new(graph.to_str().unwrap_or(&graph.to_string_lossy())).unwrap();
        let hotwords_c =
            hotwords.map(|p| CString::new(p.to_str().unwrap_or(&p.to_string_lossy())).unwrap());
        let decoding_method_c = search.to_cstring();
        let modeling_unit_c = CString::new("cjkchar").unwrap();

        let mut model_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineModelConfig>() };
        model_config.zipformer2_ctc = unsafe { zipformer.as_config() };
        model_config.tokens = tokens_c.as_ptr();
        model_config.num_threads = onnx_config.num_threads;
        model_config.provider = provider_c.as_ptr();
        model_config.debug = onnx_config.debug as i32;
        model_config.modeling_unit = modeling_unit_c.as_ptr();

        let mut rec_config = unsafe { std::mem::zeroed::<SherpaOnnxOnlineRecognizerConfig>() };
        rec_config.feat_config = SherpaOnnxFeatureConfig {
            sample_rate: 16000,
            feature_dim: 80,
        };
        rec_config.model_config = model_config;
        rec_config.decoding_method = decoding_method_c.as_ptr();
        rec_config.max_active_paths = 4;
        rec_config.enable_endpoint = 1;
        rec_config.rule1_min_trailing_silence = 2.4;
        rec_config.rule2_min_trailing_silence = 1.2;
        rec_config.rule3_min_utterance_length = 300.0;
        rec_config.ctc_fst_decoder_config = SherpaOnnxOnlineCtcFstDecoderConfig {
            graph: graph_c.as_ptr(),
            max_active: 3000,
        };

        if let Some(ref hw) = hotwords_c {
            rec_config.hotwords_file = hw.as_ptr();
            rec_config.hotwords_score = hotwords_score.unwrap_or(1.5);
        }

        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&rec_config) };
        if recognizer.is_null() {
            eyre::bail!("Failed to create recognizer");
        }

        Ok(Arc::new(Self { recognizer }))
    }

    pub fn create_stream(self: &Arc<Self>) -> Result<RecognizerStream> {
        RecognizerStream::new(Arc::clone(self))
    }
}

unsafe impl Send for Recognizer {}
unsafe impl Sync for Recognizer {}

impl Drop for Recognizer {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

pub struct RecognizerStream {
    recognizer: Arc<Recognizer>,
    stream: *const sherpa_rs_sys::SherpaOnnxOnlineStream,
}

impl RecognizerStream {
    pub fn new(recognizer: Arc<Recognizer>) -> Result<Self> {
        let stream = unsafe { SherpaOnnxCreateOnlineStream(recognizer.recognizer) };
        if stream.is_null() {
            eyre::bail!("Failed to create stream");
        }

        Ok(Self { recognizer, stream })
    }
}

impl Drop for RecognizerStream {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOnlineStream(self.stream);
        }
    }
}

impl Stream for RecognizerStream {
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
            sherpa_rs_sys::SherpaOnnxDecodeOnlineStream(self.recognizer.recognizer, self.stream);
        }
    }

    fn is_ready(&mut self) -> bool {
        unsafe {
            sherpa_rs_sys::SherpaOnnxIsOnlineStreamReady(self.recognizer.recognizer, self.stream)
                == 1
        }
    }

    fn get_result(&mut self) -> String {
        unsafe {
            let raw_result = sherpa_rs_sys::SherpaOnnxGetOnlineStreamResult(
                self.recognizer.recognizer,
                self.stream,
            );
            if raw_result.is_null() {
                return String::new();
            }
            let result = raw_result.read();
            let text = CStr::from_ptr(result.text).to_string_lossy().to_string();

            sherpa_rs_sys::SherpaOnnxDestroyOnlineRecognizerResult(raw_result);
            text
        }
    }

    fn is_endpoint(&mut self) -> bool {
        unsafe { SherpaOnnxOnlineStreamIsEndpoint(self.recognizer.recognizer, self.stream) == 1 }
    }

    fn reset(&mut self) {
        unsafe {
            SherpaOnnxOnlineStreamReset(self.recognizer.recognizer, self.stream);
        }
    }

    fn finish(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxOnlineStreamInputFinished(self.stream);
        }
    }
}
