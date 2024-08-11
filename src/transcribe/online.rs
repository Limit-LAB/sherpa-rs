use std::{
    ffi::{CStr, CString},
    path::Path,
};

use sherpa_rs_sys::{
    SherpaOnnxCreateDisplay, SherpaOnnxCreateOnlineRecognizer, SherpaOnnxCreateOnlineStream,
    SherpaOnnxDecodeOnlineStream, SherpaOnnxDestroyOnlineRecognizer,
    SherpaOnnxDestroyOnlineRecognizerResult, SherpaOnnxDestroyOnlineStream,
    SherpaOnnxFeatureConfig, SherpaOnnxGetOnlineStreamResult, SherpaOnnxIsOnlineStreamReady,
    SherpaOnnxOnlineCtcFstDecoderConfig, SherpaOnnxOnlineModelConfig,
    SherpaOnnxOnlineParaformerModelConfig, SherpaOnnxOnlineRecognizer,
    SherpaOnnxOnlineRecognizerConfig, SherpaOnnxOnlineStream, SherpaOnnxOnlineStreamAcceptWaveform,
    SherpaOnnxOnlineStreamIsEndpoint, SherpaOnnxOnlineStreamReset,
    SherpaOnnxOnlineTransducerModelConfig, SherpaOnnxOnlineZipformer2CtcModelConfig,
};

use crate::get_default_provider;

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

    fn as_config(self) -> SherpaOnnxOnlineParaformerModelConfig {
        SherpaOnnxOnlineParaformerModelConfig {
            encoder: self.encoder.into_raw(),
            decoder: self.decoder.into_raw(),
        }
    }

    fn model_type(&self) -> CString {
        CString::new("paraformer").unwrap()
    }
}

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

/// BPE not supported yet
#[derive(Debug)]
pub struct OnlineRecognizer {
    recognizer: *mut SherpaOnnxOnlineRecognizer,
    stream: *mut SherpaOnnxOnlineStream,
}

impl Drop for OnlineRecognizer {
    fn drop(&mut self) {
        unsafe {
            SherpaOnnxDestroyOnlineStream(self.stream);
            SherpaOnnxDestroyOnlineRecognizer(self.recognizer);
        }
    }
}

impl OnlineRecognizer {
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

    pub fn is_ready(&self) -> bool {
        unsafe { SherpaOnnxIsOnlineStreamReady(self.recognizer, self.stream) != 0 }
    }

    pub fn decode(&self) {
        unsafe {
            SherpaOnnxDecodeOnlineStream(self.recognizer, self.stream);
        }
    }

    pub fn get_result(&self) -> String {
        unsafe {
            let result_ptr = SherpaOnnxGetOnlineStreamResult(self.recognizer, self.stream);
            let raw_result = result_ptr.read();
            let text = CStr::from_ptr(raw_result.text);
            let string = text.to_str().unwrap().to_string();
            SherpaOnnxDestroyOnlineRecognizerResult(result_ptr);
            string
        }
    }

    pub fn is_endpoint(&self) -> bool {
        unsafe { SherpaOnnxOnlineStreamIsEndpoint(self.recognizer, self.stream) != 0 }
    }

    pub fn reset(&self) {
        unsafe {
            SherpaOnnxOnlineStreamReset(self.recognizer, self.stream);
        }
    }

    // pub fn shit() {
    // while (!stop) {
    //     const std::vector<float> &samples = alsa.Read(chunk);
    //     SherpaOnnxOnlineStreamAcceptWaveform(stream, expected_sample_rate,
    //                                          samples.data(), samples.size());
    //     while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    //       SherpaOnnxDecodeOnlineStream(recognizer, stream);
    //     }

    //     const SherpaOnnxOnlineRecognizerResult *r =
    //         SherpaOnnxGetOnlineStreamResult(recognizer, stream);

    //     std::string text = r->text;
    //     SherpaOnnxDestroyOnlineRecognizerResult(r);

    //     if (!text.empty() && last_text != text) {
    //       last_text = text;

    //       std::transform(text.begin(), text.end(), text.begin(),
    //                      [](auto c) { return std::tolower(c); });

    //       SherpaOnnxPrint(display, segment_index, text.c_str());
    //       fflush(stderr);
    //     }

    //     if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
    //       if (!text.empty()) {
    //         ++segment_index;
    //       }
    //       SherpaOnnxOnlineStreamReset(recognizer, stream);
    //     }
    //   }
    // }
}
