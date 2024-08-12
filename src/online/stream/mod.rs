pub mod keyword_spotter;
pub mod recognizer;

use std::ffi::CStr;

use sherpa_rs_sys::{SherpaOnnxOnlineStreamIsEndpoint, SherpaOnnxOnlineStreamReset};

pub trait OnlineStream {
    fn accept_waveform(&mut self, sample_rate: i32, samples: Vec<f32>);
    fn decode_stream(&mut self);
    fn get_result(&mut self) -> String;
    fn is_ready(&mut self) -> bool;

    /// better only use for recognizer
    fn is_endpoint(&mut self) -> bool {
        true
    }
    /// better only use for recognizer
    fn reset(&mut self) {}
}
