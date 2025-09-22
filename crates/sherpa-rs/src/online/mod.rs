pub mod keyword_spotter;
pub mod paraformer;
pub mod recognizer;
pub mod transducer;
pub mod zipformer2_ctc;

pub trait Stream {
    fn accept_waveform(&mut self, sample_rate: i32, samples: impl AsRef<[f32]>);
    fn decode_stream(&mut self);
    fn get_result(&mut self) -> String;
    fn is_ready(&mut self) -> bool;

    /// better only use for recognizer
    fn is_endpoint(&mut self) -> bool {
        true
    }
    fn reset(&mut self);
}
