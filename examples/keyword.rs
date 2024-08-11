use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleRate};
use sherpa_rs::keyword_spotting::KeywordSpotter;
use sherpa_rs::transcribe::online::{OnlineRecognizer, Paraformer, Search, Transducer};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), anyhow::Error> {
    let host = cpal::default_host();

    // Set up the input device and stream with the default input config.
    let device = host
        .default_input_device()
        .expect("failed to find input device");

    println!("Input device: {}", device.name()?);
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    let encoder = Path::new(
        "//home/lemonxh/下载/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    );
    let decoder = Path::new(
        "/home/lemonxh/下载/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    );
    let joiner = Path::new(
        "/home/lemonxh/下载/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    );
    let tokens = Path::new(
        "/home/lemonxh/下载/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
    );

    let keywords = Path::new(
        "/home/lemonxh/下载/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/keywords.txt",
    );

    let tr = Transducer::new(encoder, decoder, joiner);
    let spotter =
        KeywordSpotter::from_transducer(tr, Some("cpu"), tokens, false, keywords, None, None);

    println!("Begin recording...");
    let (recorder, receiver) = std::sync::mpsc::channel();
    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };
    let stream = device.build_input_stream_raw(
        &config,
        cpal::SampleFormat::F32,
        move |x, _| {
            let x = x.as_slice().unwrap().to_vec();
            recorder.clone().send(x).unwrap();
        },
        err_fn,
        None,
    )?;
    stream.play()?;

    println!("Creating recognizer...");

    recognizer(spotter, receiver);
}

fn recognizer(online_rec: KeywordSpotter, receiver: Receiver<Vec<f32>>) -> ! {
    let mut last_text = String::new();
    loop {
        let samples = receiver.recv().unwrap();
        online_rec.accept_waveform(16000, samples);

        while online_rec.is_ready() {
            online_rec.decode_stream();
        }

        let result = online_rec.get_result();
        if !result.is_empty() && last_text != result {
            last_text = result.clone();
            println!("{}", result.to_lowercase());
        }
    }
}
