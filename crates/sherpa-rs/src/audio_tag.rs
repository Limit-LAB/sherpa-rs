use eyre::{bail, Result};

use crate::{
    get_default_provider,
    utils::{cstr_to_string, cstring_from_str},
};

#[derive(Debug, Clone)]
pub struct AudioTagEvent {
    pub name: String,
    pub index: i32,
    pub prob: f32,
}

#[derive(Debug, Default, Clone)]
pub struct AudioTagConfig {
    pub model: String,
    pub labels: String,
    pub top_k: i32,
    pub ced: Option<String>,
    pub debug: bool,
    pub num_threads: Option<i32>,
    pub provider: Option<String>,
}

pub struct AudioTag {
    audio_tag: *const sherpa_rs_sys::SherpaOnnxAudioTagging,
    config: AudioTagConfig,
}

impl AudioTag {
    pub fn new(config: AudioTagConfig) -> Result<Self> {
        let config_clone = config.clone();

        let model = cstring_from_str(&config.model);
        let ced = cstring_from_str(&config.ced.unwrap_or_default());
        let labels = cstring_from_str(&config.labels);
        let provider = cstring_from_str(&config.provider.unwrap_or(get_default_provider()));

        let sherpa_config = sherpa_rs_sys::SherpaOnnxAudioTaggingConfig {
            model: sherpa_rs_sys::SherpaOnnxAudioTaggingModelConfig {
                zipformer: sherpa_rs_sys::SherpaOnnxOfflineZipformerAudioTaggingModelConfig {
                    model: model.as_ptr(),
                },
                ced: ced.as_ptr(),
                num_threads: config.num_threads.unwrap_or(1),
                debug: config.debug.into(),
                provider: provider.as_ptr(),
            },
            labels: labels.as_ptr(),
            top_k: config.top_k,
        };
        let audio_tag = unsafe { sherpa_rs_sys::SherpaOnnxCreateAudioTagging(&sherpa_config) };

        if audio_tag.is_null() {
            bail!("Failed to create audio tagging");
        }
        Ok(Self {
            audio_tag,
            config: config_clone,
        })
    }

    pub fn compute(
        &mut self,
        samples: impl AsRef<[f32]>,
        sample_rate: u32,
    ) -> Result<Vec<AudioTagEvent>> {
        let mut events = Vec::new();
        unsafe {
            let stream = sherpa_rs_sys::SherpaOnnxAudioTaggingCreateOfflineStream(self.audio_tag);
            if stream.is_null() {
                bail!("Failed to create SherpaOnnxOfflineStream");
            }

            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate as i32,
                samples.as_ref().as_ptr(),
                samples.as_ref().len() as i32,
            );

            let results = sherpa_rs_sys::SherpaOnnxAudioTaggingCompute(
                self.audio_tag,
                stream,
                self.config.top_k,
            );
            if results.is_null() {
                bail!("Failed to compute audio tagging");
            }

            for i in 0..self.config.top_k {
                let event = *results.add(i as _).read();
                events.push(AudioTagEvent {
                    name: cstr_to_string(event.name as _),
                    index: event.index,
                    prob: event.prob,
                });
            }

            sherpa_rs_sys::SherpaOnnxAudioTaggingFreeResults(results);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);
        }
        Ok(events)
    }
}

unsafe impl Send for AudioTag {}
unsafe impl Sync for AudioTag {}

impl Drop for AudioTag {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyAudioTagging(self.audio_tag);
        }
    }
}
