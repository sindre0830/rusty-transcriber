use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::diarization::Diarization;
use crate::io::{
    ModelInput, RetrieveBinaryOptions, retrieve_batch_files, retrieve_binary, samples_fingerprint,
};
use crate::stt::{PostProcessContext, Stt};
use crate::vad::{Vad, VadOptions};

mod diarization;
pub mod io;
mod stt;
mod utils;
mod vad;

#[derive(Clone, Debug)]
pub struct TranscriptOptions {
    pub cache_dir: PathBuf,
}

impl TranscriptOptions {
    pub fn new() -> Self {
        Self {
            cache_dir: PathBuf::from(".cache").join("rusty_transcriber"),
        }
    }
}

impl Default for TranscriptOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub speaker_id: usize,
}

pub struct Transcript {
    pub segments: Vec<TranscriptSegment>,
    options: TranscriptOptions,
    transcriber_model_path: Option<PathBuf>,
    diarization_model_path: Option<PathBuf>,
}

impl Transcript {
    pub fn new(options: TranscriptOptions) -> Self {
        Self {
            segments: Vec::new(),
            options,
            transcriber_model_path: None,
            diarization_model_path: None,
        }
    }

    pub fn prepare_transcriber_model(
        mut self,
        input: ModelInput,
        model_options: RetrieveBinaryOptions,
    ) -> Result<Self> {
        let model_path = match input {
            ModelInput::Path(p) => p,
            ModelInput::Url(u) => retrieve_binary(&u, &model_options)?,
            ModelInput::BatchUrls(urls) => retrieve_batch_files(&urls, &model_options)?,
        };

        self.transcriber_model_path = Some(model_path);
        Ok(self)
    }

    pub fn prepare_diarization_model(
        mut self,
        input: ModelInput,
        model_options: RetrieveBinaryOptions,
    ) -> Result<Self> {
        let model_path = match input {
            ModelInput::Path(p) => p,
            ModelInput::Url(u) => retrieve_binary(&u, &model_options)?,
            ModelInput::BatchUrls(urls) => retrieve_batch_files(&urls, &model_options)?,
        };

        self.diarization_model_path = Some(model_path);
        Ok(self)
    }

    pub fn transcribe(mut self, samples: &[f32], sample_rate: u32, channels: u8) -> Result<Self> {
        let transcriber_model_path = self
            .transcriber_model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("transcriber model not prepared"))?;

        let diarization_model_path = self
            .diarization_model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("diarization model not prepared"))?;

        let audio_fp = samples_fingerprint(samples);
        let path = cache_path(&self.options.cache_dir, &audio_fp);

        if let Some(cached) = try_load_cached_segments(&path)? {
            self.segments = cached;
            return Ok(self);
        }

        let vad = Vad::from_earshot(samples, sample_rate, channels, &VadOptions::default())?;

        let diarization = Diarization::new()
            .from_sortformer(samples, sample_rate, channels, diarization_model_path)?
            .post_process_default_with_vad(&vad.segments);

        let stt = Stt::new()
            .from_parakeet_tdt(
                samples,
                sample_rate,
                channels,
                transcriber_model_path,
                &vad.segments,
            )?
            .post_process_default(PostProcessContext {
                diarization: Some(&diarization.segments),
                vad: Some(&vad.segments),
            });

        self.segments = stt
            .segments
            .into_iter()
            .map(|seg| {
                let speaker_id = diarization
                    .speaker_for_window(seg.start_ms, seg.end_ms)
                    .unwrap_or(usize::MAX);

                TranscriptSegment {
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    text: seg.text,
                    speaker_id,
                }
            })
            .collect();

        store_cached_segments(&path, &self.segments)?;

        Ok(self)
    }
}

fn cache_path(cache_dir: &Path, key: &str) -> PathBuf {
    let mut h = blake3::Hasher::new();
    h.update(key.as_bytes());
    let name = h.finalize().to_hex().to_string();
    cache_dir.join("transcripts").join(format!("{}.json", name))
}

fn try_load_cached_segments(path: &Path) -> Result<Option<Vec<TranscriptSegment>>> {
    if !path.exists() {
        return Ok(None);
    }

    let data = fs::read(path).context("failed to read transcript cache file")?;
    let segments: Vec<TranscriptSegment> =
        serde_json::from_slice(&data).context("failed to parse transcript cache json")?;
    Ok(Some(segments))
}

fn store_cached_segments(path: &Path, segments: &[TranscriptSegment]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("failed to create transcript cache directory")?;
    }

    let data = serde_json::to_vec(segments).context("failed to serialize transcript cache json")?;
    fs::write(path, data).context("failed to write transcript cache file")?;
    Ok(())
}
