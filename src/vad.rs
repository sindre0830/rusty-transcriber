use anyhow::{Context, Result, bail};
use earshot::{VoiceActivityDetector, VoiceActivityProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadSegment {
    pub start_ms: i64,
    pub end_ms: i64,
}

#[derive(Debug, Clone)]
pub struct Vad {
    pub segments: Vec<VadSegment>,
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct VadOptions {
    pub pad_ms: i64,
    pub min_speech_ms: i64,
    pub min_silence_ms_to_split: i64,
    pub merge_gap_ms: i64,
    pub profile: VoiceActivityProfile,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            pad_ms: 80,
            min_speech_ms: 300,
            min_silence_ms_to_split: 200,
            merge_gap_ms: 40,
            profile: VoiceActivityProfile::VERY_AGGRESSIVE,
        }
    }
}

impl Vad {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn from_earshot(
        samples: &[f32],
        sample_rate: u32,
        channels: u8,
        opts: &VadOptions,
    ) -> Result<Self> {
        if sample_rate != 16_000 {
            bail!("earshot VAD only supports 16 kHz sample rate");
        }
        if channels != 1 {
            bail!("earshot VAD only supports mono audio");
        }

        const FRAME_SAMPLES: usize = 320; // 20ms @ 16kHz
        const FRAME_MS: i64 = 20;

        if samples.len() < FRAME_SAMPLES {
            return Ok(Self {
                segments: Vec::new(),
            });
        }

        let audio_ms = ((samples.len() as i64) * 1000) / 16_000;

        let mut vad = VoiceActivityDetector::new(opts.profile.clone());

        let mut segments: Vec<VadSegment> = Vec::new();

        let mut current_start_ms: Option<i64> = None;
        let mut last_speech_frame_end_ms: i64 = 0;
        let mut pending_silence_ms: i64 = 0;

        let mut pcm: [i16; FRAME_SAMPLES] = [0; FRAME_SAMPLES];

        for (frame_idx, frame) in samples.chunks_exact(FRAME_SAMPLES).enumerate() {
            let start_ms = (frame_idx as i64) * FRAME_MS;
            let end_ms = start_ms + FRAME_MS;

            fill_pcm_16k_mono_frame(&mut pcm, frame);

            let is_speech = vad
                .predict_16khz(&pcm)
                .context("earshot VAD failed (predict_16khz)")?;

            match (current_start_ms, is_speech) {
                (None, true) => {
                    current_start_ms = Some(start_ms);
                    last_speech_frame_end_ms = end_ms;
                    pending_silence_ms = 0;
                }
                (Some(_), true) => {
                    last_speech_frame_end_ms = end_ms;
                    pending_silence_ms = 0;
                }
                (Some(_), false) => {
                    pending_silence_ms += FRAME_MS;

                    if pending_silence_ms >= opts.min_silence_ms_to_split {
                        let seg_start = current_start_ms.take().expect("segment start exists");
                        let seg_end = last_speech_frame_end_ms;

                        push_segment_with_padding(
                            &mut segments,
                            seg_start,
                            seg_end,
                            audio_ms,
                            opts.pad_ms,
                            opts.min_speech_ms,
                        );

                        pending_silence_ms = 0;
                    }
                }
                (None, false) => {
                    // still silence
                }
            }
        }

        // flush last segment
        if let Some(seg_start) = current_start_ms.take() {
            let seg_end = last_speech_frame_end_ms;
            push_segment_with_padding(
                &mut segments,
                seg_start,
                seg_end,
                audio_ms,
                opts.pad_ms,
                opts.min_speech_ms,
            );
        }

        // normalize
        segments.sort_by_key(|s| s.start_ms);
        segments = merge_overlapping_or_close(segments, opts.merge_gap_ms);

        for s in &mut segments {
            if s.start_ms < 0 {
                s.start_ms = 0;
            }
            if s.end_ms > audio_ms {
                s.end_ms = audio_ms;
            }
        }
        segments.retain(|s| s.start_ms >= 0 && s.start_ms < s.end_ms);

        Ok(Self { segments })
    }
}

fn fill_pcm_16k_mono_frame(dst: &mut [i16; 320], src: &[f32]) {
    debug_assert_eq!(src.len(), 320);

    for (i, &s) in src.iter().enumerate() {
        let mut x = if s.is_finite() { s } else { 0.0 };

        x = x.clamp(-1.0, 1.0);

        let scaled = (x * i16::MAX as f32).round();
        let scaled_i32 = scaled as i32;

        dst[i] = scaled_i32.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

fn push_segment_with_padding(
    out: &mut Vec<VadSegment>,
    start_ms: i64,
    end_ms: i64,
    audio_ms: i64,
    pad_ms: i64,
    min_speech_ms: i64,
) {
    if end_ms <= start_ms {
        return;
    }

    let padded_start = (start_ms - pad_ms).max(0);
    let padded_end = (end_ms + pad_ms).min(audio_ms);

    if padded_end <= padded_start {
        return;
    }

    if padded_end - padded_start < min_speech_ms {
        return;
    }

    out.push(VadSegment {
        start_ms: padded_start,
        end_ms: padded_end,
    });
}

fn merge_overlapping_or_close(mut segments: Vec<VadSegment>, merge_gap_ms: i64) -> Vec<VadSegment> {
    if segments.is_empty() {
        return segments;
    }

    segments.sort_by_key(|s| s.start_ms);

    let mut merged: Vec<VadSegment> = Vec::with_capacity(segments.len());
    merged.push(segments[0]);

    for seg in segments.into_iter().skip(1) {
        let last = merged.last_mut().expect("non-empty");

        if seg.start_ms <= last.end_ms + merge_gap_ms {
            last.end_ms = last.end_ms.max(seg.end_ms);
        } else {
            merged.push(seg);
        }
    }

    merged
}
