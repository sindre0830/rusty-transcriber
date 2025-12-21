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

impl Vad {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Detect speech regions using `earshot` VAD.
    ///
    /// Assumptions:
    /// - `samples` is mono 16 kHz audio
    /// - `samples` are in [-1.0, 1.0]
    pub fn from_earshot(mut self, samples: &[f32]) -> Result<Self> {
        // 20ms frames at 16 kHz => 320 samples
        const FRAME_SAMPLES: usize = 320;
        const FRAME_MS: i64 = 20;

        // tweakables
        const PAD_MS: i64 = 80; // expand around detected speech
        const MIN_SPEECH_MS: i64 = 300; // drop tiny speech bursts
        const MIN_SILENCE_MS_TO_SPLIT: i64 = 200; // silence required to split segments

        if samples.len() < FRAME_SAMPLES {
            self.segments.clear();
            return Ok(self);
        }

        let mut vad = VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE);

        let mut current_start_ms: Option<i64> = None;
        let mut last_speech_frame_end_ms: i64 = 0;
        let mut pending_silence_ms: i64 = 0;

        for (frame_idx, frame) in samples.chunks_exact(FRAME_SAMPLES).enumerate() {
            let start_ms = (frame_idx as i64) * FRAME_MS;
            let end_ms = start_ms + FRAME_MS;

            let mut pcm: [i16; FRAME_SAMPLES] = [0; FRAME_SAMPLES];
            for (i, s) in frame.iter().copied().enumerate() {
                // clamp, then scale
                let clamped = s.max(-1.0).min(1.0);
                pcm[i] = (clamped * i16::MAX as f32) as i16;
            }

            let is_speech = vad
                .predict_16khz(&pcm)
                .context("earshot VAD failed (predict_16khz)")?;

            match (current_start_ms, is_speech) {
                (None, true) => {
                    // start a new speech region
                    current_start_ms = Some(start_ms);
                    last_speech_frame_end_ms = end_ms;
                    pending_silence_ms = 0;
                }
                (Some(_), true) => {
                    // continue speech
                    last_speech_frame_end_ms = end_ms;
                    pending_silence_ms = 0;
                }
                (Some(_), false) => {
                    // inside a speech region, but current frame is silence
                    pending_silence_ms += FRAME_MS;

                    // if silence is long enough, close the segment
                    if pending_silence_ms >= MIN_SILENCE_MS_TO_SPLIT {
                        let seg_start = current_start_ms.take().expect("segment start exists");
                        let seg_end = last_speech_frame_end_ms;

                        push_segment_with_padding(
                            &mut self.segments,
                            seg_start,
                            seg_end,
                            PAD_MS,
                            MIN_SPEECH_MS,
                        );

                        pending_silence_ms = 0;
                    }
                }
                (None, false) => {
                    // still silence
                }
            }
        }

        // flush last open segment (if any)
        if let Some(seg_start) = current_start_ms.take() {
            let seg_end = last_speech_frame_end_ms;
            push_segment_with_padding(
                &mut self.segments,
                seg_start,
                seg_end,
                PAD_MS,
                MIN_SPEECH_MS,
            );
        }

        // normalize: sort + merge overlaps (padding can create overlap)
        self.segments.sort_by_key(|s| s.start_ms);
        self.segments = merge_overlapping(self.segments);

        // sanity: non-negative, increasing
        for s in &self.segments {
            if s.start_ms >= s.end_ms {
                bail!("invalid VAD segment: start_ms >= end_ms");
            }
        }

        Ok(self)
    }
}

fn push_segment_with_padding(
    out: &mut Vec<VadSegment>,
    start_ms: i64,
    end_ms: i64,
    pad_ms: i64,
    min_speech_ms: i64,
) {
    let padded_start = (start_ms - pad_ms).max(0);
    let padded_end = end_ms + pad_ms;

    if padded_end - padded_start < min_speech_ms {
        return;
    }

    out.push(VadSegment {
        start_ms: padded_start,
        end_ms: padded_end,
    });
}

fn merge_overlapping(mut segments: Vec<VadSegment>) -> Vec<VadSegment> {
    if segments.is_empty() {
        return segments;
    }

    segments.sort_by_key(|s| s.start_ms);

    let mut merged: Vec<VadSegment> = Vec::with_capacity(segments.len());
    merged.push(segments[0]);

    for seg in segments.into_iter().skip(1) {
        let last = merged.last_mut().expect("non-empty");
        if seg.start_ms <= last.end_ms {
            last.end_ms = last.end_ms.max(seg.end_ms);
        } else {
            merged.push(seg);
        }
    }

    merged
}
