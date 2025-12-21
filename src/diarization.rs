use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use parakeet_rs::sortformer::Sortformer;

use crate::utils::seconds_to_ms;
use crate::vad::VadSegment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiarizationSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker_id: usize,
}

#[derive(Debug, Clone)]
pub struct Diarization {
    pub segments: Vec<DiarizationSegment>,
}

impl Diarization {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn from_sortformer(
        mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: u8,
        model_path: &Path,
    ) -> Result<Self> {
        let mut diarizer =
            Sortformer::new(model_path).context("failed to load sortformer diarization model")?;

        let segments = diarizer
            .diarize(samples.to_vec(), sample_rate, channels as u16)
            .context("sortformer diarization failed")?;

        self.segments = segments
            .into_iter()
            .map(|s| DiarizationSegment {
                start_ms: seconds_to_ms(s.start),
                end_ms: seconds_to_ms(s.end),
                speaker_id: s.speaker_id,
            })
            .collect();
        Ok(self)
    }

    pub fn sort_by_start(mut self) -> Self {
        self.segments.sort_by_key(|s| s.start_ms);
        self
    }

    pub fn drop_short(mut self, min_duration_ms: i64) -> Self {
        self.segments
            .retain(|s| s.end_ms - s.start_ms >= min_duration_ms);
        self
    }

    pub fn merge_same_speaker_gaps(mut self, max_gap_ms: i64) -> Self {
        self.segments.sort_by_key(|s| s.start_ms);

        let mut merged: Vec<DiarizationSegment> = Vec::with_capacity(self.segments.len());

        for seg in self.segments {
            if let Some(last) = merged.last_mut() {
                let gap = seg.start_ms - last.end_ms;

                if last.speaker_id == seg.speaker_id && gap <= max_gap_ms {
                    last.end_ms = last.end_ms.max(seg.end_ms);
                    continue;
                }
            }

            merged.push(seg);
        }

        self.segments = merged;
        self
    }

    /// This enforces monophonic diarization (one speaker at a time).
    pub fn drop_contained_segments(mut self) -> Self {
        if self.segments.len() < 2 {
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut result: Vec<DiarizationSegment> = Vec::with_capacity(self.segments.len());

        for seg in self.segments {
            if let Some(last) = result.last_mut() {
                let fully_inside = seg.start_ms >= last.start_ms && seg.end_ms <= last.end_ms;

                if fully_inside {
                    // same speaker: merge defensively
                    if seg.speaker_id == last.speaker_id {
                        last.end_ms = last.end_ms.max(seg.end_ms);
                    }
                    // different speaker: drop inner segment
                    continue;
                }
            }

            result.push(seg);
        }

        self.segments = result;
        self
    }

    /// Re-index speakers to 0..N by first appearance (better UX / stable output).
    pub fn normalize_speaker_ids(mut self) -> Self {
        self.segments.sort_by_key(|s| s.start_ms);

        let mut map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0usize;

        for seg in &mut self.segments {
            let new_id = *map.entry(seg.speaker_id).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });

            seg.speaker_id = new_id;
        }

        self
    }

    /// Intersects diarization segments with VAD speech segments.
    /// This trims diarization to only regions where speech exists.
    pub fn trim_to_vad(mut self, vad: &[VadSegment]) -> Self {
        if self.segments.is_empty() || vad.is_empty() {
            self.segments.clear();
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut vad_sorted: Vec<VadSegment> = vad.to_vec();
        vad_sorted.sort_by_key(|s| s.start_ms);

        let mut out: Vec<DiarizationSegment> = Vec::new();

        let mut i = 0usize;
        let mut j = 0usize;

        while i < self.segments.len() && j < vad_sorted.len() {
            let d = self.segments[i];
            let v = vad_sorted[j];

            if d.end_ms <= v.start_ms {
                i += 1;
                continue;
            }
            if v.end_ms <= d.start_ms {
                j += 1;
                continue;
            }

            let start = d.start_ms.max(v.start_ms);
            let end = d.end_ms.min(v.end_ms);

            if end > start {
                out.push(DiarizationSegment {
                    start_ms: start,
                    end_ms: end,
                    speaker_id: d.speaker_id,
                });
            }

            if d.end_ms <= v.end_ms {
                i += 1;
            } else {
                j += 1;
            }
        }

        self.segments = out;
        self
    }

    /// Drops diarization segments that contain too little speech according to VAD.
    /// `min_ratio` is overlap_ms / diar_dur_ms, recommended 0.20..0.35.
    pub fn drop_low_vad_overlap(mut self, vad: &[VadSegment], min_ratio: f32) -> Self {
        if self.segments.is_empty() || vad.is_empty() {
            self.segments.clear();
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut vad_sorted: Vec<VadSegment> = vad.to_vec();
        vad_sorted.sort_by_key(|s| s.start_ms);

        self.segments.retain(|d| {
            let dur = (d.end_ms - d.start_ms).max(0);
            if dur <= 0 {
                return false;
            }

            let overlap = total_overlap_ms(d.start_ms, d.end_ms, &vad_sorted);
            (overlap as f32) / (dur as f32) >= min_ratio
        });

        self
    }

    /// VAD-aware default post-processing chain.
    /// Trims diarization to speech regions and optionally drops low-overlap segments.
    pub fn post_process_default_with_vad(self, vad: &[VadSegment]) -> Self {
        self.sort_by_start()
            .drop_short(300)
            .trim_to_vad(vad)
            .drop_low_vad_overlap(vad, 0.25)
            .merge_same_speaker_gaps(1000)
            .drop_contained_segments()
            .normalize_speaker_ids()
    }

    /// Returns the speaker_id that overlaps most with the given time window.
    /// If no overlap exists, returns None.
    pub fn speaker_for_window(&self, start_ms: i64, end_ms: i64) -> Option<usize> {
        if start_ms >= end_ms {
            return None;
        }

        let mut overlap_per_speaker: HashMap<usize, i64> = HashMap::new();

        for seg in &self.segments {
            let overlap_start = start_ms.max(seg.start_ms);
            let overlap_end = end_ms.min(seg.end_ms);

            if overlap_end > overlap_start {
                let overlap = overlap_end - overlap_start;
                *overlap_per_speaker.entry(seg.speaker_id).or_insert(0) += overlap;
            }
        }

        overlap_per_speaker
            .into_iter()
            .max_by_key(|(_, overlap)| *overlap)
            .map(|(speaker_id, _)| speaker_id)
    }
}

fn total_overlap_ms(start_ms: i64, end_ms: i64, vad: &[VadSegment]) -> i64 {
    if start_ms >= end_ms {
        return 0;
    }

    let mut total = 0i64;

    for v in vad {
        if v.end_ms <= start_ms {
            continue;
        }
        if v.start_ms >= end_ms {
            break;
        }

        let lo = start_ms.max(v.start_ms);
        let hi = end_ms.min(v.end_ms);

        if hi > lo {
            total += hi - lo;
        }
    }

    total
}
