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

#[derive(Debug, Clone, Default)]
pub struct Diarization {
    pub segments: Vec<DiarizationSegment>,
}

#[derive(Debug, Clone, Copy)]
pub struct DiarizationOptions {
    pub min_segment_ms: i64,
    pub merge_same_speaker_gap_ms: i64,
    pub min_vad_overlap_ratio: f32,
}

impl Default for DiarizationOptions {
    fn default() -> Self {
        Self {
            min_segment_ms: 300,
            merge_same_speaker_gap_ms: 1000,
            min_vad_overlap_ratio: 0.25,
        }
    }
}

impl Diarization {
    pub fn from_sortformer(
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

        let mut out = Self {
            segments: segments
                .into_iter()
                .map(|s| DiarizationSegment {
                    start_ms: seconds_to_ms(s.start),
                    end_ms: seconds_to_ms(s.end),
                    speaker_id: s.speaker_id,
                })
                .collect(),
        };

        out.normalize_basic();
        Ok(out)
    }

    pub fn sort_by_start(mut self) -> Self {
        if self.segments.len() > 1 {
            self.segments.sort_by_key(|s| s.start_ms);
        }
        self
    }

    pub fn drop_short(mut self, min_duration_ms: i64) -> Self {
        self.segments
            .retain(|s| (s.end_ms - s.start_ms) >= min_duration_ms);
        self
    }

    pub fn merge_same_speaker_gaps(mut self, max_gap_ms: i64) -> Self {
        if self.segments.len() < 2 {
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut merged: Vec<DiarizationSegment> = Vec::with_capacity(self.segments.len());

        for seg in self.segments.into_iter() {
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

    pub fn drop_contained_segments(mut self) -> Self {
        if self.segments.len() < 2 {
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut result: Vec<DiarizationSegment> = Vec::with_capacity(self.segments.len());

        for seg in self.segments.into_iter() {
            if let Some(last) = result.last_mut() {
                let fully_inside = seg.start_ms >= last.start_ms && seg.end_ms <= last.end_ms;

                if fully_inside {
                    if seg.speaker_id == last.speaker_id {
                        last.end_ms = last.end_ms.max(seg.end_ms);
                    }
                    continue;
                }
            }

            result.push(seg);
        }

        self.segments = result;
        self
    }

    pub fn normalize_speaker_ids(mut self) -> Self {
        if self.segments.is_empty() {
            return self;
        }

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

    pub fn trim_to_vad(mut self, vad: &[VadSegment]) -> Self {
        if vad.is_empty() || self.segments.is_empty() {
            self.segments.clear();
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut vad_sorted: Vec<VadSegment> = vad.to_vec();
        vad_sorted.sort_by_key(|s| s.start_ms);

        let mut out: Vec<DiarizationSegment> = Vec::with_capacity(self.segments.len());

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
        self.normalize_basic();
        self
    }

    pub fn drop_low_vad_overlap(mut self, vad: &[VadSegment], min_ratio: f32) -> Self {
        if vad.is_empty() || self.segments.is_empty() {
            self.segments.clear();
            return self;
        }

        self.segments.sort_by_key(|s| s.start_ms);

        let mut vad_sorted: Vec<VadSegment> = vad.to_vec();
        vad_sorted.sort_by_key(|s| s.start_ms);

        let mut j = 0usize;

        self.segments.retain(|d| {
            let dur = d.end_ms - d.start_ms;
            if dur <= 0 {
                return false;
            }

            while j < vad_sorted.len() && vad_sorted[j].end_ms <= d.start_ms {
                j += 1;
            }

            let mut k = j;
            let mut overlap = 0i64;

            while k < vad_sorted.len() {
                let v = &vad_sorted[k];

                if v.start_ms >= d.end_ms {
                    break;
                }

                let lo = d.start_ms.max(v.start_ms);
                let hi = d.end_ms.min(v.end_ms);

                if hi > lo {
                    overlap += hi - lo;
                }

                k += 1;
            }

            (overlap as f32) / (dur as f32) >= min_ratio
        });

        self
    }

    pub fn post_process(self, vad: &[VadSegment], opts: &DiarizationOptions) -> Self {
        self.sort_by_start()
            .drop_short(opts.min_segment_ms)
            .trim_to_vad(vad)
            .drop_low_vad_overlap(vad, opts.min_vad_overlap_ratio)
            .merge_same_speaker_gaps(opts.merge_same_speaker_gap_ms)
            .drop_contained_segments()
            .normalize_speaker_ids()
    }

    pub fn speaker_for_window(&self, start_ms: i64, end_ms: i64) -> Option<usize> {
        if start_ms >= end_ms || self.segments.is_empty() {
            return None;
        }

        let mut overlap_per_speaker: HashMap<usize, i64> = HashMap::new();
        let mut best_speaker: Option<usize> = None;
        let mut best_overlap: i64 = 0;

        for seg in &self.segments {
            if seg.end_ms <= start_ms {
                continue;
            }
            if seg.start_ms >= end_ms {
                break;
            }

            let overlap_start = start_ms.max(seg.start_ms);
            let overlap_end = end_ms.min(seg.end_ms);

            if overlap_end <= overlap_start {
                continue;
            }

            let overlap = overlap_end - overlap_start;
            let entry = overlap_per_speaker.entry(seg.speaker_id).or_insert(0);
            *entry += overlap;

            if *entry > best_overlap {
                best_overlap = *entry;
                best_speaker = Some(seg.speaker_id);
            }
        }

        best_speaker
    }

    fn normalize_basic(&mut self) {
        self.segments.retain(|s| s.start_ms < s.end_ms);

        if self.segments.len() > 1 {
            self.segments.sort_by_key(|s| s.start_ms);
        }
    }
}
