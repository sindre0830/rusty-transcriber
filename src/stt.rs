use std::path::Path;

use anyhow::{Context, Result};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use rusty_paragraphizer::ParagraphGrouper;

use crate::diarization::DiarizationSegment;
use crate::utils::seconds_to_ms;
use crate::vad::VadSegment;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Stt {
    pub segments: Vec<SttSegment>,
}

#[derive(Debug, Clone, Copy)]
pub struct SttOptions {
    pub vad_window_pad_ms: i64,
    pub vad_window_merge_gap_ms: i64,
    pub max_window_ms: i64,
    pub timestamp_mode: TimestampMode,

    pub merge_sentence_fragments_gap_ms: i64,
    pub merge_fragmented_segments_gap_ms: i64,
    pub merge_dangling_segments_gap_ms: i64,

    pub diar_boundary_overlap_segments: u32,
    pub diar_boundary_min_overlap_ms: i64,

    pub require_same_speaker_for_merges: bool,
    pub speaker_min_overlap_ms: i64,
    pub speaker_min_overlap_ratio: f32,

    pub enable_finance_normalization: bool,
    pub enable_discourse_cleanup: bool,
    pub phrase_dedupe_max_ngram: usize,

    pub remove_fillers: bool,
    pub filler_max_len: usize,

    pub enable_paragraphs: bool,
    pub paragraph_split_threshold: f32,
    pub paragraph_min_sentences: usize,
    pub paragraph_rank_window: usize,
    pub paragraph_block_depth: usize,
}

impl Default for SttOptions {
    fn default() -> Self {
        Self {
            vad_window_pad_ms: 120,
            vad_window_merge_gap_ms: 250,
            max_window_ms: 35_000,
            timestamp_mode: TimestampMode::Sentences,

            merge_sentence_fragments_gap_ms: 800,
            merge_fragmented_segments_gap_ms: 600,
            merge_dangling_segments_gap_ms: 1200,

            diar_boundary_overlap_segments: 2,
            diar_boundary_min_overlap_ms: 200,

            require_same_speaker_for_merges: true,
            speaker_min_overlap_ms: 300,
            speaker_min_overlap_ratio: 0.35,

            enable_finance_normalization: true,
            enable_discourse_cleanup: true,
            phrase_dedupe_max_ngram: 4,

            remove_fillers: true,
            filler_max_len: 4,

            enable_paragraphs: true,
            paragraph_split_threshold: 0.55,
            paragraph_min_sentences: 2,
            paragraph_rank_window: 2,
            paragraph_block_depth: 2,
        }
    }
}

impl Stt {
    pub fn from_parakeet_tdt(
        samples: &[f32],
        sample_rate: u32,
        channels: u8,
        model_path: &Path,
        vad_segments: &[VadSegment],
        opts: &SttOptions,
    ) -> Result<Self> {
        let mut stt_model = ParakeetTDT::from_pretrained(model_path, None)
            .context("failed to load TDT STT model")?;

        let total_frames = (samples.len() / channels as usize) as i64;
        let total_ms = (total_frames * 1000) / sample_rate as i64;

        let windows = build_vad_windows(vad_segments, total_ms, opts);

        let mut out: Vec<SttSegment> = Vec::new();

        for w in windows {
            let (chunk, base_ms) =
                slice_by_ms(samples, sample_rate, channels, w.start_ms, w.end_ms);

            if chunk.is_empty() {
                continue;
            }

            let r = stt_model
                .transcribe_samples(
                    chunk.to_vec(),
                    sample_rate,
                    channels as u16,
                    Some(opts.timestamp_mode),
                )
                .context("TDT transcription failed")?;

            for t in r.tokens {
                out.push(SttSegment {
                    start_ms: base_ms + seconds_to_ms(t.start),
                    end_ms: base_ms + seconds_to_ms(t.end),
                    text: t.text,
                });
            }
        }

        out.sort_by_key(|s| (s.start_ms, s.end_ms));

        Ok(Self { segments: out })
    }

    pub fn post_process(
        self,
        diarization: &[DiarizationSegment],
        vad: &[VadSegment],
        opts: &SttOptions,
    ) -> Self {
        let mut out = self
            .format_numbers_and_spacing()
            .normalize_hyphenated_compounds();

        if opts.enable_finance_normalization {
            out = out.normalize_finance_terms();
        }

        out = out
            .merge_sentence_fragments(opts.merge_sentence_fragments_gap_ms, diarization, opts)
            .merge_fragmented_segments(opts.merge_fragmented_segments_gap_ms, diarization, opts)
            .merge_dangling_segments(opts.merge_dangling_segments_gap_ms, diarization, vad, opts)
            .dedupe_stutters_with_options(opts);

        if opts.remove_fillers {
            out = out.remove_fillers(opts);
        }

        if opts.enable_discourse_cleanup {
            out = out.cleanup_discourse_collisions();
        }

        out = out
            .normalize_punctuation_spacing()
            .finalize_sentence_casing();

        if opts.enable_paragraphs {
            out = out.group_into_paragraphs(diarization, opts);
        }

        out
    }

    pub fn format_numbers_and_spacing(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = format_numbers_and_spacing_text(&seg.text);
        }
        self
    }

    pub fn normalize_hyphenated_compounds(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = normalize_hyphenated_compounds_text(&seg.text);
        }
        self
    }

    pub fn normalize_finance_terms(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = normalize_finance_terms_text(&seg.text);
        }
        self
    }

    pub fn merge_sentence_fragments(
        mut self,
        max_gap_ms: i64,
        diarization: &[DiarizationSegment],
        opts: &SttOptions,
    ) -> Self {
        self.segments = merge_sentence_fragments_impl(self.segments, max_gap_ms, diarization, opts);
        self
    }

    pub fn merge_fragmented_segments(
        mut self,
        max_gap_ms: i64,
        diarization: &[DiarizationSegment],
        opts: &SttOptions,
    ) -> Self {
        merge_fragmented_segments_impl(&mut self.segments, max_gap_ms, diarization, opts);
        self
    }

    pub fn merge_dangling_segments(
        mut self,
        max_gap_ms: i64,
        diarization: &[DiarizationSegment],
        _vad: &[VadSegment],
        opts: &SttOptions,
    ) -> Self {
        merge_dangling_segments_impl(&mut self.segments, max_gap_ms, diarization, opts);
        self
    }

    pub fn dedupe_stutters_with_options(mut self, opts: &SttOptions) -> Self {
        for seg in &mut self.segments {
            seg.text = dedupe_stutters_text(&seg.text, opts.phrase_dedupe_max_ngram);
        }
        self
    }

    pub fn cleanup_discourse_collisions(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = cleanup_discourse_collisions_text(&seg.text);
        }
        self
    }

    pub fn normalize_punctuation_spacing(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = normalize_punctuation_spacing_text(&seg.text);
        }
        self
    }

    pub fn remove_fillers(mut self, opts: &SttOptions) -> Self {
        for seg in &mut self.segments {
            seg.text = remove_fillers_text(&seg.text, opts.filler_max_len);
        }
        self
    }

    pub fn finalize_sentence_casing(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = finalize_sentence_casing_text(&seg.text);
        }
        self
    }

    pub fn group_into_paragraphs(
        mut self,
        diarization: &[DiarizationSegment],
        opts: &SttOptions,
    ) -> Self {
        if self.segments.len() < 2 {
            return self;
        }

        self.segments.sort_by_key(|s| (s.start_ms, s.end_ms));

        let mut out: Vec<SttSegment> = Vec::with_capacity(self.segments.len());

        let mut run: Vec<SttSegment> = Vec::new();
        let mut prev_spk: Option<usize> = None;

        for seg in self.segments.into_iter() {
            let cur_spk = segment_speaker_id(diarization, &seg, opts);

            let speaker_changed = match (prev_spk, cur_spk) {
                (Some(a), Some(b)) => a != b,
                _ => false,
            };

            if !run.is_empty() && speaker_changed {
                out.extend(paragraphize_run(&run, opts));
                run.clear();
            }

            prev_spk = cur_spk.or(prev_spk);
            run.push(seg);
        }

        if !run.is_empty() {
            out.extend(paragraphize_run(&run, opts));
        }

        self.segments = out;
        self
    }
}

#[derive(Debug, Clone, Copy)]
struct VadWindow {
    start_ms: i64,
    end_ms: i64,
}

fn build_vad_windows(
    vad_segments: &[VadSegment],
    total_ms: i64,
    opts: &SttOptions,
) -> Vec<VadWindow> {
    let mut segs: Vec<VadWindow> = vad_segments
        .iter()
        .copied()
        .map(|s| VadWindow {
            start_ms: (s.start_ms - opts.vad_window_pad_ms).max(0),
            end_ms: (s.end_ms + opts.vad_window_pad_ms).min(total_ms),
        })
        .filter(|s| s.end_ms > s.start_ms)
        .collect();

    segs.sort_by_key(|s| s.start_ms);

    let mut merged: Vec<VadWindow> = Vec::with_capacity(segs.len());
    for s in segs {
        if let Some(last) = merged.last_mut() {
            let gap = s.start_ms - last.end_ms;
            if gap <= opts.vad_window_merge_gap_ms {
                last.end_ms = last.end_ms.max(s.end_ms);
                continue;
            }
        }
        merged.push(s);
    }

    let mut out: Vec<VadWindow> = Vec::new();
    for s in merged {
        let mut cur_start = s.start_ms;
        while cur_start < s.end_ms {
            let cur_end = (cur_start + opts.max_window_ms).min(s.end_ms);
            out.push(VadWindow {
                start_ms: cur_start,
                end_ms: cur_end,
            });
            cur_start = cur_end;
        }
    }

    out
}

fn slice_by_ms(
    samples: &[f32],
    sample_rate: u32,
    channels: u8,
    start_ms: i64,
    end_ms: i64,
) -> (&[f32], i64) {
    let sr = sample_rate as i64;
    let ch = channels as i64;

    let start_frames = (start_ms * sr) / 1000;
    let end_frames = (end_ms * sr) / 1000;

    let start_idx = (start_frames * ch).max(0) as usize;
    let end_idx = (end_frames * ch).max(0) as usize;

    let start_idx = start_idx.min(samples.len());
    let end_idx = end_idx.min(samples.len());

    if end_idx <= start_idx {
        return (&[], start_ms);
    }

    (&samples[start_idx..end_idx], start_ms)
}

fn merge_sentence_fragments_impl(
    mut segments: Vec<SttSegment>,
    max_gap_ms: i64,
    diarization: &[DiarizationSegment],
    opts: &SttOptions,
) -> Vec<SttSegment> {
    if segments.len() < 2 {
        return segments;
    }

    segments.sort_by_key(|s| (s.start_ms, s.end_ms));

    let mut out: Vec<SttSegment> = Vec::with_capacity(segments.len());
    for seg in segments {
        if let Some(last) = out.last_mut() {
            let gap = seg.start_ms - last.end_ms;

            let strong_merge =
                gap <= (max_gap_ms / 2).max(150) && is_strong_continuation(&last.text, &seg.text);

            if (gap <= max_gap_ms && should_merge_text(&last.text, &seg.text) || strong_merge)
                && !crosses_speaker_boundary(last.end_ms, seg.start_ms, diarization, opts)
            {
                last.end_ms = last.end_ms.max(seg.end_ms);
                last.text = join_text(&last.text, &seg.text);
                continue;
            }
        }

        out.push(seg);
    }

    out
}

fn is_strong_continuation(prev: &str, next: &str) -> bool {
    let prev = prev.trim();
    let next = next.trim();

    if prev.is_empty() || next.is_empty() {
        return true;
    }

    let next_first = next.chars().next().unwrap_or('\0');
    if next_first.is_ascii_lowercase() || next_first.is_ascii_digit() {
        return true;
    }

    let next_word = first_ascii_word(next).to_ascii_lowercase();
    matches!(
        next_word.as_str(),
        "million" | "billion" | "trillion" | "percent" | "dollars" | "usd" | "eur"
    )
}

fn crosses_speaker_boundary(
    a_ms: i64,
    b_ms: i64,
    diarization: &[DiarizationSegment],
    opts: &SttOptions,
) -> bool {
    if diarization.is_empty() {
        return false;
    }

    let lo = a_ms.min(b_ms);
    let hi = a_ms.max(b_ms);

    let mut overlaps = 0u32;
    for s in diarization {
        let ov = overlap_ms(lo, hi, s.start_ms, s.end_ms);
        if ov >= opts.diar_boundary_min_overlap_ms {
            overlaps += 1;
            if overlaps >= opts.diar_boundary_overlap_segments {
                return true;
            }
        }
    }

    false
}

fn should_merge_text(prev: &str, next: &str) -> bool {
    let prev = prev.trim();
    let next = next.trim();

    if prev.is_empty() || next.is_empty() {
        return true;
    }

    let next_first = next.chars().next().unwrap_or('\0');

    if next_first.is_ascii_lowercase() || next_first.is_ascii_digit() {
        return true;
    }

    if matches!(
        next_first,
        '$' | '€' | '£' | '%' | ')' | ']' | '}' | ',' | '.' | ':' | ';'
    ) {
        return true;
    }

    let next_word = first_ascii_word(next).to_ascii_lowercase();
    if is_continuation_word(&next_word) {
        return true;
    }

    prev_ends_with_numberish(prev)
}

fn is_continuation_word(word: &str) -> bool {
    matches!(
        word,
        "million"
            | "billion"
            | "trillion"
            | "dollar"
            | "dollars"
            | "usd"
            | "eur"
            | "percent"
            | "year"
            | "years"
            | "week"
            | "weeks"
            | "day"
            | "days"
            | "hour"
            | "hours"
    )
}

fn prev_ends_with_numberish(s: &str) -> bool {
    let last = last_ascii_token(s);
    if last.is_empty() {
        return false;
    }

    last.chars().any(|c| c.is_ascii_digit())
        && last
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '%' | '$' | '€' | '£'))
}

fn join_text(a: &str, b: &str) -> String {
    let a = a.trim_end();
    let b = b.trim_start();

    if a.is_empty() {
        return b.to_string();
    }
    if b.is_empty() {
        return a.to_string();
    }

    let a_last = a.chars().last().unwrap_or('\0');
    let b_first = b.chars().next().unwrap_or('\0');

    if a_last.is_whitespace() {
        format!("{}{}", a.trim_end(), b)
    } else if matches!(b_first, ',' | '.' | ':' | ';' | ')' | ']' | '}' | '%') {
        format!("{}{}", a, b)
    } else {
        format!("{} {}", a, b)
    }
}

fn first_ascii_word(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.is_empty() || !ch.is_whitespace() {
            break;
        }
    }
    out
}

fn last_ascii_token(s: &str) -> String {
    let mut token: Vec<char> = Vec::new();

    for ch in s.chars().rev() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | ',' | '%' | '$' | '€' | '£') {
            token.push(ch);
        } else if !token.is_empty() {
            break;
        }
    }

    token.reverse();
    token.into_iter().collect()
}

fn format_numbers_and_spacing_text(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len() + 8);

    let mut prev_emitted: Option<char> = None;
    let mut prev_non_space: Option<char> = None;

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];

        // insert space between alpha<->digit when appropriate
        if let Some(prev) = prev_non_space {
            if prev.is_ascii_alphabetic() && c.is_ascii_digit() {
                let prev_upper = prev.to_ascii_uppercase();
                let glue = matches!(prev_upper, 'Q' | 'H') || is_prev_token_glued_prefix(&out);

                if !glue && !out.ends_with(' ') {
                    out.push(' ');
                    prev_emitted = Some(' ');
                }
            } else if prev.is_ascii_digit() && c.is_ascii_alphabetic() {
                let glue =
                    looks_like_ordinal_suffix(&chars, i) || is_prev_token_glued_number_prefix(&out);

                if !glue && !out.ends_with(' ') {
                    out.push(' ');
                    prev_emitted = Some(' ');
                }
            }
        }

        // "4 .47" -> "4.47"
        if c.is_ascii_digit()
            && i + 3 < chars.len()
            && chars[i + 1].is_whitespace()
            && chars[i + 2] == '.'
            && chars[i + 3].is_ascii_digit()
        {
            out.push(c);
            out.push('.');
            prev_emitted = Some('.');
            prev_non_space = Some('.');
            i += 3;
            continue;
        }

        // "4 . 47" -> "4.47"
        if c.is_ascii_digit()
            && i + 4 < chars.len()
            && chars[i + 1].is_whitespace()
            && chars[i + 2] == '.'
            && chars[i + 3].is_whitespace()
            && chars[i + 4].is_ascii_digit()
        {
            out.push(c);
            out.push('.');
            prev_emitted = Some('.');
            prev_non_space = Some('.');
            i += 4;
            continue;
        }

        // "4. 47" -> "4.47"
        if c.is_ascii_digit() && i + 2 < chars.len() && chars[i + 1] == '.' {
            let mut j = i + 2;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }

            if j < chars.len() && chars[j].is_ascii_digit() {
                out.push(c);
                out.push('.');
                prev_emitted = Some('.');
                prev_non_space = Some('.');
                i = j;
                continue;
            }
        }

        // remove space before percent: "600 %" -> "600%"
        if c.is_whitespace() && i + 1 < chars.len() && chars[i + 1] == '%' {
            i += 1;
            continue;
        }

        // collapse whitespace
        if c.is_whitespace() {
            if !matches!(prev_emitted, Some(' ')) {
                out.push(' ');
                prev_emitted = Some(' ');
            }
            i += 1;
            continue;
        }

        out.push(c);
        prev_emitted = Some(c);
        prev_non_space = Some(c);
        i += 1;
    }

    out.trim().to_string()
}

fn is_prev_token_glued_number_prefix(out: &str) -> bool {
    let token = last_ascii_word(out);
    if token.len() != 2 {
        return false;
    }

    let mut it = token.chars();
    let first = it.next().unwrap_or('\0').to_ascii_uppercase();
    let second = it.next().unwrap_or('\0');

    matches!(first, 'Q' | 'H') && second.is_ascii_digit()
}

fn looks_like_ordinal_suffix(chars: &[char], idx: usize) -> bool {
    if idx >= chars.len() {
        return false;
    }

    let c0 = chars[idx].to_ascii_lowercase();
    let c1 = chars
        .get(idx + 1)
        .copied()
        .unwrap_or('\0')
        .to_ascii_lowercase();

    matches!((c0, c1), ('s', 't') | ('n', 'd') | ('r', 'd') | ('t', 'h'))
}

fn is_prev_token_glued_prefix(out: &str) -> bool {
    let token = last_ascii_word(out);
    matches!(token.as_str(), "FY" | "CY")
}

fn last_ascii_word(out: &str) -> String {
    let mut token: Vec<char> = Vec::new();

    for ch in out.chars().rev() {
        if ch.is_ascii_alphanumeric() {
            token.push(ch);
        } else if !token.is_empty() {
            break;
        }
    }

    token.reverse();
    token.into_iter().collect()
}

fn normalize_hyphenated_compounds_text(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];

        if c == '-' {
            let prev = out.chars().rev().find(|ch| !ch.is_whitespace());

            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let next = if j < chars.len() {
                Some(chars[j])
            } else {
                None
            };

            let should_join = match (prev, next) {
                (Some(p), Some(n)) => {
                    p.is_ascii_alphanumeric()
                        && n.is_ascii_alphanumeric()
                        && (p.is_ascii_alphabetic() || n.is_ascii_alphabetic())
                }
                _ => false,
            };

            if should_join {
                while out.ends_with(' ') {
                    out.pop();
                }

                out.push('-');
                i = j;
                continue;
            }
        }

        out.push(c);
        i += 1;
    }

    out
}

fn normalize_finance_terms_text(input: &str) -> String {
    // simple, conservative normalizations (no external deps)
    let mut s = input.to_string();

    // normalize Q12025 -> Q1 2025 and H12025 -> H1 2025
    s = normalize_quarter_half_year_tokens(&s);

    // normalize common EBITDA variants when they appear as a token-ish word
    s = replace_token_case_insensitive(&s, "epita", "EBITDA");
    s = replace_token_case_insensitive(&s, "ebita", "EBITDA");

    // normalize EPS tokenization
    s = replace_token_case_insensitive(&s, "e p s", "EPS");
    s = replace_token_case_insensitive(&s, "e. p. s.", "EPS");

    // remove stray single-letter 'f' before numbers: "a f 5.16" -> "a 5.16"
    s = drop_stray_f_before_number(&s);

    s
}

fn normalize_quarter_half_year_tokens(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];

        if (c == 'Q' || c == 'q' || c == 'H' || c == 'h')
            && i + 5 < chars.len()
            && chars[i + 1].is_ascii_digit()
            && chars[i + 2].is_ascii_digit()
            && chars[i + 3].is_ascii_digit()
            && chars[i + 4].is_ascii_digit()
            && chars[i + 5].is_ascii_digit()
        {
            out.push(c.to_ascii_uppercase());
            out.push(chars[i + 1]);
            out.push(' ');
            out.push(chars[i + 2]);
            out.push(chars[i + 3]);
            out.push(chars[i + 4]);
            out.push(chars[i + 5]);
            i += 6;
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

fn replace_token_case_insensitive(input: &str, needle: &str, replacement: &str) -> String {
    let needle = needle.to_ascii_lowercase();

    let mut out: Vec<String> = Vec::new();
    for t in input.split_whitespace() {
        let norm = normalize_token(t);
        if norm == needle {
            out.push(replacement.to_string());
        } else {
            out.push(t.to_string());
        }
    }

    out.join(" ")
}

fn drop_stray_f_before_number(input: &str) -> String {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.len() < 3 {
        return input.to_string();
    }

    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0usize;

    while i < tokens.len() {
        let t = tokens[i];

        if i + 1 < tokens.len() && normalize_token(t) == "f" && starts_with_number(tokens[i + 1]) {
            i += 1;
            continue;
        }

        out.push(t.to_string());
        i += 1;
    }

    out.join(" ")
}

fn starts_with_number(s: &str) -> bool {
    s.chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
}

// ------------------------------
// diarization-aware merge gates
// ------------------------------

#[derive(Debug, Clone, Copy)]
struct DominantSpeaker {
    speaker_id: usize,
    overlap_ms: i64,
    ratio: f32,
    is_confident: bool,
}

fn dominant_speaker(
    diar: &[DiarizationSegment],
    start_ms: i64,
    end_ms: i64,
) -> Option<DominantSpeaker> {
    if start_ms >= end_ms {
        return None;
    }

    let dur = (end_ms - start_ms).max(1) as f32;

    let mut best: Option<(usize, i64)> = None;
    for s in diar {
        let ov = overlap_ms(start_ms, end_ms, s.start_ms, s.end_ms);
        if ov <= 0 {
            continue;
        }

        let cand = (s.speaker_id, ov);
        if best.map(|b| cand.1 > b.1).unwrap_or(true) {
            best = Some(cand);
        }
    }

    best.map(|(speaker_id, overlap)| DominantSpeaker {
        speaker_id,
        overlap_ms: overlap,
        ratio: (overlap as f32) / dur,
        is_confident: false,
    })
}

fn overlap_ms(a0: i64, a1: i64, b0: i64, b1: i64) -> i64 {
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    (hi - lo).max(0)
}

// ------------------------------
// merge_fragmented_segments
// ------------------------------

fn merge_fragmented_segments_impl(
    segments: &mut Vec<SttSegment>,
    max_gap_ms: i64,
    diarization: &[DiarizationSegment],
    opts: &SttOptions,
) {
    if segments.len() < 2 {
        return;
    }

    let mut i = 0usize;
    while i + 1 < segments.len() {
        let a = segments[i].clone();
        let b = segments[i + 1].clone();

        if !can_merge_across_speaker(diarization, &a, &b, opts) {
            i += 1;
            continue;
        }

        let near = b.start_ms <= a.end_ms + max_gap_ms;
        if !near {
            i += 1;
            continue;
        }

        if let Some(merged_text) = try_merge_text(&a.text, &b.text) {
            segments[i] = SttSegment {
                start_ms: a.start_ms.min(b.start_ms),
                end_ms: a.end_ms.max(b.end_ms),
                text: merged_text,
            };
            segments.remove(i + 1);
            continue;
        }

        i += 1;
    }
}

fn try_merge_text(a: &str, b: &str) -> Option<String> {
    let (a_lead, a_rest) = split_leading_discourse(a);
    let (b_lead, b_rest) = split_leading_discourse(b);

    let a_core = normalize_for_match(&a_rest);
    let b_core = normalize_for_match(&b_rest);

    if a_core.is_empty() || b_core.is_empty() {
        return None;
    }

    let (short_is_a, short_core, long_core) = if a_core.len() <= b_core.len() {
        (true, a_core.as_str(), b_core.as_str())
    } else {
        (false, b_core.as_str(), a_core.as_str())
    };

    if !long_core.starts_with(short_core) {
        return None;
    }

    let long_original = if short_is_a { b } else { a };

    let merged = match (a_lead, b_lead) {
        (Some(al), Some(bl)) if normalize_discourse(&al) != normalize_discourse(&bl) => {
            let rest = remove_leading_discourse(long_original);
            format!("{} {}", al.trim(), rest.trim())
        }
        (Some(al), None) if short_is_a => {
            let rest = remove_leading_discourse(long_original);
            format!("{} {}", al.trim(), rest.trim())
        }
        _ => long_original.trim().to_string(),
    };

    Some(merged)
}

fn split_leading_discourse(input: &str) -> (Option<String>, String) {
    let s = input.trim_start();
    let mut iter = s.split_whitespace();
    let first = match iter.next() {
        Some(w) => w,
        None => return (None, String::new()),
    };

    let first_clean = normalize_discourse(first);

    if !is_discourse_word(&first_clean) {
        return (None, s.to_string());
    }

    let rest = s[first.len()..].trim_start().to_string();
    (Some(first.to_string()), rest)
}

fn remove_leading_discourse(input: &str) -> String {
    let (_, rest) = split_leading_discourse(input);
    rest
}

fn normalize_discourse(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

fn is_discourse_word(word: &str) -> bool {
    matches!(
        word,
        "now" | "but" | "so" | "and" | "well" | "uh" | "um" | "yeah" | "okay" | "ok"
    )
}

fn normalize_for_match(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_was_space = false;

    for c in input.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_was_space = false;
            continue;
        }

        if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ------------------------------
// merge_dangling_segments
// ------------------------------

fn merge_dangling_segments_impl(
    segments: &mut Vec<SttSegment>,
    max_gap_ms: i64,
    diarization: &[DiarizationSegment],
    opts: &SttOptions,
) {
    if segments.len() < 2 {
        return;
    }

    let mut i = 0usize;
    while i + 1 < segments.len() {
        let a = segments[i].clone();
        let b = segments[i + 1].clone();

        if !can_merge_across_speaker(diarization, &a, &b, opts) {
            i += 1;
            continue;
        }

        let gap_ms = b.start_ms - a.end_ms;
        if gap_ms > max_gap_ms {
            i += 1;
            continue;
        }

        let a_text = a.text.trim();
        let b_text = b.text.trim();

        if a_text.is_empty() || b_text.is_empty() {
            i += 1;
            continue;
        }

        let bad_terminal = ends_with_bad_terminal(a_text);
        let short_tail = is_short_tail(b_text);

        if !bad_terminal && !short_tail {
            i += 1;
            continue;
        }

        let left = if bad_terminal {
            a_text.trim_end_matches(['.', ',', ';', ':'])
        } else {
            a_text
        };

        let merged_text = join_sentences(left, b_text);

        segments[i] = SttSegment {
            start_ms: a.start_ms,
            end_ms: a.end_ms.max(b.end_ms),
            text: merged_text,
        };

        segments.remove(i + 1);
    }
}

fn can_merge_across_speaker(
    diarization: &[DiarizationSegment],
    a: &SttSegment,
    b: &SttSegment,
    opts: &SttOptions,
) -> bool {
    if !opts.require_same_speaker_for_merges || diarization.is_empty() {
        return true;
    }

    let da = dominant_speaker(diarization, a.start_ms, a.end_ms);
    let db = dominant_speaker(diarization, b.start_ms, b.end_ms);

    let (Some(mut da), Some(mut db)) = (da, db) else {
        return true;
    };

    da.is_confident =
        da.overlap_ms >= opts.speaker_min_overlap_ms || da.ratio >= opts.speaker_min_overlap_ratio;
    db.is_confident =
        db.overlap_ms >= opts.speaker_min_overlap_ms || db.ratio >= opts.speaker_min_overlap_ratio;

    !(da.is_confident && db.is_confident && da.speaker_id != db.speaker_id)
}

fn ends_with_bad_terminal(text: &str) -> bool {
    if !text.ends_with('.') {
        return false;
    }

    let last_word = text
        .trim_end_matches('.')
        .split_whitespace()
        .next_back()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase();

    matches!(
        last_word.as_str(),
        "about" | "of" | "the" | "to" | "with" | "at" | "in" | "as" | "for" | "and" | "or"
    )
}

fn is_short_tail(text: &str) -> bool {
    let cleaned = text
        .trim_matches(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?')
        .trim();

    if cleaned.is_empty() {
        return false;
    }

    let word_count = cleaned.split_whitespace().count();
    word_count <= 3 || cleaned.len() <= 12
}

fn join_sentences(a: &str, b: &str) -> String {
    let a = a.trim_end();
    let b = b.trim_start();

    if a.is_empty() {
        return b.to_string();
    }
    if b.is_empty() {
        return a.to_string();
    }

    let needs_space = !a.ends_with(' ')
        && !b.starts_with(',')
        && !b.starts_with('.')
        && !b.starts_with('!')
        && !b.starts_with('?');

    if needs_space {
        format!("{} {}", a, b)
    } else {
        format!("{}{}", a, b)
    }
}

// ------------------------------
// dedupe_stutters
// ------------------------------

fn dedupe_stutters_text(input: &str, max_ngram: usize) -> String {
    let mut tokens: Vec<String> = input.split_whitespace().map(|s| s.to_string()).collect();
    if tokens.is_empty() {
        return input.trim().to_string();
    }

    tokens = remove_adjacent_duplicates(tokens);
    tokens = remove_adjacent_repeated_ngrams(tokens, max_ngram);

    if tokens.len() >= 2 && is_single_letter(&tokens[0]) && eq_loose(&tokens[1], "yes") {
        tokens.remove(0);
    }

    tokens = drop_known_pair(tokens, "we've", "we're");
    remove_adjacent_pair(&mut tokens, "not", "no", true);
    remove_adjacent_pair(&mut tokens, "we've", "we're", true);

    if tokens.len() >= 2 && eq_loose(&tokens[0], "not") && eq_loose(&tokens[1], "no") {
        tokens.remove(0);
    }

    let mut out = tokens.join(" ");

    out = out.replace(" ,", ",");
    out = out.replace(" .", ".");
    out = out.replace(" !", "!");
    out = out.replace(" ?", "?");

    out = fix_comma_spacing(&out);

    out.trim().to_string()
}

fn remove_adjacent_duplicates(tokens: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    for t in tokens {
        if let Some(last) = out.last()
            && normalize_token(last) == normalize_token(&t)
        {
            continue;
        }
        out.push(t);
    }
    out
}

fn remove_adjacent_repeated_ngrams(mut tokens: Vec<String>, max_ngram: usize) -> Vec<String> {
    if tokens.len() < 4 || max_ngram < 2 {
        return tokens;
    }

    let max_ngram = max_ngram.min(6);

    let mut i = 0usize;
    while i + 1 < tokens.len() {
        let remaining = tokens.len() - i;
        let mut removed_any = false;

        for n in (2..=max_ngram).rev() {
            if remaining < 2 * n {
                continue;
            }

            let a = &tokens[i..i + n];
            let b = &tokens[i + n..i + 2 * n];

            if ngram_eq(a, b) {
                tokens.drain(i + n..i + 2 * n);
                removed_any = true;
                break;
            }
        }

        if !removed_any {
            i += 1;
        }
    }

    tokens
}

fn ngram_eq(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (x, y) in a.iter().zip(b.iter()) {
        if normalize_token(x) != normalize_token(y) {
            return false;
        }
    }

    true
}

fn drop_known_pair(tokens: Vec<String>, a: &str, b: &str) -> Vec<String> {
    if tokens.len() < 2 {
        return tokens;
    }

    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0usize;

    while i < tokens.len() {
        if i + 1 < tokens.len() && eq_loose(&tokens[i], a) && eq_loose(&tokens[i + 1], b) {
            out.push(tokens[i + 1].clone());
            i += 2;
            continue;
        }

        out.push(tokens[i].clone());
        i += 1;
    }

    out
}

fn is_single_letter(s: &str) -> bool {
    let s = s.trim_matches(|c: char| !c.is_ascii_alphabetic());
    s.len() == 1
}

fn eq_loose(a: &str, b: &str) -> bool {
    normalize_token(a) == normalize_token(b)
}

fn normalize_token(s: &str) -> String {
    normalize_token_keep(s, |c| c.is_ascii_alphanumeric())
}

fn normalize_filler_token(s: &str) -> String {
    normalize_token_keep(s, |c| c.is_ascii_alphabetic())
}

fn remove_adjacent_pair(tokens: &mut Vec<String>, a: &str, b: &str, remove_first: bool) {
    let mut i = 0usize;
    while i + 1 < tokens.len() {
        if eq_loose(&tokens[i], a) && eq_loose(&tokens[i + 1], b) {
            if remove_first {
                tokens.remove(i);
            } else {
                tokens.remove(i + 1);
            }
            continue;
        }
        i += 1;
    }
}

fn fix_comma_spacing(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        out.push(c);

        if c == ',' {
            let next = chars.get(i + 1).copied();
            if let Some(n) = next
                && !n.is_whitespace()
                && n != ','
                && n != '.'
                && n != '!'
                && n != '?'
            {
                out.push(' ');
            }
        }

        i += 1;
    }

    out
}

fn cleanup_discourse_collisions_text(input: &str) -> String {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.len() < 2 {
        return input.trim().to_string();
    }

    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0usize;

    while i < tokens.len() {
        let t = tokens[i];

        if i + 1 < tokens.len() && normalize_token(t) == "from" && is_filler_token(tokens[i + 1]) {
            i += 1;
            continue;
        }

        out.push(t.to_string());
        i += 1;
    }

    normalize_punctuation_spacing_text(&out.join(" "))
}

fn is_filler_token(s: &str) -> bool {
    matches!(
        normalize_token(s).as_str(),
        "yeah" | "uh" | "um" | "er" | "ah" | "like"
    )
}

fn normalize_punctuation_spacing_text(input: &str) -> String {
    let mut out = input.trim().to_string();

    // remove spaces before punctuation
    out = out.replace(" ,", ",");
    out = out.replace(" .", ".");
    out = out.replace(" !", "!");
    out = out.replace(" ?", "?");
    out = out.replace(" ;", ";");
    out = out.replace(" :", ":");

    let chars: Vec<char> = out.chars().collect();
    let mut fixed = String::with_capacity(out.len() + 8);

    let mut prev_emitted: Option<char> = None;

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        fixed.push(c);

        if matches!(c, '.' | '!' | '?') && i + 1 < chars.len() {
            let n = chars[i + 1];

            let is_decimal_dot = c == '.'
                && prev_emitted.map(|p| p.is_ascii_digit()).unwrap_or(false)
                && n.is_ascii_digit();

            if !is_decimal_dot && n.is_ascii_alphanumeric() {
                fixed.push(' ');
            }
        }

        prev_emitted = Some(c);
        i += 1;
    }

    fixed.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn remove_fillers_text(input: &str, filler_max_len: usize) -> String {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.is_empty() {
        return input.trim().to_string();
    }

    let mut out: Vec<String> = Vec::with_capacity(tokens.len());

    for t in tokens {
        let norm = normalize_filler_token(t);

        let is_short = norm.len() <= filler_max_len;
        let is_filler = is_short && is_filler_token_norm(&norm);

        if is_filler {
            continue;
        }

        out.push(t.to_string());
    }

    normalize_punctuation_spacing_text(&out.join(" "))
}

fn is_filler_token_norm(norm: &str) -> bool {
    matches!(norm, "uh" | "um" | "er" | "ah" | "eh" | "hmm" | "mmm")
}

fn finalize_sentence_casing_text(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut chars: Vec<char> = trimmed.chars().collect();
    let mut idx = 0usize;

    while idx < chars.len() {
        if chars[idx].is_ascii_alphabetic() {
            break;
        }
        idx += 1;
    }

    if idx < chars.len() {
        chars[idx] = chars[idx].to_ascii_uppercase();
    }

    chars.into_iter().collect()
}

fn normalize_token_keep<F>(s: &str, keep: F) -> String
where
    F: Fn(char) -> bool,
{
    s.trim_matches(|c: char| !keep(c))
        .chars()
        .filter(|&c| keep(c))
        .collect::<String>()
        .to_ascii_lowercase()
}

fn segment_speaker_id(
    diarization: &[DiarizationSegment],
    seg: &SttSegment,
    opts: &SttOptions,
) -> Option<usize> {
    if diarization.is_empty() {
        return None;
    }

    let mut dom = dominant_speaker(diarization, seg.start_ms, seg.end_ms)?;

    dom.is_confident = dom.overlap_ms >= opts.speaker_min_overlap_ms
        || dom.ratio >= opts.speaker_min_overlap_ratio;

    if dom.is_confident {
        Some(dom.speaker_id)
    } else {
        None
    }
}

fn paragraphize_run(run: &[SttSegment], opts: &SttOptions) -> Vec<SttSegment> {
    if run.len() <= 1 {
        return run.to_vec();
    }

    let sentences: Vec<String> = run.iter().map(|s| s.text.trim().to_string()).collect();

    let grouper = ParagraphGrouper::new()
        .split_threshold(opts.paragraph_split_threshold)
        .min_sentences(opts.paragraph_min_sentences)
        .rank_window(opts.paragraph_rank_window)
        .block_depth(opts.paragraph_block_depth);

    let paragraphs = match grouper.group(sentences) {
        Ok(p) if !p.is_empty() => p,
        _ => {
            return run.to_vec();
        }
    };

    map_paragraphs_back_to_time_ranges(run, &paragraphs)
}

fn map_paragraphs_back_to_time_ranges(
    run: &[SttSegment],
    paragraphs: &[String],
) -> Vec<SttSegment> {
    let mut out: Vec<SttSegment> = Vec::with_capacity(paragraphs.len());
    let mut i = 0usize;

    for p in paragraphs {
        if i >= run.len() {
            break;
        }

        let p_norm = normalize_ws(p);

        let start_idx = i;
        let mut acc = String::new();
        let mut end_idx = i;

        while end_idx < run.len() {
            if !acc.is_empty() {
                acc.push(' ');
            }
            acc.push_str(run[end_idx].text.trim());

            if normalize_ws(&acc) == p_norm {
                break;
            }

            end_idx += 1;
        }

        if end_idx >= run.len() {
            end_idx = start_idx;
        }

        let start_ms = run[start_idx].start_ms;
        let end_ms = run[end_idx].end_ms;

        out.push(SttSegment {
            start_ms,
            end_ms,
            text: p.trim().to_string(),
        });

        i = end_idx + 1;
    }

    while i < run.len() {
        out.push(run[i].clone());
        i += 1;
    }

    out
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
