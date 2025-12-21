use std::path::Path;

use anyhow::{Context, Result};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};

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

#[derive(Clone, Copy)]
pub struct PostProcessContext<'a> {
    pub diarization: Option<&'a [DiarizationSegment]>,
    pub vad: Option<&'a [VadSegment]>,
}

impl Stt {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    pub fn from_parakeet_tdt(
        mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: u8,
        model_path: &Path,
        vad_segments: &[VadSegment],
    ) -> Result<Self> {
        let mut stt_model = ParakeetTDT::from_pretrained(model_path, None)
            .context("failed to load TDT STT model")?;

        let total_frames = (samples.len() / channels as usize) as i64;
        let total_ms = (total_frames * 1000) / sample_rate as i64;

        let windows = build_vad_windows(vad_segments, total_ms);

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
                    Some(TimestampMode::Sentences),
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
        self.segments = out;
        Ok(self)
    }

    pub fn post_process_default(self, ctx: PostProcessContext<'_>) -> Self {
        self.format_numbers_and_spacing()
            .normalize_hyphenated_compounds()
            .merge_sentence_fragments(800, ctx.diarization)
            .merge_fragmented_segments(600, ctx.diarization)
            .merge_dangling_segments(1200, ctx.diarization, ctx.vad)
            .dedupe_stutters()
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

    /// Merge small “sentence fragments” such as:
    /// - "We're about 4.47" + "million in the quarter."
    /// while respecting optional diarization boundaries.
    pub fn merge_sentence_fragments(
        mut self,
        max_gap_ms: i64,
        diarization: Option<&[DiarizationSegment]>,
    ) -> Self {
        self.segments = merge_sentence_fragments_impl(self.segments, max_gap_ms, diarization);
        self
    }

    /// Merge fragmented segments that look like a prefix-overlap duplication,
    /// while respecting diarization speaker identity (if provided).
    pub fn merge_fragmented_segments(
        mut self,
        max_gap_ms: i64,
        diarization: Option<&[DiarizationSegment]>,
    ) -> Self {
        merge_fragmented_segments_impl(&mut self.segments, max_gap_ms, diarization);
        self
    }

    /// Merge “dangling” short follow-ups and weak terminals, while respecting diarization
    /// speaker identity (if provided). `vad` is accepted for future tightening, but
    /// current heuristics don't require it.
    pub fn merge_dangling_segments(
        mut self,
        max_gap_ms: i64,
        diarization: Option<&[DiarizationSegment]>,
        _vad: Option<&[VadSegment]>,
    ) -> Self {
        merge_dangling_segments_impl(&mut self.segments, max_gap_ms, diarization);
        self
    }

    pub fn dedupe_stutters(mut self) -> Self {
        for seg in &mut self.segments {
            seg.text = dedupe_stutters_text(&seg.text);
        }
        self
    }
}

#[derive(Debug, Clone, Copy)]
struct VadWindow {
    start_ms: i64,
    end_ms: i64,
}

fn build_vad_windows(vad_segments: &[VadSegment], total_ms: i64) -> Vec<VadWindow> {
    let pad_ms: i64 = 120;
    let merge_gap_ms: i64 = 250;
    let max_window_ms: i64 = 35_000;

    let mut segs: Vec<VadWindow> = vad_segments
        .iter()
        .copied()
        .map(|s| VadWindow {
            start_ms: (s.start_ms - pad_ms).max(0),
            end_ms: (s.end_ms + pad_ms).min(total_ms),
        })
        .filter(|s| s.end_ms > s.start_ms)
        .collect();

    segs.sort_by_key(|s| s.start_ms);

    let mut merged: Vec<VadWindow> = Vec::with_capacity(segs.len());
    for s in segs {
        if let Some(last) = merged.last_mut() {
            let gap = s.start_ms - last.end_ms;
            if gap <= merge_gap_ms {
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
            let cur_end = (cur_start + max_window_ms).min(s.end_ms);
            out.push(VadWindow {
                start_ms: cur_start,
                end_ms: cur_end,
            });
            cur_start = cur_end;
        }
    }

    out
}

fn slice_by_ms<'a>(
    samples: &'a [f32],
    sample_rate: u32,
    channels: u8,
    start_ms: i64,
    end_ms: i64,
) -> (&'a [f32], i64) {
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
    diarization: Option<&[DiarizationSegment]>,
) -> Vec<SttSegment> {
    if segments.len() < 2 {
        return segments;
    }

    segments.sort_by_key(|s| (s.start_ms, s.end_ms));

    let mut out: Vec<SttSegment> = Vec::with_capacity(segments.len());
    for seg in segments {
        if let Some(last) = out.last_mut() {
            let gap = seg.start_ms - last.end_ms;

            if gap <= max_gap_ms
                && should_merge_text(&last.text, &seg.text)
                && !crosses_speaker_boundary(last.end_ms, seg.start_ms, diarization)
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

/// Returns true if there is a diarization boundary between [a_ms, b_ms].
/// We don't assign speakers; we just avoid merging across a change.
fn crosses_speaker_boundary(
    a_ms: i64,
    b_ms: i64,
    diarization: Option<&[DiarizationSegment]>,
) -> bool {
    let Some(d) = diarization else {
        return false;
    };

    if d.is_empty() {
        return false;
    }

    let lo = a_ms.min(b_ms);
    let hi = a_ms.max(b_ms);

    // a speaker change implies the end of one segment and start of another within the interval.
    // robust check: if the interval overlaps 2+ diar segments, treat as boundary-crossing.
    let mut overlaps = 0u32;
    for s in d {
        if ranges_overlap(lo, hi, s.start_ms, s.end_ms) {
            overlaps += 1;
            if overlaps >= 2 {
                return true;
            }
        }
    }

    false
}

fn ranges_overlap(a0: i64, a1: i64, b0: i64, b1: i64) -> bool {
    let (a0, a1) = if a0 <= a1 { (a0, a1) } else { (a1, a0) };
    let (b0, b1) = if b0 <= b1 { (b0, b1) } else { (b1, b0) };
    a0 < b1 && b0 < a1
}

fn should_merge_text(prev: &str, next: &str) -> bool {
    let prev = prev.trim();
    let next = next.trim();

    if prev.is_empty() || next.is_empty() {
        return true;
    }

    let next_first = next.chars().next().unwrap_or('\0');

    // continuation heuristics
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
        } else if !out.is_empty() {
            break;
        } else if !ch.is_whitespace() {
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

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];

        // helper: last non-space character we emitted
        let last_non_space = out.chars().rev().find(|ch| !ch.is_whitespace());

        // insert space between alpha<->digit, except for common glued patterns (Q1, H1, 2nd, 1st, etc)
        if let Some(prev) = last_non_space {
            if prev.is_ascii_alphabetic() && c.is_ascii_digit() {
                let prev_upper = prev.to_ascii_uppercase();

                // only single-letter glue prefixes here
                // FY/CY are handled by is_prev_token_glued_prefix (token-based)
                let glue = matches!(prev_upper, 'Q' | 'H') || is_prev_token_glued_prefix(&out);

                if !glue && !out.ends_with(' ') {
                    out.push(' ');
                }
            } else if prev.is_ascii_digit() && c.is_ascii_alphabetic() {
                let glue =
                    looks_like_ordinal_suffix(&chars, i) || is_prev_token_glued_number_prefix(&out);
                if !glue && !out.ends_with(' ') {
                    out.push(' ');
                }
            }
        }

        // fix "digit space dot digit" -> "digit.dotdigit"  (e.g. "1 .47")
        if c.is_ascii_digit()
            && i + 3 < chars.len()
            && chars[i + 1].is_whitespace()
            && chars[i + 2] == '.'
            && chars[i + 3].is_ascii_digit()
        {
            out.push(c);
            out.push('.');
            i += 3;
            continue;
        }

        // fix "digit space dot space digit" -> "digit.dotdigit" (e.g. "5 . 16")
        if c.is_ascii_digit()
            && i + 4 < chars.len()
            && chars[i + 1].is_whitespace()
            && chars[i + 2] == '.'
            && chars[i + 3].is_whitespace()
            && chars[i + 4].is_ascii_digit()
        {
            out.push(c);
            out.push('.');
            i += 4;
            continue;
        }

        // remove space before percent: "600 %" -> "600%"
        if c.is_whitespace() && i + 1 < chars.len() && chars[i + 1] == '%' {
            i += 1;
            continue;
        }

        // collapse whitespace
        if c.is_whitespace() {
            if !out.ends_with(' ') {
                out.push(' ');
            }
            i += 1;
            continue;
        }

        out.push(c);
        i += 1;
    }

    out.trim().to_string()
}

/// optional hook: keep patterns like "Q1" already formed, so you can build "Q12025" without spaces
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

/// returns true if the upcoming letters at `idx` look like an ordinal suffix: st/nd/rd/th
/// idx is the index of the first suffix character (current char) in the original input.
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

/// optional hook: keep prefixes like "FY" or "CY" glued to numbers (FY25, CY2025)
fn is_prev_token_glued_prefix(out: &str) -> bool {
    let token = last_ascii_word(out);
    matches!(token.as_str(), "FY" | "CY")
}

/// gets the last contiguous [A-Za-z0-9] token from `out` (already emitted text)
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
            // find previous emitted non-space
            let prev = out.chars().rev().find(|ch| !ch.is_whitespace());

            // find next non-space in input
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
                    // join if both sides are alnum and at least one side is a letter
                    // (avoids changing "10 - 20" into "10-20")
                    p.is_ascii_alphanumeric()
                        && n.is_ascii_alphanumeric()
                        && (p.is_ascii_alphabetic() || n.is_ascii_alphabetic())
                }
                _ => false,
            };

            if should_join {
                // remove trailing spaces before '-'
                while out.ends_with(' ') {
                    out.pop();
                }

                out.push('-');

                // skip spaces after '-' (we already emitted the hyphen)
                i = j;
                continue;
            }
        }

        out.push(c);
        i += 1;
    }

    out
}

// ------------------------------
// diarization-aware merge gates
// ------------------------------

fn same_diarization_speaker(
    diarization: Option<&[DiarizationSegment]>,
    a0: i64,
    a1: i64,
    b0: i64,
    b1: i64,
) -> bool {
    let Some(d) = diarization else {
        return false;
    };

    if d.is_empty() {
        return false;
    }

    let sa = diarization_speaker_for_window(d, a0, a1);
    let sb = diarization_speaker_for_window(d, b0, b1);

    matches!((sa, sb), (Some(x), Some(y)) if x == y)
}

fn diarization_speaker_for_window(
    diar: &[DiarizationSegment],
    start_ms: i64,
    end_ms: i64,
) -> Option<usize> {
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

    best.map(|b| b.0)
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
    diarization: Option<&[DiarizationSegment]>,
) {
    if segments.len() < 2 {
        return;
    }

    let mut i = 0usize;
    while i + 1 < segments.len() {
        let a = segments[i].clone();
        let b = segments[i + 1].clone();

        if !same_diarization_speaker(diarization, a.start_ms, a.end_ms, b.start_ms, b.end_ms) {
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
    diarization: Option<&[DiarizationSegment]>,
) {
    if segments.len() < 2 {
        return;
    }

    let mut i = 0usize;
    while i + 1 < segments.len() {
        let a = segments[i].clone();
        let b = segments[i + 1].clone();

        if !same_diarization_speaker(diarization, a.start_ms, a.end_ms, b.start_ms, b.end_ms) {
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
            a_text.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':'))
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

fn ends_with_bad_terminal(text: &str) -> bool {
    // merges things like "... talking about." + "AI a lot."
    // only triggers if the segment ends with '.' and the last word is a weak terminal.
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
    // merges very short follow-ups (often chunk boundary split),
    // like "AI a lot." or "System." or "More importantly,"
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
    // joins with sensible punctuation/spaces, avoids ".,"
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

fn dedupe_stutters_text(input: &str) -> String {
    let mut tokens: Vec<String> = input.split_whitespace().map(|s| s.to_string()).collect();
    if tokens.is_empty() {
        return input.trim().to_string();
    }

    // remove immediate duplicate tokens: "the the" -> "the"
    tokens = remove_adjacent_duplicates(tokens);

    // fix "y yes" -> "yes"
    if tokens.len() >= 2 && is_single_letter(&tokens[0]) && eq_loose(&tokens[1], "yes") {
        tokens.remove(0);
    }

    // fix "we've we're" -> keep second
    tokens = drop_known_pair(tokens, "we've", "we're");
    remove_adjacent_pair(&mut tokens, "not", "no", true);
    remove_adjacent_pair(&mut tokens, "we've", "we're", true);

    // fix "not no X" -> "no X" (conservative, only for immediate "not no")
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
        if let Some(last) = out.last() {
            if normalize_token(last) == normalize_token(&t) {
                continue;
            }
        }
        out.push(t);
    }
    out
}

fn drop_known_pair(tokens: Vec<String>, a: &str, b: &str) -> Vec<String> {
    if tokens.len() < 2 {
        return tokens;
    }

    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0usize;

    while i < tokens.len() {
        if i + 1 < tokens.len() && eq_loose(&tokens[i], a) && eq_loose(&tokens[i + 1], b) {
            // drop the first token, keep the second
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
    s.trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
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
            if let Some(n) = next {
                if !n.is_whitespace() && n != ',' && n != '.' && n != '!' && n != '?' {
                    out.push(' ');
                }
            }
        }

        i += 1;
    }

    out
}
