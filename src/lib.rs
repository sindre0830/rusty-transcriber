use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::NamedTempFile;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, install_logging_hooks,
};

pub fn merge_sentence_segments(segments: &[TranscriptSegment]) -> Vec<TranscriptSegment> {
    let mut merged = Vec::new();
    let mut current = None::<TranscriptSegment>;

    for (i, seg) in segments.iter().enumerate() {
        let is_last = i + 1 == segments.len();
        let next = segments.get(i + 1);

        match current.as_mut() {
            Some(cur) => {
                // append this segment
                cur.text.push(' ');
                cur.text.push_str(seg.text.trim());

                cur.end_ms = seg.end_ms;

                // check if this segment ends a sentence
                let ends_sentence = seg.text.trim_end().ends_with(['.', '!', '?']);

                // check if next segment likely starts a new one
                let next_starts_capital = next
                    .and_then(|n| n.text.trim().chars().next())
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);

                if ends_sentence || (next_starts_capital && !is_last) {
                    // finalize sentence
                    merged.push(cur.clone());
                    current = None;
                }
            }
            None => {
                // start new merged segment
                let new_seg = seg.clone();

                // same logic for single-sentence segments
                let ends_sentence = seg.text.trim_end().ends_with(['.', '!', '?']);
                let next_starts_capital = next
                    .and_then(|n| n.text.trim().chars().next())
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);

                if ends_sentence || (next_starts_capital && !is_last) {
                    merged.push(new_seg);
                } else {
                    current = Some(new_seg);
                }
            }
        }
    }

    // flush any unfinished segment
    if let Some(cur) = current {
        merged.push(cur);
    }

    merged
}

/// represents one transcribed segment with timing and text
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start_ms: i32,
    pub end_ms: i32,
    pub text: String,
}

/// transcribe an in-memory mono 16 kHz f32 PCM buffer and return transcript lines (with caching).
pub fn transcribe_to_file(
    model_path: &Path,
    samples: &[f32],
    cache_dir: &Path,
    language: Option<&str>,
    translate_to_english: bool,
    n_threads: i32,
) -> Result<Vec<TranscriptSegment>> {
    install_logging_hooks();

    // basic input checks
    if samples.is_empty() {
        anyhow::bail!("audio buffer is empty");
    }

    // ensure output directory exists
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create output directory: {}", cache_dir.display()))?;

    // compute cache file path
    let file_stem = hash_samples(samples);
    let out_path = cache_dir.join(format!("{file_stem}.txt"));

    // if cached file exists, read & return
    if out_path.exists() {
        return read_segments(&out_path);
    }

    // load model and create state
    let model_str = model_path
        .to_str()
        .ok_or_else(|| anyhow!("invalid UTF-8 in model path: {}", model_path.display()))?;

    let ctx = WhisperContext::new_with_params(model_str, WhisperContextParameters::default())
        .map_err(|e| anyhow!("failed to load model '{}': {e}", model_path.display()))?;

    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow!("failed to create state: {e}"))?;

    // set decode parameters
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    params.set_n_threads(n_threads.max(1));
    params.set_translate(translate_to_english);
    params.set_language(language);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // run inference
    state
        .full(params, samples)
        .map_err(|e| anyhow!("inference failed: {e}"))?;

    // write segments atomically: write to .tmp, then rename to .txt
    let tmp_path = tmp_path_for(&out_path);
    {
        let mut file = File::create(&tmp_path)
            .with_context(|| format!("failed to create {}", tmp_path.display()))?;

        let n = state.full_n_segments();
        for i in 0..n {
            let seg = state
                .get_segment(i)
                .ok_or_else(|| anyhow!("no segment at index {i}"))?;

            let text = seg.to_string();
            let t0 = seg.start_timestamp();
            let t1 = seg.end_timestamp();

            // format: [start_ms - end_ms]: text
            writeln!(file, "[{} - {}]: {}", t0, t1, text)
                .with_context(|| "failed to write segment")?;
        }
        file.flush().ok();
    }

    fs::rename(&tmp_path, &out_path).with_context(|| {
        format!(
            "failed to move {} -> {}",
            tmp_path.display(),
            out_path.display()
        )
    })?;

    // return freshly written lines
    read_segments(&out_path)
}

/// read a transcript file into a vector of segments.
///
/// expected line format:
/// ```text
/// [123.45 - 678.90]: hello world
/// ```
pub fn read_segments(path: &Path) -> Result<Vec<TranscriptSegment>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut segments = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed reading {}", path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // find the bracketed time range and the following text
        let parsed =
            parse_segment_line(line).with_context(|| format!("line {} malformed", idx + 1))?;
        segments.push(parsed);
    }

    Ok(segments)
}

/// parse one line like `[12.34 - 56.78]: hello there`
fn parse_segment_line(line: &str) -> Result<TranscriptSegment> {
    // format: [start - end]: text
    // example: [0.00 - 12.34]: hello world
    let Some((range_part, text_part)) = line.split_once("]:") else {
        return Err(anyhow!("missing ']: ' separator"));
    };

    let range_part = range_part.trim_start_matches('[').trim();
    let text = text_part.trim().to_string();

    let Some((start_str, end_str)) = range_part.split_once('-') else {
        return Err(anyhow!("missing '-' in time range"));
    };

    let start_ms: i32 = start_str
        .trim()
        .parse()
        .map_err(|_| anyhow!("invalid start time '{}'", start_str))?;
    let end_ms: i32 = end_str
        .trim()
        .parse()
        .map_err(|_| anyhow!("invalid end time '{}'", end_str))?;

    Ok(TranscriptSegment {
        start_ms,
        end_ms,
        text,
    })
}

/// generate a temp file path next to the final path.
fn tmp_path_for(final_path: &Path) -> PathBuf {
    let p = final_path.to_path_buf();
    if let Some(ext) = p.extension() {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("whisper")
            .to_string();
        let parent = p.parent().unwrap_or_else(|| Path::new("."));
        return parent.join(format!("{stem}.tmp.{}", ext.to_string_lossy()));
    }
    p.with_extension("tmp")
}

pub fn hash_samples(samples: &[f32]) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr() as *const u8,
            std::mem::size_of_val(samples),
        )
    };

    let hash = blake3::hash(bytes);
    hash.to_hex().to_string()
}

pub struct RetrieveBinaryOptions {
    pub cache_dir: PathBuf,
    pub filename_override: Option<String>,
    pub expected_sha256: Option<String>,
    pub timeout: Duration,
    pub user_agent: Option<String>,
}

impl Default for RetrieveBinaryOptions {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".cache").join("bin"),
            filename_override: None,
            expected_sha256: None,
            timeout: Duration::from_secs(60),
            user_agent: None,
        }
    }
}

/// retrieves a binary file and caches it locally.
///
/// - Downloads only if missing or checksum mismatch.
/// - Verifies optional SHA-256 checksum.
/// - Returns the final cached file path.
///
/// Example cache layout: `.cache/bin/ggml-base.en.bin`
pub fn retrieve_binary(url: &str, opts: &RetrieveBinaryOptions) -> Result<PathBuf> {
    // choose cache base
    fs::create_dir_all(&opts.cache_dir).context("failed to create cache directory")?;

    // infer filename
    let filename = opts
        .filename_override
        .as_ref()
        .map(|s| s.to_string())
        .or_else(|| url.split('/').next_back().map(|s| s.to_string()))
        .ok_or_else(|| anyhow!("could not infer filename from url"))?;

    let target_path = opts.cache_dir.join(&filename);

    // verify existing file if available
    if target_path.exists() {
        if let Some(exp) = &opts.expected_sha256 {
            if verify_sha256(&target_path, exp)? {
                return Ok(target_path);
            } else {
                fs::remove_file(&target_path).ok();
            }
        } else {
            return Ok(target_path);
        }
    }

    // build client
    let client = Client::builder()
        .timeout(opts.timeout)
        .redirect(Policy::limited(10))
        .build()
        .context("failed to build http client")?;

    let mut req = client.get(url);
    if let Some(ua) = &opts.user_agent {
        req = req.header(reqwest::header::USER_AGENT, ua);
    }

    // download
    let resp = req.send().context("request failed")?;
    if !resp.status().is_success() {
        bail!("bad status {} for {}", resp.status(), url);
    }
    let bytes = resp.bytes().context("reading body failed")?;

    // checksum verify
    if let Some(exp) = &opts.expected_sha256 {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got = hex::encode(hasher.finalize());
        if got != *exp {
            bail!("checksum mismatch: expected {}, got {}", exp, got);
        }
    }

    // atomic write
    let parent = target_path
        .parent()
        .ok_or_else(|| anyhow!("invalid target path"))?;
    let mut tmp = NamedTempFile::new_in(parent)?;
    std::io::Write::write_all(&mut tmp, &bytes)?;
    tmp.persist(&target_path)
        .map_err(|e| anyhow!("persist failed: {}", e))?;

    Ok(target_path)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<bool> {
    let data = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let got = hex::encode(hasher.finalize());
    Ok(got == expected)
}
