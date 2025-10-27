use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::NamedTempFile;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, install_logging_hooks,
};

/// options for configuring a transcription
pub struct Options<'a> {
    pub language: Option<&'a str>,
    pub translate_to_english: bool,
    pub n_threads: i32,
    pub cache_dir: PathBuf,
    pub model_fingerprint: Option<String>,
}

impl<'a> Default for Options<'a> {
    fn default() -> Self {
        Self {
            language: None,
            translate_to_english: false,
            n_threads: 4,
            cache_dir: PathBuf::from(".cache"),
            model_fingerprint: None,
        }
    }
}

/// model source type (local or remote)
pub enum ModelInput {
    Path(PathBuf),
    Url(String),
}

/// represents one transcribed segment
#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// a transcript containing multiple segments
#[derive(Debug, Clone)]
pub struct Transcript {
    pub segments: Vec<Segment>,
}

impl Transcript {
    /// merges consecutive segments into full sentences
    pub fn merge_sentences(&self) -> Self {
        let mut merged = Vec::new();
        let mut current: Option<Segment> = None;

        for (i, seg) in self.segments.iter().enumerate() {
            let is_last = i + 1 == self.segments.len();
            let next = self.segments.get(i + 1);

            match &mut current {
                Some(cur) => {
                    cur.text.push(' ');
                    cur.text.push_str(seg.text.trim());
                    cur.end_ms = seg.end_ms;

                    let ends_sentence = seg.text.trim_end().ends_with(['.', '!', '?']);
                    let next_caps = next
                        .and_then(|n| n.text.trim().chars().next())
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false);

                    if ends_sentence || (next_caps && !is_last) {
                        merged.push(cur.clone());
                        current = None;
                    }
                }
                None => {
                    let n = seg.clone();
                    let ends = n.text.trim_end().ends_with(['.', '!', '?']);
                    let next_caps = next
                        .and_then(|n| n.text.trim().chars().next())
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false);

                    if ends || (next_caps && !is_last) {
                        merged.push(n);
                    } else {
                        current = Some(n);
                    }
                }
            }
        }

        if let Some(cur) = current {
            merged.push(cur);
        }

        Self { segments: merged }
    }
}

/// transcriber builder, handles setup of model, audio, and caching
pub struct TranscriberBuilder<'a> {
    options: Options<'a>,
    model_path: Option<PathBuf>,
}

impl<'a> TranscriberBuilder<'a> {
    pub fn new(options: Options<'a>) -> Self {
        Self {
            options,
            model_path: None,
        }
    }

    pub fn load_model(mut self, input: ModelInput) -> Result<Self> {
        let model_path = match input {
            ModelInput::Path(p) => p,
            ModelInput::Url(u) => {
                let opts = RetrieveBinaryOptions::default();
                retrieve_binary(&u, &opts)?
            }
        };

        self.model_path = Some(model_path);
        Ok(self)
    }

    pub fn transcribe(self, samples: Vec<f32>) -> Result<Transcript> {
        let model_input = self.model_path.ok_or_else(|| anyhow!("model not loaded"))?;

        let model_fp = if let Some(f) = &self.options.model_fingerprint {
            f.clone()
        } else {
            model_fingerprint(&model_input)?
        };

        let model_str = model_input
            .to_str()
            .ok_or_else(|| anyhow!("invalid utf-8 in model path"))?;

        let cache_dir = &self.options.cache_dir.join("rusty_transcriber");

        fs::create_dir_all(cache_dir).ok();

        install_logging_hooks();
        let ctx = WhisperContext::new_with_params(model_str, WhisperContextParameters::default())?;

        // compute stable hash for samples
        let audio_id = hash_samples(&samples);
        let opts_hash = opts_hash(
            self.options.language,
            self.options.translate_to_english,
            self.options.n_threads,
        );
        let out_path = &cache_dir
            .join("transcripts")
            .join(&model_fp)
            .join(&audio_id)
            .join(format!("{}.txt", &opts_hash));

        if out_path.exists() {
            return read_txt(out_path);
        }

        let mut state = ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        params.set_n_threads(self.options.n_threads.max(1));
        params.set_translate(self.options.translate_to_english);
        params.set_language(self.options.language);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, &samples)?;

        let n = state.full_n_segments();
        let mut segs = Vec::with_capacity(n as usize);
        for i in 0..n {
            let s = state
                .get_segment(i)
                .ok_or_else(|| anyhow!("no segment {i}"))?;
            segs.push(Segment {
                start_ms: s.start_timestamp(),
                end_ms: s.end_timestamp(),
                text: s.to_string(),
            });
        }

        let transcript = Transcript { segments: segs };
        write_txt_atomic(out_path, &transcript)?;
        Ok(transcript)
    }
}

/// configuration for downloading and caching models
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
            cache_dir: PathBuf::from(".cache"),
            filename_override: None,
            expected_sha256: None,
            timeout: Duration::from_secs(60),
            user_agent: None,
        }
    }
}

/// download binary and cache it locally
pub fn retrieve_binary(url: &str, opts: &RetrieveBinaryOptions) -> Result<PathBuf> {
    let cache_dir = &opts.cache_dir.join("rusty_transcriber").join("bin");
    fs::create_dir_all(cache_dir).context("failed to create cache directory")?;

    let filename = opts
        .filename_override
        .as_ref()
        .map(|s| s.to_string())
        .or_else(|| url.split('/').next_back().map(|s| s.to_string()))
        .ok_or_else(|| anyhow!("could not infer filename from url"))?;

    let target_path = cache_dir.join(&filename);

    if target_path.exists() {
        if let Some(exp) = &opts.expected_sha256 {
            if verify_sha256(&target_path, exp)? {
                return Ok(target_path);
            } else {
                let _ = fs::remove_file(&target_path);
            }
        } else {
            return Ok(target_path);
        }
    }

    let client = Client::builder()
        .timeout(opts.timeout)
        .redirect(Policy::limited(10))
        .build()
        .context("failed to build http client")?;

    let mut req = client.get(url);
    if let Some(ua) = &opts.user_agent {
        req = req.header(reqwest::header::USER_AGENT, ua);
    }

    let resp = req.send().context("request failed")?;
    if !resp.status().is_success() {
        bail!("bad status {} for {}", resp.status(), url);
    }
    let bytes = resp.bytes().context("reading body failed")?;

    if let Some(exp) = &opts.expected_sha256 {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got = hex::encode(hasher.finalize());
        if got != *exp {
            bail!("checksum mismatch: expected {}, got {}", exp, got);
        }
    }

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

/// compute a fingerprint for the model (for cache namespace)
pub fn model_fingerprint(path: &Path) -> Result<String> {
    let data = fs::read(path)?;
    let mut h = blake3::Hasher::new();
    h.update(&data);
    Ok(h.finalize().to_hex().to_string())
}

/// compute stable hash of pcm samples (content-addressed cache key)
pub fn hash_samples(samples: &[f32]) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr() as *const u8,
            std::mem::size_of_val(samples),
        )
    };
    let mut h = blake3::Hasher::new();
    h.update(bytes);
    h.finalize().to_hex().to_string()
}

/// compute hash of options affecting transcript output
fn opts_hash(language: Option<&str>, translate: bool, n_threads: i32) -> String {
    let mut h = blake3::Hasher::new();
    h.update(&[translate as u8]);
    h.update(&n_threads.to_le_bytes());
    if let Some(lang) = language {
        h.update(lang.as_bytes());
    }
    h.finalize().to_hex().to_string()
}

/// read cached transcript from txt
fn read_txt(path: &Path) -> Result<Transcript> {
    use std::io::{BufRead, BufReader};
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut segments = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed reading {}", path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let (range, text) = line
            .split_once("]:")
            .ok_or_else(|| anyhow!("missing ']: ' on line {}", idx + 1))?;
        let range = range.trim_start_matches('[').trim();
        let (start, end) = range
            .split_once('-')
            .ok_or_else(|| anyhow!("missing '-' in time range"))?;
        let start_ms: i64 = start.trim().parse().context("invalid start")?;
        let end_ms: i64 = end.trim().parse().context("invalid end")?;

        segments.push(Segment {
            start_ms,
            end_ms,
            text: text.trim().to_string(),
        });
    }

    Ok(Transcript { segments })
}

/// write transcript atomically to txt
fn write_txt_atomic(path: &Path, t: &Transcript) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).ok();
    }
    let tmp = tmp_path_for(path);
    {
        use std::io::Write;
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("failed to create {}", tmp.display()))?;
        for s in &t.segments {
            writeln!(f, "[{} - {}]: {}", s.start_ms, s.end_ms, s.text)?;
        }
        let _ = f.flush();
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("failed to move {} -> {}", tmp.display(), path.display()))
}

fn tmp_path_for(final_path: &Path) -> PathBuf {
    let p = final_path.to_path_buf();
    if let Some(ext) = p.extension() {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("whisper");
        let parent = p.parent().unwrap_or_else(|| Path::new("."));
        return parent.join(format!("{stem}.tmp.{}", ext.to_string_lossy()));
    }
    p.with_extension("tmp")
}
