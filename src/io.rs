use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

/// model source type (local or remote)
pub enum ModelInput {
    Path(PathBuf),
    Url(String),
    BatchUrls(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct RetrieveBinaryOptions {
    pub cache_dir: PathBuf,
    pub namespace: PathBuf,
    pub filename_override: Option<String>,
    pub expected_sha256: Option<String>,
    pub timeout: Duration,
    pub user_agent: Option<String>,
    pub retries: usize,
}

impl RetrieveBinaryOptions {
    pub fn new() -> Self {
        Self {
            cache_dir: PathBuf::from(".cache"),
            namespace: PathBuf::from("rusty_transcriber/bin"),
            filename_override: None,
            expected_sha256: None,
            timeout: Duration::from_secs(60),
            user_agent: None,
            retries: 2,
        }
    }

    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    pub fn namespace(mut self, ns: impl Into<PathBuf>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn filename_override(mut self, name: impl Into<String>) -> Self {
        self.filename_override = Some(name.into());
        self
    }

    pub fn expected_sha256(mut self, checksum: impl Into<String>) -> Self {
        self.expected_sha256 = Some(checksum.into());
        self
    }

    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = duration;
        self
    }

    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }

    pub fn retries(mut self, retries: usize) -> Self {
        self.retries = retries;
        self
    }
}

impl Default for RetrieveBinaryOptions {
    fn default() -> Self {
        Self::new()
    }
}

pub fn retrieve_binary(url: &str, opts: &RetrieveBinaryOptions) -> Result<PathBuf> {
    let cache_dir = opts.cache_dir.join(&opts.namespace);
    fs::create_dir_all(&cache_dir).context("failed to create cache directory")?;

    let filename = opts
        .filename_override
        .clone()
        .or_else(|| infer_filename_from_url(url))
        .ok_or_else(|| anyhow!("could not infer filename from url"))?;

    let target_path = cache_dir.join(&filename);

    if target_path.exists() {
        if let Some(exp) = &opts.expected_sha256 {
            if verify_sha256_streaming(&target_path, exp)? {
                return Ok(target_path);
            }

            let _ = fs::remove_file(&target_path);
        } else {
            return Ok(target_path);
        }
    }

    let client = Client::builder()
        .timeout(opts.timeout)
        .redirect(Policy::limited(10))
        .build()
        .context("failed to build http client")?;

    for attempt in 0..=opts.retries {
        match download_to_path(&client, url, opts, &target_path) {
            Ok(path) => return Ok(path),
            Err(_err) if attempt < opts.retries => {
                // simple backoff
                let backoff_ms = 250u64.saturating_mul(1u64 << attempt);
                std::thread::sleep(Duration::from_millis(backoff_ms));
                continue;
            }
            Err(err) => return Err(err),
        }
    }

    Err(anyhow!("download attempts exhausted"))
}

fn download_to_path(
    client: &Client,
    url: &str,
    opts: &RetrieveBinaryOptions,
    target_path: &Path,
) -> Result<PathBuf> {
    let mut req = client.get(url);
    if let Some(ua) = &opts.user_agent {
        req = req.header(reqwest::header::USER_AGENT, ua);
    }

    let mut resp = req
        .send()
        .with_context(|| format!("request failed: {}", url))?;
    if !resp.status().is_success() {
        bail!("bad status {} for {}", resp.status(), url);
    }

    let parent = target_path
        .parent()
        .ok_or_else(|| anyhow!("invalid target path"))?;
    let mut tmp = NamedTempFile::new_in(parent).context("failed to create temp file")?;

    let mut sha = Sha256::new();
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = resp
            .read(&mut buf)
            .context("failed reading response body")?;
        if n == 0 {
            break;
        }

        sha.update(&buf[..n]);
        tmp.write_all(&buf[..n])
            .context("failed writing temp file")?;
    }

    if let Some(exp) = &opts.expected_sha256 {
        let got = hex::encode(sha.finalize());
        if got != *exp {
            bail!("checksum mismatch: expected {}, got {}", exp, got);
        }
    }

    tmp.persist(target_path)
        .map_err(|e| anyhow!("persist failed: {}", e))?;

    Ok(target_path.to_path_buf())
}

fn infer_filename_from_url(url: &str) -> Option<String> {
    let without_fragment = url.split('#').next()?;
    let without_query = without_fragment.split('?').next()?;
    without_query
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn verify_sha256_streaming(path: &Path, expected: &str) -> Result<bool> {
    let mut file = fs::File::open(path).context("failed to open file for sha256")?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = file
            .read(&mut buf)
            .context("failed reading file for sha256")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    let got = hex::encode(hasher.finalize());
    Ok(got == expected)
}

pub fn model_fingerprint(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).context("failed to open file for fingerprint")?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = file
            .read(&mut buf)
            .context("failed reading file for fingerprint")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(hasher.finalize().to_hex().to_string())
}

pub fn samples_fingerprint(samples: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();

    let bytes = unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr() as *const u8,
            std::mem::size_of_val(samples),
        )
    };
    hasher.update(bytes);

    hasher.finalize().to_hex().to_string()
}

/// Download multiple files from URLs into a single subfolder and return the folder path.
/// Creates a unique folder based on the combined URLs, useful for models requiring multiple files.
pub fn retrieve_batch_files(urls: &[String], opts: &RetrieveBinaryOptions) -> Result<PathBuf> {
    if urls.is_empty() {
        bail!("batch urls cannot be empty");
    }

    // Create a unique folder name based on all URLs combined
    let mut hasher = blake3::Hasher::new();
    for url in urls {
        hasher.update(url.as_bytes());
        hasher.update(b"|"); // separator
    }
    let folder_name = hasher.finalize().to_hex().to_string();

    let cache_dir = opts.cache_dir.join(&opts.namespace);
    let model_dir = cache_dir.join(folder_name);

    // Extract filenames from URLs
    let filenames: Vec<String> = urls
        .iter()
        .map(|url| {
            infer_filename_from_url(url)
                .ok_or_else(|| anyhow!("could not infer filename from url: {}", url))
        })
        .collect::<Result<Vec<_>>>()?;

    // Check if all files already exist
    let all_exist = filenames.iter().all(|f| model_dir.join(f).exists());

    if all_exist {
        return Ok(model_dir);
    }

    // Create the model directory
    fs::create_dir_all(&model_dir).context("failed to create batch download directory")?;

    // Download each file
    for (url, filename) in urls.iter().zip(filenames.iter()) {
        let file_path = model_dir.join(filename);

        // Skip if file already exists
        if file_path.exists() {
            continue;
        }

        // Create a temporary RetrieveBinaryOptions for this file
        let file_opts = RetrieveBinaryOptions {
            cache_dir: model_dir.clone(),
            namespace: PathBuf::from(""),
            filename_override: Some(filename.clone()),
            expected_sha256: None,
            timeout: opts.timeout,
            user_agent: opts.user_agent.clone(),
            retries: opts.retries,
        };

        retrieve_binary(url, &file_opts)
            .with_context(|| format!("failed to download batch file: {}", filename))?;
    }

    Ok(model_dir)
}
