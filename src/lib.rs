use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::NamedTempFile;

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
