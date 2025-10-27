# Rusty Transcriber

**Rusty Transcriber** is a lightweight Rust crate for converting raw PCM audio data into text transcripts using speech-to-text models.
Currently supports OpenAI Whisper models through [whisper-rs](https://codeberg.org/tazz4843/whisper-rs).

## Prerequisites

* `libclang` required for whisper-rs

---

## Usage Guide

To include this crate in your project, add it to your dependencies:

```bash
cargo add --git https://github.com/sindre0830/rusty-transcriber.git --tag v0.1.0 rusty-transcriber
```

Or manually in your `Cargo.toml`:

```toml
[dependencies]
rusty-transcriber = { git = "https://github.com/sindre0830/rusty-transcriber.git", tag = "v0.1.0" }
```

### Example

This examples uses the [rusty-pcm-resolver](https://github.com/sindre0830/rusty-pcm-resolver) crate for preparing audio samples.

```rust
use rusty_pcm_resolver::PcmResolver;
use rusty_pcm_resolver::domain::MediaInput;

use rusty_transcriber::{ModelInput, Options, RetrieveBinaryOptions, TranscriberBuilder};

fn main() -> anyhow::Result<()> {
    let pcm_resolver_options =
        rusty_pcm_resolver::Options::new(MediaInput::Url("https://url.to/audio".into()));
    let samples = PcmResolver::new(pcm_resolver_options)
        .resolve_media()?
        .convert_to_pcm()?
        .load()?;
    println!("Samples: {}", samples.len());

    let model_options = RetrieveBinaryOptions::default();
    let options = Options::new().language("en").threads(6);
    let transcript = TranscriberBuilder::new(options)
        .load_model(
            ModelInput::Url(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".into(),
            ),
            model_options,
        )?
        .transcribe(samples)?
        .merge_sentences();

    println!("Segments: {}", transcript.segments.len());
    for seg in &transcript.segments {
        println!("[{:.2} - {:.2}] {}", seg.start_ms, seg.end_ms, seg.text);
    }

    Ok(())
}
```

---

## API Overview

### Core Flow (Builder)

| Method                             | Input                                 | Output                       | Description                                                                                   |
| ---------------------------------- | ------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------- |
| `TranscriberBuilder::new(options)` | `Options`                             | `TranscriberBuilder`         | Create a builder with runtime/config options.                                                 |
| `load_model(input, model_options)` | `ModelInput`, `RetrieveBinaryOptions` | `Result<TranscriberBuilder>` | Resolve the Whisper model path. Downloads if URL is given.                                    |
| `transcribe(samples)`              | `Vec<f32>` (mono 16 kHz)              | `Result<Transcript>`         | Run Whisper on in-memory PCM samples. Uses cached transcript if available.                    |
| `merge_sentences()`                | —                                     | `Transcript`                 | Combine adjacent short fragments into sentence-level segments. Optional post-processing step. |

> Order: **new -> load_model -> transcribe -> merge_sentences (optional)**

---

### Configuration

#### `Options` (transcription runtime)

| Field                  | Type             | Default  | Description                                          |
| ---------------------- | ---------------- | -------- | ---------------------------------------------------- |
| `language`             | `Option<String>` | `None`   | Target language code (e.g., `"en"`).                 |
| `translate_to_english` | `bool`           | `false`  | Translate output to English if supported.            |
| `n_threads`            | `i32`            | `4`      | Number of CPU threads used by Whisper.               |
| `cache_dir`            | `PathBuf`        | `.cache` | Root directory for transcript caching.               |
| `model_fingerprint`    | `Option<String>` | `None`   | Optional override; otherwise computed automatically. |

#### `RetrieveBinaryOptions` (model download)

| Field               | Type             | Default  | Description                       |
| ------------------- | ---------------- | -------- | --------------------------------- |
| `cache_dir`         | `PathBuf`        | `.cache` | Base model cache directory.       |
| `filename_override` | `Option<String>` | `None`   | Custom filename for the download. |
| `expected_sha256`   | `Option<String>` | `None`   | Optional integrity verification.  |
| `timeout`           | `Duration`       | `60 s`   | HTTP timeout.                     |
| `user_agent`        | `Option<String>` | `None`   | Custom User-Agent header.         |

---

### Types

| Type         | Description                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| `ModelInput` | `Path(PathBuf)` or `Url(String)` — where to load the Whisper model from.                                |
| `Segment`    | `{ start_ms: i64, end_ms: i64, text: String }` — individual decoded chunk.                              |
| `Transcript` | `{ segments: Vec<Segment> }` — full transcription result with `.merge_sentences()` for post-processing. |

---

### Utilities

| Function            | Input                            | Output            | Description                                                      |
| ------------------- | -------------------------------- | ----------------- | ---------------------------------------------------------------- |
| `retrieve_binary`   | `&str`, `&RetrieveBinaryOptions` | `Result<PathBuf>` | Downloads and caches a binary file (e.g., model).                |
| `model_fingerprint` | `&Path`                          | `Result<String>`  | Returns a `blake3` fingerprint of model bytes.                   |
| `hash_samples`      | `&[f32]`                         | `String`          | Computes `blake3` hash of PCM content for deterministic caching. |

---

## Development Guide

### Commands

| Command            | Description                             | Example                                                    |
| ------------------ | --------------------------------------- | ---------------------------------------------------------- |
| **Build**          | Compiles the crate in release mode      | `cargo build --release`                                    |
| **Run Tests**      | Executes all unit tests                 | `cargo test`                                               |
| **Lint (Clippy)**  | Checks for style and performance issues | `cargo clippy --all-targets --all-features -- -D warnings` |
| **Format Code**    | Formats the entire codebase             | `cargo fmt`                                                |
| **Doc Generation** | Builds local documentation              | `cargo doc --open`                                         |

---

### Upgrading Dependencies

To upgrade all dependencies to the latest compatible versions:

```bash
cargo update
```

Or for a specific crate:

```bash
cargo update -p crate-name
```
