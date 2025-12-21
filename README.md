# Rusty Transcriber

**Rusty Transcriber** is a lightweight Rust crate for converting raw PCM audio data into text transcripts using speech-to-text models.
Currently supports Nvidia Parakeet models through [parakeet-rs](https://github.com/altunenes/parakeet-rs).

The Parakeet ONNX models (downloaded separately from HuggingFace) are licensed under CC-BY-4.0 by NVIDIA. This library does not distribute the models.

---

## Usage Guide

To include this crate in your project, add it to your dependencies:

```bash
cargo add --git https://github.com/sindre0830/rusty-transcriber.git --tag v1.0.0 rusty-transcriber
```

Or manually in your `Cargo.toml`:

```toml
[dependencies]
rusty-transcriber = { git = "https://github.com/sindre0830/rusty-transcriber.git", tag = "v1.0.0" }
```

### Example

This examples uses the [rusty-pcm-resolver](https://github.com/sindre0830/rusty-pcm-resolver) crate for preparing audio samples.

```rust
use anyhow::Context;
use rusty_pcm_resolver::PcmResolver;
use rusty_pcm_resolver::domain::MediaInput;

use rusty_transcriber::{Transcript, TranscriptOptions};

fn main() -> anyhow::Result<()> {
    let program_start = std::time::Instant::now();

    let sample_rate: u32 = 16_000;
    let channels: u8 = 1;

    let pcm_resolver_options =
        rusty_pcm_resolver::Options::new(MediaInput::Url("https://url.to/audio".into()))
            .sample_rate(sample_rate)
            .channels(channels);

    let samples = PcmResolver::new(pcm_resolver_options)
        .resolve_media()?
        .convert_to_pcm()?
        .load()?;
    println!("Samples: {}", samples.len());

    let options = TranscriptOptions::new();
    let transcript = Transcript::new(options)
        .prepare_transcriber_model(
            rusty_transcriber::io::ModelInput::BatchUrls(vec![
                "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx".into(),
                "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data".into(),
                "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx".into(),
                "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt".into(),
            ]),
            rusty_transcriber::io::RetrieveBinaryOptions::default(),
        )?
        .prepare_diarization_model(
            rusty_transcriber::io::ModelInput::Url(
                "https://huggingface.co/altunenes/parakeet-rs/resolve/main/diar_streaming_sortformer_4spk-v2.1.onnx".into(),
            ),
            rusty_transcriber::io::RetrieveBinaryOptions::default(),
        )?
        .transcribe(&samples, sample_rate, channels)
        .context("transcription failed")?;

    println!("Segments: {}", transcript.segments.len());
    for seg in &transcript.segments {
        println!(
            "[{} {} - {}] {}",
            seg.speaker_id, seg.start_ms, seg.end_ms, seg.text
        );
    }

    println!(
        "\nTotal program execution time: {:.2?}",
        program_start.elapsed()
    );

    Ok(())
}
```

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
