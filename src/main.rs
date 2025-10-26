use rusty_pcm_resolver::{MediaInput, resolve_pcm};
use std::path::PathBuf;

use rusty_transcriber::{
    RetrieveBinaryOptions, merge_sentence_segments, retrieve_binary, transcribe_to_file,
};

fn main() -> anyhow::Result<()> {
    let opts = RetrieveBinaryOptions::default();
    let model_path = retrieve_binary(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        &opts,
    )?;
    println!("Model ready at: {}", model_path.display());

    let samples = resolve_pcm(
        MediaInput::Url("https://5b54b4fce3488.streamlock.net/vods3/_definst_/mp4:amazons3/gowebflash/gorilla/gorilla_250814v2.mp4/playlist.m3u8".to_string()),
        None,
        Some(16_000),
        Some(1),
    )?;
    println!("Loaded {} samples from remote URL", samples.len());

    let segments = transcribe_to_file(
        &model_path,
        &samples,
        &PathBuf::from(".cache").join("transcripts"),
        Some("en"),
        false,
        8,
    )?;

    for seg in &segments {
        println!("[{:.2} - {:.2}] {}", seg.start_ms, seg.end_ms, seg.text);
    }

    println!();

    let merged_segments = merge_sentence_segments(&segments);
    for seg in &merged_segments {
        println!("[{:.2} - {:.2}] {}", seg.start_ms, seg.end_ms, seg.text);
    }

    Ok(())
}
