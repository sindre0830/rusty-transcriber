use anyhow::Context;
use rusty_pcm_resolver::PcmResolver;
use rusty_pcm_resolver::domain::MediaInput;

use rusty_transcriber::{Transcript, TranscriptOptions};

fn main() -> anyhow::Result<()> {
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
            "[{} {:.2} - {:.2}] {}",
            seg.speaker_id, seg.start_ms, seg.end_ms, seg.text
        );
    }

    Ok(())
}
