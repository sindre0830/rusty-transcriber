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
