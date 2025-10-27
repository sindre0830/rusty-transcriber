use rusty_pcm_resolver::PcmResolver;
use rusty_pcm_resolver::domain::MediaInput;

use rusty_transcriber::{ModelInput, Options, TranscriberBuilder};

fn main() -> anyhow::Result<()> {
    let pcm_resolver_options =
        rusty_pcm_resolver::Options::new(MediaInput::Url("https://url.to/audio".into()));
    let samples = PcmResolver::new(pcm_resolver_options)
        .resolve_media()?
        .convert_to_pcm()?
        .load()?;
    println!("Samples: {}", samples.len());

    let options = Options {
        language: Some("en"),
        translate_to_english: false,
        n_threads: 6,
        cache_dir: ".cache".into(),
        model_fingerprint: None,
    };
    let transcript = TranscriberBuilder::new(options)
        .load_model(ModelInput::Url("https://url.to/whisper.model.bin".into()))?
        .transcribe(samples)?
        .merge_sentences();

    println!("Segments: {}", transcript.segments.len());
    for seg in &transcript.segments {
        println!("[{:.2} - {:.2}] {}", seg.start_ms, seg.end_ms, seg.text);
    }

    Ok(())
}
