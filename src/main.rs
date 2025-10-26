use rusty_transcriber::{RetrieveBinaryOptions, retrieve_binary};

fn main() -> anyhow::Result<()> {
    let opts = RetrieveBinaryOptions::default();

    let model = retrieve_binary(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        &opts,
    )?;

    println!("Model ready at: {}", model.display());
    Ok(())
}
