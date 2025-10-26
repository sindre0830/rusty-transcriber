# Rusty Transcriber

**Rusty Transcriber** is a lightweight Rust crate for converting raw PCM audio data into text transcripts using speech-to-text models.

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

---

## API Overview

### Core Functions

---

## Development Guide

### Prerequisites

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
