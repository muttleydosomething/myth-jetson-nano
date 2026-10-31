//! Chatterbox Turbo conversion boundary: native text, embedding input, logits
//! processing, generation, TensorRT ownership and retained-voice waveform
//! synthesis and complete new-voice reference preparation. Production service
//! replacement remains separate.
pub mod audio_output;
pub mod audio_preparation;
pub mod campplus;
pub mod conditioning;
pub mod conformer;
mod conformer_trt;
pub mod generation;
pub mod gpu;
pub mod hift;
mod hift_trt;
pub mod meanflow;
mod meanflow_trt;
pub mod npy;
pub mod pipeline;
pub mod s3_tokenizer;
pub mod s3gen;
pub mod sampling;
pub mod service;
pub mod tensorrt;
pub mod text;
pub mod tokenizer;
mod trace;
pub mod voice_encoder;
pub mod voice_preparation;
pub mod workshop;

pub type Result<T> = std::result::Result<T, String>;
