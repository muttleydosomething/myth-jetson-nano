//! Native HTTP response codecs. Reference decoding belongs to audio_preparation;
//! lossless inference receipts retain pipeline::wav_bytes. This boundary alone
//! implements the operational service's PCM16 WAV, MP3 and Ogg/Opus outputs.
//! Stable C ABIs: libsndfile/docs/api.md and lameproject/lame/include/lame.h.
//! Codec state is request-owned, inside the service's single inference permit.
use crate::{npy::F32Npy, Result};
use libloading::Library;
use std::{
    ffi::{c_int, c_short, c_ulong, c_void},
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::fd::{AsRawFd, FromRawFd},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wav,
    Mp3,
    Opus,
}
impl Format {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "opus" => Ok(Self::Opus),
            _ => Err("Supported response formats are wav, mp3 and opus".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
        }
    }
    pub fn content_type(self) -> &'static str {
        // Opus is transported in its Ogg container; use the actual media type.
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg",
        }
    }
}
fn checkpoint(cancelled: &mut impl FnMut() -> bool) -> Result<()> {
    if cancelled() {
        Err("Speech request cancelled".into())
    } else {
        Ok(())
    }
}
fn append(bytes: &mut Vec<u8>, chunk: &[u8]) -> Result<()> {
    bytes.try_reserve(chunk.len()).map_err(|e| e.to_string())?;
    bytes.extend_from_slice(chunk);
    Ok(())
}
fn pcm16(values: &[f32]) -> Vec<i16> {
    // The deployed NumPy path clips, multiplies in FP32, then truncates toward
    // zero. A float-to-PCM library conversion may round instead.
    values
        .iter()
        .map(|v| (v.clamp(-1., 1.) * 32767.) as i16)
        .collect()
}
/// Reject invalid controls before expensive inference. The deployed positive-rate
/// contract has no arbitrary utterance or tempo cap; output-size admission below
/// still rejects overflow, empty rounded output and failed allocations.
pub fn validate_speed(speed: f64) -> Result<()> {
    if !speed.is_finite() || speed <= 0. {
        return Err("Speech speed must be finite and positive".into());
    }
    Ok(())
}
fn validate_pcm(pcm: &F32Npy) -> Result<()> {
    if pcm.shape.len() != 2
        || pcm.shape[0] != 1
        || pcm.shape[1] == 0
        || pcm.shape[1] != pcm.values.len()
        || pcm.values.iter().any(|v| !v.is_finite())
    {
        return Err("Response processing requires finite mono samples [1,N]".into());
    }
    Ok(())
}

/// Native equivalent of the deployed librosa 0.11 time_stretch defaults:
/// centered zero-padded 2048-point periodic Hann, 512-sample hop, interpolated
/// magnitudes, unwrapped phase advance, and window-normalized overlap/add.
/// The phase-vocoder algorithm derives from librosa (ISC; see LICENSE).
/// Only two input spectra and one inverse frame are retained, independent of
/// utterance length. Output samples/normalization are the only growing buffers.
/// Normal speed borrows the input so its waveform bits and allocation stay intact.
pub fn tempo<'a>(
    pcm: &'a F32Npy,
    speed: f64,
    mut cancelled: impl FnMut() -> bool,
) -> Result<std::borrow::Cow<'a, F32Npy>> {
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::{borrow::Cow, f64::consts::TAU};
    checkpoint(&mut cancelled)?;
    validate_speed(speed)?;
    validate_pcm(pcm)?;
    if speed == 1. {
        return Ok(Cow::Borrowed(pcm));
    }
    const FFT: usize = 2048;
    const HOP: usize = FFT / 4;
    const BINS: usize = FFT / 2 + 1;
    // Match Python round, including halfway-to-even. Check before float-to-usize
    // conversion: Rust's saturating cast would otherwise conceal overflow.
    let length = (pcm.values.len() as f64 / speed).round_ties_even();
    if !length.is_finite() || length < 1. || length >= (isize::MAX as usize / 4) as f64 {
        return Err("Tempo output sample count is empty or exceeds addressable memory".into());
    }
    let length = length as usize;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|e| format!("Tempo output: {e}"))?;
    values.resize(length, 0f32);
    let mut norm = Vec::new();
    norm.try_reserve_exact(length)
        .map_err(|e| format!("Tempo normalization: {e}"))?;
    norm.resize(length, 0f32);
    let window: Vec<f64> = (0..FFT)
        .map(|i| 0.5 - 0.5 * (TAU * i as f64 / FFT as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(FFT);
    let inverse = planner.plan_fft_inverse(FFT);
    // Explicit reusable scratch avoids per-frame planner or FFT allocation.
    let mut scratch = vec![
        Complex::default();
        forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len())
    ];
    let mut frame = vec![Complex::default(); FFT];
    let mut left = vec![Complex::<f32>::default(); BINS];
    let mut right = left.clone();
    let mut phases = vec![0f32; BINS];
    let input_frames = 1 + pcm.values.len() / HOP;
    let mut cached = None;
    let mut output_frame = 0usize;
    // Read one centered input spectrum. Frames beyond the input analysis count
    // are zero, exactly as phase_vocoder's two trailing padding columns.
    let mut spectrum = |index: usize, bins: &mut [Complex<f32>], scratch: &mut [Complex<f64>]| {
        bins.fill(Complex::default());
        if index >= input_frames {
            return;
        }
        let center = index * HOP;
        for (i, slot) in frame.iter_mut().enumerate() {
            let source = center as i128 + i as i128 - (FFT / 2) as i128;
            let value = if source >= 0 && source < pcm.values.len() as i128 {
                pcm.values[source as usize] as f64
            } else {
                0.
            };
            *slot = Complex::new(value * window[i], 0.);
        }
        forward.process_with_scratch(&mut frame, scratch);
        for (out, value) in bins.iter_mut().zip(&frame) {
            *out = Complex::new(value.re as f32, value.im as f32);
        }
    };
    spectrum(0, &mut left, &mut scratch);
    for (phase, bin) in phases.iter_mut().zip(&left) {
        *phase = bin.arg();
    }
    // Separate inverse workspace permits the input-spectrum closure to retain
    // its analysis frame. No spectrum matrix or duration-sized time-step array.
    let mut reconstructed = vec![Complex::<f64>::default(); FFT];
    loop {
        checkpoint(&mut cancelled)?;
        let step = output_frame as f64 * speed;
        let start = output_frame
            .checked_mul(HOP)
            .ok_or("Tempo frame offset overflow")?;
        if step >= input_frames as f64 || start >= length.saturating_add(FFT / 2) {
            break;
        }
        let index = step.floor() as usize;
        if cached != Some(index) {
            if cached.and_then(|i: usize| i.checked_add(1)) == Some(index) {
                std::mem::swap(&mut left, &mut right);
            } else {
                spectrum(index, &mut left, &mut scratch);
            }
            spectrum(index + 1, &mut right, &mut scratch);
            cached = Some(index);
        }
        let alpha = step.fract();
        for bin in 0..BINS {
            let magnitude =
                ((1. - alpha) * left[bin].norm() as f64 + alpha * right[bin].norm() as f64) as f32;
            let value = Complex::<f32>::from_polar(magnitude, phases[bin]);
            reconstructed[bin] = Complex::new(value.re as f64, value.im as f64);
            let advance = TAU * HOP as f64 * bin as f64 / FFT as f64;
            // Match the deployed FP32 angle subtraction/accumulator, with
            // FP64 phase-unwrapping arithmetic and nearest-even wrap.
            let delta = (right[bin].arg() - left[bin].arg()) as f64 - advance;
            let delta = delta - TAU * (delta / TAU).round_ties_even();
            phases[bin] = (phases[bin] as f64 + advance + delta) as f32;
        }
        // A real inverse FFT ignores imaginary DC/Nyquist components.
        reconstructed[0].im = 0.;
        reconstructed[FFT / 2].im = 0.;
        for bin in 1..FFT / 2 {
            reconstructed[FFT - bin] = reconstructed[bin].conj();
        }
        inverse.process_with_scratch(&mut reconstructed, &mut scratch);
        for i in 0..FFT {
            let destination = start as i128 + i as i128 - (FFT / 2) as i128;
            if destination < 0 || destination >= length as i128 {
                continue;
            }
            let destination = destination as usize;
            // scipy's inverse is FP32, followed by its FP64 Hann multiplication.
            let sample = (reconstructed[i].re / FFT as f64) as f32;
            values[destination] = (values[destination] as f64 + sample as f64 * window[i]) as f32;
            norm[destination] += (window[i] * window[i]) as f32;
        }
        output_frame += 1;
    }
    for (out, denominator) in values.chunks_mut(4096).zip(norm.chunks(4096)) {
        checkpoint(&mut cancelled)?;
        for (value, &weight) in out.iter_mut().zip(denominator) {
            if weight > f32::MIN_POSITIVE {
                *value /= weight;
            }
            if !value.is_finite() {
                return Err("Tempo waveform overflow".into());
            }
        }
    }
    checkpoint(&mut cancelled)?;
    Ok(Cow::Owned(F32Npy {
        shape: vec![1, length],
        values,
    }))
}
pub fn encode(
    pcm: &F32Npy,
    format: Format,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Vec<u8>> {
    checkpoint(&mut cancelled)?;
    validate_pcm(pcm)?;
    let bytes = match format {
        Format::Wav => wav(&pcm.values, &mut cancelled),
        Format::Mp3 => mp3(&pcm.values, &mut cancelled),
        Format::Opus => opus(&pcm.values, &mut cancelled),
    }?;
    checkpoint(&mut cancelled)?;
    Ok(bytes)
}
fn wav(values: &[f32], cancelled: &mut impl FnMut() -> bool) -> Result<Vec<u8>> {
    let size = values
        .len()
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("PCM16 waveform exceeds RIFF size")?;
    let riff = size
        .checked_add(36)
        .ok_or("PCM16 waveform exceeds RIFF size")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(riff as usize + 8)
        .map_err(|e| e.to_string())?;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24000u32.to_le_bytes());
    bytes.extend_from_slice(&48000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&size.to_le_bytes());
    for chunk in values.chunks(4096) {
        checkpoint(cancelled)?;
        for sample in pcm16(chunk) {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    Ok(bytes)
}

type LameClose = unsafe extern "C" fn(*mut c_void) -> c_int;
struct Lame {
    pointer: *mut c_void,
    close: LameClose,
}
impl Drop for Lame {
    fn drop(&mut self) {
        // SAFETY: this handle is owned, and the library outlives the guard.
        unsafe {
            (self.close)(self.pointer);
        }
    }
}
fn mp3(values: &[f32], cancelled: &mut impl FnMut() -> bool) -> Result<Vec<u8>> {
    type Init = unsafe extern "C" fn() -> *mut c_void;
    type Set = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
    type SetSamples = unsafe extern "C" fn(*mut c_void, c_ulong) -> c_int;
    type InitParams = unsafe extern "C" fn(*mut c_void) -> c_int;
    type Encode = unsafe extern "C" fn(
        *mut c_void,
        *const c_short,
        *const c_short,
        c_int,
        *mut u8,
        c_int,
    ) -> c_int;
    type Flush = unsafe extern "C" fn(*mut c_void, *mut u8, c_int) -> c_int;
    type Tag = unsafe extern "C" fn(*const c_void, *mut u8, usize) -> usize;
    // SAFETY: fixed installed soname and stable lame.h signatures. Every source
    // chunk and destination buffer remains valid for each synchronous call.
    unsafe {
        let lib = Library::new("libmp3lame.so.0").map_err(|e| format!("MP3 codec: {e}"))?;
        let init = lib.get::<Init>(b"lame_init\0").map_err(|e| e.to_string())?;
        let close = *lib
            .get::<LameClose>(b"lame_close\0")
            .map_err(|e| e.to_string())?;
        let pointer = init();
        if pointer.is_null() {
            return Err("MP3 codec allocation failed".into());
        }
        let handle = Lame { pointer, close };
        for (symbol, value) in [
            (b"lame_set_num_channels\0".as_slice(), 1),
            (b"lame_set_in_samplerate\0".as_slice(), 24000),
            (b"lame_set_out_samplerate\0".as_slice(), 24000),
            (b"lame_set_brate\0".as_slice(), 64),
            (b"lame_set_quality\0".as_slice(), 3),
            (b"lame_set_bWriteVbrTag\0".as_slice(), 1),
        ] {
            if lib.get::<Set>(symbol).map_err(|e| e.to_string())?(handle.pointer, value) != 0 {
                return Err("MP3 codec configuration failed".into());
            }
        }
        let frames = c_ulong::try_from(values.len()).map_err(|_| "MP3 sample count overflow")?;
        if lib
            .get::<SetSamples>(b"lame_set_num_samples\0")
            .map_err(|e| e.to_string())?(handle.pointer, frames)
            != 0
            || lib
                .get::<InitParams>(b"lame_init_params\0")
                .map_err(|e| e.to_string())?(handle.pointer)
                < 0
        {
            return Err("MP3 codec initialization failed".into());
        }
        let encode = lib
            .get::<Encode>(b"lame_encode_buffer\0")
            .map_err(|e| e.to_string())?;
        let flush = lib
            .get::<Flush>(b"lame_encode_flush\0")
            .map_err(|e| e.to_string())?;
        let tag = lib
            .get::<Tag>(b"lame_get_lametag_frame\0")
            .map_err(|e| e.to_string())?;
        // LAME requires >=1.25*N+7200 bytes and >=7200 for flush.
        let mut buffer = vec![0u8; 16384];
        let mut bytes = Vec::new();
        for chunk in values.chunks(4096) {
            checkpoint(cancelled)?;
            let pcm = pcm16(chunk);
            let count = encode(
                handle.pointer,
                pcm.as_ptr(),
                pcm.as_ptr(),
                pcm.len() as c_int,
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
            );
            if count < 0 || count as usize > buffer.len() {
                return Err(format!("MP3 encoding failed: {count}"));
            }
            append(&mut bytes, &buffer[..count as usize])?;
        }
        checkpoint(cancelled)?;
        let count = flush(handle.pointer, buffer.as_mut_ptr(), buffer.len() as c_int);
        if count < 0 || count as usize > buffer.len() {
            return Err(format!("MP3 flush failed: {count}"));
        }
        append(&mut bytes, &buffer[..count as usize])?;
        // The seekable response buffer permits replacing LAME's initial Info
        // frame. Preserve encoder delay/padding so decoders recover exact length.
        let count = tag(handle.pointer, buffer.as_mut_ptr(), buffer.len());
        if count == 0 || count > buffer.len() || count > bytes.len() {
            return Err("MP3 gapless tag was not produced".into());
        }
        bytes[..count].copy_from_slice(&buffer[..count]);
        Ok(bytes)
    }
}

#[repr(C)]
#[derive(Default)]
struct SfInfo {
    frames: i64,
    samplerate: c_int,
    channels: c_int,
    format: c_int,
    sections: c_int,
    seekable: c_int,
}
fn opus(values: &[f32], cancelled: &mut impl FnMut() -> bool) -> Result<Vec<u8>> {
    type Open = unsafe extern "C" fn(c_int, c_int, *mut SfInfo, c_int) -> *mut c_void;
    type Close = unsafe extern "C" fn(*mut c_void) -> c_int;
    type Write = unsafe extern "C" fn(*mut c_void, *const f32, i64) -> i64;
    // A private, anonymous seekable descriptor lets libsndfile finalize its Ogg
    // pages without callbacks, a pathname, a subprocess or a temporary recording.
    // SAFETY: static nul-terminated name; File takes ownership exactly once.
    let fd = unsafe { libc::memfd_create(b"myth-opus\0".as_ptr().cast(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(format!(
            "Opus response allocation: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut file = unsafe { File::from_raw_fd(fd) };
    // SAFETY: stable sndfile.h ABI and owned descriptor. close_desc=0 leaves
    // File responsible for the descriptor; sf_close is called on every outcome.
    unsafe {
        let lib = Library::new("libsndfile.so.1").map_err(|e| format!("Opus codec: {e}"))?;
        let open = lib
            .get::<Open>(b"sf_open_fd\0")
            .map_err(|e| e.to_string())?;
        let close = lib.get::<Close>(b"sf_close\0").map_err(|e| e.to_string())?;
        let write = lib
            .get::<Write>(b"sf_writef_float\0")
            .map_err(|e| e.to_string())?;
        let mut info = SfInfo {
            samplerate: 24000,
            channels: 1,
            format: 0x200000 | 0x0064,
            ..Default::default()
        };
        let pointer = open(file.as_raw_fd(), 0x20, &mut info, 0);
        if pointer.is_null() {
            return Err("Opus codec could not open Ogg/Opus output".into());
        }
        // This closure deliberately owns all fallible encoding operations so
        // even cancellation closes/finalizes native state before returning.
        let result = (|| {
            for chunk in values.chunks(4096) {
                checkpoint(cancelled)?;
                if write(pointer, chunk.as_ptr(), chunk.len() as i64) != chunk.len() as i64 {
                    return Err("Opus codec short write".to_string());
                }
            }
            Ok(())
        })();
        let closed = close(pointer);
        result?;
        if closed != 0 {
            return Err(format!("Opus codec close failed: {closed}"));
        }
    }
    checkpoint(cancelled)?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.is_empty() {
        return Err("Opus codec produced empty output".into());
    }
    Ok(bytes)
}
