//! New-reference audio stage, before the three established neural encoders.
//! Rust owns mono mixing, K-weighting, gating, normalization and reference cuts.
//! Existing libsndfile/SoXR C ABIs supply codec decoding and HQ resampling,
//! matching librosa's native backends without an interpreter or subprocess.
use crate::Result;
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_uint, c_ulong, c_void, CStr},
    fs::OpenOptions,
    os::fd::AsRawFd,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    ptr,
};

// Exact public stable ABIs from sndfile.h and soxr.h. Libraries remain owned by
// AudioPreparation while function pointers or native handles can be used.
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
#[repr(C)]
struct Quality {
    precision: f64,
    phase_response: f64,
    passband_end: f64,
    stopband_begin: f64,
    reserved: *mut c_void,
    flags: c_ulong,
}
type SfOpen = unsafe extern "C" fn(c_int, c_int, *mut SfInfo, c_int) -> *mut c_void;
type SfClose = unsafe extern "C" fn(*mut c_void) -> c_int;
type SfRead = unsafe extern "C" fn(*mut c_void, *mut f32, i64) -> i64;
type SfError = unsafe extern "C" fn(*mut c_void) -> c_int;
type SfStrError = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type QualitySpec = unsafe extern "C" fn(c_ulong, c_ulong) -> Quality;
type OneShot = unsafe extern "C" fn(
    f64,
    f64,
    c_uint,
    *const c_void,
    usize,
    *mut usize,
    *mut c_void,
    usize,
    *mut usize,
    *const c_void,
    *const Quality,
    *const c_void,
) -> *const c_char;

pub struct AudioPreparation {
    sndfile: Library,
    soxr: Library,
}
struct SoundFile {
    pointer: *mut c_void,
    close: SfClose,
}
impl Drop for SoundFile {
    fn drop(&mut self) {
        // SAFETY: sf_open_fd created this owned handle. The library and borrowed
        // descriptor outlive this guard; sf_close does not close the descriptor.
        unsafe {
            (self.close)(self.pointer);
        }
    }
}
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub channels: usize,
    pub mono: Vec<f32>,
}
pub struct PreparedAudio {
    /// Full normalized recording: loudness must be measured before the 10 s cut.
    pub pcm24: Vec<f32>,
    /// Full normalized/resampled recording for T3 speaker embedding.
    pub pcm16: Vec<f32>,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    /// Silence/below-gate input has no finite LUFS value and is left unchanged.
    pub loudness_lufs: Option<f64>,
    pub gain: f64,
}
impl PreparedAudio {
    pub fn decoder_reference(&self) -> &[f32] {
        &self.pcm24[..self.pcm24.len().min(10 * 24000)]
    }
    pub fn t3_reference(&self) -> &[f32] {
        &self.pcm16[..self.pcm16.len().min(15 * 16000)]
    }
}
fn native_error(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return "unknown native audio error".into();
    }
    // SAFETY: the ABI returns a nul-terminated, library-owned error string.
    unsafe { CStr::from_ptr(pointer).to_string_lossy().into_owned() }
}
impl AudioPreparation {
    pub fn open() -> Result<Self> {
        // SAFETY: load the installed native audio sonames, never user-selected
        // executable code. No symbol or handle outlives these Library objects.
        unsafe {
            Ok(Self {
                sndfile: Library::new("libsndfile.so.1").map_err(|e| format!("libsndfile: {e}"))?,
                soxr: Library::new("libsoxr.so.0").map_err(|e| format!("SoXR: {e}"))?,
            })
        }
    }
    pub fn decode(&self, path: &Path) -> Result<DecodedAudio> {
        // Open without waiting on a FIFO, then check the descriptor itself.
        // A path-only precheck would race replacement between check and open.
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .map_err(|e| format!("Audio source: {e}"))?;
        if !file.metadata().map_err(|e| e.to_string())?.is_file() {
            return Err("Audio source must be a regular recording".into());
        }
        // SAFETY: symbol signatures match sndfile.h; input/output buffers and
        // SfInfo remain valid for each call. Handle guard closes on every exit.
        unsafe {
            let open = self
                .sndfile
                .get::<SfOpen>(b"sf_open_fd\0")
                .map_err(|e| e.to_string())?;
            let close = *self
                .sndfile
                .get::<SfClose>(b"sf_close\0")
                .map_err(|e| e.to_string())?;
            let read = self
                .sndfile
                .get::<SfRead>(b"sf_readf_float\0")
                .map_err(|e| e.to_string())?;
            let error = self
                .sndfile
                .get::<SfError>(b"sf_error\0")
                .map_err(|e| e.to_string())?;
            let strerror = self
                .sndfile
                .get::<SfStrError>(b"sf_strerror\0")
                .map_err(|e| e.to_string())?;
            let mut info = SfInfo::default();
            // SFM_READ = 0x10. close_desc = 0 leaves File as descriptor owner.
            let pointer = open(file.as_raw_fd(), 0x10, &mut info, 0);
            if pointer.is_null() {
                return Err(format!(
                    "Audio decode: {}",
                    native_error(strerror(ptr::null_mut()))
                ));
            }
            let handle = SoundFile { pointer, close };
            if info.samplerate <= 0 || info.channels <= 0 || info.frames <= 0 {
                return Err("Audio source has invalid rate, channels or frame count".into());
            }
            let channels = info.channels as usize;
            let block_samples = channels
                .checked_mul(4096)
                .ok_or("Audio channel size overflow")?;
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(block_samples)
                .map_err(|e| format!("Audio workspace: {e}"))?;
            buffer.resize(block_samples, 0.0f32);
            let mut mono = Vec::new();
            // Decode bounded blocks instead of trusting container header lengths
            // for an allocation. There is no recording-duration or format cap.
            loop {
                let frames = read(handle.pointer, buffer.as_mut_ptr(), 4096);
                if !(0..=4096).contains(&frames) {
                    return Err("Invalid native audio read length".into());
                }
                if frames == 0 {
                    break;
                }
                mono.try_reserve(frames as usize)
                    .map_err(|e| format!("Audio samples: {e}"))?;
                for frame in buffer[..frames as usize * channels].chunks_exact(channels) {
                    if frame.iter().any(|x| !x.is_finite()) {
                        return Err("Nonfinite decoded audio".into());
                    }
                    // librosa.to_mono averages the decoded FP32 channels before
                    // resampling. Keep this order; mixing after SRC differs.
                    let mean = frame.iter().copied().sum::<f32>() / channels as f32;
                    if !mean.is_finite() {
                        return Err("Mono mix overflow".into());
                    }
                    mono.push(mean);
                }
            }
            if error(handle.pointer) != 0 {
                return Err(format!(
                    "Audio decode: {}",
                    native_error(strerror(handle.pointer))
                ));
            }
            if mono.is_empty() {
                return Err("Empty decoded audio".into());
            }
            Ok(DecodedAudio {
                sample_rate: info.samplerate as u32,
                channels,
                mono,
            })
        }
    }
    pub fn resample(&self, audio: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>> {
        if input_rate == 0
            || output_rate == 0
            || audio.is_empty()
            || audio.iter().any(|v| !v.is_finite())
        {
            return Err("Resampling requires finite nonempty PCM and positive rates".into());
        }
        if input_rate == output_rate {
            return Ok(audio.to_vec());
        }
        // librosa fixes the final length to ceil(N * target / source); SoXR
        // itself rounds to nearest. Preserve the final zero pad when required.
        let count = (audio.len() as u128 * output_rate as u128).div_ceil(input_rate as u128);
        let count = usize::try_from(count).map_err(|_| "Resampled audio length overflow")?;
        let capacity = count.checked_add(1).ok_or("Resampled capacity overflow")?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(capacity)
            .map_err(|e| format!("Resampling workspace: {e}"))?;
        output.resize(capacity, 0.0f32);
        // SAFETY: soxr.h specifies the struct return, buffer/count types and
        // null-default FP32 interleaved I/O. Explicit HQ overrides oneshot's LQ.
        unsafe {
            let quality = self
                .soxr
                .get::<QualitySpec>(b"soxr_quality_spec\0")
                .map_err(|e| e.to_string())?;
            let oneshot = self
                .soxr
                .get::<OneShot>(b"soxr_oneshot\0")
                .map_err(|e| e.to_string())?;
            let specification = quality(4, 0); // SOXR_HQ = SOXR_20_BITQ = 4
            let (mut used, mut produced) = (0, 0);
            let error = oneshot(
                input_rate as f64,
                output_rate as f64,
                1,
                audio.as_ptr().cast(),
                audio.len(),
                &mut used,
                output.as_mut_ptr().cast(),
                output.len(),
                &mut produced,
                ptr::null(),
                &specification,
                ptr::null(),
            );
            if !error.is_null() {
                return Err(format!("SoXR: {}", native_error(error)));
            }
            if used != audio.len() || produced > capacity || produced.abs_diff(count) > 1 {
                return Err("Incomplete SoXR resampling".into());
            }
            output.truncate(produced);
        }
        output.resize(count, 0.0);
        if output.iter().any(|v| !v.is_finite()) {
            return Err("Resampling overflow".into());
        }
        Ok(output)
    }
    pub fn prepare(&self, path: &Path, normalise: bool) -> Result<PreparedAudio> {
        let decoded = self.decode(path)?;
        let mut pcm24 = self.resample(&decoded.mono, decoded.sample_rate, 24000)?;
        // Turbo's strict >5 s precondition applies after librosa's length fix.
        if pcm24.len() <= 5 * 24000 {
            return Err("Audio prompt must be longer than 5 seconds".into());
        }
        let (loudness_lufs, gain) = if normalise {
            normalize_loudness(&mut pcm24)?
        } else {
            (None, 1.0)
        };
        let pcm16 = self.resample(&pcm24, 24000, 16000)?;
        Ok(PreparedAudio {
            pcm24,
            pcm16,
            source_sample_rate: decoded.sample_rate,
            source_channels: decoded.channels,
            loudness_lufs,
            gain,
        })
    }
}

// K-weighting derived from pyloudnorm's MIT-licensed default Meter/IIRfilter.
// Match SciPy lfilter's f64 transposed direct-form II, then round to FP32 after
// each stage, as the deployed Meter assigns back into its FP32 input array.
fn filter(audio: &mut [f32], b: [f64; 3], a: [f64; 3]) -> Result<()> {
    let b = b.map(|x| x / a[0]);
    let a = a.map(|x| x / a[0]);
    let (mut z1, mut z2) = (0.0, 0.0);
    for value in audio {
        let x = *value as f64;
        let y = b[0] * x + z1;
        z1 = b[1] * x - a[1] * y + z2;
        z2 = b[2] * x - a[2] * y;
        *value = y as f32;
        if !value.is_finite() {
            return Err("Loudness weighting overflow".into());
        }
    }
    Ok(())
}
pub fn normalize_loudness(audio: &mut [f32]) -> Result<(Option<f64>, f64)> {
    if audio.len() < 9600 || audio.iter().any(|v| !v.is_finite()) {
        return Err("Loudness requires finite mono 24 kHz PCM of at least 400 ms".into());
    }
    let mut weighted = audio.to_vec();
    let a_gain = 10f64.powf(4.0 / 40.0);
    let w = 2.0 * std::f64::consts::PI * (1500.0 / 24000.0);
    let c = w.cos();
    let alpha = w.sin() / (2.0 / 2f64.sqrt());
    let r = 2.0 * a_gain.sqrt() * alpha;
    filter(
        &mut weighted,
        [
            a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * c + r),
            -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * c),
            a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * c - r),
        ],
        [
            (a_gain + 1.0) - (a_gain - 1.0) * c + r,
            2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * c),
            (a_gain + 1.0) - (a_gain - 1.0) * c - r,
        ],
    )?;
    let w = 2.0 * std::f64::consts::PI * (38.0 / 24000.0);
    let c = w.cos();
    let alpha = w.sin(); // Q = 0.5
    filter(
        &mut weighted,
        [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0],
        [1.0 + alpha, -2.0 * c, 1.0 - alpha],
    )?;
    // NumPy round uses ties-to-even, including a possibly short final block.
    // Keep pyloudnorm's floating-point boundary arithmetic and 9600 denominator.
    let seconds = audio.len() as f64 / 24000.0;
    let blocks = ((seconds - 0.4) / (0.4 * 0.25)).round_ties_even() as usize + 1;
    let mut powers = Vec::with_capacity(blocks);
    for j in 0..blocks {
        let lo = (0.4 * (j as f64 * 0.25) * 24000.0) as usize;
        let hi = ((0.4 * (j as f64 * 0.25 + 1.0) * 24000.0) as usize).min(weighted.len());
        let sum: f64 = weighted[lo..hi]
            .iter()
            .map(|x| {
                let square = *x * *x; // NumPy square runs in FP32 before reduction.
                square as f64
            })
            .sum();
        let power = (sum as f32) as f64 / 9600.0;
        if !power.is_finite() {
            return Err("Loudness power overflow".into());
        }
        powers.push(power);
    }
    let loudness = |power: f64| -0.691 + 10.0 * power.log10();
    let absolute: Vec<f64> = powers
        .iter()
        .copied()
        .filter(|p| loudness(*p) >= -70.0)
        .collect();
    if absolute.is_empty() {
        return Ok((None, 1.0));
    }
    let relative = loudness(absolute.iter().sum::<f64>() / absolute.len() as f64) - 10.0;
    let gated: Vec<f64> = powers
        .into_iter()
        .filter(|p| loudness(*p) > relative && loudness(*p) > -70.0)
        .collect();
    if gated.is_empty() {
        return Ok((None, 1.0));
    }
    let measured = loudness(gated.iter().sum::<f64>() / gated.len() as f64);
    let gain = 10f64.powf((-27.0 - measured) / 20.0);
    if !gain.is_finite() || gain <= 0.0 {
        return Ok((Some(measured), 1.0));
    }
    // NumPy applies the scalar in the FP32 array dtype, with no peak limiter.
    let gain32 = gain as f32;
    if !gain32.is_finite() || audio.iter().any(|v| !(*v * gain32).is_finite()) {
        return Err("Loudness normalization overflow".into());
    }
    for value in audio {
        *value *= gain32;
    }
    Ok((Some(measured), gain))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn audio_preparation_validation() {
        let engine = AudioPreparation::open().unwrap();
        for (input, a, b) in [(vec![], 1, 2), (vec![f32::NAN], 1, 2), (vec![0.0], 0, 1)] {
            assert!(engine.resample(&input, a, b).is_err());
        }
        assert_eq!(
            engine.resample(&[0.25, -0.5], 24000, 24000).unwrap(),
            [0.25, -0.5]
        );
        assert_eq!(engine.resample(&[0.0; 17], 24000, 16000).unwrap().len(), 12);
        assert!(normalize_loudness(&mut [0.0; 10]).is_err());
        assert!(normalize_loudness(&mut [f32::NAN; 9600]).is_err());
        assert!(normalize_loudness(&mut [f32::MAX; 9600]).is_err());
        let mut silence = vec![0.0; 144000];
        assert_eq!(normalize_loudness(&mut silence).unwrap(), (None, 1.0));
        assert!(silence.iter().all(|v| *v == 0.0));
    }
    #[test]
    fn audio_preparation_rejects_fifo() {
        let path = std::env::temp_dir().join(format!(
            "chatterbox-audio-fifo-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: test-owned nul-terminated path; mode creates only this FIFO.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let result = AudioPreparation::open().unwrap().decode(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(result.err().unwrap().contains("regular recording"));
    }
}
