//! Read only the C-contiguous, little-endian FP16/FP32 NumPy arrays emitted by
//! the established Chatterbox capture. No Python interpreter is needed at runtime.
use crate::Result;
use half::f16;
use std::{fs::File, io::Read, path::Path};

#[derive(Debug)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub halves: Vec<u16>,
}

/// S3Gen's retained mel/speaker state is FP32. Keep its original bits instead
/// of routing it through the FP16 embedding conversion used by the T3 engine.
#[derive(Debug, Clone)]
pub struct F32Npy {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl F32Npy {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let _trace = crate::trace::span("tensor_parse_validate");
        let raw = RawNpy::read_from(&mut &bytes[..])?;
        if raw.bytes_per_value != 4 {
            return Err("S3Gen reference arrays must be FP32".into());
        }
        let values: Vec<f32> = raw
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite S3Gen reference value".into());
        }
        Ok(Self {
            shape: raw.shape,
            values,
        })
    }
}

// Share NumPy envelope/shape validation; the caller selects whether conversion
// to FP16 is appropriate. Reference loading must never quantise FP32 values.
struct RawNpy {
    shape: Vec<usize>,
    bytes_per_value: usize,
    data: Vec<u8>,
}
impl Npy {
    pub fn read(path: &Path) -> Result<Self> {
        let mut source = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::read_from(&mut source)
    }

    /// Parse the exact admitted bytes without reopening a mutable asset.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let _trace = crate::trace::span("tensor_parse_validate");
        Self::read_from(&mut &bytes[..])
    }

    fn read_from(source: &mut impl Read) -> Result<Self> {
        Self::from_raw(RawNpy::read_from(source)?)
    }

    #[cfg(test)]
    fn read_data(header: &str, source: &mut impl Read) -> Result<Self> {
        Self::from_raw(RawNpy::read_data(header, source)?)
    }

    fn from_raw(raw: RawNpy) -> Result<Self> {
        let mut halves = Vec::with_capacity(raw.data.len() / raw.bytes_per_value);
        for item in raw.data.chunks_exact(raw.bytes_per_value) {
            let value = if raw.bytes_per_value == 2 {
                f16::from_bits(u16::from_le_bytes([item[0], item[1]]))
            } else {
                f16::from_f32(f32::from_le_bytes(item.try_into().unwrap()))
            };
            if !value.is_finite() {
                return Err("Nonfinite or overflowing embedding value".into());
            }
            halves.push(value.to_bits());
        }
        Ok(Self {
            shape: raw.shape,
            halves,
        })
    }

    /// Turbo's embedding width is fixed by the admitted GPT2-medium engine.
    pub fn prompt_frames(&self) -> Result<usize> {
        if self.shape.len() != 3
            || self.shape[0] != 1
            || self.shape[2] != 1024
            || !(1..=512).contains(&self.shape[1])
        {
            return Err("Prompt must have shape [1,1..512,1024]".into());
        }
        Ok(self.shape[1])
    }

    pub fn speech_row(&self, token: u32) -> Result<&[u16]> {
        if self.shape != [6563, 1024] {
            return Err("Speech embedding must have shape [6563,1024]".into());
        }
        if token >= 6563 {
            return Err("Speech token outside embedding vocabulary".into());
        }
        let start = token as usize * 1024;
        Ok(&self.halves[start..start + 1024])
    }
}

impl RawNpy {
    fn read_from(source: &mut impl Read) -> Result<Self> {
        let mut prefix = [0u8; 8];
        source.read_exact(&mut prefix).map_err(|e| e.to_string())?;
        if &prefix[..6] != b"\x93NUMPY" || prefix[7] != 0 {
            return Err("Unsupported NumPy signature/version".into());
        }
        let length_bytes = match prefix[6] {
            1 => 2,
            2 | 3 => 4,
            _ => return Err("Unsupported NumPy version".into()),
        };
        let mut length = [0u8; 4];
        source
            .read_exact(&mut length[..length_bytes])
            .map_err(|e| e.to_string())?;
        let header_len = u32::from_le_bytes(length) as usize;
        if header_len > 65536 {
            return Err("NumPy header exceeds 64 KiB".into());
        }
        let mut header = vec![0u8; header_len];
        source.read_exact(&mut header).map_err(|e| e.to_string())?;
        let header = std::str::from_utf8(&header).map_err(|e| e.to_string())?;
        Self::read_data(header, source)
    }

    fn read_data(header: &str, source: &mut impl Read) -> Result<Self> {
        let field = |name: &str| -> Result<&str> {
            let key = format!("'{name}'");
            let double_key = format!("\"{name}\"");
            let rest = header
                .find(&key)
                .map(|i| &header[i + key.len()..])
                .or_else(|| {
                    header
                        .find(&double_key)
                        .map(|i| &header[i + double_key.len()..])
                })
                .ok_or_else(|| format!("NumPy header lacks {name}"))?;
            rest.trim_start()
                .strip_prefix(':')
                .map(str::trim_start)
                .ok_or_else(|| format!("Malformed NumPy field {name}"))
        };
        let dtype = field("descr")?;
        let bytes_per_value = if dtype.starts_with("'<f2'") || dtype.starts_with("\"<f2\"") {
            2
        } else if dtype.starts_with("'<f4'") || dtype.starts_with("\"<f4\"") {
            4
        } else {
            return Err("NumPy dtype must be little-endian f2 or f4".into());
        };
        if !field("fortran_order")?.starts_with("False") {
            return Err("Fortran-order arrays are unsupported".into());
        }
        let shape_field = field("shape")?;
        let shape_end = shape_field.find(')').ok_or("Malformed NumPy shape")?;
        let shape_text = shape_field[..shape_end]
            .strip_prefix('(')
            .ok_or("Malformed NumPy shape")?;
        let shape: Vec<usize> = shape_text
            .split(',')
            .filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<_>>()?;
        if shape.is_empty() || shape.contains(&0) {
            return Err("Empty NumPy shape".into());
        }
        let count = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or("NumPy shape overflow")?;
        let byte_count = count
            .checked_mul(bytes_per_value)
            .ok_or("NumPy byte count overflow")?;
        // Read bounded chunks instead of trusting the declared shape to allocate
        // before physical data exists. A truncated array must fail explicitly.
        let mut data = Vec::new();
        let read_limit = (byte_count as u64)
            .checked_add(1)
            .ok_or("NumPy byte count overflow")?;
        source
            .take(read_limit)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        if data.len() != byte_count {
            return Err("NumPy data length differs from shape".into());
        }
        Ok(Self {
            shape,
            bytes_per_value,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_half_array_and_reject_truncation() {
        let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1, 2), }";
        let data = [0u8, 0x3c, 0, 0xc0];
        let array = Npy::read_data(header, &mut &data[..]).unwrap();
        assert_eq!(array.shape, [1, 2]);
        assert_eq!(array.halves, [0x3c00, 0xc000]);
        assert!(Npy::read_data(header, &mut &data[..2]).is_err());
        assert!(Npy::read_data(header, &mut &[0u8; 5][..]).is_err());
    }
    #[test]
    fn capture_float_conversion_and_header_errors() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let data: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let a = Npy::read_data(header, &mut &data[..]).unwrap();
        assert_eq!(
            a.halves,
            [f16::from_f32(1.5).to_bits(), f16::from_f32(-2.0).to_bits()]
        );
        for header in [
            "{'descr': '>f2', 'fortran_order': False, 'shape': (2,), }",
            "{'descr': '<f2', 'fortran_order': True, 'shape': (2,), }",
            "{'descr': '<f2', 'fortran_order': False, 'shape': (0,), }",
            "{'descr': '<f2', 'fortran_order': False, 'shape': (18446744073709551615,2), }",
        ] {
            assert!(Npy::read_data(header, &mut &data[..]).is_err());
        }
    }

    #[test]
    fn s3gen_reference_fp32_parser_preserves_bits_and_rejects_fp16() {
        let make = |dtype: &str, data: &[u8]| {
            let header =
                format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': (1,), }}\n");
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(data);
            bytes
        };
        for value in [-0.0f32, 1.0000001, 100000.125, f32::from_bits(1)] {
            let array = F32Npy::from_bytes(&make("<f4", &value.to_le_bytes())).unwrap();
            assert_eq!(array.values[0].to_bits(), value.to_bits());
        }
        assert!(F32Npy::from_bytes(&make("<f2", &0x3c00u16.to_le_bytes())).is_err());
        assert!(F32Npy::from_bytes(&make("<f4", &f32::NAN.to_le_bytes())).is_err());
    }
}
