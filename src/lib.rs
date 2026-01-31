//! Hex float conversion for f32 and f64.
//!
//! This crate provides two traits:
//! - [`ToHex`] for formatting floats as hex strings (`0x1.8p+1`)
//! - [`FromHex`] for parsing hex strings back to floats
//!
//! The format follows the [IEEE 754] hex float specification—the same format
//! used by C's `%a` printf specifier, Java's `Double.toHexString()`, and
//! the WebAssembly text format.
//!
//! [IEEE 754]: https://en.wikipedia.org/wiki/Hexadecimal_floating_point
//!
//! # Examples
//!
//! ```
//! use fhex::{ToHex, FromHex};
//!
//! // Formatting
//! let hex = 3.0_f64.to_hex();
//! assert_eq!(hex, "0x1.8p+1");
//!
//! // Parsing
//! let value = f64::from_hex("0x1.8p+1").unwrap();
//! assert_eq!(value, 3.0);
//!
//! // Round-trip
//! let original = std::f64::consts::PI;
//! let roundtrip = f64::from_hex(&original.to_hex()).unwrap();
//! assert_eq!(original, roundtrip);
//! ```
//!
//! # Format
//!
//! Floating point numbers are represented as `±0xh.hhhp±d`, where:
//! - `±` is the sign (`-` for negative, omitted for positive)
//! - `0x` is the hex prefix
//! - `h.hhh` is the significand in hexadecimal
//! - `p±d` is the exponent in decimal (base 2)
//!
//! Special values:
//! - `±0x0p+0` for zero
//! - `±inf` for infinity
//! - `nan` for quiet NaN
//! - `nan:0x...` for NaN with payload (signalling NaN)
//!
//! # Panics
//!
//! Neither [`ToHex::to_hex`] nor [`FromHex::from_hex`] will panic. All inputs
//! are handled gracefully.

/// Trait for converting floating-point numbers to hexadecimal strings.
///
/// # Examples
///
/// ```
/// use fhex::ToHex;
///
/// assert_eq!(1.0_f32.to_hex(), "0x1p+0");
/// assert_eq!((-3.5_f64).to_hex(), "-0x1.cp+1");
/// assert_eq!(f64::INFINITY.to_hex(), "inf");
/// assert_eq!(f64::NAN.to_hex(), "nan");
/// ```
pub trait ToHex {
    /// Converts the floating-point number to a hexadecimal string.
    #[must_use]
    fn to_hex(self) -> String;
}

/// Trait for parsing hexadecimal strings to floating-point numbers.
///
/// # Examples
///
/// ```
/// use fhex::FromHex;
///
/// assert_eq!(f64::from_hex("0x1p+0"), Some(1.0));
/// assert_eq!(f64::from_hex("0x1.8p+1"), Some(3.0));
/// assert_eq!(f64::from_hex("-0x1.4p+3"), Some(-10.0));
/// assert_eq!(f64::from_hex("inf"), Some(f64::INFINITY));
/// assert!(f64::from_hex("nan").unwrap().is_nan());
/// ```
///
/// # Accepted Formats
///
/// - Hex floats: `0x1.8p+1`, `0X1P-10`, `-0x1.abcdefp+100`
/// - Special values: `inf`, `-inf`, `nan`, `NaN`
/// - NaN with payload: `nan:0x123` (preserves signalling NaN bits)
/// - With underscores: `0x1_0p+0` (for readability)
/// - Whitespace: leading/trailing whitespace is trimmed
///
/// # Returns `None` When
///
/// - Empty string or only whitespace
/// - Missing `0x` prefix for hex floats
/// - No digits in mantissa (`0x.p+0`, `0xp+0`)
/// - Invalid characters
/// - NaN payload is zero or exceeds significand size
pub trait FromHex: Sized {
    /// Parses a hexadecimal string to a floating-point number.
    ///
    /// Returns `None` if the string is not a valid hex float.
    /// See trait documentation for accepted formats and error cases.
    #[must_use]
    fn from_hex(s: &str) -> Option<Self>;
}

impl ToHex for f32 {
    fn to_hex(self) -> String {
        to_hex(FLOAT32, self.to_bits() as u64)
    }
}

impl ToHex for f64 {
    fn to_hex(self) -> String {
        to_hex(FLOAT64, self.to_bits())
    }
}

impl FromHex for f32 {
    fn from_hex(s: &str) -> Option<Self> {
        from_hex_f32(s)
    }
}

impl FromHex for f64 {
    fn from_hex(s: &str) -> Option<Self> {
        from_hex_f64(s)
    }
}

// =============================================================================
// ToHex implementation
// =============================================================================

/// IEEE 754 floating-point format parameters for bit manipulation.
///
/// This struct captures the bit layout of f32/f64 for extracting and formatting
/// the sign, exponent, and significand components. Designed for const construction
/// so both FLOAT32 and FLOAT64 can be compile-time constants.
struct Floater {
    /// Total bits in the format (32 or 64)
    bits: u64,
    /// Bits in the significand (23 for f32, 52 for f64)
    sig_bits: u64,
    /// Bit position of sign bit (bits - 1)
    sign_shift: u64,
    /// Mask to extract significand bits
    sig_mask: u64,
    /// Maximum hex digits needed to represent significand
    max_sig_digits: u64,
    /// Mask to extract exponent bits
    exp_mask: u64,
    /// Maximum biased exponent (indicates inf/nan)
    max_exp: i64,
    /// Minimum biased exponent (indicates subnormal/zero)
    min_exp: i64,
    /// Exponent bias for the format
    exp_bias: i64,
    /// Bit pattern for quiet NaN (vs signalling NaN)
    quiet_nan_tag: u64,
}

impl Floater {
    const fn new(bits: u32, sig_bits: u32) -> Self {
        let exp_bits: u64 = bits as u64 - sig_bits as u64 - 1;
        let sign_shift: u64 = (bits as u64) - 1;
        let sig_mask = (1_u64 << sig_bits) - 1;
        // Maximum hex digits = sig_bits * log₁₀(2) ≈ sig_bits * 0.30103
        // We use 3010/10000 as an integer approximation for const evaluation
        let max_sig_digits = sig_bits as u64 * 3010 / 10000;
        let exp_mask = (1 << exp_bits) - 1;
        let max_exp = 1 << (exp_bits - 1);
        let min_exp = -max_exp + 1;
        let exp_bias = -min_exp;
        let quiet_nan_tag = 1 << (sig_bits - 1);

        Self {
            bits: bits as u64,
            sig_bits: sig_bits as u64,
            sign_shift,
            sig_mask,
            max_sig_digits,
            exp_mask,
            max_exp,
            min_exp,
            exp_bias,
            quiet_nan_tag,
        }
    }
}

const HEX_DIGITS: &[u8] = b"0123456789abcdef";
const FLOAT32: Floater = Floater::new(32, 23);
const FLOAT64: Floater = Floater::new(64, 52);

fn to_hex(typ: Floater, bits: u64) -> String {
    let mut buffer = String::new();

    let mut exponent = (((bits >> typ.sig_bits) & typ.exp_mask) as i64) - typ.exp_bias;
    let mut significand = bits & typ.sig_mask;

    if bits >> typ.sign_shift != 0 {
        buffer.push('-');
    }

    if exponent == typ.max_exp {
        write_nan_or_infinity(&mut buffer, significand, &typ);
    } else {
        let is_zero = significand == 0 && exponent == typ.min_exp;
        buffer.push_str("0x");
        buffer.push(if is_zero { '0' } else { '1' });

        // Shift significand up so the top 4-bits are at the top
        significand <<= (typ.bits - typ.sig_bits) as i32;

        if significand != 0 {
            write_significand(&mut buffer, &mut exponent, significand, &typ);
        }
        write_exponent(&mut buffer, is_zero, exponent);
    }
    buffer
}

fn write_nan_or_infinity(buffer: &mut String, significand: u64, typ: &Floater) {
    let num_nybbles = typ.bits / 4;
    let top_nybble_shift = typ.bits - 4;
    let top_nybble = 0xf << top_nybble_shift;

    if significand == 0 {
        buffer.push_str("inf");
    } else {
        buffer.push_str("nan");
        if significand != typ.quiet_nan_tag {
            buffer.push_str(":0x");
            // Skip leading zeroes
            let mut significand = significand;
            let mut num_nybbles = num_nybbles;
            while (significand & top_nybble) == 0 {
                significand <<= 4;
                num_nybbles -= 1;
            }
            while num_nybbles > 0 {
                let nybble = (significand >> top_nybble_shift) & 0xf;
                buffer.push(HEX_DIGITS[nybble as usize] as char);
                significand <<= 4;
                num_nybbles -= 1;
            }
        }
    }
}

fn write_significand(buffer: &mut String, exponent: &mut i64, mut significand: u64, typ: &Floater) {
    let top_nybble_shift = typ.bits - 4;

    if *exponent == typ.min_exp {
        // Subnormal; shift the significand up, and shift out the implicit 1
        let leading_zeroes = significand.leading_zeros();
        let leading_zeroes = if typ.bits == 32 {
            leading_zeroes.saturating_sub(32)
        } else {
            leading_zeroes
        };
        if leading_zeroes < typ.sign_shift as u32 {
            significand <<= (leading_zeroes + 1) as i32;
            // f32 significand is stored in u64; mask to prevent overflow into upper bits
            if typ.bits == 32 {
                significand &= 0xffffffff;
            }
        } else {
            significand = 0;
        }
        *exponent -= leading_zeroes as i64;
    }

    buffer.push('.');
    for i in 0..typ.max_sig_digits {
        if significand == 0 {
            if i == 0 {
                buffer.pop(); // Remove '.'
            }
            break;
        }
        let nybble = (significand >> top_nybble_shift) & 0xf;
        buffer.push(HEX_DIGITS[nybble as usize] as char);
        significand <<= 4;
        // f32 significand is stored in u64; mask to prevent overflow into upper bits
        if typ.bits == 32 {
            significand &= 0xffffffff;
        }
    }
}

/// Write the exponent part of a hex float (e.g., "p+10", "p-1023").
///
/// Uses manual digit extraction instead of format!() to avoid std::fmt overhead
/// and enable potential no_std support. Exponents range from -1074 to +1023 for
/// f64, so we only need to handle up to 4 digits.
fn write_exponent(buffer: &mut String, is_zero: bool, mut exponent: i64) {
    buffer.push('p');
    if is_zero {
        buffer.push_str("+0");
    } else {
        if exponent < 0 {
            buffer.push('-');
            exponent = -exponent;
        } else {
            buffer.push('+');
        }
        // Extract digits from most significant to least (max 4 digits: -1074 to +1023)
        if exponent >= 1000 {
            buffer.push('1');
        }
        if exponent >= 100 {
            let digit = (exponent / 100) % 10;
            buffer.push((b'0' + digit as u8) as char);
        }
        if exponent >= 10 {
            let digit = (exponent / 10) % 10;
            buffer.push((b'0' + digit as u8) as char);
        }
        let digit = exponent % 10;
        buffer.push((b'0' + digit as u8) as char);
    }
}

// =============================================================================
// FromHex implementation
// =============================================================================

/// Result of parsing the prefix of a hex float string.
enum ParsedPrefix<'a> {
    /// Infinity (positive or negative based on `negative` flag)
    Inf { negative: bool },
    /// Quiet NaN (positive or negative)
    Nan { negative: bool },
    /// NaN with payload
    NanPayload { negative: bool, payload: &'a str },
    /// Regular hex float (the remaining string after "0x" prefix)
    HexFloat { negative: bool, mantissa: &'a str },
}

/// Parse the prefix of a hex float string, handling sign and special values.
/// Returns the parsed result or None if the format is invalid.
fn parse_prefix(s: &str) -> Option<ParsedPrefix<'_>> {
    let s = s.trim();

    // Handle sign
    let (negative, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };

    // Handle special values
    if s.eq_ignore_ascii_case("inf") {
        return Some(ParsedPrefix::Inf { negative });
    }

    if s.eq_ignore_ascii_case("nan") {
        return Some(ParsedPrefix::Nan { negative });
    }

    // Handle nan:0x... payload
    if let Some(payload) = s.strip_prefix("nan:0x").or_else(|| s.strip_prefix("nan:0X")) {
        return Some(ParsedPrefix::NanPayload { negative, payload });
    }

    // Parse hex float: 0xh.hhhp±d
    let mantissa = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    Some(ParsedPrefix::HexFloat { negative, mantissa })
}

fn from_hex_f64(s: &str) -> Option<f64> {
    match parse_prefix(s)? {
        ParsedPrefix::Inf { negative } => Some(if negative { f64::NEG_INFINITY } else { f64::INFINITY }),
        ParsedPrefix::Nan { negative } => Some(if negative { -f64::NAN } else { f64::NAN }),
        ParsedPrefix::NanPayload { negative, payload } => {
            let payload = u64::from_str_radix(payload, 16).ok()?;
            // Payload must fit in 52 bits and be non-zero
            if payload == 0 || payload > 0xfffffffffffff {
                return None;
            }
            let bits = 0x7ff0000000000000_u64 | payload;
            let value = f64::from_bits(bits);
            Some(if negative { -value } else { value })
        }
        ParsedPrefix::HexFloat { negative, mantissa } => parse_hex_float_f64(mantissa, negative),
    }
}

fn from_hex_f32(s: &str) -> Option<f32> {
    match parse_prefix(s)? {
        ParsedPrefix::Inf { negative } => Some(if negative { f32::NEG_INFINITY } else { f32::INFINITY }),
        ParsedPrefix::Nan { negative } => Some(if negative { -f32::NAN } else { f32::NAN }),
        ParsedPrefix::NanPayload { negative, payload } => {
            let payload = u32::from_str_radix(payload, 16).ok()?;
            // Payload must fit in 23 bits and be non-zero
            if payload == 0 || payload > 0x7fffff {
                return None;
            }
            let bits = 0x7f800000_u32 | payload;
            let value = f32::from_bits(bits);
            Some(if negative { -value } else { value })
        }
        ParsedPrefix::HexFloat { negative, mantissa } => parse_hex_float_f32(mantissa, negative),
    }
}

/// Parse hex float mantissa and exponent (after 0x prefix).
fn parse_hex_float_f64(s: &str, negative: bool) -> Option<f64> {
    // Split mantissa and exponent at 'p' or 'P'
    let (mantissa_str, exp_str) = if let Some(p_pos) = s.find(['p', 'P']) {
        (&s[..p_pos], &s[p_pos + 1..])
    } else {
        (s, "+0")
    };

    // Parse exponent (base 2)
    let exp_str = exp_str.strip_prefix('+').unwrap_or(exp_str);
    let exponent: i32 = exp_str.parse().ok()?;

    // Split mantissa into integer and fractional parts
    let (int_str, frac_str) = if let Some(dot_pos) = mantissa_str.find('.') {
        (&mantissa_str[..dot_pos], &mantissa_str[dot_pos + 1..])
    } else {
        (mantissa_str, "")
    };

    // Filter underscores for WAT compatibility
    let int_clean: String = int_str.chars().filter(|&c| c != '_').collect();
    let frac_clean: String = frac_str.chars().filter(|&c| c != '_').collect();

    // Must have at least one hex digit
    if int_clean.is_empty() && frac_clean.is_empty() {
        return None;
    }

    // Parse integer part
    let int_val = if int_clean.is_empty() {
        0u64
    } else {
        u64::from_str_radix(&int_clean, 16).ok()?
    };

    // Build mantissa: integer part + fractional part
    let mut mantissa = int_val as f64;

    // Add fractional part: each hex digit contributes digit / 16^position
    for (i, c) in frac_clean.chars().enumerate() {
        let digit = c.to_digit(16)? as f64;
        // 16^(i+1) = 2^(4*(i+1))
        let shift = 4 * (i as u32 + 1);
        if shift < 64 {
            let divisor = (1_u64 << shift) as f64;
            mantissa += digit / divisor;
        }
        // Beyond 64 bits of precision, additional digits don't matter for f64
    }

    if mantissa == 0.0 {
        return Some(if negative { -0.0 } else { 0.0 });
    }

    // Apply exponent using repeated multiply/divide to maintain precision
    // Use steps of 30 bits to stay well within f64 and u64 range
    let value = if exponent >= 0 {
        let mut result = mantissa;
        let mut exp = exponent;
        while exp > 0 {
            let step = exp.min(30);
            result *= (1_u64 << step) as f64;
            exp -= step;
        }
        result
    } else {
        let mut result = mantissa;
        let mut exp = -exponent;
        while exp > 0 {
            let step = exp.min(30);
            result /= (1_u64 << step) as f64;
            exp -= step;
        }
        result
    };

    Some(if negative { -value } else { value })
}

fn parse_hex_float_f32(s: &str, negative: bool) -> Option<f32> {
    // Use f64 for intermediate precision, then convert
    let value = parse_hex_float_f64(s, false)?;
    let result = value as f32;
    Some(if negative { -result } else { result })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ToHex tests (existing)
    // =========================================================================

    #[test]
    fn test_f32_to_hex() {
        let cases: &[([u8; 4], &str)] = &[
            ([0x00, 0x00, 0x00, 0x80], "-0x0p+0"),
            ([0x00, 0x00, 0x00, 0x00], "0x0p+0"),
            ([0x01, 0x00, 0x80, 0xd8], "-0x1.000002p+50"),
            ([0x01, 0x00, 0x80, 0xa6], "-0x1.000002p-50"),
            ([0x01, 0x00, 0x80, 0x58], "0x1.000002p+50"),
            ([0x01, 0x00, 0x80, 0x26], "0x1.000002p-50"),
            ([0x01, 0x00, 0x00, 0x7f], "0x1.000002p+127"),
            ([0xb4, 0xa2, 0x11, 0x52], "0x1.234568p+37"),
            ([0xb4, 0xa2, 0x91, 0x5b], "0x1.234568p+56"),
            ([0x99, 0x76, 0x96, 0xfe], "-0x1.2ced32p+126"),
            ([0x99, 0x76, 0x96, 0x7e], "0x1.2ced32p+126"),
            ([0x03, 0x00, 0x00, 0x80], "-0x1.8p-148"),
            ([0x03, 0x00, 0x00, 0x00], "0x1.8p-148"),
            ([0x00, 0x00, 0x00, 0xff], "-0x1p+127"),
            ([0x00, 0x00, 0x00, 0x7f], "0x1p+127"),
            ([0x02, 0x00, 0x00, 0x80], "-0x1p-148"),
            ([0x02, 0x00, 0x00, 0x00], "0x1p-148"),
            ([0x01, 0x00, 0x00, 0x80], "-0x1p-149"),
            ([0x01, 0x00, 0x00, 0x00], "0x1p-149"),
            ([0x00, 0x00, 0x80, 0xd8], "-0x1p+50"),
            ([0x00, 0x00, 0x80, 0xa6], "-0x1p-50"),
            ([0x00, 0x00, 0x80, 0x58], "0x1p+50"),
            ([0x00, 0x00, 0x80, 0x26], "0x1p-50"),
            ([0x00, 0x00, 0x80, 0x7f], "inf"),
            ([0x00, 0x00, 0x80, 0xff], "-inf"),
            ([0x00, 0x00, 0xc0, 0x7f], "nan"),
            ([0x01, 0x00, 0x80, 0x7f], "nan:0x1"),
            ([0xff, 0xff, 0xff, 0x7f], "nan:0x7fffff"),
            ([0x00, 0x00, 0x80, 0x3f], "0x1p+0"),
            ([0x00, 0x00, 0x80, 0xbf], "-0x1p+0"),
            ([0xff, 0xff, 0x7f, 0x7f], "0x1.fffffep+127"),
            ([0xff, 0xff, 0x7f, 0xff], "-0x1.fffffep+127"),
            ([0xa4, 0x70, 0x9d, 0x3f], "0x1.3ae148p+0"),
        ];
        for (bytes, expected) in cases {
            let float = f32::from_ne_bytes(*bytes);
            let result = float.to_hex();
            assert_eq!(&result, *expected, "f32 {:?} -> {}", bytes, result);
        }
    }

    #[test]
    fn test_f64_to_hex() {
        let cases: &[([u8; 8], &str)] = &[
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80], "-0x0p+0"),
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], "0x0p+0"),
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0xc3],
                "-0x1.0000000000001p+60",
            ),
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0x43],
                "0x1.0000000000001p+60",
            ),
            (
                [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0xff],
                "-0x1.fffffffffffffp+1023",
            ),
            (
                [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0x7f],
                "0x1.fffffffffffffp+1023",
            ),
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0xff], "-0x1p+1023"),
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x7f], "0x1p+1023"),
            ([0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80], "-0x1p-1073"),
            ([0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], "0x1p-1073"),
            ([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80], "-0x1p-1074"),
            ([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], "0x1p-1074"),
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f], "inf"),
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff], "-inf"),
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f], "nan"),
            ([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f], "nan:0x1"),
            ([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f], "nan:0xfffffffffffff"),
        ];
        for (bytes, expected) in cases {
            let float = f64::from_ne_bytes(*bytes);
            let result = float.to_hex();
            assert_eq!(&result, *expected, "f64 {:?} -> {}", bytes, result);
        }
    }

    // =========================================================================
    // FromHex tests
    // =========================================================================

    #[test]
    fn test_f64_from_hex_basic() {
        assert_eq!(f64::from_hex("0x0p+0"), Some(0.0));
        assert_eq!(f64::from_hex("0x1p+0"), Some(1.0));
        assert_eq!(f64::from_hex("0x1p+1"), Some(2.0));
        assert_eq!(f64::from_hex("0x1.8p+1"), Some(3.0));
        assert_eq!(f64::from_hex("0x1.4p+3"), Some(10.0));
        assert_eq!(f64::from_hex("-0x1.4p+3"), Some(-10.0));
    }

    #[test]
    fn test_f64_from_hex_special() {
        assert_eq!(f64::from_hex("inf"), Some(f64::INFINITY));
        assert_eq!(f64::from_hex("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(f64::from_hex("INF"), Some(f64::INFINITY));
        assert!(f64::from_hex("nan").unwrap().is_nan());
        assert!(f64::from_hex("NaN").unwrap().is_nan());
        assert!(f64::from_hex("-nan").unwrap().is_nan());
    }

    #[test]
    fn test_f64_from_hex_nan_payload() {
        let value = f64::from_hex("nan:0x1").unwrap();
        assert!(value.is_nan());
        // Verify payload is preserved
        let bits = value.to_bits();
        assert_eq!(bits & 0xfffffffffffff, 1);
    }

    #[test]
    fn test_f32_from_hex_basic() {
        assert_eq!(f32::from_hex("0x0p+0"), Some(0.0));
        assert_eq!(f32::from_hex("0x1p+0"), Some(1.0));
        assert_eq!(f32::from_hex("0x1.8p+1"), Some(3.0));
        assert_eq!(f32::from_hex("inf"), Some(f32::INFINITY));
        assert!(f32::from_hex("nan").unwrap().is_nan());
    }

    #[test]
    fn test_f64_from_hex_with_underscores() {
        // WAT format allows underscores in numbers
        assert_eq!(f64::from_hex("0x1_0p+0"), Some(16.0));
        assert_eq!(f64::from_hex("0x1.8_0p+1"), Some(3.0));
    }

    #[test]
    fn test_from_hex_invalid() {
        assert_eq!(f64::from_hex(""), None);
        assert_eq!(f64::from_hex("0x"), None);
        assert_eq!(f64::from_hex("0x."), None);
        assert_eq!(f64::from_hex("0xp+0"), None);
        assert_eq!(f64::from_hex("hello"), None);
        assert_eq!(f64::from_hex("nan:0x0"), None); // Zero payload invalid
    }

    #[test]
    fn test_from_hex_whitespace() {
        // Leading/trailing whitespace should be trimmed
        assert_eq!(f64::from_hex("  0x1p+0  "), Some(1.0));
        assert_eq!(f64::from_hex("\t0x1.8p+1\n"), Some(3.0));
        assert_eq!(f64::from_hex("  inf  "), Some(f64::INFINITY));
    }

    #[test]
    fn test_from_hex_case_insensitive() {
        // Hex digits should be case-insensitive
        assert_eq!(f64::from_hex("0xABC"), f64::from_hex("0xabc"));
        assert_eq!(f64::from_hex("0xAbC"), f64::from_hex("0xabc"));
        // 0x prefix case
        assert_eq!(f64::from_hex("0X1p+0"), Some(1.0));
        // Exponent indicator case
        assert_eq!(f64::from_hex("0x1P+0"), Some(1.0));
        // Special values case
        assert_eq!(f64::from_hex("INF"), Some(f64::INFINITY));
        assert_eq!(f64::from_hex("Inf"), Some(f64::INFINITY));
        assert!(f64::from_hex("NAN").unwrap().is_nan());
        assert!(f64::from_hex("NaN").unwrap().is_nan());
    }

    #[test]
    fn test_subnormal_roundtrip() {
        // Explicit subnormal tests for FromHex
        // f64 subnormals have exponent -1074 to -1023
        let subnormals = [
            f64::MIN_POSITIVE / 2.0,    // Smallest subnormal * 2^51
            f64::MIN_POSITIVE / 1024.0, // Smaller subnormal
            5e-324_f64,                 // Near smallest subnormal
        ];
        for &v in &subnormals {
            let hex = v.to_hex();
            let parsed = f64::from_hex(&hex);
            assert!(parsed.is_some(), "Failed to parse subnormal: {}", hex);
            assert_eq!(
                v.to_bits(),
                parsed.unwrap().to_bits(),
                "Subnormal roundtrip failed: {} -> {}",
                v,
                hex
            );
        }

        // f32 subnormals
        let subnormals_f32 = [f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE / 1024.0];
        for &v in &subnormals_f32 {
            let hex = v.to_hex();
            let parsed = f32::from_hex(&hex);
            assert!(parsed.is_some(), "Failed to parse f32 subnormal: {}", hex);
            assert_eq!(
                v.to_bits(),
                parsed.unwrap().to_bits(),
                "f32 subnormal roundtrip failed: {} -> {}",
                v,
                hex
            );
        }
    }

    // =========================================================================
    // Round-trip tests
    // =========================================================================

    #[test]
    fn test_f64_roundtrip() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            std::f64::consts::PI,
            std::f64::consts::E,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for &v in &values {
            let hex = v.to_hex();
            let parsed = f64::from_hex(&hex).unwrap();
            if v.is_nan() {
                assert!(parsed.is_nan(), "NaN roundtrip failed");
            } else {
                assert_eq!(v, parsed, "Roundtrip failed for {}: {} -> {}", v, hex, parsed);
            }
        }
    }

    #[test]
    fn test_f32_roundtrip() {
        let values = [
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            std::f32::consts::PI,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for &v in &values {
            let hex = v.to_hex();
            let parsed = f32::from_hex(&hex).unwrap();
            if v.is_nan() {
                assert!(parsed.is_nan(), "NaN roundtrip failed");
            } else {
                assert_eq!(v, parsed, "Roundtrip failed for {}: {} -> {}", v, hex, parsed);
            }
        }
    }

    #[test]
    fn test_nan_roundtrip() {
        // Quiet NaN
        let nan = f64::NAN;
        let hex = nan.to_hex();
        let parsed = f64::from_hex(&hex).unwrap();
        assert!(parsed.is_nan());

        // NaN with payload
        let payload = 0x123_u64;
        let bits = 0x7ff0000000000000_u64 | payload;
        let nan_with_payload = f64::from_bits(bits);
        let hex = nan_with_payload.to_hex();
        let parsed = f64::from_hex(&hex).unwrap();
        assert!(parsed.is_nan());
        assert_eq!(parsed.to_bits() & 0xfffffffffffff, payload);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Round-trip property: to_hex -> from_hex should preserve the value.
        #[test]
        fn f64_roundtrip(v in any::<f64>()) {
            let hex = v.to_hex();
            if let Some(parsed) = f64::from_hex(&hex) {
                if v.is_nan() {
                    prop_assert!(parsed.is_nan());
                } else {
                    prop_assert_eq!(v.to_bits(), parsed.to_bits(),
                        "Roundtrip failed: {} -> {} -> {}", v, hex, parsed);
                }
            } else {
                // from_hex should succeed for any valid to_hex output
                prop_assert!(false, "from_hex failed for {}", hex);
            }
        }

        #[test]
        fn f32_roundtrip(v in any::<f32>()) {
            let hex = v.to_hex();
            if let Some(parsed) = f32::from_hex(&hex) {
                if v.is_nan() {
                    prop_assert!(parsed.is_nan());
                } else {
                    prop_assert_eq!(v.to_bits(), parsed.to_bits(),
                        "Roundtrip failed: {} -> {} -> {}", v, hex, parsed);
                }
            } else {
                prop_assert!(false, "from_hex failed for {}", hex);
            }
        }

        /// to_hex output should always be parseable.
        #[test]
        fn to_hex_is_parseable_f64(v in any::<f64>()) {
            let hex = v.to_hex();
            prop_assert!(f64::from_hex(&hex).is_some(), "Unparseable: {}", hex);
        }

        #[test]
        fn to_hex_is_parseable_f32(v in any::<f32>()) {
            let hex = v.to_hex();
            prop_assert!(f32::from_hex(&hex).is_some(), "Unparseable: {}", hex);
        }
    }
}
