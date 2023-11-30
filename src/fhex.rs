//! This module provides the `ToHex` trait for converting floating-point numbers
//! to their hexadecimal representation. The trait is implemented for `f32` and `f64` types.
//!
//! The hexadecimal representation follows the format specified in the IEEE 754 standard.
//! The returned string starts with "0x" for non-zero numbers and "0" for zero.
//! The exponent is represented by 'p'.

/// `ToHex` is a trait that provides a method for converting floating-point
/// numbers to their hexadecimal representation.
///
/// This trait is implemented for `f32` and `f64` types.
///
/// # Examples
///
/// ```
/// use fhex::ToHex;
///
/// let num: f32 = 1.23;
/// let hex = num.to_hex();
/// println!("{}", hex);
/// ```
///
/// This will print the hexadecimal representation of `1.23`.
pub trait ToHex {
    /// Converts the floating-point number to a hexadecimal string of the format `±0xh.hhhp±d`.
    ///
    /// The returned string starts with "0x" for positive numbers and "-0x" for negative numbers.
    /// The string uses lowercase letters for the hexadecimal digits 'a' to 'f'.
    /// The exponent is represented by 'p', as specified in the IEEE 754 standard.
    /// NaN and Infinity and -Infinity are represented by "nan", "inf" and "-inf" respectively.
    /// Signalling NaN is represented by "nan:0x" followed by the significand in hexadecimal.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhex::ToHex;
    ///
    /// let num: f32 = 1.23;
    /// assert_eq!(num.to_hex(), "0x1.3ae148p+0");
    /// ```
    fn to_hex(self) -> String;
}

impl ToHex for f32 {
    fn to_hex(self) -> String {
        to_hex(FLOAT32, self.to_uint())
    }
}

impl ToHex for f64 {
    fn to_hex(self) -> String {
        to_hex(FLOAT64, self.to_uint())
    }
}

struct Floater {
    bits: u64,
    sig_bits: u64,
    #[allow(dead_code)] // TODO:
    exp_bits: u64,
    sign_shift: u64,
    sig_mask: u64,
    max_sig_digits: u64,
    #[allow(dead_code)] // TODO:
    sig_plus_one_bits: u64,
    #[allow(dead_code)] // TODO:
    sig_plus_one_mask: u64,
    exp_mask: u64,
    max_exp: i64,
    min_exp: i64,
    exp_bias: i64,
    quiet_nan_tag: u64,
}

impl Floater {
    const fn new(bits: u32, sig_bits: u32) -> Self {
        let exp_bits: u64 = bits as u64 - sig_bits as u64 - 1;
        let sign_shift: u64 = (bits as u64) - 1;
        let sig_mask = ((1 as u64) << sig_bits) - 1;
        let max_sig_digits = sig_bits as u64 * 3010 / 10000;
        let sig_plus_one_bits = sig_bits as u64 + 1;
        let sig_plus_one_mask = (1 << sig_plus_one_bits) - 1;
        let exp_mask = (1 << exp_bits) - 1;
        let max_exp = 1 << (exp_bits - 1);
        let min_exp = -(max_exp as i64) + 1;
        let exp_bias = -min_exp;
        let quiet_nan_tag = 1 << (sig_bits - 1);

        Self {
            bits: bits as u64,
            sig_bits: sig_bits as u64,
            exp_bits,
            sign_shift,
            sig_mask,
            max_sig_digits,
            sig_plus_one_bits,
            sig_plus_one_mask,
            exp_mask,
            max_exp,
            min_exp,
            exp_bias,
            quiet_nan_tag,
        }
    }
}

const HEX_DIGITS: &str = "0123456789abcdef";

fn to_hex(typ: Floater, bits: u64) -> String {
    let mut buffer = String::new();

    let mut exponent = (((bits >> typ.sig_bits) & typ.exp_mask) as i64) - typ.exp_bias;
    let mut significand = bits & typ.sig_mask;

    if bits >> typ.sign_shift != 0 {
        // negative sign bit
        buffer.push('-');
    }
    if exponent == typ.max_exp {
        // exponent indicates we have special values
        handle_nan_and_infinity(&mut buffer, significand, &typ);
    } else {
        let is_zero = significand == 0 && exponent == typ.min_exp;
        buffer.push_str("0x");
        buffer.push(if is_zero { '0' } else { '1' });

        // shift significand up so the top 4-bits are at the top
        significand <<= (typ.bits - typ.sig_bits) as i32;

        if significand != 0 {
            // there's a significand to print, we should get a `.xxx`
            handle_significand(&mut buffer, &mut exponent, significand, &typ);
        }
        // add the exponent, so we get a `p+xxx` or `p-xxx
        handle_exponent(&mut buffer, is_zero, exponent)
    }
    buffer
}

fn handle_nan_and_infinity(buffer: &mut String, significand: u64, typ: &Floater) {
    let num_nybbles = typ.bits / 4;
    let top_nybble_shift = typ.bits - 4;
    let top_nybble = 0xf << top_nybble_shift;

    // Infinity or nan.
    if significand == 0 {
        buffer.push_str("inf");
    } else {
        buffer.push_str("nan");
        if significand != typ.quiet_nan_tag {
            buffer.push_str(":0x");
            // Skip leading zeroes.
            let mut significand = significand;
            let mut num_nybbles = num_nybbles;
            while (significand & top_nybble) == 0 {
                significand <<= 4;
                num_nybbles -= 1;
            }
            while num_nybbles > 0 {
                let nybble = (significand >> top_nybble_shift) & 0xf;
                buffer.push(HEX_DIGITS.chars().nth(nybble as usize).unwrap());
                significand <<= 4;
                num_nybbles -= 1;
            }
        }
    }
}

fn handle_significand(
    buffer: &mut String,
    exponent: &mut i64,
    mut significand: u64,
    typ: &Floater,
) {
    let top_nybble_shift = typ.bits - 4;

    if *exponent == typ.min_exp {
        // Subnormal; shift the significand up, and shift out the implicit 1.
        let leading_zeroes = significand.leading_zeros();
        let leading_zeroes = if typ.bits == 32 {
            leading_zeroes.saturating_sub(32)
        } else {
            leading_zeroes
        };
        if leading_zeroes < typ.sign_shift as u32 {
            significand <<= (leading_zeroes + 1) as i32;
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
                // remove '.'
                buffer.pop();
            }
            break;
        }
        let nybble = (significand >> top_nybble_shift) & 0xf;
        buffer.push(HEX_DIGITS.chars().nth(nybble as usize).unwrap());
        significand <<= 4;
        if typ.bits == 32 {
            significand &= 0xffffffff;
        }
    }
}

fn handle_exponent(buffer: &mut String, is_zero: bool, mut exponent: i64) {
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
        if exponent >= 1000 {
            buffer.push('1');
        }
        if exponent >= 100 {
            let digit = (exponent / 100) % 10;
            buffer.push((('0' as u8) + digit as u8) as char);
        }
        if exponent >= 10 {
            let digit = (exponent / 10) % 10;
            buffer.push((('0' as u8) + digit as u8) as char);
        }
        let digit = exponent % 10;
        buffer.push((('0' as u8) + digit as u8) as char);
    }
}

const FLOAT32: Floater = Floater::new(4 * 8, 23);
const FLOAT64: Floater = Floater::new(8 * 8, 52);

trait ToUint64<U> {
    fn to_uint(self) -> u64;
}

impl ToUint64<u32> for f32 {
    fn to_uint(self) -> u64 {
        self.to_bits() as u64
    }
}

impl ToUint64<u64> for f64 {
    fn to_uint(self) -> u64 {
        self.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use crate::fhex::ToHex;

    #[test]
    fn test_f32_hex() {
        let cases = [
            ([0x00, 0x00, 0x00, 0x80], "-0x0p+0"), // Go: "-0x0p+00" (no match) // C++: -0x0p+0
            ([0x00, 0x00, 0x00, 0x00], "0x0p+0"),  // Go: "0x0p+00" (no match) // C++: 0x0p+0
            ([0x01, 0x00, 0x80, 0xd8], "-0x1.000002p+50"), // Go: "-0x1.000002p+50" // C++: -0x1.000002p+50
            ([0x01, 0x00, 0x80, 0xa6], "-0x1.000002p-50"), // Go: "-0x1.000002p-50" // C++: -0x1.000002p-50
            ([0x01, 0x00, 0x80, 0x58], "0x1.000002p+50"), // Go: "0x1.000002p+50" // C++: 0x1.000002p+50
            ([0x01, 0x00, 0x80, 0x26], "0x1.000002p-50"), // Go: "0x1.000002p-50" // C++: 0x1.000002p-50
            ([0x01, 0x00, 0x00, 0x7f], "0x1.000002p+127"), // Go: "0x1.000002p+127" // C++: 0x1.000002p+127
            ([0x02, 0x00, 0x80, 0xd8], "-0x1.000004p+50"), // Go: "-0x1.000004p+50" // C++: -0x1.000004p+50
            ([0x02, 0x00, 0x80, 0xa6], "-0x1.000004p-50"), // Go: "-0x1.000004p-50" // C++: -0x1.000004p-50
            ([0x02, 0x00, 0x80, 0x58], "0x1.000004p+50"), // Go: "0x1.000004p+50" // C++: 0x1.000004p+50
            ([0x02, 0x00, 0x80, 0x26], "0x1.000004p-50"), // Go: "0x1.000004p-50" // C++: 0x1.000004p-50
            ([0x03, 0x00, 0x80, 0xd8], "-0x1.000006p+50"), // Go: "-0x1.000006p+50" // C++: -0x1.000006p+50
            ([0x03, 0x00, 0x80, 0xa6], "-0x1.000006p-50"), // Go: "-0x1.000006p-50" // C++: -0x1.000006p-50
            ([0x03, 0x00, 0x80, 0x58], "0x1.000006p+50"), // Go: "0x1.000006p+50" // C++: 0x1.000006p+50
            ([0x03, 0x00, 0x80, 0x26], "0x1.000006p-50"), // Go: "0x1.000006p-50" // C++: 0x1.000006p-50
            ([0xb4, 0xa2, 0x11, 0x52], "0x1.234568p+37"), // Go: "0x1.234568p+37" // C++: 0x1.234568p+37
            ([0xb4, 0xa2, 0x91, 0x5b], "0x1.234568p+56"), // Go: "0x1.234568p+56" // C++: 0x1.234568p+56
            ([0xb4, 0xa2, 0x11, 0x65], "0x1.234568p+75"), // Go: "0x1.234568p+75" // C++: 0x1.234568p+75
            ([0x99, 0x76, 0x96, 0xfe], "-0x1.2ced32p+126"), // Go: "-0x1.2ced32p+126" // C++: -0x1.2ced32p+126
            ([0x99, 0x76, 0x96, 0x7e], "0x1.2ced32p+126"), // Go: "0x1.2ced32p+126" // C++: 0x1.2ced32p+126
            ([0x03, 0x00, 0x00, 0x80], "-0x1.8p-148"),     // Go: "-0x1.8p-148" // C++: -0x1.8p-148
            ([0x03, 0x00, 0x00, 0x00], "0x1.8p-148"),      // Go: "0x1.8p-148" // C++: 0x1.8p-148
            ([0xff, 0x2f, 0x59, 0x2d], "0x1.b25ffep-37"), // Go: "0x1.b25ffep-37" // C++: 0x1.b25ffep-37
            ([0xa3, 0x79, 0xeb, 0x4c], "0x1.d6f346p+26"), // Go: "0x1.d6f346p+26" // C++: 0x1.d6f346p+26
            ([0x7b, 0x4d, 0x7f, 0x6c], "0x1.fe9af6p+89"), // Go: "0x1.fe9af6p+89" // C++: 0x1.fe9af6p+89
            ([0x00, 0x00, 0x00, 0xff], "-0x1p+127"),      // Go: "-0x1p+127" // C++: -0x1p+127
            ([0x00, 0x00, 0x00, 0x7f], "0x1p+127"),       // Go: "0x1p+127" // C++: 0x1p+127
            ([0x02, 0x00, 0x00, 0x80], "-0x1p-148"),      // Go: "-0x1p-148" // C++: -0x1p-148
            ([0x02, 0x00, 0x00, 0x00], "0x1p-148"),       // Go: "0x1p-148" // C++: 0x1p-148
            ([0x01, 0x00, 0x00, 0x80], "-0x1p-149"),      // Go: "-0x1p-149" // C++: -0x1p-149
            ([0x01, 0x00, 0x00, 0x00], "0x1p-149"),       // Go: "0x1p-149" // C++: 0x1p-149
            ([0x00, 0x00, 0x80, 0xd8], "-0x1p+50"),       // Go: "-0x1p+50" // C++: -0x1p+50
            ([0x00, 0x00, 0x80, 0xa6], "-0x1p-50"),       // Go: "-0x1p-50" // C++: -0x1p-50
            ([0x00, 0x00, 0x80, 0x58], "0x1p+50"),        // Go: "0x1p+50" // C++: 0x1p+50
            ([0x00, 0x00, 0x80, 0x26], "0x1p-50"),        // Go: "0x1p-50" // C++: 0x1p-50
            ([0x00, 0x00, 0x80, 0x7f], "inf"),            // Go: "+Inf" (no match) // C++: inf
            ([0x00, 0x00, 0x80, 0xff], "-inf"),           // Go: "-Inf" (no match) // C++: -inf
            ([0x00, 0x00, 0xc0, 0x7f], "nan"),            // Go: "NaN" (no match) // C++: nan
            ([0x01, 0x00, 0x80, 0x7f], "nan:0x1"), // Go: "NaN" (no match) // C++: nan (no match)
            ([0xff, 0xff, 0xff, 0x7f], "nan:0x7fffff"), // Go: "NaN" (no match) // C++: nan (no match)
            ([0x00, 0x00, 0x80, 0x3f], "0x1p+0"),       // Go: "0x1p+00" (no match) // C++: 0x1p+0
            ([0x00, 0x00, 0x80, 0xbf], "-0x1p+0"),      // Go: "-0x1p+00" (no match) // C++: -0x1p+0
            ([0xff, 0xff, 0x7f, 0x7f], "0x1.fffffep+127"), // Go: "0x1.fffffep+127" // C++: 0x1.fffffep+127
            ([0xff, 0xff, 0x7f, 0xff], "-0x1.fffffep+127"), // Go: "-0x1.fffffep+127" // C++: -0x1.fffffep+127
            ([0x00, 0x00, 0x80, 0x4b], "0x1p+24"),          // Go: "0x1p+24" // C++: 0x1p+24
            ([0x00, 0x00, 0x80, 0xcb], "-0x1p+24"),         // Go: "-0x1p+24" // C++: -0x1p+24
            ([0xa4, 0x70, 0x9d, 0x3f], "0x1.3ae148p+0"), // Go: "0x1.3ae148p+00" (no match) // C++: 0x1.3ae148p+0
        ];
        for case in &cases {
            let (bytes, expected) = case;
            let float = f32::from_ne_bytes(*bytes);
            let result = float.to_hex();
            println!(
                "Float32: [{}] {} -> {}",
                bytes
                    .iter()
                    .map(|b| format!("{:02X}", b))
                    .collect::<Vec<String>>()
                    .join(" "),
                float,
                result
            );
            assert_eq!(result, *expected);
        }
    }

    #[test]
    fn test_f64_hex() {
        let cases = [
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80], "-0x0p+0"), // Go: "-0x0p+00" (no match) // C++: -0x0p+0
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], "0x0p+0"), // Go: "0x0p+00" (no match) // C++: 0x0p+0
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0xc3],
                "-0x1.0000000000001p+60",
            ), // Go: "-0x1.0000000000001p+60" // C++: -0x1.0000000000001p+60
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0x43],
                "0x1.0000000000001p+60",
            ), // Go: "0x1.0000000000001p+60" // C++: 0x1.0000000000001p+60
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0xe5],
                "-0x1.0000000000001p+600",
            ), // Go: "-0x1.0000000000001p+600" // C++: -0x1.0000000000001p+600
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x9a],
                "-0x1.0000000000001p-600",
            ), // Go: "-0x1.0000000000001p-600" // C++: -0x1.0000000000001p-600
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x65],
                "0x1.0000000000001p+600",
            ), // Go: "0x1.0000000000001p+600" // C++: 0x1.0000000000001p+600
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x1a],
                "0x1.0000000000001p-600",
            ), // Go: "0x1.0000000000001p-600" // C++: 0x1.0000000000001p-600
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6],
                "-0x1.0000000000001p+97",
            ), // Go: "-0x1.0000000000001p+97" // C++: -0x1.0000000000001p+97
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46],
                "0x1.0000000000001p+97",
            ), // Go: "0x1.0000000000001p+97" // C++: 0x1.0000000000001p+97
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0xfe],
                "-0x1.0000000000001p+999",
            ), // Go: "-0x1.0000000000001p+999" // C++: -0x1.0000000000001p+999
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x7e],
                "0x1.0000000000001p+999",
            ), // Go: "0x1.0000000000001p+999" // C++: 0x1.0000000000001p+999
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0xc3],
                "-0x1.0000000000002p+60",
            ), // Go: "-0x1.0000000000002p+60" // C++: -0x1.0000000000002p+60
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0x43],
                "0x1.0000000000002p+60",
            ), // Go: "0x1.0000000000002p+60" // C++: 0x1.0000000000002p+60
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0xe5],
                "-0x1.0000000000002p+600",
            ), // Go: "-0x1.0000000000002p+600" // C++: -0x1.0000000000002p+600
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x9a],
                "-0x1.0000000000002p-600",
            ), // Go: "-0x1.0000000000002p-600" // C++: -0x1.0000000000002p-600
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x65],
                "0x1.0000000000002p+600",
            ), // Go: "0x1.0000000000002p+600" // C++: 0x1.0000000000002p+600
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x1a],
                "0x1.0000000000002p-600",
            ), // Go: "0x1.0000000000002p-600" // C++: 0x1.0000000000002p-600
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6],
                "-0x1.0000000000002p+97",
            ), // Go: "-0x1.0000000000002p+97" // C++: -0x1.0000000000002p+97
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46],
                "0x1.0000000000002p+97",
            ), // Go: "0x1.0000000000002p+97" // C++: 0x1.0000000000002p+97
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0xfe],
                "-0x1.0000000000002p+999",
            ), // Go: "-0x1.0000000000002p+999" // C++: -0x1.0000000000002p+999
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x7e],
                "0x1.0000000000002p+999",
            ), // Go: "0x1.0000000000002p+999" // C++: 0x1.0000000000002p+999
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x80],
                "-0x1.0000000000003p-1022",
            ), // Go: "-0x1.0000000000003p-1022" // C++: -0x1.0000000000003p-1022
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00],
                "0x1.0000000000003p-1022",
            ), // Go: "0x1.0000000000003p-1022" // C++: 0x1.0000000000003p-1022
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0xe5],
                "-0x1.0000000000003p+600",
            ), // Go: "-0x1.0000000000003p+600" // C++: -0x1.0000000000003p+600
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x9a],
                "-0x1.0000000000003p-600",
            ), // Go: "-0x1.0000000000003p-600" // C++: -0x1.0000000000003p-600
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x65],
                "0x1.0000000000003p+600",
            ), // Go: "0x1.0000000000003p+600" // C++: 0x1.0000000000003p+600
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x1a],
                "0x1.0000000000003p-600",
            ), // Go: "0x1.0000000000003p-600" // C++: 0x1.0000000000003p-600
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6],
                "-0x1.0000000000003p+97",
            ), // Go: "-0x1.0000000000003p+97" // C++: -0x1.0000000000003p+97
            (
                [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46],
                "0x1.0000000000003p+97",
            ), // Go: "0x1.0000000000003p+97" // C++: 0x1.0000000000003p+97
            (
                [0xa0, 0xc8, 0xeb, 0x85, 0xf3, 0xcc, 0xe1, 0xff],
                "-0x1.1ccf385ebc8ap+1023",
            ), // Go: "-0x1.1ccf385ebc8ap+1023" // C++: -0x1.1ccf385ebc8ap+1023
            (
                [0xa0, 0xc8, 0xeb, 0x85, 0xf3, 0xcc, 0xe1, 0x7f],
                "0x1.1ccf385ebc8ap+1023",
            ), // Go: "0x1.1ccf385ebc8ap+1023" // C++: 0x1.1ccf385ebc8ap+1023
            (
                [0xdf, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0xc2, 0x43],
                "0x1.23456789abcdfp+61",
            ), // Go: "0x1.23456789abcdfp+61" // C++: 0x1.23456789abcdfp+61
            (
                [0xdf, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0xf2, 0x44],
                "0x1.23456789abcdfp+80",
            ), // Go: "0x1.23456789abcdfp+80" // C++: 0x1.23456789abcdfp+80
            (
                [0xdf, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x22, 0x46],
                "0x1.23456789abcdfp+99",
            ), // Go: "0x1.23456789abcdfp+99" // C++: 0x1.23456789abcdfp+99
            (
                [0x11, 0x43, 0x2b, 0xd6, 0xff, 0x25, 0xab, 0x3d],
                "0x1.b25ffd62b4311p-37",
            ), // Go: "0x1.b25ffd62b4311p-37" // C++: 0x1.b25ffd62b4311p-37
            (
                [0x12, 0xec, 0x36, 0xd6, 0xff, 0x25, 0xab, 0x3d],
                "0x1.b25ffd636ec12p-37",
            ), // Go: "0x1.b25ffd636ec12p-37" // C++: 0x1.b25ffd636ec12p-37
            (
                [0x58, 0xa4, 0x0c, 0x54, 0x34, 0x6f, 0x9d, 0x41],
                "0x1.d6f34540ca458p+26",
            ), // Go: "0x1.d6f34540ca458p+26" // C++: 0x1.d6f34540ca458p+26
            (
                [0x00, 0x00, 0x00, 0x54, 0x34, 0x6f, 0x9d, 0x41],
                "0x1.d6f3454p+26",
            ), // Go: "0x1.d6f3454p+26" // C++: 0x1.d6f3454p+26
            (
                [0xfa, 0x16, 0x5e, 0x5b, 0xaf, 0xe9, 0x8f, 0x45],
                "0x1.fe9af5b5e16fap+89",
            ), // Go: "0x1.fe9af5b5e16fap+89" // C++: 0x1.fe9af5b5e16fap+89
            (
                [0xd5, 0xcb, 0x6b, 0x5b, 0xaf, 0xe9, 0x8f, 0x45],
                "0x1.fe9af5b6bcbd5p+89",
            ), // Go: "0x1.fe9af5b6bcbd5p+89" // C++: 0x1.fe9af5b6bcbd5p+89
            (
                [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0xff],
                "-0x1.fffffffffffffp+1023",
            ), // Go: "-0x1.fffffffffffffp+1023" // C++: -0x1.fffffffffffffp+1023
            (
                [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0x7f],
                "0x1.fffffffffffffp+1023",
            ), // Go: "0x1.fffffffffffffp+1023" // C++: 0x1.fffffffffffffp+1023
            (
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0xff],
                "-0x1p+1023",
            ), // Go: "-0x1p+1023" // C++: -0x1p+1023
            (
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x7f],
                "0x1p+1023",
            ), // Go: "0x1p+1023" // C++: 0x1p+1023
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80],
                "-0x1p-1073",
            ), // Go: "-0x1p-1073" // C++: -0x0.0000000000002p-1022 (no match)
            (
                [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                "0x1p-1073",
            ), // Go: "0x1p-1073" // C++: 0x0.0000000000002p-1022 (no match)
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80],
                "-0x1p-1074",
            ), // Go: "-0x1p-1074" // C++: -0x0.0000000000001p-1022 (no match)
            (
                [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                "0x1p-1074",
            ), // Go: "0x1p-1074" // C++: 0x0.0000000000001p-1022 (no match)
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0xc3], "-0x1p+60"), // Go: "-0x1p+60" // C++: -0x1p+60
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0x43], "0x1p+60"), // Go: "0x1p+60" // C++: 0x1p+60
            (
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0xe5],
                "-0x1p+600",
            ), // Go: "-0x1p+600" // C++: -0x1p+600
            (
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x9a],
                "-0x1p-600",
            ), // Go: "-0x1p-600" // C++: -0x1p-600
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x65], "0x1p+600"), // Go: "0x1p+600" // C++: 0x1p+600
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x1a], "0x1p-600"), // Go: "0x1p-600" // C++: 0x1p-600
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6], "-0x1p+97"), // Go: "-0x1p+97" // C++: -0x1p+97
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46], "0x1p+97"), // Go: "0x1p+97" // C++: 0x1p+97
            (
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0xfe],
                "-0x1p+999",
            ), // Go: "-0x1p+999" // C++: -0x1p+999
            ([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x7e], "0x1p+999"), // Go: "0x1p+999" // C++: 0x1p+999
            ([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f], "nan:0x1"), // Go: "NaN" (no match) // C++: nan (no match)
            (
                [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f],
                "nan:0xfffffffffffff",
            ), // Go: "NaN" (no match) // C++: nan (no match)
        ];
        for case in &cases {
            let (bytes, expected) = case;
            let float = f64::from_ne_bytes(*bytes);
            let result = float.to_hex();
            println!("Float64: {} -> {}", float, result);
            assert_eq!(result, *expected);
        }
    }
}

/*
// Go test program. Change the `float32` to `float64` if running the f64 cases.
// Copy and paste in the test case block, surrounded by ``, comma separated.

package main

import (
    "bytes"
    "encoding/binary"
    "fmt"
    "regexp"
    "strconv"
    "strings"
)

func main() {
    cs := []string{
        `([0x00, 0x00, 0x00, 0x80], "-0x0p+0"),`,
        `...`,
        `([0x00, 0x00, 0x80, 0xcb], "-0x1.000002p-1"),`,
    }

    r := regexp.MustCompile(`\(\[([0-9a-fx, ]+)\], "([^"]+)"\)`)
    for _, c := range cs {
        matches := r.FindStringSubmatch(c)
        if len(matches) <= 2 {
            panic("bad match")
        }
        byteStrings := strings.Split(matches[1], ",")
        byts := make([]byte, len(byteStrings))
        for i, bs := range byteStrings {
            b, _ := strconv.ParseUint(strings.TrimSpace(bs), 0, 8)
            byts[i] = byte(b)
        }
        buf := bytes.NewReader(byts)
        var f float32 // change to float64
        if err := binary.Read(buf, binary.LittleEndian, &f); err != nil {
            panic(err)
        }
        gofmt := fmt.Sprintf("%x", f)
        nomatch := ""
        if matches[2] != gofmt {
            nomatch = " (no match)"
        }
        fmt.Printf("%s, // Go: %q%s\n", c, gofmt, nomatch)
    }
}
*/

/*
// Go test program. Change the `float` to `double` if running the f64 cases.
// Copy and paste in the test case block, with '"' replaced with '\"' and
// surrounded by "", comma separated.

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>

int main() {
    std::vector<std::string> cs = {
        "([0x00, 0x00, 0x00, 0x80], \"-0x0p+0\"), // Go: \"-0x0p+00\" (no match)",
        "...",
        "([0x00, 0x00, 0x80, 0xcb], \"-0x1p+24\"), // Go: \"-0x1p+24\"",
    };

    std::regex r("\\(\\[([0-9a-fx, ]+)\\], \"([^\"]+)\"\\)");
    for (const auto& c : cs) {
        std::smatch matches;
        if (std::regex_search(c, matches, r) && matches.size() > 2) {
            std::stringstream ss(matches[1]);
            std::string byteString;
            std::vector<unsigned char> bytes;
            while (std::getline(ss, byteString, ',')) {
                bytes.push_back(std::stoul(byteString, nullptr, 0));
            }
            float f;
            std::memcpy(&f, bytes.data(), sizeof(f));
            std::stringstream hexfloatStream;
            hexfloatStream << std::hexfloat << f;
            std::string cppfmt = hexfloatStream.str();
            std::string nomatch = "";
            if (matches[2] != cppfmt) {
                nomatch = " (no match)";
            }
            std::cout << c << " // C++: " << cppfmt << nomatch << "\n";
        } else {
            std::cout << "bad match\n";
        }
    }
    return 0;
}
*/
