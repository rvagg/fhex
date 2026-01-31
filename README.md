# fhex

[![Crates.io](https://img.shields.io/crates/v/fhex.svg)](https://crates.io/crates/fhex)
[![Documentation](https://docs.rs/fhex/badge.svg)](https://docs.rs/fhex)

Hex float conversion for Rust: `ToHex` for formatting, `FromHex` for parsing.

Uses the [IEEE 754 hexadecimal floating-point](https://en.wikipedia.org/wiki/Hexadecimal_floating_point) format (`±0xh.hhhp±d`)—the same format used by C's `%a` printf specifier, Java's `Double.toHexString()`, and the WebAssembly text format.

**[Documentation](https://docs.rs/fhex)** · **[Crates.io](https://crates.io/crates/fhex)**

## Usage

```toml
[dependencies]
fhex = "2.0"
```

```rust
use fhex::{ToHex, FromHex};

// Formatting
let hex = 3.0_f64.to_hex();
assert_eq!(hex, "0x1.8p+1");

// Parsing
let value = f64::from_hex("0x1.8p+1").unwrap();
assert_eq!(value, 3.0);

// Round-trip
let original = std::f64::consts::PI;
let roundtrip = f64::from_hex(&original.to_hex()).unwrap();
assert_eq!(original, roundtrip);
```

## Format

Floating point numbers are represented as `±0xh.hhhp±d`, where:

* `±` is the sign (`-` for negative, omitted for positive)
* `0x` is the hex prefix
* `h.hhh` is the significand in hexadecimal
* `p±d` is the exponent in decimal (base 2)

Special values:

* `±0x0p+0` for zero
* `±inf` for infinity
* `nan` for quiet NaN
* `nan:0x...` for NaN with payload (signalling NaN)

## Parsing Features

* Underscores for readability: `0x1_000p+0`
* NaN payloads preserved: `nan:0x123`
* Case-insensitive: `INF`, `NaN`, `0X1P+0`
* Whitespace trimmed

## Output Differences vs Other Languages

* **Go's `%x`**: Uses `±Inf` and `NaN` (capitalised), zero-pads exponent to 2 digits
* **C++ `std::hexfloat`**: No special NaN payload handling

## License

Apache-2.0
