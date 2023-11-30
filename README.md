# fhex

`fhex` is a Rust crate that provides the `ToHex` trait for converting floating-point numbers to their hexadecimal representation. The trait is implemented for `f32` and `f64` types.

The hexadecimal representation follows the format specified for the IEEE 754 standard, `±0xh.hhhp±d`, described below.

To parse hexadecimal floating point numbers, see the [hexf](https://crates.io/crates/hexf) crate.

## Usage

Add `fhex` to your `Cargo.toml` dependencies:

```toml
[dependencies]
fhex = "1.0.0"
```

Then, import the `ToHex` trait in your code:

```rust
use fhex::ToHex;
```

You can now call the `to_hex` method on any `f32` or `f64` value:

```rust
let num: f32 = 1.23;
let hex = num.to_hex();
println!("{}", hex);
```

This will print the hexadecimal representation of 1.23.

```
0x1.3ae148p+0
```

## Format

Floating point numbers are represented in the format: `±0xh.hhhp±d`, where:

* `±` is the sign of the number, although `+` is omitted for positive numbers.
* `0x` is the prefix for hexadecimal numbers.
* `h.hhh` is the significand (also known as the mantissa), represented as a hexadecimal number.
* `p` indicates that the following number is the exponent.
* `±d` is the exponent, represented as a decimal number.

Special cases:

* `±0x0p+0` is used to represent zero.
* `±inf` is used to represent infinity.
* `nan` is used to represent NaN (a "quiet NaN").
* `nan:0x` followed by a hexadecimal number is used to represent a NaN with a payload (a "signalling NaN").

### Known differences with other implementations

* Go's `%x` format specifier for floating point numbers uses the format `±0xh.hhhp±d`, but it insists on printing the exponent with at least two digits, zero padded. Infinity is represented as `±Inf` and NaN as `NaN`. There is no special treatment of signalling NaNs.
* C++'s `std::hexfloat` format specifier for floating point numbers uses the format `±0xh.hhhp±d`. There is no special treatment of signalling NaNs, they are just `nan`.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.
