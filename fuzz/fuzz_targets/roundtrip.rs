#![no_main]

use libfuzzer_sys::fuzz_target;

use fhex::{FromHex, ToHex};

fuzz_target!(|data: (f64, f32)| {
    let (f64_val, f32_val) = data;

    // f64 round-trip: to_hex should always produce parseable output
    let hex = f64_val.to_hex();
    let parsed = f64::from_hex(&hex);
    assert!(parsed.is_some(), "Failed to parse f64 to_hex output: {}", hex);

    // For non-NaN values, bits should match exactly
    if !f64_val.is_nan() {
        assert_eq!(
            f64_val.to_bits(),
            parsed.unwrap().to_bits(),
            "f64 roundtrip mismatch: {} -> {} -> {:?}",
            f64_val,
            hex,
            parsed
        );
    }

    // f32 round-trip
    let hex = f32_val.to_hex();
    let parsed = f32::from_hex(&hex);
    assert!(parsed.is_some(), "Failed to parse f32 to_hex output: {}", hex);

    if !f32_val.is_nan() {
        assert_eq!(
            f32_val.to_bits(),
            parsed.unwrap().to_bits(),
            "f32 roundtrip mismatch: {} -> {} -> {:?}",
            f32_val,
            hex,
            parsed
        );
    }
});
