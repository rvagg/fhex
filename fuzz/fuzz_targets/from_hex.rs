#![no_main]

use libfuzzer_sys::fuzz_target;

use fhex::FromHex;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to a string (invalid UTF-8 becomes replacement chars)
    let source = String::from_utf8_lossy(data);

    // Try parsing as f64 - should never panic
    let _ = f64::from_hex(&source);

    // Try parsing as f32 - should never panic
    let _ = f32::from_hex(&source);
});
