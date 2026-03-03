use sesoko::yolo_candle::{build_candle_detector_with_default, resolve_default_model_path_from};

#[test]
fn resolve_default_model_path_from_yields_error_with_no_model() {
    // A freshly-created temp directory has no safetensors file.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let result = resolve_default_model_path_from(tmp.path());
    assert!(
        result.is_err(),
        "expected error when no model file exists in the ancestor chain"
    );
}

#[test]
fn build_candle_detector_with_default_explicit_missing_path_errors() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let missing = tmp.path().join("nonexistent_weights.safetensors");
    let result = build_candle_detector_with_default(Some(missing.as_path()));
    assert!(result.is_err(), "expected error for missing weights file");
}
