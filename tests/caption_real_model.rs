mod common;

use sesoko::caption::{
    CandleVlmCaptionModel, CaptionModel, CaptionsFile, DEFAULT_CAPTION_MODEL_DIRNAME,
    DEFAULT_CAPTION_PROMPT,
};
use sesoko::image_utils::open_image;

fn default_model_dir() -> std::path::PathBuf {
    // Allow overriding via env var for testing different model variants
    if let Ok(dir) = std::env::var("SESOKO_MODEL_DIR") {
        return std::path::PathBuf::from(dir);
    }
    common::workspace_root()
        .join("models")
        .join(DEFAULT_CAPTION_MODEL_DIRNAME)
}

fn guard() -> bool {
    std::env::var("SESOKO_RUN_REAL_MODEL_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn real_model_captions_rumcajs() {
    if !guard() {
        return;
    }

    let model_dir = default_model_dir();
    assert!(
        model_dir.join("config.json").is_file(),
        "BF16 model not found at {}.\n\
         Download with:\n  \
         hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct",
        model_dir.display()
    );

    let model =
        CandleVlmCaptionModel::load(&model_dir, DEFAULT_CAPTION_PROMPT).expect("load model");

    let fixture = common::fixture_dir().join("rumcajs.jpg");
    assert!(fixture.is_file(), "fixture missing: {}", fixture.display());
    let image = open_image(&fixture).expect("open image");

    let caption = model.generate_caption(&image).expect("generate caption");
    assert!(!caption.trim().is_empty(), "caption must not be empty");
    println!("rumcajs.jpg caption: {caption}");
}

#[test]
fn real_model_captions_woman_webp() {
    if !guard() {
        return;
    }

    let model_dir = default_model_dir();
    assert!(
        model_dir.join("config.json").is_file(),
        "BF16 model not found at {}.\n\
         Download with:\n  \
         hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct",
        model_dir.display()
    );

    let model =
        CandleVlmCaptionModel::load(&model_dir, DEFAULT_CAPTION_PROMPT).expect("load model");

    let fixture = common::fixture_dir().join("woman.webp");
    assert!(fixture.is_file(), "fixture missing: {}", fixture.display());
    let image = open_image(&fixture).expect("open image");

    let caption = model.generate_caption(&image).expect("generate caption");
    assert!(!caption.trim().is_empty(), "caption must not be empty");
    println!("woman.webp caption: {caption}");
}

/// Verifies that `CaptionsFile` round-trips through TOML serialisation.
/// This test does not require any model weights.
#[test]
fn captions_file_write_and_parse_roundtrip() {
    use std::collections::BTreeMap;

    let mut inner: BTreeMap<String, String> = BTreeMap::new();
    inner.insert("photo.jpg".to_string(), "A judo throw.".to_string());
    inner.insert("kata.jpg".to_string(), "A kata sequence.".to_string());

    let mut outer: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    outer.insert("/images/martial".to_string(), inner.clone());

    let original = CaptionsFile(outer);

    // Serialize to TOML
    let toml_str = toml::to_string_pretty(&original.0).expect("serialize to TOML");
    assert!(!toml_str.is_empty());

    // Deserialize back
    let parsed: BTreeMap<String, BTreeMap<String, String>> =
        toml::from_str(&toml_str).expect("parse TOML");
    let roundtripped = CaptionsFile(parsed);

    assert_eq!(
        roundtripped.0["/images/martial"]["photo.jpg"],
        "A judo throw."
    );
    assert_eq!(
        roundtripped.0["/images/martial"]["kata.jpg"],
        "A kata sequence."
    );
}
