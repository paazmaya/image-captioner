use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::image_utils::{get_image_files, open_image, resize_image_aspect_ratio};

// Re-export the mistralrs-based implementation and shared constants.
pub use crate::caption_mistral::{
    MistralRsVlmCaptionModel, DEFAULT_CAPTION_MODEL_DIRNAME, DEFAULT_CAPTION_PROMPT,
};
// Candle/Qwen3 backend (commented out — compile with caption_qwen3 module to re-enable):
// pub use crate::caption_qwen3::{CandleVlmCaptionModel, DEFAULT_CAPTION_MODEL_DIRNAME, DEFAULT_CAPTION_PROMPT};

/// A trait for models that can generate a text caption for an image.
///
/// Implement this trait to plug in alternative or mock caption backends.
pub trait CaptionModel: Send + Sync {
    /// Generates a text caption for `image`.
    ///
    /// # Arguments
    ///
    /// * `image` – The source image to describe.
    ///
    /// # Returns
    ///
    /// A non-empty caption string on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails or produces no usable output.
    fn generate_caption(&self, image: &DynamicImage) -> Result<String>;
}

// ── Path resolution ─────────────────────────────────────────────────────────

/// Resolves the model directory path.
///
/// When `model_path` is `None`, returns `<cwd>/models/Qwen3-VL-2B-Instruct`.
/// When `model_path` is `Some(p)`, returns `p` unchanged.
///
/// This function does **not** verify that the returned path exists.
///
/// # Arguments
///
/// * `model_path` – Optional override path.  When `Some`, used as-is.
///
/// # Returns
///
/// A `PathBuf` pointing at (or expected to point at) the model directory.
///
/// # Errors
///
/// Returns an error only if `model_path` is `None` and the current working
/// directory cannot be determined.
///
/// # Examples
///
/// ```no_run
/// use sesoko::caption::resolve_caption_model_path;
///
/// // None → resolves to <cwd>/models/Qwen3-VL-2B-Instruct
/// let p = resolve_caption_model_path(None).unwrap();
/// assert!(p.ends_with("Qwen3-VL-2B-Instruct"));
/// ```
///
/// ```
/// use std::path::Path;
/// use sesoko::caption::resolve_caption_model_path;
///
/// // Some(path) → returned unchanged
/// let custom = Path::new("/my/custom/model");
/// let resolved = resolve_caption_model_path(Some(custom)).unwrap();
/// assert_eq!(resolved, custom);
/// ```
pub fn resolve_caption_model_path(model_path: Option<&Path>) -> Result<PathBuf> {
    let resolved = match model_path {
        Some(path) => path.to_path_buf(),
        None => {
            let cwd = env::current_dir().context("failed to read current working directory")?;
            cwd.join("models").join(DEFAULT_CAPTION_MODEL_DIRNAME)
        }
    };
    Ok(resolved)
}

/// Configuration for a batch captioning run.
#[derive(Debug, Clone)]
pub struct CaptionOptions {
    /// Directory containing source images to caption.
    pub folder: PathBuf,
    /// Path to the output TOML file that accumulates captions.
    pub output: PathBuf,
    /// When `true`, no per-image `.txt` sidecar files are written.
    pub no_sidecar: bool,
    /// Optional directory into which sidecar `.txt` files are written,
    /// mirroring the source structure.  When `None` (and `no_sidecar` is
    /// `false`) sidecars are written next to the source images.
    pub sidecar_dir: Option<PathBuf>,
}

/// A TOML-serialisable store mapping folder paths to per-image captions.
///
/// The outer key is the canonical path of the image folder; the inner key
/// is the relative path of each image within that folder.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use sesoko::caption::CaptionsFile;
///
/// let mut captions = CaptionsFile::default();
/// captions.0
///     .entry("images/".to_string())
///     .or_default()
///     .insert("photo.jpg".to_string(), "A person practising judo.".to_string());
/// assert_eq!(captions.0["images/"]["photo.jpg"], "A person practising judo.");
/// ```
///
/// ```
/// use sesoko::caption::CaptionsFile;
///
/// // Default is empty
/// let cf = CaptionsFile::default();
/// assert!(cf.0.is_empty());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CaptionsFile(pub BTreeMap<String, BTreeMap<String, String>>);

/// Captions every image in `options.folder` using `model` and persists the
/// results to a TOML file at `options.output`.
///
/// # Arguments
///
/// * `model` – Any [`CaptionModel`] implementation.  Pass a [`CandleVlmCaptionModel`]
///   for real inference or a stub/mock for testing.
/// * `options` – Configuration controlling input/output paths and sidecar
///   behaviour.
///
/// # Returns
///
/// A [`CaptionsFile`] containing **all** captions written so far (the TOML
/// file is loaded and merged on each run, so re-running is idempotent for
/// unchanged images).
///
/// # Errors
///
/// Returns an error if `options.folder` is not a directory, no images are
/// found, or any image fails to open.
///
/// # Examples
///
/// ```no_run
/// use std::path::PathBuf;
/// use sesoko::caption::{run_caption_folder, CaptionOptions, CandleVlmCaptionModel, DEFAULT_CAPTION_PROMPT};
///
/// let model = CandleVlmCaptionModel::load("models/Qwen3-VL-2B-Instruct", DEFAULT_CAPTION_PROMPT).unwrap();
/// let captions = run_caption_folder(&model, &CaptionOptions {
///     folder: PathBuf::from("images/"),
///     output: PathBuf::from("captions.toml"),
///     no_sidecar: false,
///     sidecar_dir: None,
/// }).unwrap();
/// println!("{} folder(s) captioned", captions.0.len());
/// ```
///
/// ```no_run
/// use std::path::PathBuf;
/// use sesoko::caption::{run_caption_folder, CaptionOptions};
///
/// // Non-existent folder returns an error
/// struct Stub;
/// impl sesoko::caption::CaptionModel for Stub {
///     fn generate_caption(&self, _: &image::DynamicImage) -> anyhow::Result<String> {
///         Ok("caption".to_string())
///     }
/// }
/// let result = run_caption_folder(&Stub, &CaptionOptions {
///     folder: PathBuf::from("/nonexistent"),
///     output: PathBuf::from("/tmp/out.toml"),
///     no_sidecar: true,
///     sidecar_dir: None,
/// });
/// assert!(result.is_err());
/// ```
pub fn run_caption_folder(
    model: &dyn CaptionModel,
    options: &CaptionOptions,
) -> Result<CaptionsFile> {
    if !options.folder.is_dir() {
        bail!("{} is not a valid directory", options.folder.display());
    }

    let image_files = get_image_files(&options.folder, true)?;
    if image_files.is_empty() {
        bail!("No image files found in {}", options.folder.display());
    }

    let mut output_path = options.output.clone();
    if output_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("toml"))
        != Some(true)
    {
        output_path.set_extension("toml");
    }

    let mut captions = if output_path.exists() {
        read_existing_captions(&output_path).unwrap_or_default()
    } else {
        CaptionsFile::default()
    };

    let absolute_folder = options.folder.canonicalize()?.to_string_lossy().to_string();
    captions.0.entry(absolute_folder.clone()).or_default();

    let sidecar_base = if options.no_sidecar {
        None
    } else {
        match &options.sidecar_dir {
            Some(dir) => {
                fs::create_dir_all(dir)?;
                Some(dir.clone())
            }
            None => None,
        }
    };

    for image_path in image_files {
        let image = open_image(&image_path)?;
        let resized = resize_image_aspect_ratio(&image, 896);
        let caption = model.generate_caption(&resized)?;

        let relative_path = image_path
            .strip_prefix(&options.folder)
            .with_context(|| {
                format!(
                    "failed to compute relative path for {}",
                    image_path.display()
                )
            })?
            .to_string_lossy()
            .replace('\\', "/");

        if let Some(folder_table) = captions.0.get_mut(&absolute_folder) {
            folder_table.insert(relative_path, caption.clone());
        }

        write_captions_file(&output_path, &captions)?;

        if !options.no_sidecar {
            let sidecar_path =
                build_sidecar_path(&image_path, &options.folder, sidecar_base.as_deref());
            if let Some(parent) = sidecar_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::File::create(&sidecar_path)
                .with_context(|| format!("failed to create sidecar {}", sidecar_path.display()))?;
            file.write_all(caption.as_bytes())?;
        }
    }

    Ok(captions)
}

/// Append `.txt` to the existing extension: `image.jpg` → `image.jpg.txt`.
fn append_txt_ext(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "{}.txt",
        path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
    ))
}

fn build_sidecar_path(image_path: &Path, folder: &Path, sidecar_base: Option<&Path>) -> PathBuf {
    match sidecar_base {
        Some(base) => {
            let relative = image_path.strip_prefix(folder).unwrap_or(image_path);
            base.join(append_txt_ext(relative))
        }
        None => append_txt_ext(image_path),
    }
}

fn read_existing_captions(path: &Path) -> Result<CaptionsFile> {
    let content = fs::read_to_string(path)?;
    let parsed = toml::from_str::<BTreeMap<String, BTreeMap<String, String>>>(&content)?;
    Ok(CaptionsFile(parsed))
}

fn write_captions_file(path: &Path, captions: &CaptionsFile) -> Result<()> {
    let content = toml::to_string_pretty(&captions.0)?;
    fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};
    use tempfile::tempdir;

    struct StubModel;

    impl CaptionModel for StubModel {
        fn generate_caption(&self, _image: &DynamicImage) -> Result<String> {
            Ok("Stub caption".to_string())
        }
    }

    #[test]
    fn caption_writes_toml_and_sidecar() {
        let dir = tempdir().unwrap();
        let img_path = dir.path().join("img.jpg");
        DynamicImage::ImageRgb8(RgbImage::new(16, 16))
            .save(&img_path)
            .unwrap();

        let output = dir.path().join("captions.toml");
        let options = CaptionOptions {
            folder: dir.path().to_path_buf(),
            output: output.clone(),
            no_sidecar: false,
            sidecar_dir: None,
        };

        let captions = run_caption_folder(&StubModel, &options).unwrap();
        assert!(output.exists());
        assert!(img_path.with_extension("jpg.txt").exists());

        let folder = dir
            .path()
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert_eq!(captions.0[&folder]["img.jpg"], "Stub caption");
    }

    #[test]
    fn resolve_caption_model_path_uses_default_models_folder() {
        let temp = tempdir().unwrap();
        let model_dir = temp
            .path()
            .join("models")
            .join(DEFAULT_CAPTION_MODEL_DIRNAME);
        fs::create_dir_all(&model_dir).unwrap();

        let cwd = env::current_dir().unwrap();
        env::set_current_dir(temp.path()).unwrap();

        let resolved = resolve_caption_model_path(None).unwrap();
        // Use canonicalize on both sides to handle macOS /private/var symlink
        assert_eq!(
            resolved.canonicalize().unwrap(),
            model_dir.canonicalize().unwrap()
        );

        env::set_current_dir(cwd).unwrap();
    }

    #[test]
    fn resolve_caption_model_path_accepts_directory_argument() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("custom");
        fs::create_dir_all(&model_dir).unwrap();

        let resolved = resolve_caption_model_path(Some(&model_dir)).unwrap();
        assert_eq!(resolved, model_dir);
    }

    #[test]
    fn resolve_caption_model_path_none_returns_default_subpath() {
        let resolved = resolve_caption_model_path(None).unwrap();
        // The last component must be the model dirname
        assert_eq!(
            resolved.file_name().and_then(|s| s.to_str()),
            Some(DEFAULT_CAPTION_MODEL_DIRNAME)
        );
    }

    #[test]
    fn caption_options_fields_accessible() {
        let opts = CaptionOptions {
            folder: std::path::PathBuf::from("/images"),
            output: std::path::PathBuf::from("/out.toml"),
            no_sidecar: true,
            sidecar_dir: None,
        };
        assert_eq!(opts.folder, std::path::PathBuf::from("/images"));
        assert!(opts.no_sidecar);
        assert!(opts.sidecar_dir.is_none());
    }

    #[test]
    fn captions_file_insert_and_retrieve() {
        let mut cf = CaptionsFile::default();
        cf.0.entry("folder".to_string())
            .or_default()
            .insert("img.jpg".to_string(), "A judo throw.".to_string());
        assert_eq!(cf.0["folder"]["img.jpg"], "A judo throw.");
    }

    #[test]
    fn run_caption_folder_no_sidecar_mode() {
        let dir = tempdir().unwrap();
        let img_path = dir.path().join("img.jpg");
        DynamicImage::ImageRgb8(RgbImage::new(16, 16))
            .save(&img_path)
            .unwrap();
        let output = dir.path().join("captions.toml");
        let options = CaptionOptions {
            folder: dir.path().to_path_buf(),
            output: output.clone(),
            no_sidecar: true,
            sidecar_dir: None,
        };
        run_caption_folder(&StubModel, &options).unwrap();
        // TOML file written
        assert!(output.exists());
        // No sidecar file next to the image
        assert!(!img_path.with_extension("jpg.txt").exists());
    }
}
