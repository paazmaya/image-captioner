use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::image_utils::{get_image_files, open_image, resize_image_aspect_ratio};

/// Default model directory name searched under `models/` when no explicit
/// path is supplied to [`resolve_caption_model_path`].
///
/// # Examples
///
/// ```
/// use sesoko::caption::DEFAULT_CAPTION_MODEL_DIRNAME;
///
/// assert_eq!(DEFAULT_CAPTION_MODEL_DIRNAME, "Qwen3-VL-2B-Instruct");
/// ```
pub const DEFAULT_CAPTION_MODEL_DIRNAME: &str = "Qwen3-VL-2B-Instruct";

/// Default prompt sent to the VLM when captioning martial-arts imagery.
///
/// # Examples
///
/// ```
/// use sesoko::caption::DEFAULT_CAPTION_PROMPT;
///
/// assert!(!DEFAULT_CAPTION_PROMPT.is_empty());
/// assert!(DEFAULT_CAPTION_PROMPT.len() < 512);
/// ```
pub const DEFAULT_CAPTION_PROMPT: &str = "Describe the image as a caption with less than 255 characters. It contains Japanese martial arts. What martial art is shown? Describe clothing, belt, technique, the surrounding area, what kind of emotions the person might show? Respond with plain text only, no formatting or markdown. Be firm, no guessing.";

// ── Special token IDs (Qwen3-VL tokenizer) ─────────────────────────────────
const EOS: u32 = 151643; // <|endoftext|>
const IM_START: u32 = 151644; // <|im_start|>
const IM_END: u32 = 151645; // <|im_end|>
const VISION_START: u32 = 151652; // <|vision_start|>
const VISION_END: u32 = 151653; // <|vision_end|>
const IMAGE_PAD: u32 = 151655; // <|image_pad|>

// ── Preprocessing constants ─────────────────────────────────────────────────
const PATCH_SIZE: usize = 16;
const TEMPORAL_PATCH_SIZE: usize = 2;
const SPATIAL_MERGE_SIZE: usize = 2;
/// Features per patch = C × T_temporal × pH × pW = 3 × 2 × 16 × 16
const PIXEL_DIM: usize = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE; // 1536
/// smart_resize alignment factor = patch_size × spatial_merge_size = 32
const SMART_RESIZE_FACTOR: u32 = (PATCH_SIZE * SPATIAL_MERGE_SIZE) as u32;
const MIN_PIXELS: u64 = 3_136; // 56 × 56 for practical CPU inference
const MAX_PIXELS: u64 = 1_003_520; // 28² × 1280 for practical CPU inference

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

// ── Image preprocessing ─────────────────────────────────────────────────────

/// Resize `(h, w)` so that `h × w` is within `[MIN_PIXELS, MAX_PIXELS]` and
/// both dimensions are multiples of `SMART_RESIZE_FACTOR`. Aspect ratio is
/// preserved as closely as possible.
fn smart_resize(h: u32, w: u32) -> (u32, u32) {
    let f = SMART_RESIZE_FACTOR as f64;
    let mut h_bar = (((h as f64) / f).round() as u32).max(1) * SMART_RESIZE_FACTOR;
    let mut w_bar = (((w as f64) / f).round() as u32).max(1) * SMART_RESIZE_FACTOR;

    let area = (h_bar as u64) * (w_bar as u64);
    if area > MAX_PIXELS {
        let scale = (MAX_PIXELS as f64 / area as f64).sqrt();
        h_bar = (((h as f64) * scale / f).floor() as u32).max(1) * SMART_RESIZE_FACTOR;
        w_bar = (((w as f64) * scale / f).floor() as u32).max(1) * SMART_RESIZE_FACTOR;
    } else if area < MIN_PIXELS {
        let scale = (MIN_PIXELS as f64 / area as f64).sqrt();
        h_bar = (((h as f64) * scale / f).ceil() as u32).max(1) * SMART_RESIZE_FACTOR;
        w_bar = (((w as f64) * scale / f).ceil() as u32).max(1) * SMART_RESIZE_FACTOR;
    }
    (h_bar, w_bar)
}

/// Preprocess one image into tensors the vision encoder expects.
///
/// Returns
/// - `pixel_values`: `[N, 1536]` f32, where `N = num_patches_h × num_patches_w`
/// - `image_grid_thw`: `[1, 3]` u32 = `[[T, H, W]]` with `T=1`
/// - `num_img_tokens`: image tokens after spatial merge = `H × W / spatial_merge²`
fn preprocess_image(img: &DynamicImage, device: &Device) -> Result<(Tensor, Tensor, usize)> {
    let (orig_w, orig_h) = (img.width(), img.height());
    let (new_h, new_w) = smart_resize(orig_h, orig_w);
    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let nh = new_h as usize;
    let nw = new_w as usize;
    let num_patches_h = nh / PATCH_SIZE;
    let num_patches_w = nw / PATCH_SIZE;
    let n_patches = num_patches_h * num_patches_w;

    // Build channels-first f32 array [C, H, W], normalised to [-1, +1].
    let mut chw = vec![0f32; 3 * nh * nw];
    for (idx, pixel) in rgb.pixels().enumerate() {
        let row = idx / nw;
        let col = idx % nw;
        for c in 0..3usize {
            chw[c * nh * nw + row * nw + col] = pixel[c] as f32 / 127.5 - 1.0;
        }
    }

    // Build pixel_values [N, 1536].
    // The PatchEmbed in the vision model reshapes its input as
    //   (N, C, T_temporal, pH, pW)  ← index = c*(T*pH*pW) + t*(pH*pW) + y*pW + x
    // For a static image we duplicate the frame (T=TEMPORAL_PATCH_SIZE=2).
    let mut pv = vec![0f32; n_patches * PIXEL_DIM];
    for pi in 0..num_patches_h {
        for pj in 0..num_patches_w {
            let pidx = pi * num_patches_w + pj;
            for c in 0..3usize {
                let c_stride = TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
                for py in 0..PATCH_SIZE {
                    for px in 0..PATCH_SIZE {
                        let v =
                            chw[c * nh * nw + (pi * PATCH_SIZE + py) * nw + pj * PATCH_SIZE + px];
                        let spatial = py * PATCH_SIZE + px;
                        for t in 0..TEMPORAL_PATCH_SIZE {
                            pv[pidx * PIXEL_DIM
                                + c * c_stride
                                + t * PATCH_SIZE * PATCH_SIZE
                                + spatial] = v;
                        }
                    }
                }
            }
        }
    }

    let pixel_values = Tensor::from_vec(pv, (n_patches, PIXEL_DIM), device)?;
    let image_grid_thw = Tensor::from_vec(
        vec![1u32, num_patches_h as u32, num_patches_w as u32],
        (1, 3),
        device,
    )?;
    let num_img_tokens = n_patches / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE);
    Ok((pixel_values, image_grid_thw, num_img_tokens))
}

// ── Prompt tokenisation ─────────────────────────────────────────────────────

/// Build the full input token sequence using the Qwen3-VL chat template:
/// ```text
/// <|im_start|>system\n{SYSTEM}<|im_end|>\n
/// <|im_start|>user\n<|vision_start|>[N × <|image_pad|>]<|vision_end|>{prompt}<|im_end|>\n
/// <|im_start|>assistant\n
/// ```
///
/// Returns `(token_ids, continuous_img_pad)` where `continuous_img_pad` is the
/// per-batch-item list of `(start, end)` index ranges that will be filled with
/// image embeddings by `Qwen3VLModel::forward`.
#[allow(clippy::type_complexity)]
fn build_prompt_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    n_img_tokens: usize,
) -> Result<(Vec<u32>, Vec<Vec<(usize, usize)>>)> {
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenize {:?}: {e}", s))?
            .get_ids()
            .to_vec())
    };

    let system_ids = encode("You are a helpful assistant.")?;
    let sys_role = encode("system\n")?;
    let user_role = encode("user\n")?;
    let asst_role = encode("assistant\n")?;
    let newline = encode("\n")?;
    let prompt_ids = encode(prompt)?;

    let mut tokens: Vec<u32> = Vec::new();

    // <|im_start|>system\n…<|im_end|>\n
    tokens.push(IM_START);
    tokens.extend_from_slice(&sys_role);
    tokens.extend_from_slice(&system_ids);
    tokens.push(IM_END);
    tokens.extend_from_slice(&newline);

    // <|im_start|>user\n<|vision_start|>[image pads]<|vision_end|>{prompt}<|im_end|>\n
    tokens.push(IM_START);
    tokens.extend_from_slice(&user_role);
    tokens.push(VISION_START);
    let img_pad_start = tokens.len();
    tokens.extend(std::iter::repeat_n(IMAGE_PAD, n_img_tokens));
    let img_pad_end = tokens.len();
    tokens.push(VISION_END);
    tokens.extend_from_slice(&prompt_ids);
    tokens.push(IM_END);
    tokens.extend_from_slice(&newline);

    // <|im_start|>assistant\n
    tokens.push(IM_START);
    tokens.extend_from_slice(&asst_role);

    let continuous_img_pad = vec![vec![(img_pad_start, img_pad_end)]];
    Ok((tokens, continuous_img_pad))
}

// ── Greedy sampler ──────────────────────────────────────────────────────────

fn argmax_token(logits: &Tensor) -> Result<u32> {
    // logits: [batch, vocab_size] — forward_embeds already selects the last token
    let row: Vec<f32> = logits.i(0)?.to_dtype(DType::F32)?.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
    indexed
        .sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "[sesoko] top-5 tokens: {:?} | special={:.3}/{:.3}/{:.3} (EOS/IM_START/IM_END)",
        &indexed[..5.min(indexed.len())],
        row[EOS as usize],
        row[IM_START as usize],
        row[IM_END as usize]
    );
    Ok(indexed.first().map(|(i, _)| *i as u32).unwrap_or(IM_END))
}

// ── CandleVlmCaptionModel ───────────────────────────────────────────────────

/// Standalone VLM caption backend using `candle-transformers` and the
/// `Qwen/Qwen3-VL-2B-Instruct` (BF16) safetensors model.
///
/// Construct with [`CandleVlmCaptionModel::load`] then pass to
/// [`run_caption_folder`] or call [`CaptionModel::generate_caption`]
/// directly.
pub struct CandleVlmCaptionModel {
    model: Qwen3VLModel,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    config: Config,
    device: Device,
    prompt: String,
}

impl CandleVlmCaptionModel {
    /// Load the model from `model_dir`.
    ///
    /// `model_dir` must contain `config.json`, `tokenizer.json`, and one or
    /// more `*.safetensors` weight files.  The **BF16** model
    /// `Qwen/Qwen3-VL-2B-Instruct` is required; download it with:
    /// ```sh
    /// hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct
    /// ```
    ///
    /// # Arguments
    ///
    /// * `model_dir` – Path to the directory containing the model files.
    /// * `prompt` – System prompt sent to the VLM before each image.
    ///   Use [`DEFAULT_CAPTION_PROMPT`] for the built-in martial-arts prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.json` or `tokenizer.json` are absent,
    /// no `.safetensors` files are found, or the model cannot be built.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::caption::{CandleVlmCaptionModel, DEFAULT_CAPTION_PROMPT};
    ///
    /// let model = CandleVlmCaptionModel::load(
    ///     "models/Qwen3-VL-2B-Instruct",
    ///     DEFAULT_CAPTION_PROMPT,
    /// ).unwrap();
    /// ```
    ///
    /// ```no_run
    /// use sesoko::caption::CandleVlmCaptionModel;
    ///
    /// // Missing config.json returns an error
    /// assert!(
    ///     CandleVlmCaptionModel::load("/nonexistent/model", "describe this").is_err()
    /// );
    /// ```
    pub fn load(model_dir: impl AsRef<Path>, prompt: impl Into<String>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !config_path.is_file() {
            bail!(
                "config.json not found in {}.\n\
                 Download the model with:\n  \
                 hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct",
                model_dir.display()
            );
        }

        let config: Config = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        let mut st_paths: Vec<PathBuf> = fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        st_paths.sort();

        if st_paths.is_empty() {
            bail!(
                "No .safetensors files found in {}.\n\
                 Download the model with:\n  \
                 hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct",
                model_dir.display()
            );
        }

        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&st_paths, DType::F32, &device)? };

        let model = Qwen3VLModel::new(&config, vb).with_context(|| "building Qwen3VLModel")?;

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            prompt: prompt.into(),
        })
    }
}

impl CaptionModel for CandleVlmCaptionModel {
    fn generate_caption(&self, image: &DynamicImage) -> Result<String> {
        const MAX_NEW_TOKENS: usize = 256;

        // Allow text-only testing mode (skips vision encoder; image replaced with N pads)
        let text_only_mode = std::env::var("SESOKO_TEXT_ONLY")
            .map(|v| v == "1")
            .unwrap_or(false);

        let (pixel_values_opt, image_grid_thw_opt, n_img_tokens, continuous_img_pad_arg) =
            if text_only_mode {
                // Text-only: feed 4 dummy image pad tokens but no pixel values
                eprintln!("[sesoko] TEXT-ONLY mode: skipping vision encoder");
                (None, None, 4usize, vec![vec![]])
            } else {
                let (pv, grid, n) = preprocess_image(image, &self.device)?;
                let (_, cip) = build_prompt_tokens(&self.tokenizer, &self.prompt, n)?;
                (Some(pv), Some(grid), n, cip)
            };

        let (token_ids, continuous_img_pad) =
            build_prompt_tokens(&self.tokenizer, &self.prompt, n_img_tokens)?;
        let continuous_img_pad_for_forward = if text_only_mode {
            continuous_img_pad_arg
        } else {
            continuous_img_pad
        };

        let seq_len = token_ids.len();
        let input_ids = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;

        // ── Prefill ─────────────────────────────────────────────────────────
        let logits = self.model.forward(
            &input_ids,
            pixel_values_opt,
            None, // pixel_values_videos
            image_grid_thw_opt,
            None,          // video_grid_thw
            vec![seq_len], // seqlens
            continuous_img_pad_for_forward,
            vec![vec![]], // continuous_vid_pad
            &[0usize],    // seqlen_offsets (no KV cache yet)
        )?;

        let next = argmax_token(&logits)?;
        eprintln!("[sesoko] prefill first token: {next}");
        let mut generated = vec![next];
        let mut past_len = seq_len;

        // ── Decode loop ──────────────────────────────────────────────────────
        loop {
            let last = *generated.last().unwrap();
            if last == IM_END || last == EOS || generated.len() >= MAX_NEW_TOKENS {
                break;
            }
            let decode_ids = Tensor::from_vec(vec![last], (1, 1), &self.device)?;
            let logits = self.model.forward(
                &decode_ids,
                None, // pixel_values — use KV cache from prefill
                None,
                None,
                None,
                vec![1],      // seqlens
                vec![vec![]], // continuous_img_pad (no image in decode step)
                vec![vec![]], // continuous_vid_pad
                &[past_len],  // seqlen_offsets = number of already-processed tokens
            )?;
            let token = argmax_token(&logits)?;
            eprintln!("[sesoko] decode step {}: token {token}", generated.len());
            generated.push(token);
            past_len += 1;
        }

        while matches!(generated.last(), Some(&IM_END) | Some(&EOS)) {
            generated.pop();
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decoding generated tokens: {e}"))?;

        let cleaned = clean_caption_output(&text);
        if cleaned.is_empty() {
            bail!("model generated an empty caption");
        }
        Ok(cleaned)
    }
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

// ── Shared helpers ───────────────────────────────────────────────────────────

fn clean_caption_output(raw_output: &str) -> String {
    let mut text = raw_output.trim().to_string();
    if let Some(pos) = text.rfind("assistant") {
        let (_, right) = text.split_at(pos + "assistant".len());
        text = right.trim().to_string();
    }
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
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
    fn smart_resize_aligns_to_factor() {
        let (h, w) = smart_resize(100, 150);
        assert_eq!(h % SMART_RESIZE_FACTOR, 0, "h={h} not aligned");
        assert_eq!(w % SMART_RESIZE_FACTOR, 0, "w={w} not aligned");
    }

    #[test]
    fn smart_resize_respects_pixel_bounds() {
        let (h, w) = smart_resize(4096, 4096);
        let area = (h as u64) * (w as u64);
        // Allow small overshoot due to rounding to `factor` grid
        assert!(area <= MAX_PIXELS + (SMART_RESIZE_FACTOR as u64) * 2 * 4096);
    }

    #[test]
    fn preprocess_image_produces_correct_shapes() {
        // 224 is a multiple of SMART_RESIZE_FACTOR (32), so no scaling needed.
        let img = DynamicImage::ImageRgb8(RgbImage::new(224, 224));
        let (pv, grid, n_tokens) = preprocess_image(&img, &Device::Cpu).unwrap();

        let g = grid.to_vec2::<u32>().unwrap();
        assert_eq!(g.len(), 1);
        let (t, h, w) = (g[0][0], g[0][1] as usize, g[0][2] as usize);
        assert_eq!(t, 1);
        assert_eq!(h, 224 / PATCH_SIZE);
        assert_eq!(w, 224 / PATCH_SIZE);
        assert_eq!(pv.dims(), &[h * w, PIXEL_DIM]);
        assert_eq!(n_tokens, h * w / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE));
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
