//! Caption backend using the [aha](https://github.com/jhqxxx/aha) Qwen3-VL implementation.
//!
//! This module wraps aha's lower-level `Qwen3VLModel` and `Qwen3VLProcessor`
//! directly, bypassing aha's heavyweight `Qwen3VLGenerateModel` wrapper (which
//! requires an OpenAI-compatible `ChatCompletionParameters` type) in favour of
//! a simple greedy-decode inference loop.
//!
//! # Usage
//!
//! ```no_run
//! use sesoko::caption_crane::{AhaCaptionModel, DEFAULT_CAPTION_PROMPT};
//!
//! let model = AhaCaptionModel::load(
//!     "models/Qwen3-VL-2B-Instruct",
//!     DEFAULT_CAPTION_PROMPT,
//! )
//! .unwrap();
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
use tokenizers::Tokenizer;

use aha::models::qwen3vl::{
    config::{PreprocessorConfig, Qwen3VLConfig},
    model::Qwen3VLModel,
    processor::Qwen3VLProcessor,
};

use crate::caption::CaptionModel;

/// Default model directory name searched under `models/` when no explicit
/// path is supplied to [`crate::caption::resolve_caption_model_path`].
///
/// # Examples
///
/// ```
/// use sesoko::caption_crane::DEFAULT_CAPTION_MODEL_DIRNAME;
///
/// assert_eq!(DEFAULT_CAPTION_MODEL_DIRNAME, "Qwen3-VL-2B-Instruct");
/// ```
pub const DEFAULT_CAPTION_MODEL_DIRNAME: &str = "Qwen3-VL-2B-Instruct";

/// Default prompt sent to the VLM when captioning martial-arts imagery.
///
/// # Examples
///
/// ```
/// use sesoko::caption_crane::DEFAULT_CAPTION_PROMPT;
///
/// assert!(!DEFAULT_CAPTION_PROMPT.is_empty());
/// assert!(DEFAULT_CAPTION_PROMPT.len() < 512);
/// ```
pub const DEFAULT_CAPTION_PROMPT: &str = "Describe the image as a caption with less than 255 characters. It contains Japanese martial arts. What martial art is shown? What weapons? Describe clothing, belt, technique, the surrounding area, what kind of emotions the person might show? Respond with plain text only, no formatting or markdown. Be firm, no guessing.";

// ── Special token IDs (Qwen3-VL tokenizer) ──────────────────────────────────
const EOS: u32 = 151_643; // <|endoftext|>
const IM_START: u32 = 151_644; // <|im_start|>
const IM_END: u32 = 151_645; // <|im_end|>
const VISION_START: u32 = 151_652; // <|vision_start|>
const VISION_END: u32 = 151_653; // <|vision_end|>
const IMAGE_PAD: u32 = 151_655; // <|image_pad|>

// ── Token-sequence helpers ───────────────────────────────────────────────────

fn tok_encode(tokenizer: &Tokenizer, s: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(s, false)
        .map_err(|e| anyhow::anyhow!("tokenize {:?}: {e}", s))?
        .get_ids()
        .to_vec())
}

/// Build the full input token sequence using the Qwen3-VL chat template:
///
/// ```text
/// <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
/// <|im_start|>user\n<|vision_start|>[N × <|image_pad|>]<|vision_end|>{prompt}<|im_end|>\n
/// <|im_start|>assistant\n
/// ```
fn build_input_token_ids(
    tokenizer: &Tokenizer,
    prompt: &str,
    n_img_tokens: usize,
) -> Result<Vec<u32>> {
    let system_ids = tok_encode(tokenizer, "You are a helpful assistant.")?;
    let sys_role = tok_encode(tokenizer, "system\n")?;
    let user_role = tok_encode(tokenizer, "user\n")?;
    let asst_role = tok_encode(tokenizer, "assistant\n")?;
    let newline = tok_encode(tokenizer, "\n")?;
    let prompt_ids = tok_encode(tokenizer, prompt)?;

    let mut tokens: Vec<u32> = Vec::new();

    // <|im_start|>system\n…<|im_end|>\n
    tokens.push(IM_START);
    tokens.extend_from_slice(&sys_role);
    tokens.extend_from_slice(&system_ids);
    tokens.push(IM_END);
    tokens.extend_from_slice(&newline);

    // <|im_start|>user\n<|vision_start|>[N×<|image_pad|>]<|vision_end|>{prompt}<|im_end|>\n
    tokens.push(IM_START);
    tokens.extend_from_slice(&user_role);
    tokens.push(VISION_START);
    tokens.extend(std::iter::repeat_n(IMAGE_PAD, n_img_tokens));
    tokens.push(VISION_END);
    tokens.extend_from_slice(&prompt_ids);
    tokens.push(IM_END);
    tokens.extend_from_slice(&newline);

    // <|im_start|>assistant\n
    tokens.push(IM_START);
    tokens.extend_from_slice(&asst_role);

    Ok(tokens)
}

// ── Greedy sampler ───────────────────────────────────────────────────────────

/// Returns the index of the maximum value in a flat 1-D logit tensor.
fn argmax_token(logits: &Tensor) -> Result<u32> {
    let row: Vec<f32> = logits.to_vec1()?;
    Ok(row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i as u32)
        .unwrap_or(IM_END))
}

// ── Output cleaning ──────────────────────────────────────────────────────────

pub(crate) fn clean_caption_output(raw: &str) -> String {
    let mut text = raw.trim().to_string();
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

// ── AhaCaptionModel ──────────────────────────────────────────────────────────

/// Caption backend backed by [aha](https://github.com/jhqxxx/aha)'s
/// `Qwen3VLModel` and `Qwen3VLProcessor`.
///
/// `Qwen3VLModel::forward` requires `&mut self`, so the model is stored behind
/// a [`Mutex`] to satisfy the `&self` requirement of [`CaptionModel`] (and the
/// `Send + Sync` bounds needed for multi-threaded use).
///
/// Construct with [`AhaCaptionModel::load`] then pass to
/// [`crate::caption::run_caption_folder`] or call
/// [`CaptionModel::generate_caption`] directly.
pub struct AhaCaptionModel {
    model: Mutex<Qwen3VLModel>,
    processor: Qwen3VLProcessor,
    tokenizer: Tokenizer,
    device: Device,
    /// Image normalisation mean, shape `[3, 1, 1]`, F32.
    img_mean: Tensor,
    /// Image normalisation std, shape `[3, 1, 1]`, F32.
    img_std: Tensor,
    /// `spatial_merge_size` from the vision config (typically 2).
    merge_size: usize,
    prompt: String,
}

impl AhaCaptionModel {
    /// Load the Qwen3-VL model from `model_dir`.
    ///
    /// `model_dir` must contain:
    /// - `config.json`
    /// - `tokenizer.json`
    /// - `preprocessor_config.json`
    /// - `video_preprocessor_config.json`
    /// - one or more `*.safetensors` weight files
    ///
    /// The BF16 model `Qwen/Qwen3-VL-2B-Instruct` is expected; weights are
    /// loaded as F32 for CPU-compatible inference.  Download with:
    ///
    /// ```sh
    /// hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any required file is absent, cannot be parsed,
    /// or the model fails to build from the weight files.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::caption_crane::{AhaCaptionModel, DEFAULT_CAPTION_PROMPT};
    ///
    /// let model = AhaCaptionModel::load(
    ///     "models/Qwen3-VL-2B-Instruct",
    ///     DEFAULT_CAPTION_PROMPT,
    /// )
    /// .unwrap();
    /// ```
    ///
    /// ```no_run
    /// use sesoko::caption_crane::AhaCaptionModel;
    ///
    /// // Missing config.json returns an error
    /// assert!(AhaCaptionModel::load("/nonexistent/model", "describe this").is_err());
    /// ```
    pub fn load(model_dir: impl AsRef<Path>, prompt: impl Into<String>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_dir_str = model_dir
            .to_str()
            .with_context(|| format!("model dir path is not valid UTF-8: {}", model_dir.display()))?;

        // ── config.json ──────────────────────────────────────────────────────
        let config_path = model_dir.join("config.json");
        if !config_path.is_file() {
            bail!(
                "config.json not found in {}.\n\
                 Download the model with:\n  \
                 hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct",
                model_dir.display()
            );
        }

        let config: Qwen3VLConfig = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .with_context(|| format!("parsing {}", config_path.display()))?;

        let merge_size = config.vision_config.spatial_merge_size;

        // ── preprocessor_config.json — for mean/std tensors ─────────────────
        // Qwen3VLProcessor reads this file too but stores the values privately;
        // we read it ourselves so we can pass mean/std to process_images().
        let preproc_path = model_dir.join("preprocessor_config.json");
        let preproc: PreprocessorConfig = serde_json::from_str(
            &fs::read_to_string(&preproc_path)
                .with_context(|| format!("reading {}", preproc_path.display()))?,
        )
        .with_context(|| format!("parsing {}", preproc_path.display()))?;

        let device = Device::Cpu;
        let img_mean = Tensor::from_slice(&preproc.image_mean, (3, 1, 1), &device)?;
        let img_std = Tensor::from_slice(&preproc.image_std, (3, 1, 1), &device)?;

        // ── aha processor ────────────────────────────────────────────────────
        // Reads preprocessor_config.json and video_preprocessor_config.json.
        let processor = Qwen3VLProcessor::new(model_dir_str, &device, DType::F32)?;

        // ── tokenizer.json ───────────────────────────────────────────────────
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "loading tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        // ── *.safetensors weights ────────────────────────────────────────────
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

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&st_paths, DType::F32, &device)? };

        let model =
            Qwen3VLModel::new(config, vb).with_context(|| "building Qwen3VLModel via aha")?;

        Ok(Self {
            model: Mutex::new(model),
            processor,
            tokenizer,
            device,
            img_mean,
            img_std,
            merge_size,
            prompt: prompt.into(),
        })
    }
}

impl CaptionModel for AhaCaptionModel {
    fn generate_caption(&self, image: &DynamicImage) -> Result<String> {
        const MAX_NEW_TOKENS: usize = 256;

        // ── Image preprocessing ──────────────────────────────────────────────
        // process_images handles smart-resize, temporal duplication (T=2),
        // patch extraction, and normalisation.
        let vision_input = self
            .processor
            .process_images(vec![image.clone()], &self.img_mean, &self.img_std)?;

        let pixel_values = &vision_input.data; // [N_patches, 1536]
        let image_grid_thw = &vision_input.grid_thw; // [1, 3] — [[T, H, W]]

        // n_img_tokens = T*H*W / merge_size² (after spatial merge in vision encoder)
        let grid_vals = image_grid_thw.to_vec2::<u32>()?;
        let n_img_tokens = grid_vals[0].iter().map(|&x| x as usize).product::<usize>()
            / (self.merge_size * self.merge_size);

        // ── Tokenisation ─────────────────────────────────────────────────────
        let token_ids =
            build_input_token_ids(&self.tokenizer, &self.prompt, n_img_tokens)?;
        let seq_len = token_ids.len();
        let input_ids = Tensor::from_vec(token_ids, (1, seq_len), &self.device)?;
        let cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;

        // ── Inference ────────────────────────────────────────────────────────
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("model mutex poisoned"))?;

        // Wrap inference in a closure so clear_kv_cache() is always called,
        // even on error paths that return early.
        let result = (|| -> Result<Vec<u32>> {
            // Prefill: encode the whole prompt + image patches at once.
            let logits = model.forward(
                &input_ids,
                Some(pixel_values),
                Some(image_grid_thw),
                None, // pixel_values_video
                None, // video_grid_thw
                Some(&cache_position),
                0, // seqlen_offset
            )?;
            // forward returns [1, 1, vocab_size] — squeeze to [vocab_size]
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = argmax_token(&logits)?;

            let mut generated = vec![next_token];
            let mut seqlen_offset = seq_len;

            // Greedy decode loop — one token at a time, reusing KV cache.
            loop {
                let last = *generated.last().unwrap();
                if last == IM_END || last == EOS || generated.len() >= MAX_NEW_TOKENS {
                    break;
                }
                let decode_ids = Tensor::from_vec(vec![last], (1, 1), &self.device)?;
                let step_cache_pos =
                    Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
                let logits = model.forward(
                    &decode_ids,
                    None, // pixel_values — from KV cache now
                    None,
                    None,
                    None,
                    Some(&step_cache_pos),
                    seqlen_offset,
                )?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let token = argmax_token(&logits)?;
                generated.push(token);
                seqlen_offset += 1;
            }

            Ok(generated)
        })();

        // Always clear the KV cache regardless of success or failure.
        model.clear_kv_cache();
        drop(model);

        let mut generated = result?;

        // ── Decode tokens → text ─────────────────────────────────────────────
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

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_caption_strips_assistant_prefix() {
        let raw = "assistant\n\nA karateka in white gi performs a kick.";
        let clean = clean_caption_output(raw);
        assert!(
            !clean.contains("assistant"),
            "should strip 'assistant' prefix, got: {clean}"
        );
        assert!(
            clean.contains("kick"),
            "should preserve caption body, got: {clean}"
        );
    }

    #[test]
    fn clean_caption_joins_blank_lines() {
        let raw = "  A karateka.\n\n  White gi.\n";
        let clean = clean_caption_output(raw);
        assert_eq!(clean, "A karateka. White gi.");
    }

    #[test]
    fn clean_caption_empty_input_returns_empty() {
        assert_eq!(clean_caption_output(""), "");
        assert_eq!(clean_caption_output("  \n\n  "), "");
    }
}
