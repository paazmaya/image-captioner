use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use mistralrs::{
    blocking::BlockingModel, IsqType, RequestBuilder, TextMessageRole, VisionMessages,
    VisionModelBuilder,
};

use crate::caption::CaptionModel;

/// Default model directory name searched under `models/` when no explicit
/// path is supplied to [`crate::caption::resolve_caption_model_path`].
pub const DEFAULT_CAPTION_MODEL_DIRNAME: &str = "Qwen3-VL-2B-Instruct";

/// Default prompt sent to the VLM when captioning martial-arts imagery.
pub const DEFAULT_CAPTION_PROMPT: &str = "Describe the image as a caption with less than 255 characters. It contains Japanese martial arts. What martial art is shown? What weapons? Describe clothing, belt, technique, the surrounding area, what kind of emotions the person might show? Respond with plain text only, no formatting or markdown. Be firm, no guessing.";

/// Strip any chat-template artefacts from the raw model output.
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

/// Caption backend that uses `mistralrs` for Qwen3-VL-2B-Instruct inference.
///
/// Uses Q4K in-situ quantization (ISQ) at load time, roughly halving RAM
/// usage compared with the BF16 weights stored on disk.
///
/// Construct with [`MistralRsVlmCaptionModel::load`] and pass to
/// [`crate::caption::run_caption_folder`] or call [`CaptionModel::generate_caption`]
/// directly.
pub struct MistralRsVlmCaptionModel {
    model: BlockingModel,
    prompt: String,
}

impl MistralRsVlmCaptionModel {
    /// Load the model from `model_dir`.
    ///
    /// `model_dir` must contain `config.json`, `tokenizer.json`, and the model
    /// safetensors (e.g. `models/Qwen3-VL-2B-Instruct`).
    /// Q4K ISQ quantization is applied in-memory at load time.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory does not exist or the model cannot be loaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::caption_mistral::{MistralRsVlmCaptionModel, DEFAULT_CAPTION_PROMPT};
    ///
    /// let model = MistralRsVlmCaptionModel::load(
    ///     "models/Qwen3-VL-2B-Instruct",
    ///     DEFAULT_CAPTION_PROMPT,
    /// ).unwrap();
    /// ```
    pub fn load(model_dir: impl AsRef<Path>, prompt: impl Into<String>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            bail!(
                "model directory not found: {}\n\
                 Download with:\n  \
                 hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct",
                model_dir.display()
            );
        }

        // Canonicalize to an absolute path so mistralrs / hf-hub can locate files.
        let model_path = model_dir
            .canonicalize()
            .with_context(|| format!("cannot resolve path: {}", model_dir.display()))?
            .to_string_lossy()
            .into_owned();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("failed to create tokio runtime")?;

        let async_model = rt
            .block_on(
                VisionModelBuilder::new(&model_path)
                    // If auto-detection of loader type fails for the local config.json,
                    // add: .with_loader_type(mistralrs::VisionLoaderType::Qwen3VL)
                    .with_isq(IsqType::Q4K)
                    .build(),
            )
            .context("failed to load Qwen3-VL model with mistralrs")?;

        let model = BlockingModel::new(async_model, Arc::new(rt));

        Ok(Self {
            model,
            prompt: prompt.into(),
        })
    }
}

impl CaptionModel for MistralRsVlmCaptionModel {
    fn generate_caption(&self, image: &DynamicImage) -> Result<String> {
        let messages = VisionMessages::new()
            .add_message(TextMessageRole::System, "You are a helpful assistant.")
            .add_image_message(
                TextMessageRole::User,
                self.prompt.as_str(),
                vec![image.clone()],
            );

        let request = RequestBuilder::from(messages).set_sampler_max_len(256);

        let response = self
            .model
            .send_chat_request(request)
            .context("mistralrs inference request failed")?;

        let raw = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        let cleaned = clean_caption_output(&raw);
        if cleaned.is_empty() {
            bail!("model generated an empty caption");
        }
        Ok(cleaned)
    }
}
