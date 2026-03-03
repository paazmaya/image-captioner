use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{non_maximum_suppression, Bbox};
use image::DynamicImage;

use crate::crop::{default_class_names, BoundingBox, Detector, DEFAULT_COCO_CLASSES};
use crate::yolo_model::{Multiples, YoloV8};

const DEFAULT_SAFETENSORS_CANDIDATES: &[&str] = &[
    "models/candle-yolo-v8/yolov8n.safetensors",
    "yolov8n.safetensors",
];

/// Confidence threshold applied to raw class scores during detection.
///
/// Predictions below this value are discarded before NMS.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.25;
/// Intersection-over-union threshold used during non-maximum suppression.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.45;
/// Side length (in pixels) to which input images are resized before
/// being fed into the YOLOv8 backbone.
pub const MODEL_INPUT_SIZE: usize = 640;

/// Wrap a candle expression, mapping its error to an `anyhow` error with a label.
macro_rules! candle_op {
    ($expr:expr, $label:literal) => {
        $expr.map_err(|e| anyhow::anyhow!("{}: {}", $label, e))?
    };
}

/// A YOLOv8 object detector powered by [candle](https://github.com/huggingface/candle),
/// running inference on CPU using safetensors weights.
///
/// Construct with [`CandleYoloDetector::load`] or the higher-level
/// [`build_candle_detector`] / [`build_candle_detector_with_default`] helpers.
pub struct CandleYoloDetector {
    model: YoloV8,
    device: Device,
    class_names: Vec<String>,
    num_classes: usize,
}

impl CandleYoloDetector {
    /// Load a detector from a `.safetensors` weights file.
    ///
    /// # Arguments
    ///
    /// * `path` – Path to a `yolov8n.safetensors` weights file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, cannot be memory-mapped,
    /// or the weight tensors do not match the YOLOv8-nano architecture.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::yolo_candle::CandleYoloDetector;
    ///
    /// let detector = CandleYoloDetector::load_safetensors(
    ///     "models/candle-yolo-v8/yolov8n.safetensors"
    /// ).unwrap();
    /// ```
    ///
    /// ```no_run
    /// use sesoko::yolo_candle::CandleYoloDetector;
    ///
    /// // Missing file returns an error
    /// assert!(CandleYoloDetector::load_safetensors("/nonexistent.safetensors").is_err());
    /// ```
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            anyhow::bail!("safetensors model not found: {}", path.display());
        }
        let device = Device::Cpu;
        let multiples = Multiples::n();
        let num_classes = DEFAULT_COCO_CLASSES.len();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)
                .with_context(|| format!("failed to mmap safetensors: {}", path.display()))?
        };
        let model = YoloV8::load(vb, multiples, num_classes)
            .map_err(|e| anyhow::anyhow!("failed to load YoloV8 model: {e}"))?;
        Ok(Self {
            model,
            device,
            num_classes,
            class_names: default_class_names(),
        })
    }

    /// Load a detector from a `.safetensors` weights file.
    ///
    /// This is an alias for [`CandleYoloDetector::load_safetensors`].
    ///
    /// Download weights with:
    /// ```sh
    /// hf download lmz/candle-yolo-v8 yolov8n.safetensors --local-dir models/candle-yolo-v8
    /// ```
    ///
    /// # Arguments
    ///
    /// * `model_path` – Path to the `.safetensors` weights file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is absent or incompatible with the
    /// YOLOv8-nano architecture.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::yolo_candle::CandleYoloDetector;
    ///
    /// let detector = CandleYoloDetector::load("models/candle-yolo-v8/yolov8n.safetensors").unwrap();
    /// ```
    ///
    /// ```no_run
    /// use sesoko::yolo_candle::CandleYoloDetector;
    ///
    /// assert!(CandleYoloDetector::load("/no/such/file.safetensors").is_err());
    /// ```
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_safetensors(model_path)
    }

    /// Run YOLO detection on `image` and return all detected class/bbox pairs.
    ///
    /// The image is resized to fit within [`MODEL_INPUT_SIZE`] × [`MODEL_INPUT_SIZE`]
    /// while preserving the aspect ratio.  Bounding boxes in the returned list
    /// are in the coordinate space of the **original** (un-resized) image.
    ///
    /// # Arguments
    ///
    /// * `image` – Input image (any size; will be resized internally).
    /// * `confidence_threshold` – Minimum class confidence; predictions below
    ///   this value are discarded.  Typical value: [`DEFAULT_CONFIDENCE_THRESHOLD`].
    /// * `nms_threshold` – IoU threshold for non-maximum suppression.
    ///   Typical value: [`DEFAULT_NMS_THRESHOLD`].
    ///
    /// # Returns
    ///
    /// A `Vec` of `(class_index, Bbox)` tuples, one entry per surviving detection.
    ///
    /// # Errors
    ///
    /// Returns an error if the model forward pass fails (e.g., tensor shape
    /// mismatch) or if the output tensor cannot be read back to the CPU.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::yolo_candle::{CandleYoloDetector, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD};
    /// use image::DynamicImage;
    ///
    /// let detector = CandleYoloDetector::load("yolov8n.safetensors").unwrap();
    /// let img = image::open("photo.jpg").unwrap();
    /// let detections = detector.detect(&img, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD).unwrap();
    /// println!("Detected {} objects", detections.len());
    /// ```
    ///
    /// ```no_run
    /// use sesoko::yolo_candle::{CandleYoloDetector, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD};
    ///
    /// let detector = CandleYoloDetector::load("yolov8n.safetensors").unwrap();
    /// let blank = image::DynamicImage::ImageRgb8(image::RgbImage::new(640, 480));
    /// // A blank image should yield no detections above the default threshold
    /// let detections = detector.detect(&blank, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD).unwrap();
    /// assert!(detections.is_empty());
    /// ```
    pub fn detect(
        &self,
        image: &DynamicImage,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<(usize, Bbox<()>)>> {
        let (orig_w, orig_h) = (image.width() as f32, image.height() as f32);

        let (scaled_w, scaled_h) = {
            let w = image.width() as usize;
            let h = image.height() as usize;
            if w < h {
                let w = (w * MODEL_INPUT_SIZE / h) / 32 * 32;
                (w.max(32), MODEL_INPUT_SIZE)
            } else {
                let h = (h * MODEL_INPUT_SIZE / w) / 32 * 32;
                (MODEL_INPUT_SIZE, h.max(32))
            }
        };

        let resized = image.resize_exact(
            scaled_w as u32,
            scaled_h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let raw = resized.to_rgb8().into_raw();

        let image_t = candle_op!(
            candle_op!(
                Tensor::from_vec(raw, (scaled_h, scaled_w, 3), &self.device),
                "build input tensor"
            )
            .permute((2, 0, 1)),
            "permute"
        );
        let image_t = candle_op!(
            candle_op!(
                candle_op!(image_t.unsqueeze(0), "unsqueeze").to_dtype(DType::F32),
                "to_dtype"
            ) * (1. / 255.),
            "normalize"
        );

        let predictions = candle_op!(
            candle_op!(self.model.forward(&image_t), "model forward").squeeze(0),
            "squeeze"
        );

        extract_bboxes(
            &predictions,
            orig_w,
            orig_h,
            scaled_w as f32,
            scaled_h as f32,
            confidence_threshold,
            nms_threshold,
            self.num_classes,
        )
    }
}

impl Detector for CandleYoloDetector {
    fn detect_first_box_for_class(
        &self,
        image: &DynamicImage,
        class_name: &str,
    ) -> Result<Option<BoundingBox>> {
        let class_idx = match self.class_names.iter().position(|c| c == class_name) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let detections = self
            .detect(image, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD)
            .context("YOLO detection failed")?;

        let best = detections
            .into_iter()
            .filter(|(cls, _)| *cls == class_idx)
            .max_by(|a, b| a.1.confidence.total_cmp(&b.1.confidence));

        Ok(best.map(|(_, b)| BoundingBox {
            x1: b.xmin,
            y1: b.ymin,
            x2: b.xmax,
            y2: b.ymax,
        }))
    }

    fn class_names(&self) -> Vec<String> {
        self.class_names.clone()
    }
}

// ---------------------------------------------------------------------------
// Post-processing helpers
// ---------------------------------------------------------------------------

/// Extract and NMS-filter bounding boxes from the raw model output tensor.
///
/// `pred` has shape `[pred_size, npreds]` where `pred_size = 4 + num_classes`.
#[allow(clippy::too_many_arguments)]
fn extract_bboxes(
    pred: &Tensor,
    orig_w: f32,
    orig_h: f32,
    scaled_w: f32,
    scaled_h: f32,
    confidence_threshold: f32,
    nms_threshold: f32,
    num_classes: usize,
) -> Result<Vec<(usize, Bbox<()>)>> {
    let pred = candle_op!(pred.to_device(&Device::Cpu), "pred to cpu");
    let (pred_size, npreds) = candle_op!(pred.dims2(), "pred dims2");

    if pred_size != 4 + num_classes {
        anyhow::bail!(
            "unexpected prediction tensor size {pred_size}, expected {} \
             (4 + {} classes)",
            4 + num_classes,
            num_classes
        );
    }

    let w_ratio = orig_w / scaled_w;
    let h_ratio = orig_h / scaled_h;

    let mut bboxes: Vec<Vec<Bbox<()>>> = (0..num_classes).map(|_| vec![]).collect();

    for idx in 0..npreds {
        let col: Vec<f32> = pred
            .i((.., idx))
            .and_then(Vec::<f32>::try_from)
            .map_err(|e| anyhow::anyhow!("read prediction column {idx}: {e}"))?;

        let confidence = col[4..].iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if confidence <= confidence_threshold {
            continue;
        }

        let class_idx = col[4..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if col[4 + class_idx] <= 0. {
            continue;
        }

        // col[0..4] = cx, cy, w, h in scaled-image pixel coordinates.
        bboxes[class_idx].push(Bbox {
            xmin: (col[0] - col[2] / 2.) * w_ratio,
            ymin: (col[1] - col[3] / 2.) * h_ratio,
            xmax: (col[0] + col[2] / 2.) * w_ratio,
            ymax: (col[1] + col[3] / 2.) * h_ratio,
            confidence,
            data: (),
        });
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    Ok(bboxes
        .into_iter()
        .enumerate()
        .flat_map(|(cls, bs)| bs.into_iter().map(move |b| (cls, b)))
        .collect())
}

// ---------------------------------------------------------------------------
// Public builder API
// ---------------------------------------------------------------------------

/// Builds a [`Detector`] trait object backed by candle YOLO v8 inference.
///
/// # Arguments
///
/// * `model_path` – Path to a `.safetensors` weights file.
///
/// # Errors
///
/// Returns an error if the weights file cannot be loaded.
///
/// # Examples
///
/// ```no_run
/// use sesoko::yolo_candle::build_candle_detector;
///
/// let detector = build_candle_detector("models/candle-yolo-v8/yolov8n.safetensors").unwrap();
/// ```
///
/// ```no_run
/// use sesoko::yolo_candle::build_candle_detector;
///
/// assert!(build_candle_detector("/nonexistent/weights.safetensors").is_err());
/// ```
pub fn build_candle_detector(model_path: impl AsRef<Path>) -> Result<Box<dyn Detector>> {
    let detector = CandleYoloDetector::load(model_path)
        .context("failed to initialize Candle YOLO detector")?;
    Ok(Box::new(detector))
}

/// Resolves the default YOLO model path by searching upward from the current
/// working directory.
///
/// Equivalent to `resolve_default_model_path_from(&std::env::current_dir())`.
///
/// # Errors
///
/// Returns an error if the current directory is unreadable or no model file
/// is found anywhere in the ancestor chain.
///
/// # Examples
///
/// ```no_run
/// use sesoko::yolo_candle::resolve_default_model_path;
///
/// // Succeeds when run from inside a workspace that has the model downloaded
/// let path = resolve_default_model_path().unwrap();
/// assert!(path.exists());
/// ```
///
/// ```no_run
/// use sesoko::yolo_candle::resolve_default_model_path;
///
/// // When no model exists the function returns an error
/// // (example shown in docstring only; outcome depends on filesystem state)
/// let result = resolve_default_model_path();
/// println!("resolved: {:?}", result);
/// ```
pub fn resolve_default_model_path() -> Result<PathBuf> {
    let cwd = env::current_dir().context("failed to read current working directory")?;
    resolve_default_model_path_from(&cwd)
}

/// Resolves the default YOLO model path by walking upward from `start_dir`.
///
/// Searches each ancestor directory for `models/candle-yolo-v8/yolov8n.safetensors`
/// and then for a bare `yolov8n.safetensors` in the root.
///
/// # Arguments
///
/// * `start_dir` – Directory from which the upward search begins.
///
/// # Errors
///
/// Returns an error if no matching `.safetensors` file is found in any ancestor
/// directory.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use sesoko::yolo_candle::resolve_default_model_path_from;
///
/// // Will find the model when started from inside a project that has downloaded it
/// let path = resolve_default_model_path_from(Path::new("src/")).unwrap();
/// assert!(path.exists());
/// ```
///
/// ```
/// use std::path::Path;
/// use sesoko::yolo_candle::resolve_default_model_path_from;
///
/// // A directory with no model anywhere in its ancestor chain returns an error
/// let tmp = tempfile::tempdir().unwrap();
/// assert!(resolve_default_model_path_from(tmp.path()).is_err());
/// ```
pub fn resolve_default_model_path_from(start_dir: &Path) -> Result<PathBuf> {
    for search_dir in start_dir.ancestors() {
        for candidate in DEFAULT_SAFETENSORS_CANDIDATES {
            let p = search_dir.join(candidate);
            if p.exists() {
                return Ok(p);
            }
        }
    }
    anyhow::bail!(
        "Could not find YOLO safetensors model. Expected at: models/candle-yolo-v8/yolov8n.safetensors\n\
         Download with:\n\
           hf download lmz/candle-yolo-v8 yolov8n.safetensors --local-dir models/candle-yolo-v8"
    )
}

/// Builds a [`Detector`] from an optional explicit path, falling back to the
/// default model search when `model_path` is `None`.
///
/// # Arguments
///
/// * `model_path` – Optional explicit path to a `.safetensors` weights file.
///   When `None`, the default path is resolved via [`resolve_default_model_path`].
///
/// # Errors
///
/// Returns an error if the weights file cannot be found or loaded.
///
/// # Examples
///
/// ```no_run
/// use sesoko::yolo_candle::build_candle_detector_with_default;
///
/// // Explicit path
/// let d = build_candle_detector_with_default(
///     Some(std::path::Path::new("models/candle-yolo-v8/yolov8n.safetensors"))
/// ).unwrap();
/// ```
///
/// ```no_run
/// use sesoko::yolo_candle::build_candle_detector_with_default;
///
/// // None → resolved automatically from the current working directory
/// let d = build_candle_detector_with_default(None).unwrap();
/// ```
pub fn build_candle_detector_with_default(model_path: Option<&Path>) -> Result<Box<dyn Detector>> {
    match model_path {
        Some(path) => build_candle_detector(path),
        None => {
            let default_path = resolve_default_model_path()
                .context("failed to resolve default YOLO model path")?;
            build_candle_detector(default_path)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn resolves_default_model_from_models_subdir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let nested = temp.path().join("a/b/c");
        fs::create_dir_all(&nested).expect("create nested directories");
        let model_dir = temp.path().join("models/candle-yolo-v8");
        fs::create_dir_all(&model_dir).expect("create model dir");
        let model = model_dir.join("yolov8n.safetensors");
        fs::write(&model, b"stub").expect("write model file");
        let found = resolve_default_model_path_from(&nested).expect("found");
        assert_eq!(found, model);
    }

    #[test]
    fn resolves_default_model_from_root_safetensors() {
        let temp = tempfile::tempdir().expect("tempdir");
        let nested = temp.path().join("a/b/c");
        fs::create_dir_all(&nested).expect("create nested directories");
        let model = temp.path().join("yolov8n.safetensors");
        fs::write(&model, b"stub").expect("write model file");
        let found = resolve_default_model_path_from(&nested).expect("found");
        assert_eq!(found, model);
    }

    #[test]
    fn load_safetensors_missing_file_errors() {
        let result = CandleYoloDetector::load_safetensors("/nonexistent/path/weights.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn build_candle_detector_missing_file_errors() {
        let result = build_candle_detector("/nonexistent/path/weights.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn resolve_default_model_path_from_not_found_errors() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Empty temp dir: no safetensors file anywhere in the tree
        let result = resolve_default_model_path_from(temp.path());
        assert!(result.is_err(), "expected error when model is absent");
    }

    #[test]
    fn build_candle_detector_with_default_propagates_error() {
        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("no_such.safetensors");
        let result = build_candle_detector_with_default(Some(missing.as_path()));
        assert!(result.is_err());
    }
}
