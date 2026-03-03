use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use image::imageops::{crop_imm, resize, FilterType};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::image_utils::{crop_to_square, get_image_files, open_image, save_image_optimized};

/// The 80 COCO object class names, ordered by class index.
///
/// These are the labels used when training the standard YOLOv8 model on the
/// [COCO dataset](https://cocodataset.org/).
///
/// # Examples
///
/// ```
/// use sesoko::crop::DEFAULT_COCO_CLASSES;
///
/// assert_eq!(DEFAULT_COCO_CLASSES.len(), 80);
/// assert_eq!(DEFAULT_COCO_CLASSES[0], "person");
/// ```
pub const DEFAULT_COCO_CLASSES: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Returns all default COCO class names as owned `String`s.
///
/// # Examples
///
/// ```
/// use sesoko::crop::default_class_names;
///
/// let names = default_class_names();
/// assert_eq!(names.len(), 80);
/// assert!(names.contains(&"person".to_string()));
/// ```
///
/// ```
/// use sesoko::crop::{default_class_names, DEFAULT_COCO_CLASSES};
///
/// // The owned strings match the string slices in the constant
/// let names = default_class_names();
/// for (owned, &borrowed) in names.iter().zip(DEFAULT_COCO_CLASSES.iter()) {
///     assert_eq!(owned.as_str(), borrowed);
/// }
/// ```
pub fn default_class_names() -> Vec<String> {
    DEFAULT_COCO_CLASSES.iter().map(|s| s.to_string()).collect()
}

/// Axis-aligned bounding box in pixel coordinates of the source image.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// Left edge of the box (x coordinate).
    pub x1: f32,
    /// Top edge of the box (y coordinate).
    pub y1: f32,
    /// Right edge of the box (x coordinate).
    pub x2: f32,
    /// Bottom edge of the box (y coordinate).
    pub y2: f32,
}

/// A trait for object detectors that can locate a named class in an image.
///
/// Implementations should be `Send + Sync` so they can be used across threads.
pub trait Detector: Send + Sync {
    /// Returns the bounding box of the highest-confidence detection matching
    /// `class_name`, or `None` if no such object was found.
    ///
    /// # Arguments
    ///
    /// * `image` – The image to run detection on.
    /// * `class_name` – Name of the COCO class to search for (e.g. `"person"`).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying inference engine fails.
    fn detect_first_box_for_class(
        &self,
        image: &DynamicImage,
        class_name: &str,
    ) -> Result<Option<BoundingBox>>;

    /// Returns the list of class names this detector was trained on.
    fn class_names(&self) -> Vec<String>;
}

/// Batch processing statistics returned by [`YOLOCropper::process_folder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStats {
    /// Canonical path of the input directory.
    pub base_folder: String,
    /// Canonical path of the output directory.
    pub output_folder: String,
    /// Name of the crop focus class, or `"center"` for plain centre-crop.
    pub crop_focus: String,
    /// Paths of images that were successfully cropped and saved.
    pub processed: Vec<String>,
    /// Paths of images that were skipped (class not found + `skip_if_not_found`).
    pub skipped: Vec<String>,
    /// Paths of images that could not be processed due to an error.
    pub failed: Vec<String>,
}

/// Content-aware image cropper that uses a [`Detector`] to locate a target
/// object and crop around it.
///
/// Without a detector (or when the target class is not found and
/// `skip_if_not_found` is `false`) the cropper falls back to a plain
/// centre-crop.
pub struct YOLOCropper {
    /// Name of the COCO class to focus on, or `None` for centre-crop only.
    pub crop_focus: Option<String>,
    /// Side length of the square output image in pixels.
    pub resolution: u32,
    detector: Option<Box<dyn Detector>>,
    class_names: Vec<String>,
}

impl YOLOCropper {
    /// Creates a new `YOLOCropper` without a detector.
    ///
    /// Without a detector all images are processed with a plain centre-crop.
    /// Attach a detector with [`YOLOCropper::with_detector`].
    ///
    /// # Arguments
    ///
    /// * `crop_focus` – COCO class name to focus on, e.g. `Some("person".into())`.
    ///   Pass `None` to always use the centre-crop fallback.
    /// * `resolution` – Side length of the square output image in pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// use sesoko::crop::YOLOCropper;
    ///
    /// let cropper = YOLOCropper::new(None, 512);
    /// assert_eq!(cropper.resolution, 512);
    /// assert!(cropper.crop_focus.is_none());
    /// ```
    ///
    /// ```
    /// use sesoko::crop::YOLOCropper;
    ///
    /// let cropper = YOLOCropper::new(Some("person".to_string()), 768);
    /// assert_eq!(cropper.crop_focus.as_deref(), Some("person"));
    /// ```
    pub fn new(crop_focus: Option<String>, resolution: u32) -> Self {
        Self {
            crop_focus,
            resolution,
            detector: None,
            class_names: default_class_names(),
        }
    }

    /// Attaches a detector and returns the updated `YOLOCropper`.
    ///
    /// The cropper's class-name list is replaced with the detector's.
    ///
    /// # Arguments
    ///
    /// * `detector` – A boxed [`Detector`] implementation to use for object
    ///   detection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use sesoko::crop::YOLOCropper;
    /// use sesoko::yolo_candle::build_candle_detector_with_default;
    ///
    /// let detector = build_candle_detector_with_default(None).unwrap();
    /// let cropper = YOLOCropper::new(Some("person".to_string()), 512)
    ///     .with_detector(detector);
    /// assert!(!cropper.get_available_classes().is_empty());
    /// ```
    ///
    /// ```no_run
    /// use sesoko::crop::YOLOCropper;
    /// use sesoko::yolo_candle::build_candle_detector_with_default;
    ///
    /// // Detector's class names replace the default list
    /// let detector = build_candle_detector_with_default(None).unwrap();
    /// let classes_before = YOLOCropper::new(None, 512).get_available_classes();
    /// let classes_after = YOLOCropper::new(None, 512)
    ///     .with_detector(detector)
    ///     .get_available_classes();
    /// assert_eq!(classes_before.len(), classes_after.len()); // both 80 for COCO
    /// ```
    pub fn with_detector(mut self, detector: Box<dyn Detector>) -> Self {
        self.class_names = detector.class_names();
        self.detector = Some(detector);
        self
    }

    /// Returns the list of available class names (from the attached detector,
    /// or the default COCO list when no detector is attached).
    ///
    /// # Examples
    ///
    /// ```
    /// use sesoko::crop::YOLOCropper;
    ///
    /// let classes = YOLOCropper::new(None, 512).get_available_classes();
    /// assert!(classes.contains(&"person".to_string()));
    /// ```
    ///
    /// ```
    /// use sesoko::crop::YOLOCropper;
    ///
    /// // Default list has 80 COCO classes
    /// let classes = YOLOCropper::new(None, 512).get_available_classes();
    /// assert_eq!(classes.len(), 80);
    /// ```
    pub fn get_available_classes(&self) -> Vec<String> {
        self.class_names.clone()
    }

    /// Processes a single image and returns the cropped result, or `None` if
    /// the image was skipped.
    ///
    /// Crop strategy:
    /// 1. If a `crop_focus` and detector are set, tries a content-aware crop
    ///    centred on the detected object (1.2× bounding box, square, then
    ///    padded to `resolution`).
    /// 2. Falls back to a plain centre-crop when no object is found and
    ///    `skip_if_not_found` is `false`.
    /// 3. Returns `None` when `skip_if_not_found` is `true` and no object is
    ///    found.
    ///
    /// # Arguments
    ///
    /// * `image` – Source image to crop.
    /// * `skip_if_not_found` – When `true`, returns `None` if the target class
    ///   was not detected; when `false`, falls back to centre-crop.
    ///
    /// # Examples
    ///
    /// ```
    /// use image::{DynamicImage, GenericImageView, RgbImage};
    /// use sesoko::crop::YOLOCropper;
    ///
    /// // Without a detector the image is always centre-cropped
    /// let cropper = YOLOCropper::new(None, 256);
    /// let img = DynamicImage::ImageRgb8(RgbImage::new(800, 600));
    /// let result = cropper.process_image(&img, false).unwrap();
    /// assert_eq!(result.dimensions(), (256, 256));
    /// ```
    ///
    /// ```
    /// use image::{DynamicImage, RgbImage};
    /// use sesoko::crop::YOLOCropper;
    ///
    /// // With skip_if_not_found=true and no detector, returns None for focus classes
    /// let cropper = YOLOCropper::new(Some("person".to_string()), 512);
    /// let img = DynamicImage::ImageRgb8(RgbImage::new(800, 600));
    /// assert!(cropper.process_image(&img, true).is_none());
    /// ```
    pub fn process_image(
        &self,
        image: &DynamicImage,
        skip_if_not_found: bool,
    ) -> Option<DynamicImage> {
        let mut output = if self.crop_focus.is_some() {
            match self.content_aware_crop(image) {
                Ok(Some(cropped)) => cropped,
                Ok(None) => {
                    if skip_if_not_found {
                        return None;
                    }
                    self.center_crop(image)
                }
                Err(_) => {
                    if skip_if_not_found {
                        return None;
                    }
                    self.center_crop(image)
                }
            }
        } else {
            self.center_crop(image)
        };

        output = DynamicImage::ImageRgba8(resize(
            &output,
            self.resolution,
            self.resolution,
            FilterType::Lanczos3,
        ))
        .to_rgb8()
        .into();

        Some(output)
    }

    fn content_aware_crop(&self, image: &DynamicImage) -> Result<Option<DynamicImage>> {
        let crop_focus = self
            .crop_focus
            .as_ref()
            .ok_or_else(|| anyhow!("crop focus is not set"))?;
        let detector = match &self.detector {
            Some(detector) => detector,
            None => return Ok(None),
        };

        let target_box = detector.detect_first_box_for_class(image, crop_focus)?;
        let target_box = match target_box {
            Some(target_box) => target_box,
            None => return Ok(None),
        };

        let width = image.width() as f32;
        let height = image.height() as f32;
        let w = target_box.x2 - target_box.x1;
        let h = target_box.y2 - target_box.y1;
        let cx = target_box.x1 + w / 2.0;
        let cy = target_box.y1 + h / 2.0;

        let size = (w.max(h) * 1.2).max(1.0);
        let half = size / 2.0;

        let x1 = (cx - half).max(0.0);
        let y1 = (cy - half).max(0.0);
        let x2 = (cx + half).min(width);
        let y2 = (cy + half).min(height);

        let crop_w = (x2 - x1).max(1.0) as u32;
        let crop_h = (y2 - y1).max(1.0) as u32;
        let cropped = crop_imm(image, x1 as u32, y1 as u32, crop_w, crop_h).to_image();
        let cropped = DynamicImage::ImageRgba8(cropped);

        Ok(Some(crop_to_square(&cropped)))
    }

    fn center_crop(&self, image: &DynamicImage) -> DynamicImage {
        crop_to_square(image)
    }

    /// Processes all images in `input_dir` and writes the results to
    /// `output_dir`.
    ///
    /// Output filenames are derived from the input stem plus a `.jpg` extension,
    /// with a numeric suffix appended to avoid collisions.
    ///
    /// # Arguments
    ///
    /// * `input_dir` – Directory containing source images (non-recursive scan).
    /// * `output_dir` – Directory for output JPEG files (created if absent).
    /// * `skip_if_not_found` – Passed directly to [`YOLOCropper::process_image`].
    ///
    /// # Errors
    ///
    /// Returns an error if `input_dir` cannot be read or cannot be
    /// canonicalised.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use sesoko::crop::YOLOCropper;
    ///
    /// let cropper = YOLOCropper::new(None, 512);
    /// let stats = cropper
    ///     .process_folder(Path::new("photos/"), Path::new("output/"), false)
    ///     .unwrap();
    /// println!("Processed: {}", stats.processed.len());
    /// ```
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use sesoko::crop::YOLOCropper;
    ///
    /// // skip_if_not_found=true: images where the class is absent are skipped
    /// let cropper = YOLOCropper::new(Some("person".to_string()), 512);
    /// let stats = cropper
    ///     .process_folder(Path::new("photos/"), Path::new("output/"), true)
    ///     .unwrap();
    /// println!("Skipped: {}", stats.skipped.len());
    /// ```
    pub fn process_folder(
        &self,
        input_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        skip_if_not_found: bool,
    ) -> Result<ProcessStats> {
        let input_path: PathBuf = input_dir.as_ref().canonicalize()?;
        let output_path: PathBuf = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_path)?;

        let mut stats = ProcessStats {
            base_folder: input_path.display().to_string(),
            output_folder: output_path
                .canonicalize()
                .unwrap_or(output_path.clone())
                .display()
                .to_string(),
            crop_focus: self
                .crop_focus
                .clone()
                .unwrap_or_else(|| "center".to_string()),
            processed: Vec::new(),
            skipped: Vec::new(),
            failed: Vec::new(),
        };

        let image_files = get_image_files(&input_path, false)?;

        for file_path in image_files {
            let process_result = (|| -> Result<()> {
                let image = open_image(&file_path)?;
                let processed = self.process_image(&image, skip_if_not_found);

                if let Some(processed) = processed {
                    let stem = file_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    let output_file = unique_output_path(&output_path, &stem);
                    save_image_optimized(&processed, &output_file)?;
                    stats.processed.push(file_path.display().to_string());
                } else {
                    stats.skipped.push(file_path.display().to_string());
                }

                Ok(())
            })();

            if process_result.is_err() {
                stats.failed.push(file_path.display().to_string());
            }
        }

        Ok(stats)
    }
}

/// Return `<dir>/<stem>.jpg`, or `<dir>/<stem>_1.jpg`, `<dir>/<stem>_2.jpg`, …
/// until a path that does not yet exist is found.
fn unique_output_path(dir: &Path, stem: &str) -> PathBuf {
    let candidate = dir.join(format!("{stem}.jpg"));
    if !candidate.exists() {
        return candidate;
    }
    let mut counter = 1u32;
    loop {
        let candidate = dir.join(format!("{stem}_{counter}.jpg"));
        if !candidate.exists() {
            return candidate;
        }
        counter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    struct FakeDetector;

    impl Detector for FakeDetector {
        fn detect_first_box_for_class(
            &self,
            _image: &DynamicImage,
            class_name: &str,
        ) -> Result<Option<BoundingBox>> {
            if class_name == "person" {
                Ok(Some(BoundingBox {
                    x1: 100.0,
                    y1: 100.0,
                    x2: 300.0,
                    y2: 280.0,
                }))
            } else {
                Ok(None)
            }
        }

        fn class_names(&self) -> Vec<String> {
            vec!["person".to_string(), "car".to_string()]
        }
    }

    /// Tests the content_aware_crop code path: detector returns None for an unknown class,
    /// so the cropper falls back to center crop instead of skipping.
    #[test]
    fn process_image_falls_back_without_match() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(800, 600));
        let cropper = YOLOCropper::new(Some("missing".to_string()), 512)
            .with_detector(Box::new(FakeDetector));
        let result = cropper.process_image(&image, false).unwrap();
        assert_eq!(result.dimensions(), (512, 512));
    }

    /// Tests the content_aware_crop code path: detector returns None and skip=true
    /// so process_image returns None.
    #[test]
    fn process_image_skips_if_required() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(800, 600));
        let cropper = YOLOCropper::new(Some("missing".to_string()), 512)
            .with_detector(Box::new(FakeDetector));
        let result = cropper.process_image(&image, true);
        assert!(result.is_none());
    }

    #[test]
    fn unique_output_path_no_collision() {
        let dir = tempfile::tempdir().unwrap();
        let p = unique_output_path(dir.path(), "photo");
        assert_eq!(p, dir.path().join("photo.jpg"));
    }

    #[test]
    fn unique_output_path_single_collision() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("photo.jpg"), b"").unwrap();
        let p = unique_output_path(dir.path(), "photo");
        assert_eq!(p, dir.path().join("photo_1.jpg"));
    }

    #[test]
    fn unique_output_path_multiple_collisions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("photo.jpg"), b"").unwrap();
        std::fs::write(dir.path().join("photo_1.jpg"), b"").unwrap();
        std::fs::write(dir.path().join("photo_2.jpg"), b"").unwrap();
        let p = unique_output_path(dir.path(), "photo");
        assert_eq!(p, dir.path().join("photo_3.jpg"));
    }

    #[test]
    fn bounding_box_fields_accessible() {
        let bb = BoundingBox {
            x1: 10.0,
            y1: 20.0,
            x2: 110.0,
            y2: 120.0,
        };
        assert_eq!(bb.x1, 10.0);
        assert_eq!(bb.y1, 20.0);
        assert_eq!(bb.x2, 110.0);
        assert_eq!(bb.y2, 120.0);
    }

    #[test]
    fn default_class_names_matches_const() {
        let names = default_class_names();
        assert_eq!(names.len(), DEFAULT_COCO_CLASSES.len());
        for (owned, &borrowed) in names.iter().zip(DEFAULT_COCO_CLASSES.iter()) {
            assert_eq!(owned.as_str(), borrowed);
        }
    }

    #[test]
    fn with_detector_updates_class_names() {
        let cropper =
            YOLOCropper::new(Some("person".to_string()), 512).with_detector(Box::new(FakeDetector));
        let classes = cropper.get_available_classes();
        assert_eq!(classes, vec!["person".to_string(), "car".to_string()]);
    }

    #[test]
    fn process_image_content_aware_with_match() {
        // FakeDetector returns a box for "person"; image should be cropped to 512×512
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(800, 600));
        let cropper =
            YOLOCropper::new(Some("person".to_string()), 512).with_detector(Box::new(FakeDetector));
        let result = cropper.process_image(&image, false).unwrap();
        assert_eq!(result.dimensions(), (512, 512));
    }

    #[test]
    fn process_stats_serialization() {
        let stats = ProcessStats {
            base_folder: "/input".to_string(),
            output_folder: "/output".to_string(),
            crop_focus: "person".to_string(),
            processed: vec!["a.jpg".to_string()],
            skipped: vec![],
            failed: vec![],
        };
        let json = serde_json::to_string(&stats).unwrap();
        let roundtripped: ProcessStats = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.base_folder, "/input");
        assert_eq!(roundtripped.processed, vec!["a.jpg".to_string()]);
    }
}
