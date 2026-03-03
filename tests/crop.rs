use image::{DynamicImage, GenericImageView, RgbImage};
use sesoko::crop::{default_class_names, YOLOCropper, DEFAULT_COCO_CLASSES};
use tempfile::tempdir;

#[test]
fn init_default_parameters() {
    let cropper = YOLOCropper::new(None, 512);
    assert!(cropper.crop_focus.is_none());
    assert_eq!(cropper.resolution, 512);
}

#[test]
fn init_custom_parameters() {
    let cropper = YOLOCropper::new(Some("person".to_string()), 768);
    assert_eq!(cropper.crop_focus.as_deref(), Some("person"));
    assert_eq!(cropper.resolution, 768);
}

#[test]
fn available_classes_non_empty_and_common() {
    let cropper = YOLOCropper::new(None, 512);
    let classes = cropper.get_available_classes();
    assert!(!classes.is_empty());
    assert!(classes.iter().any(|c| c == "person"));
}

#[test]
fn process_image_returns_square_resolution() {
    let cropper = YOLOCropper::new(None, 256);
    let image = DynamicImage::ImageRgb8(RgbImage::new(1000, 500));

    let processed = cropper
        .process_image(&image, false)
        .expect("processed image");
    assert_eq!(processed.dimensions(), (256, 256));
}

#[test]
fn process_image_skip_if_not_found_behaviour() {
    let cropper = YOLOCropper::new(Some("nonexistent_object".to_string()), 512);
    let image = DynamicImage::ImageRgb8(RgbImage::new(800, 600));

    let skipped = cropper.process_image(&image, true);
    assert!(skipped.is_none());

    let fallback = cropper.process_image(&image, false);
    assert!(fallback.is_some());
    assert_eq!(fallback.unwrap().dimensions(), (512, 512));
}

#[test]
fn process_folder_stats_structure_and_output_dir() {
    let input_dir = tempdir().unwrap();
    let output_dir = tempdir().unwrap();
    let output_path = output_dir.path().join("crop-output");

    DynamicImage::ImageRgb8(RgbImage::new(800, 600))
        .save(input_dir.path().join("test.jpg"))
        .unwrap();

    let cropper = YOLOCropper::new(None, 512);
    let stats = cropper
        .process_folder(input_dir.path(), &output_path, true)
        .expect("process folder");

    assert!(output_path.exists());
    assert!(!stats.base_folder.is_empty());
    assert!(!stats.output_folder.is_empty());
    assert!(!stats.crop_focus.is_empty());

    let total = stats.processed.len() + stats.skipped.len() + stats.failed.len();
    assert_eq!(total, 1);
}

#[test]
fn default_class_names_count_matches_const() {
    let names = default_class_names();
    assert_eq!(names.len(), DEFAULT_COCO_CLASSES.len());
    assert_eq!(names.len(), 80);
}

#[test]
fn process_image_result_is_square_at_given_resolution() {
    let cropper = YOLOCropper::new(None, 128);
    let img = DynamicImage::ImageRgb8(RgbImage::new(1920, 1080));
    let result = cropper.process_image(&img, false).unwrap();
    assert_eq!(result.dimensions(), (128, 128));
}
