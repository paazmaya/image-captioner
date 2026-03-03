mod common;

use image::{GenericImageView, Rgba, RgbaImage};
use sesoko::crop::YOLOCropper;
use sesoko::yolo_candle::{resolve_default_model_path_from, CandleYoloDetector};

/// Draw a filled-border rectangle onto an RGBA image.
fn draw_rect(
    img: &mut RgbaImage,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    color: Rgba<u8>,
    thickness: u32,
) {
    let (w, h) = img.dimensions();
    let x2 = x2.min(w.saturating_sub(1));
    let y2 = y2.min(h.saturating_sub(1));
    let x1 = x1.min(x2);
    let y1 = y1.min(y2);

    for t in 0..thickness {
        // top edge
        let row = y1 + t;
        if row <= y2 {
            for x in x1..=x2 {
                img.put_pixel(x, row, color);
            }
        }
        // bottom edge
        let row = y2.saturating_sub(t);
        if row >= y1 {
            for x in x1..=x2 {
                img.put_pixel(x, row, color);
            }
        }
        // left edge
        let col = x1 + t;
        if col <= x2 {
            for y in y1..=y2 {
                img.put_pixel(col, y, color);
            }
        }
        // right edge
        let col = x2.saturating_sub(t);
        if col >= x1 {
            for y in y1..=y2 {
                img.put_pixel(col, y, color);
            }
        }
    }
}

#[test]
fn detect_person_and_save_annotated_image() {
    let root = common::workspace_root();

    let model_path = resolve_default_model_path_from(&root).expect(
        "YOLO safetensors model not found.\n\
         Download with:\n  \
         hf download lmz/candle-yolo-v8 yolov8n.safetensors --local-dir models/candle-yolo-v8",
    );

    let fixture = common::fixture_dir().join("woman.webp");
    assert!(
        fixture.is_file(),
        "fixture image missing: {}",
        fixture.display()
    );

    let img = image::open(&fixture).expect("open fixture image");
    let (orig_w, orig_h) = img.dimensions();

    let detector = CandleYoloDetector::load(&model_path).expect("load raw detector");
    let detections = detector.detect(&img, 0.25, 0.45).expect("run detection");

    // COCO class 0 = "person"
    const PERSON_CLASS: usize = 0;
    let person_boxes: Vec<_> = detections
        .iter()
        .filter(|(cls, _)| *cls == PERSON_CLASS)
        .collect();

    assert!(
        !person_boxes.is_empty(),
        "expected at least one person detection in {}",
        fixture.display()
    );

    // Annotate image with green rectangles around detected persons.
    let mut annotated = img.to_rgba8();
    let green = Rgba([0u8, 255, 0, 255]);
    let thickness = 3u32;

    for (_, bbox) in &person_boxes {
        let x1 = bbox.xmin.max(0.) as u32;
        let y1 = bbox.ymin.max(0.) as u32;
        let x2 = (bbox.xmax as u32).min(orig_w.saturating_sub(1));
        let y2 = (bbox.ymax as u32).min(orig_h.saturating_sub(1));
        draw_rect(&mut annotated, x1, y1, x2, y2, green, thickness);
    }

    let output_path = common::fixture_dir().join("yolo_person_detected.png");
    annotated.save(&output_path).expect("save annotated PNG");

    println!(
        "detect_person_and_save_annotated_image: {} person(s) detected → {}",
        person_boxes.len(),
        output_path.display()
    );
}

#[test]
fn crop_person_centered_square_and_save() {
    let root = common::workspace_root();

    let model_path = resolve_default_model_path_from(&root).expect(
        "YOLO safetensors model not found.\n\
         Download with:\n  \
         hf download lmz/candle-yolo-v8 yolov8n.safetensors --local-dir models/candle-yolo-v8",
    );

    let fixture = common::fixture_dir().join("woman.webp");
    assert!(
        fixture.is_file(),
        "fixture image missing: {}",
        fixture.display()
    );

    let img = image::open(&fixture).expect("open fixture image");

    const RESOLUTION: u32 = 512;
    let detector = CandleYoloDetector::load(&model_path).expect("load YOLO detector");
    let cropper =
        YOLOCropper::new(Some("person".to_string()), RESOLUTION).with_detector(Box::new(detector));

    let cropped = cropper
        .process_image(&img, false)
        .expect("process_image returned None unexpectedly");

    let (w, h) = cropped.dimensions();
    assert_eq!(w, RESOLUTION, "cropped width should be {RESOLUTION}");
    assert_eq!(h, RESOLUTION, "cropped height should be {RESOLUTION}");

    let output_path = common::fixture_dir().join("yolo_person_crop_square.png");
    cropped.save(&output_path).expect("save cropped PNG");

    println!(
        "crop_person_centered_square_and_save: saved {RESOLUTION}×{RESOLUTION} crop → {}",
        output_path.display()
    );
}
