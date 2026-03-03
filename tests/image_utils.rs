use std::fs;

use image::{DynamicImage, GenericImageView, RgbImage, RgbaImage};
use sesoko::image_utils::{
    crop_to_square, get_image_files, image_format_from_path, is_supported_image_extension,
    open_image, resize_image_aspect_ratio, save_image_optimized,
};
use tempfile::tempdir;

#[test]
fn get_image_files_recursive_and_non_recursive() {
    let dir = tempdir().unwrap();
    let root = dir.path();

    DynamicImage::ImageRgb8(RgbImage::new(800, 400))
        .save(root.join("wide.jpg"))
        .unwrap();
    DynamicImage::ImageRgb8(RgbImage::new(400, 800))
        .save(root.join("tall.png"))
        .unwrap();

    let subdir = root.join("subdir");
    fs::create_dir_all(&subdir).unwrap();
    DynamicImage::ImageRgb8(RgbImage::new(300, 300))
        .save(subdir.join("sub_image.png"))
        .unwrap();

    let recursive = get_image_files(root, true).unwrap();
    let non_recursive = get_image_files(root, false).unwrap();

    assert_eq!(recursive.len(), 3);
    assert_eq!(non_recursive.len(), 2);
}

#[test]
fn open_image_converts_to_rgb() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rgba.png");
    DynamicImage::ImageRgba8(RgbaImage::new(200, 200))
        .save(&path)
        .unwrap();

    let image = open_image(&path).unwrap();
    assert_eq!(image.color().channel_count(), 3);
    assert_eq!(image.dimensions(), (200, 200));
}

#[test]
fn resize_preserves_aspect_ratio() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(1000, 500));
    let resized = resize_image_aspect_ratio(&image, 400);
    assert_eq!(resized.dimensions(), (400, 200));
}

#[test]
fn crop_to_square_centered() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(800, 400));
    let cropped = crop_to_square(&image);
    assert_eq!(cropped.dimensions(), (400, 400));
}

#[test]
fn save_image_as_jpeg_and_create_dirs() {
    let dir = tempdir().unwrap();
    let out = dir.path().join("a/b/out.jpg");
    let image = DynamicImage::ImageRgb8(RgbImage::new(64, 64));

    save_image_optimized(&image, &out).unwrap();

    assert!(out.exists());
    let loaded = image::open(out).unwrap();
    assert_eq!(loaded.dimensions(), (64, 64));
}

#[test]
fn is_supported_image_extension_with_real_paths() {
    let dir = tempdir().unwrap();
    let jpg = dir.path().join("file.jpg");
    let txt = dir.path().join("file.txt");
    assert!(is_supported_image_extension(&jpg));
    assert!(!is_supported_image_extension(&txt));
}

#[test]
fn image_format_from_path_roundtrip_extensions() {
    use image::ImageFormat;
    assert_eq!(
        image_format_from_path(std::path::Path::new("img.PNG")),
        Some(ImageFormat::Png)
    );
    assert_eq!(
        image_format_from_path(std::path::Path::new("img.webp")),
        Some(ImageFormat::WebP)
    );
    assert!(image_format_from_path(std::path::Path::new("data.csv")).is_none());
}
