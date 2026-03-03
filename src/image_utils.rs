use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{crop_imm, resize, FilterType};
use image::{DynamicImage, GenericImageView, ImageFormat};
use walkdir::WalkDir;

/// Lowercase, dot-prefixed file extensions recognised as images by this crate.
///
/// # Examples
///
/// ```
/// use sesoko::image_utils::SUPPORTED_IMAGE_EXTENSIONS;
///
/// assert!(SUPPORTED_IMAGE_EXTENSIONS.contains(&".jpg"));
/// assert!(!SUPPORTED_IMAGE_EXTENSIONS.contains(&"jpg")); // missing leading dot
/// ```
pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".avif",
    ".jxl", ".psd",
];

/// Returns `true` if the file extension of `path` matches any entry in
/// [`SUPPORTED_IMAGE_EXTENSIONS`] (comparison is case-insensitive).
///
/// # Arguments
///
/// * `path` – The file path whose extension is examined.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use sesoko::image_utils::is_supported_image_extension;
///
/// assert!(is_supported_image_extension(Path::new("photo.JPG")));
/// assert!(!is_supported_image_extension(Path::new("notes.txt")));
/// ```
///
/// ```
/// use std::path::Path;
/// use sesoko::image_utils::is_supported_image_extension;
///
/// // Paths without an extension return false
/// assert!(!is_supported_image_extension(Path::new("README")));
/// ```
pub fn is_supported_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let normalized = format!(".{}", ext.to_ascii_lowercase());
            SUPPORTED_IMAGE_EXTENSIONS.contains(&normalized.as_str())
        })
        .unwrap_or(false)
}

/// Collects all image files inside `folder_path`, sorted alphabetically by path.
///
/// # Arguments
///
/// * `folder_path` – Directory to search.
/// * `recursive` – When `true`, descends into sub-directories via [`walkdir`].
///
/// # Returns
///
/// A sorted `Vec<PathBuf>` containing every file whose extension matches
/// [`SUPPORTED_IMAGE_EXTENSIONS`].
///
/// # Errors
///
/// Returns an error when `folder_path` does not exist or cannot be read
/// (non-recursive mode only; `walkdir` silently skips unreadable entries).
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use sesoko::image_utils::get_image_files;
///
/// // Non-recursive: only immediate children are returned.
/// let files = get_image_files(Path::new("images/"), false).unwrap();
/// println!("Found {} image(s)", files.len());
/// ```
///
/// ```no_run
/// use std::path::Path;
/// use sesoko::image_utils::get_image_files;
///
/// // Recursive: descends into all sub-directories.
/// let files = get_image_files(Path::new("images/"), true).unwrap();
/// println!("Found {} image(s) recursively", files.len());
/// ```
pub fn get_image_files(folder_path: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut image_files = Vec::new();

    if recursive {
        for entry in WalkDir::new(folder_path).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            if path.is_file() && is_supported_image_extension(path) {
                image_files.push(path.to_path_buf());
            }
        }
    } else {
        for entry in fs::read_dir(folder_path)
            .with_context(|| format!("read_dir failed for {}", folder_path.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && is_supported_image_extension(&path) {
                image_files.push(path);
            }
        }
    }

    image_files.sort();
    Ok(image_files)
}

/// Opens an image from `image_path` and converts it to the RGB8 colour space.
///
/// Any colour space or alpha channel present in the source file is discarded;
/// the returned image always has exactly three channels.
///
/// # Arguments
///
/// * `image_path` – Path to the image file to open.
///
/// # Returns
///
/// A [`DynamicImage`] in `ImageRgb8` colour space.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or is not a recognised
/// image format.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use sesoko::image_utils::open_image;
///
/// let img = open_image(Path::new("photo.jpg")).unwrap();
/// assert_eq!(img.color().channel_count(), 3); // always RGB8
/// ```
///
/// ```no_run
/// use std::path::Path;
/// use sesoko::image_utils::open_image;
///
/// // RGBA images are automatically down-converted to RGB8
/// let img = open_image(Path::new("transparent.png")).unwrap();
/// assert_eq!(img.color().channel_count(), 3);
/// ```
pub fn open_image(image_path: &Path) -> Result<DynamicImage> {
    let image = image::open(image_path)
        .with_context(|| format!("failed to open {}", image_path.display()))?;
    Ok(DynamicImage::ImageRgb8(image.to_rgb8()))
}

/// Resizes `image` so that its longest side equals `target_size`, preserving
/// the original aspect ratio.
///
/// Uses Lanczos3 resampling. The shorter dimension may be slightly less than
/// `target_size` due to rounding. The output is always in RGB8 colour space.
///
/// # Arguments
///
/// * `image` – Source image.
/// * `target_size` – Pixel length for the longest side of the output.
///
/// # Examples
///
/// ```
/// use image::{DynamicImage, RgbImage};
/// use sesoko::image_utils::resize_image_aspect_ratio;
///
/// // Landscape: width becomes target_size
/// let img = DynamicImage::ImageRgb8(RgbImage::new(1000, 500));
/// let resized = resize_image_aspect_ratio(&img, 400);
/// assert_eq!(resized.width(), 400);
/// assert_eq!(resized.height(), 200);
/// ```
///
/// ```
/// use image::{DynamicImage, RgbImage};
/// use sesoko::image_utils::resize_image_aspect_ratio;
///
/// // Portrait: height becomes target_size
/// let img = DynamicImage::ImageRgb8(RgbImage::new(400, 800));
/// let resized = resize_image_aspect_ratio(&img, 200);
/// assert_eq!(resized.height(), 200);
/// assert_eq!(resized.width(), 100);
/// ```
pub fn resize_image_aspect_ratio(image: &DynamicImage, target_size: u32) -> DynamicImage {
    let (width, height) = image.dimensions();
    let aspect_ratio = width as f32 / height as f32;

    let (new_width, new_height) = if aspect_ratio > 1.0 {
        (target_size, (target_size as f32 / aspect_ratio) as u32)
    } else {
        ((target_size as f32 * aspect_ratio) as u32, target_size)
    };

    let resized = resize(
        image,
        new_width.max(1),
        new_height.max(1),
        FilterType::Lanczos3,
    );
    DynamicImage::ImageRgba8(resized).to_rgb8().into()
}

/// Centre-crops `image` to a square whose side equals the shorter dimension.
///
/// If the image is already square it is returned as-is (converted to RGB8).
/// Equal amounts are trimmed from each end of the longer axis.
///
/// # Arguments
///
/// * `image` – Source image to crop.
///
/// # Examples
///
/// ```
/// use image::{DynamicImage, RgbImage};
/// use sesoko::image_utils::crop_to_square;
///
/// let img = DynamicImage::ImageRgb8(RgbImage::new(800, 400));
/// let sq = crop_to_square(&img);
/// assert_eq!(sq.width(), 400);
/// assert_eq!(sq.height(), 400);
/// ```
///
/// ```
/// use image::{DynamicImage, GenericImageView, RgbImage};
/// use sesoko::image_utils::crop_to_square;
///
/// // Already-square images are returned unchanged
/// let img = DynamicImage::ImageRgb8(RgbImage::new(300, 300));
/// let sq = crop_to_square(&img);
/// assert_eq!(sq.dimensions(), (300, 300));
/// ```
pub fn crop_to_square(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    if width == height {
        return DynamicImage::ImageRgb8(image.to_rgb8());
    }

    let new_size = width.min(height);
    let left = (width - new_size) / 2;
    let top = (height - new_size) / 2;
    let cropped = crop_imm(image, left, top, new_size, new_size).to_image();
    DynamicImage::ImageRgba8(cropped).to_rgb8().into()
}

/// Saves `image` as a JPEG at `output_path` with quality 85.
///
/// Parent directories of `output_path` are created automatically when they
/// do not already exist.
///
/// # Arguments
///
/// * `image` – Image to encode and save.
/// * `output_path` – Destination file path (`.jpg` / `.jpeg` recommended).
///
/// # Errors
///
/// Returns an error if the parent directory cannot be created, the destination
/// file cannot be opened for writing, or JPEG encoding fails.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use image::{DynamicImage, RgbImage};
/// use sesoko::image_utils::save_image_optimized;
///
/// let img = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
/// save_image_optimized(&img, Path::new("/tmp/out.jpg")).unwrap();
/// ```
///
/// ```no_run
/// use std::path::Path;
/// use image::{DynamicImage, RgbImage};
/// use sesoko::image_utils::save_image_optimized;
///
/// // Nested parent directories are created on the fly
/// let img = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
/// save_image_optimized(&img, Path::new("/tmp/a/b/c/out.jpg")).unwrap();
/// ```
pub fn save_image_optimized(image: &DynamicImage, output_path: &Path) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let file = fs::File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut encoder = JpegEncoder::new_with_quality(file, 85);
    encoder
        .encode_image(image)
        .with_context(|| format!("failed to encode JPEG for {}", output_path.display()))?;

    Ok(())
}

/// Infers an [`ImageFormat`] from the file extension of `path`.
///
/// The look-up is delegated to [`ImageFormat::from_extension`] and is therefore
/// case-insensitive on most platforms.
///
/// Returns `None` when the path has no extension or the extension is not
/// recognised by the `image` crate.
///
/// # Arguments
///
/// * `path` – Path whose extension is mapped to a format.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use image::ImageFormat;
/// use sesoko::image_utils::image_format_from_path;
///
/// assert_eq!(image_format_from_path(Path::new("photo.jpg")), Some(ImageFormat::Jpeg));
/// assert_eq!(image_format_from_path(Path::new("image.png")), Some(ImageFormat::Png));
/// ```
///
/// ```
/// use std::path::Path;
/// use sesoko::image_utils::image_format_from_path;
///
/// assert_eq!(image_format_from_path(Path::new("report.pdf")), None);
/// assert_eq!(image_format_from_path(Path::new("noextension")), None);
/// ```
pub fn image_format_from_path(path: &Path) -> Option<ImageFormat> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(ImageFormat::from_extension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn is_supported_extension_returns_true_for_jpg() {
        assert!(is_supported_image_extension(std::path::Path::new(
            "photo.jpg"
        )));
    }

    #[test]
    fn is_supported_extension_returns_false_for_txt() {
        assert!(!is_supported_image_extension(std::path::Path::new(
            "docs.txt"
        )));
    }

    #[test]
    fn is_supported_extension_no_extension() {
        assert!(!is_supported_image_extension(std::path::Path::new(
            "README"
        )));
    }

    #[test]
    fn is_supported_extension_case_insensitive() {
        assert!(is_supported_image_extension(std::path::Path::new(
            "photo.JPG"
        )));
        assert!(is_supported_image_extension(std::path::Path::new(
            "scan.PNG"
        )));
    }

    #[test]
    fn image_format_from_path_returns_jpeg() {
        use image::ImageFormat;
        assert_eq!(
            image_format_from_path(std::path::Path::new("test.jpg")),
            Some(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn image_format_from_path_returns_none_for_unknown() {
        assert!(image_format_from_path(std::path::Path::new("file.xyz")).is_none());
    }

    #[test]
    fn get_image_files_nonexistent_dir_errors() {
        let result = get_image_files(std::path::Path::new("/nonexistent/dir/xyz_sesoko"), false);
        assert!(result.is_err());
    }

    #[test]
    fn crop_to_square_already_square() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(300, 300));
        let sq = crop_to_square(&img);
        assert_eq!(sq.dimensions(), (300, 300));
    }
}
