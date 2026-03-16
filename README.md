# sesoko (瀬底)

> Prepare images for training machine learning models

Two core features:

- **Image captioning** using native candle inference (Qwen3-VL-2B-Instruct, CPU)
- **Content-aware cropping** to square JPEG using YOLO v8 object detection (candle backend, CPU)

## Prerequisites

### Caption model (required for captioning)

https://huggingface.co/docs/huggingface_hub/guides/cli

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct
```

Expected path: `models/Qwen3-VL-2B-Instruct/` relative to the current working directory.
Override with `--model-path` on the `caption` subcommand.

### YOLO model (required only for content-aware cropping)

The default workflow and `sesoko crop` without `--crop-focus` use a pure center crop
and **do not need YOLO**. Only needed when you pass `--crop-focus <class>`:

```bash
hf download lmz/candle-yolo-v8 yolov8n.safetensors --local-dir models/candle-yolo-v8
```

Expected path: `models/candle-yolo-v8/yolov8n.safetensors` relative to cwd.
Override with `--model-path` on the `crop` subcommand.

## CLI

### Install

```bash
cargo install sesoko
```

### Default workflow — crop + caption in one step

Center-crops every image to a 512×512 JPEG, then writes a caption `.txt` sidecar next to
each output image and a summary `captions.toml`. No YOLO required.

```bash
sesoko ./raw_images ./output
```

**File discovery:** top-level images only (not recursive).  
**Output filenames:** `{stem}.jpg` — the original extension is replaced with `.jpg`.
If two input files share the same stem (e.g. `photo.png` and `photo.webp`), a counter
is appended to avoid overwriting: the second becomes `photo_1.jpg`, the third
`photo_2.jpg`, and so on.  
**Caption sidecars:** written as `{original_filename}.txt` (e.g. `photo.jpg` →
`photo.jpg.txt`), so caption files are always distinct even when stems collide.

### Crop images to squares

**File discovery:** top-level images only (not recursive).  
**Output filenames:** `{stem}.jpg` — the original extension is always replaced with `.jpg`.
If two input files share the same stem (e.g. `photo.png` and `photo.jpg`), a counter
is appended to the second and subsequent files: `photo_1.jpg`, `photo_2.jpg`, etc.  
**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`,
`.tif`, `.heic`, `.heif`, `.avif`, `.jxl`, `.psd`, `.ico`.

Center-crop (no YOLO, no `--crop-focus`):

```bash
sesoko crop --input-dir ./raw_images --output-dir ./cropped
```

Content-aware crop focused on the detected object class (requires YOLO weights):

```bash
sesoko crop \
    --input-dir  ./raw_images \
    --output-dir ./cropped \
    --crop-focus person \
    --resolution 512
```

Save processing statistics to a JSON file:

```bash
sesoko crop \
    --input-dir ./raw_images \
    --output-dir ./cropped \
    --crop-focus person \
    --stats stats.json
```

List all YOLO-detectable object classes:

```bash
sesoko crop --list-classes
```

Use a custom YOLO weights file:

```bash
sesoko crop \
    --input-dir ./raw_images \
    --output-dir ./cropped \
    --model-path /path/to/yolov8n.safetensors \
    --crop-focus person
```

### Caption images

**File discovery:** recursive — all images in the folder and any subfolders.  
**Sidecar filenames:** `{original_filename}.txt` (e.g. `photo.jpg` → `photo.jpg.txt`),
placed next to the source image. Files with different extensions but the same stem
(`photo.png` and `photo.jpg`) each get their own distinct sidecar.  
**TOML output:** keyed by absolute input folder path, then by path relative to that
folder. Running the command again on the same folder updates existing entries.

Generate captions for all images in a folder (recursive), writing a TOML summary and
per-image sidecar `.txt` files next to each image:

```bash
sesoko caption ./images
```

Write captions to a custom TOML output file:

```bash
sesoko caption ./images --output my_captions.toml
```

Disable sidecar `.txt` files (TOML only):

```bash
sesoko caption ./images --no-sidecar
```

Write sidecar files to a separate directory (mirror of input structure):

```bash
sesoko caption ./images --sidecar-dir ./sidecars
```

Use a custom model directory:

```bash
sesoko caption ./images --model-path /path/to/Qwen3-VL-2B-Instruct
```

## Library

Add to your project:

```bash
cargo add sesoko
```

### Content-aware cropping

```rust
use sesoko::crop::YOLOCropper;
use sesoko::yolo_candle::build_candle_detector_with_default;
use sesoko::image_utils::open_image;

fn main() -> anyhow::Result<()> {
    // Load YOLO detector (looks for models/candle-yolo-v8/yolov8n.safetensors)
    let detector = build_candle_detector_with_default(None)?;

    let cropper = YOLOCropper::new(Some("person".to_string()), 512)
        .with_detector(detector);

    // Process a single image
    let image = open_image("photo.jpg".as_ref())?;
    if let Some(cropped) = cropper.process_image(&image, /*skip_if_not_found=*/ true) {
        cropped.save("cropped.jpg")?;
    }

    // Process an entire folder
    let stats = cropper.process_folder("./input", "./output", true)?;
    println!("processed={} skipped={} failed={}",
        stats.processed.len(), stats.skipped.len(), stats.failed.len());

    Ok(())
}
```

### Pure center crop (no model needed)

```rust
use sesoko::crop::YOLOCropper;
use sesoko::image_utils::open_image;

fn main() -> anyhow::Result<()> {
    let cropper = YOLOCropper::new(None, 512); // no detector, center crop

    let image = open_image("photo.jpg".as_ref())?;
    let cropped = cropper.process_image(&image, false).unwrap();
    cropped.save("cropped.jpg")?;
    Ok(())
}
```

### Image captioning

```rust
use sesoko::caption::{CandleVlmCaptionModel, CaptionModel, CaptionOptions, run_caption_folder, DEFAULT_CAPTION_PROMPT};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Load from the default path (models/Qwen3-VL-2B-Instruct)
    // or pass an explicit directory:
    let model = CandleVlmCaptionModel::load("models/Qwen3-VL-2B-Instruct", DEFAULT_CAPTION_PROMPT)?;

    // Caption a single image
    let image = image::open("photo.jpg")?.into_rgb8().into();
    let caption = model.generate_caption(&image)?;
    println!("{caption}");

    // Caption an entire folder, writing captions.toml and sidecar .txt files
    let options = CaptionOptions {
        folder: PathBuf::from("./images"),
        output: PathBuf::from("captions.toml"),
        no_sidecar: false,
        sidecar_dir: None,
    };
    let captions = run_caption_folder(&model, &options)?;
    println!("{} folders captioned", captions.0.len());

    Ok(())
}
```

### Image utilities

```rust
use sesoko::image_utils::{
    get_image_files, open_image, resize_image_aspect_ratio,
    crop_to_square, save_image_optimized,
};

fn main() -> anyhow::Result<()> {
    // Discover all supported images recursively
    let files = get_image_files("./photos".as_ref(), true)?;

    for path in &files {
        let img = open_image(path)?;                           // always returns RGB
        let resized = resize_image_aspect_ratio(&img, 896);    // long edge → 896 px
        let square = crop_to_square(&resized);                 // center square crop
        save_image_optimized(&square, &path.with_extension("jpg"))?; // JPEG q85
    }
    Ok(())
}
```

### Implementing a custom caption backend

```rust
use sesoko::caption::{CaptionModel, CaptionOptions, run_caption_folder};
use image::DynamicImage;
use std::path::PathBuf;

struct MyCaptionBackend;

impl CaptionModel for MyCaptionBackend {
    fn generate_caption(&self, image: &DynamicImage) -> anyhow::Result<String> {
        // call your own model / API here
        Ok(format!("a martial arts image ({}×{})", image.width(), image.height()))
    }
}

fn main() -> anyhow::Result<()> {
    let model = MyCaptionBackend;
    let options = CaptionOptions {
        folder: PathBuf::from("./images"),
        output: PathBuf::from("captions.toml"),
        no_sidecar: true,
        sidecar_dir: None,
    };
    run_caption_folder(&model, &options)?;
    Ok(())
}
```

## Training z-image-turbo Lora with ai-toolkit

https://civitai.com/articles/25872/how-i-create-my-z-image-turbo-character-loras
https://github.com/ostris/ai-toolkit/tree/main
https://hub.docker.com/r/ostris/aitoolkit/tags
https://huggingface.co/ostris/zimage_turbo_training_adapter

- **Z-Image-Turbo** — from `Tongyi-MAI/Z-Image-Turbo` on HuggingFace (the bf16 `.safetensors` version is best for training)
- **Training adapter** — from `ostris/zimage_turbo_training_adapter` on HuggingFace (there's a v1 and v2; start with v1, v2 is experimental)

docker pull ostris/aitoolkit:latest

```ps1
docker run --rm --gpus all `
  -p 7273:7273 `
  -v "H:\zimage_turbo_training_adapter:/workspace/h" `
  -v "H:\ComfyUI-user-data\models\diffusion_models\z:/workspace/z" `
  -v "H:\training-budo-lora-2026-03:/workspace/train" `
  ostris/aitoolkit:0.7.24
```

| Task                      | Detail                                                      |
| ------------------------- | ----------------------------------------------------------- |
| Pull Docker image         | `ostris/aitoolkit:latest` (~10GB)                           |
| Download base model       | `Tongyi-MAI/Z-Image-Turbo` (bf16)                           |
| Download training adapter | `ostris/zimage_turbo_training_adapter` (v1)                 |
| Image resolution          | Your 512x512 is perfect for 12GB VRAM                       |
| Dataset size              | 35–50+ images, balanced across each weapon/clothing concept |
| Captioning                | Tag every concept visible in each image, consistently       |
| Steps                     | 2000 to start                                               |
| LR                        | 1e-4                                                        |
| Transformer Offload       | **Must be 0%** (bug)                                        |
| LoRA rank                 | 16                                                          |
| Inference strength        | 0.55–0.75                                                   |

## License

MIT
