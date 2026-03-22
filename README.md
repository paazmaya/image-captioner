# sesoko (瀬底)

> Prepare images for training machine learning models

Two key features:
* Image captioning for martial arts imagery using **Image-Captioning-Blip** (default), **CapRL-Qwen3VL-2B**, or **Qwen3-VL-2B-Instruct**
* Content aware cropping by using YOLO11

## Architecture

The project uses a shared image handling library (`src/lib/image_utils.py`) to provide common functionality:

- **Image Format Support**: Supports 15+ image formats including JPEG, PNG, GIF, WebP, TIFF, HEIC/HEIF, AVIF, JPEG XL, and more
- **Image Loading**: Safe image opening with automatic RGB conversion and HEIF support
- **Image Processing**: Aspect-ratio preserving resizing and square cropping without upscaling or padding
- **Optimized Output**: JPEG encoding with quality 85 optimization

## Quick Start

```bash
# Install dependencies
uv sync

# Generate captions for images
uv run python src/caption_images.py "path/to/images"

# Crop images based on object detection
uv run python src/crop_yolo.py --input-dir my_images --output-dir cropped_images
```

## Utility Scripts

This project provides two main scripts for preparing images for machine learning training:

### 1. Image Captioning

Generate detailed captions for images using a selectable vision-language model.

Available models (stored under `models/`):
| Key | Model | VRAM |
|-----|-------|------|
| `blip` *(default)* | Image-Captioning-Blip (Amirhossein75) | ~1 GB |
| `caprl` | CapRL-Qwen3VL-2B (internlm) | ~4-6 GB |
| `qwen3vl` | Qwen3-VL-2B-Instruct (Qwen) | ~4-6 GB |

**Features:**
- 🚀 Ultra-fast inference — BLIP uses beam search on a tiny model (~1 GB VRAM)
- 🔑 Keyword-rich, concise descriptions from BLIP out of the box
- 💾 Minimal VRAM — BLIP ~1 GB, Qwen3-VL models ~4-6 GB
- ✨ Detailed captions for martial arts images with Qwen3-VL models
- 🔄 Streaming writes - recovers from interruptions
- 📝 Plain text output - no markdown formatting

**Basic Usage:**

```bash
# Default model (Image-Captioning-Blip) - fast, keyword-focused
uv run python src/caption_images.py "path/to/images"

# Use CapRL-Qwen3VL-2B for detailed martial arts captions
uv run python src/caption_images.py "path/to/images" --model caprl

# Use Qwen3-VL-2B-Instruct
uv run python src/caption_images.py "path/to/images" --model qwen3vl
```

Output will be saved to `captions.toml` in the current folder and as sidecar `.txt` files alongside each image.

**Advanced Options:**

```bash
# Custom output path
uv run python src/caption_images.py "path/to/images" -o my_captions.toml

# Disable sidecar files (TOML only)
uv run python src/caption_images.py "path/to/images" --no-sidecar

# Write sidecar files to a different directory
uv run python src/caption_images.py "Dropbox\images" --sidecar-dir "captions"
```

**Output Format:**

TOML file organized by folder absolute path:
```toml
["/absolute/path/to/images/folder"]
"image1.jpg" = "Caption text here..."
"image2.jpg" = "Another caption..."
```

Sidecar `.txt` files (optional):
```
images/
  photo1.jpg
  photo1.jpg.txt     # Contains the caption
  photo2.jpg
  photo2.jpg.txt
```

**Supported Image Formats:**
- `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.heic`, `.heif`, `.avif`

**Troubleshooting:**

- **OOM (Out of Memory):** Use `--model blip` (default, ~1 GB VRAM) or reduce image size in `resize_image()` (Qwen3-VL uses 896px, BLIP uses 512px)
- **Permission denied:** Use `--no-sidecar` or `--sidecar-dir` for cloud storage (Dropbox, OneDrive, etc.)
- **Slow inference:** Ensure CUDA is properly detected with `nvidia-smi`, check Flash Attention 2 installation

### 2. Image Cropping with YOLO11

Detect and crop objects in images to square format using **YOLO11 segmentation**.

All output images are **square** without padding or upscaling. The script crops to the smaller dimension of the image, ensuring native resolution quality. Images are saved as optimized JPEG files (quality 85) for efficient storage.

**Basic Usage:**

```bash
# List all available object classes
uv run python scripts/crop_yolo.py --list-classes

# Center crop all images to square (512x512)
uv run python scripts/crop_yolo.py --input-dir my_images --output-dir cropped_images
```

**Advanced Options:**

```bash
# Crop focusing on specific objects (person, face, dog, etc.)
uv run python scripts/crop_yolo.py \
  --input-dir my_images \
  --output-dir cropped_images \
  --crop-focus person \
  --resolution 1024

# Save processing statistics
uv run python scripts/crop_yolo.py \
  --input-dir my_images \
  --output-dir cropped_images \
  --crop-focus face \
  --stats stats.json
```

**Features:**
- Content-aware object detection using YOLO11
- Square output (no padding, no upscaling) - crops to smallest dimension
- JPEG format with quality 85 and optimization for minimal file size
- Smart filtering - skips images without target object
- Detailed processing statistics (JSON output)
- Supports any YOLO-detectable object class

**Output Format:**
- All images are cropped to square (smallest input dimension)
- Then resized to the specified resolution (default 512x512)
- Saved as optimized JPEG files (quality 85) in the output directory

---

### 3. Model Precision Conversion

Convert models to reduced precision formats for smaller file sizes and faster inference.

**Basic Usage:**

```bash
# Convert to bfloat16 (default)
uv run python scripts/convert_floats.py --input H:/my-model

# Convert to float8 e4m3fn (higher precision)
uv run python scripts/convert_floats.py --input H:/my-model --dtype e4m3fn

# Convert single safetensors file
uv run python scripts/convert_floats.py --input H:/models/model.safetensors
```

**Supported Formats:** `bf16`, `e4m3fn`, `e5m2`

---

## Requirements

- **Python:** 3.11+
- **GPU:** CUDA-capable NVIDIA GPU
- **VRAM:** 
  - Image captioning (2B): ~4-6 GB
  - Image captioning (4B): ~6-8 GB
  - Cropping: ~2-4 GB

### Setup

1. Install CUDA 13 drivers from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Verify CUDA availability: `uv run python -c "import torch; print(torch.cuda.is_available())"`

## Notes

- Image captioning focuses on martial arts imagery with plain text output
- GPU cache is cleared after each image to prevent memory fragmentation
- Streaming writes enable recovery from interruptions
- All scripts process images sequentially

## License

MIT