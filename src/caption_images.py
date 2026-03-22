#!/usr/bin/env python3
"""Image captioning script supporting multiple VL models.

This script processes all images in a given folder and generates captions
using a selectable vision-language model, storing the results in a TOML file.

Supported models:
  caprl    - CapRL-Qwen3VL-2B (default) from internlm
  qwen3vl  - Qwen3-VL-2B-Instruct from Qwen
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

from lib.image_utils import (
    get_image_files,
    open_image,
    resize_image_aspect_ratio,
)

try:
    import tomli_w
except ImportError:
    tomli_w = None

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None


# Paths to locally stored models, relative to this file's location
_MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_CHOICES: dict[str, Path] = {
    # https://huggingface.co/internlm/CapRL-Qwen3VL-2B
    "caprl": _MODELS_DIR / "CapRL-Qwen3VL-2B",
    # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
    "qwen3vl": _MODELS_DIR / "Qwen3-VL-2B-Instruct",
}

DEFAULT_MODEL = "caprl"


def load_model_and_processor(
    model_key: str = DEFAULT_MODEL,
) -> tuple[Qwen3VLForConditionalGeneration, Qwen3VLProcessor, str]:
    """Load a Qwen3-VL-based model and processor from local storage.

    Args:
        model_key: One of the keys in MODEL_CHOICES (e.g. "caprl", "qwen3vl").

    Returns:
        Tuple containing:
            - Qwen3VLForConditionalGeneration: The loaded model
            - Qwen3VLProcessor: The processor for handling inputs
            - str: Device string ("cuda" or "cpu")
    """
    model_path = MODEL_CHOICES[model_key]
    model_id = str(model_path)

    print(f"Loading model '{model_key}' from {model_id}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # Device is handled automatically by device_map="auto"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    return model, processor, device


def generate_caption(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    image: Image.Image,
    device: str,
) -> str:
    """Generate caption for an image using Qwen3-VL model.

    Args:
        model: Qwen3-VL model for conditional generation
        processor: Qwen3VL processor for handling images and text
        image: PIL Image object
        device: Device to use ("cuda" or "cpu")

    Returns:
        Generated caption string
    """
    # Detailed prompt for martial arts - instruct model to respond in plain text
    prompt = "Describe the image as a caption with less than 255 characters. It contains Japanese martial arts. What martial art is shown? What weapons are used, name them? Describe clothing, belt, technique. Respond with plain text only, no formatting or markdown. Be firm, no guessing."

    # Prepare inputs using chat template (following official example)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # Apply chat template to get formatted text string, then call processor correctly
    text: str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore
    inputs: dict[str, Any] = processor(  # type: ignore
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate without trimming complexity
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Simply decode the entire output
    caption = processor.batch_decode([outputs[0]], skip_special_tokens=True)[0]

    # Remove the prompt part if it's in there
    if "assistant" in caption:
        caption = caption.split("assistant")[-1].strip()

    return caption


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Caption images using a selectable vision-language model"
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="captions.toml",
        help="Output TOML file path (default: captions.toml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CHOICES.keys()),
        default=DEFAULT_MODEL,
        help=(
            f"Vision-language model to use for captioning (default: {DEFAULT_MODEL}). "
            "caprl=CapRL-Qwen3VL-2B, qwen3vl=Qwen3-VL-2B-Instruct"
        ),
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Disable creation of sidecar .txt files with captions",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=str,
        default=None,
        help="Directory to write sidecar .txt files (preserves folder structure)",
    )

    args = parser.parse_args()

    # Validate folder
    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Error: {args.folder} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    # Get image files
    image_files = get_image_files(folder_path)
    if not image_files:
        print(f"No image files found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process")

    # Load model and processor
    model, processor, device = load_model_and_processor(args.model)

    # Determine output path and format
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".toml":
        output_path = output_path.with_suffix(".toml")

    print(f"Captions will be saved to {output_path}")
    if not args.no_sidecar:
        if args.sidecar_dir:
            print(f"Sidecar .txt files will be created in {args.sidecar_dir}")
        else:
            print("Sidecar .txt files will also be created alongside images")

    # Process images and generate captions with streaming writes
    # Get absolute path of the input folder to use as section name
    absolute_folder_path = folder_path.resolve().as_posix()

    # Load existing captions if the file exists
    if output_path.exists() and tomllib:
        try:
            with open(output_path, "rb") as f:
                captions: Dict[str, Dict[str, str]] = tomllib.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing captions file: {e}", file=sys.stderr)
            captions = {}
    else:
        captions = {}

    # Only create the section if it doesn't already exist
    if absolute_folder_path not in captions:
        captions[absolute_folder_path] = {}

    # Prepare sidecar directory if specified
    sidecar_base_dir = None
    if not args.no_sidecar and args.sidecar_dir:
        sidecar_base_dir = Path(args.sidecar_dir)
        sidecar_base_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(image_files, 1):
        try:
            print(
                f"[{idx}/{len(image_files)}] Processing {image_path.name}...",
                end=" ",
                flush=True,
            )

            # Load and resize image
            image = open_image(image_path)
            if image is None:
                print("Failed to open image")
                continue
            image = resize_image_aspect_ratio(image, target_size=896)

            # Generate caption
            caption = generate_caption(model, processor, image, device)

            # Store with relative path as key under the folder section
            relative_path = image_path.relative_to(folder_path).as_posix()
            captions[absolute_folder_path][relative_path] = caption

            # Show caption
            print(f"✓ {caption}")

            # Stream write to TOML file immediately
            if tomli_w:
                with open(output_path, "wb") as f:
                    tomli_w.dump(captions, f)
            else:
                # Fallback to JSON if tomli_w not available
                with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(captions, f, indent=2, ensure_ascii=False)

            # Write sidecar .txt file if not disabled
            if not args.no_sidecar:
                if sidecar_base_dir:
                    # Write to sidecar directory, preserving folder structure
                    relative_path_obj = image_path.relative_to(folder_path)
                    sidecar_path = sidecar_base_dir / relative_path_obj.with_suffix(
                        relative_path_obj.suffix + ".txt"
                    )
                    # Create subdirectories if needed
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Write alongside image
                    sidecar_path = image_path.with_suffix(image_path.suffix + ".txt")

                try:
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                except PermissionError:
                    print(
                        f"✗ Permission denied writing sidecar file: {sidecar_path}", file=sys.stderr
                    )
                    print(
                        "Stopping processing - cannot write sidecar files with current permissions.",
                        file=sys.stderr,
                    )
                    print(
                        "Use --no-sidecar flag to process without sidecar files.", file=sys.stderr
                    )
                    if not args.sidecar_dir:
                        print(
                            "Or use --sidecar-dir to write sidecar files to a different location.",
                            file=sys.stderr,
                        )
                    sys.exit(1)

            # Clear GPU cache periodically to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback

            print(f"✗ Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

    print(
        f"\nSuccessfully captioned {len(captions[absolute_folder_path])}/{len(image_files)} images"
    )
    print(f"Captions saved to {output_path}")


if __name__ == "__main__":
    main()
