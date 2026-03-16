use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use sesoko::caption::{
    resolve_caption_model_path, run_caption_folder, MistralRsVlmCaptionModel, CaptionOptions,
    DEFAULT_CAPTION_PROMPT,
};
use sesoko::crop::YOLOCropper;

#[derive(Parser, Debug)]
#[command(name = "sesoko")]
#[command(about = "Crop images of various formats to square JPEGs and generate caption .txt files")]
#[command(
    long_about = "Crop a folder of various format images to square JPEGs and\n\
    write a caption .txt sidecar next to each output image.\n\
    \n\
    USAGE:\n  \
    sesoko <INPUT_DIR> <OUTPUT_DIR>\n\
    \n\
    Defaults:\n  \
    - Output size : 512 × 512 px JPEG\n  \
    - Crop method : center crop (no YOLO required)\n  \
    - Caption model: models/Qwen3-VL-2B-Instruct/  (relative to cwd)\n  \
    - YOLO model  : models/candle-yolo-v8/yolov8n.safetensors (used by `crop` subcommand)\n\
    \n\
    For finer control use the subcommands below."
)]
#[command(arg_required_else_help = true)]
struct Cli {
    /// Input folder containing images (top-level only; all common formats accepted)
    input_dir: Option<PathBuf>,

    /// Output folder for cropped square JPEGs and caption .txt sidecars
    output_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Generate captions for images using Qwen3-VL")]
    Caption {
        #[arg(help = "Path to input folder containing images (searched recursively)")]
        folder: PathBuf,
        #[arg(
            short,
            long,
            default_value = "captions.toml",
            help = "Output captions file path (TOML). If extension is not .toml it will be replaced"
        )]
        output: PathBuf,
        #[arg(
            long,
            help = "Path to model artifact used for default model-name resolution"
        )]
        model_path: Option<PathBuf>,
        #[arg(long, help = "Disable writing sidecar .txt files")]
        no_sidecar: bool,
        #[arg(
            long,
            help = "Directory for sidecar files. Relative folder structure is preserved"
        )]
        sidecar_dir: Option<PathBuf>,
    },
    #[command(about = "Crop images to square JPEGs using YOLO detection")]
    Crop {
        #[arg(
            long,
            help = "Input folder containing images. Required unless --list-classes is used"
        )]
        input_dir: Option<PathBuf>,
        #[arg(
            long,
            help = "Output folder for cropped JPEG images. Required unless --list-classes is used"
        )]
        output_dir: Option<PathBuf>,
        #[arg(
            long,
            help = "YOLO model path override. Defaults to models/candle-yolo-v8/yolov8n.safetensors"
        )]
        model_path: Option<PathBuf>,
        #[arg(
            long,
            help = "Object class name to focus crop on (e.g. person). If omitted, center crop is used"
        )]
        crop_focus: Option<String>,
        #[arg(
            long,
            default_value_t = 512,
            help = "Output square resolution in pixels"
        )]
        resolution: u32,
        #[arg(long, help = "List available YOLO classes and exit")]
        list_classes: bool,
        #[arg(long, help = "Write processing stats JSON to this file")]
        stats: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Caption {
            folder,
            output,
            model_path,
            no_sidecar,
            sidecar_dir,
        }) => {
            let model_dir = resolve_caption_model_path(model_path.as_deref())?;
            let model = MistralRsVlmCaptionModel::load(model_dir, DEFAULT_CAPTION_PROMPT)?;
            let options = CaptionOptions {
                folder,
                output,
                no_sidecar,
                sidecar_dir,
            };
            run_caption_folder(&model, &options)?;
        }
        Some(Commands::Crop {
            input_dir,
            output_dir,
            model_path,
            crop_focus,
            resolution,
            list_classes,
            stats,
        }) => {
            let cropper = {
                let detector =
                    sesoko::yolo_candle::build_candle_detector_with_default(model_path.as_deref())?;
                YOLOCropper::new(crop_focus, resolution).with_detector(detector)
            };

            if list_classes {
                println!("Available YOLO object classes:");
                for class_name in cropper.get_available_classes() {
                    println!("  - {}", class_name);
                }
                return Ok(());
            }

            let (input_dir, output_dir) = match (input_dir, output_dir) {
                (Some(input_dir), Some(output_dir)) => (input_dir, output_dir),
                _ => bail!("--input-dir and --output-dir are required unless using --list-classes"),
            };

            let stats_result = cropper.process_folder(input_dir, output_dir, true)?;
            println!("Processed: {} images", stats_result.processed.len());
            println!("Skipped: {} images", stats_result.skipped.len());
            println!("Failed: {} images", stats_result.failed.len());

            if let Some(path) = stats {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let content = serde_json::to_string_pretty(&stats_result)?;
                std::fs::write(&path, content)?;
                println!("Statistics saved to: {}", path.display());
            }
        }
        None => {
            // Default: crop all images to 512px square JPEGs, then caption each one.
            let (input_dir, output_dir) = match (cli.input_dir, cli.output_dir) {
                (Some(i), Some(o)) => (i, o),
                _ => bail!(
                    "Provide INPUT_DIR and OUTPUT_DIR, or use a subcommand.\nRun `sesoko --help` for usage."
                ),
            };

            // Step 1: center-crop every image to a 512 × 512 JPEG.
            println!(
                "Cropping {} → {}...",
                input_dir.display(),
                output_dir.display()
            );
            let cropper = YOLOCropper::new(None, 512);
            let crop_stats = cropper.process_folder(&input_dir, &output_dir, false)?;
            println!(
                "Crop: {} processed, {} skipped, {} failed",
                crop_stats.processed.len(),
                crop_stats.skipped.len(),
                crop_stats.failed.len(),
            );

            // Step 2: caption the cropped images; write a .txt sidecar next to each.
            println!("Captioning images in {}...", output_dir.display());
            let model_dir = resolve_caption_model_path(None)?;
            let model = MistralRsVlmCaptionModel::load(model_dir, DEFAULT_CAPTION_PROMPT)?;
            let options = CaptionOptions {
                folder: output_dir.clone(),
                output: output_dir.join("captions.toml"),
                no_sidecar: false,
                sidecar_dir: None,
            };
            run_caption_folder(&model, &options)?;
            println!(
                "Done. Caption .txt files written next to each image in {}",
                output_dir.display()
            );
        }
    }

    Ok(())
}
