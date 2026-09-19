"""
Standalone inference script for photographic style transfer.

Content images and style images are paired by file stem. For example,
content_dir/a.jpg and style_dir/a.png are treated as one input pair.

Usage example:

uv run python infer.py \
    --content_dir ./data/test_pairs/content \
    --style_dir ./data/test_pairs/reference \
    --output_dir ./outputs
"""

import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from tqdm.auto import tqdm
from kornia.color import rgb_to_lab, lab_to_rgb

from models_style_encoder import StyleBranch
from models_lut import LUT


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Configuration of the released photographic style encoder (ViT-L/16).
VIT_ARGS = {
    "img_size": 512, "patch_size": 16, "embed_dim": 1024, "depth": 24,
    "num_heads": 16, "ffn_ratio": 4,
    "layerscale_init": 1.0e-05, "norm_layer": "rmsnorm", "ffn_layer": "swiglu",
}


# ============================================================================
# Color space helpers
# ============================================================================

def rgb2lab(rgb: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> normalized LAB [0,1]."""
    lab = rgb_to_lab(rgb)
    lab[:, 0:1, :, :] = lab[:, 0:1, :, :] / 100.0
    lab[:, 1:3, :, :] = (lab[:, 1:3, :, :] + 128.0) / 255.0
    return lab


def lab2rgb(lab: torch.Tensor) -> torch.Tensor:
    """Normalized LAB [0,1] -> RGB [0,1]."""
    rgb = lab_to_rgb(torch.cat([lab[:, 0:1] * 100.0, lab[:, 1:3] * 255.0 - 128.0], dim=1))
    return rgb


# ============================================================================
# Image loading/saving and directory handling
# ============================================================================

def load_image_tensor(path: str, resize: int | None = None) -> torch.Tensor:
    """Load an image as a float32 tensor [1, 3, H, W] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    tensor = T.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
    if resize is not None:
        tensor = F.interpolate(tensor, size=(resize, resize), mode="bilinear", align_corners=False)
    return torch.clamp(tensor, 0, 1)


def save_image_tensor(tensor: torch.Tensor, path: str, output_format: str | None = None):
    """Save a [3, H, W] or [1, 3, H, W] float tensor as an image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.float().permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    image = Image.fromarray(np.uint8(arr * 255 + 0.5))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or Path(path).suffix.lstrip(".") or "jpg").lower()
    if fmt in {"jpg", "jpeg"}:
        image.save(path, quality=100)
    elif fmt == "png":
        image.save(path, compress_level=0)
    else:
        image.save(path)


def list_images(directory: str) -> list[str]:
    """Return sorted list of image file names in a directory."""
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMG_EXTENSIONS))


def build_stem_to_name_map(directory: str) -> dict[str, str]:
    """Return a unique mapping from file stem to file name for images in a directory."""
    stem_to_name: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for file_name in list_images(directory):
        stem = os.path.splitext(file_name)[0]
        if stem in stem_to_name:
            duplicates.setdefault(stem, [stem_to_name[stem]]).append(file_name)
            continue
        stem_to_name[stem] = file_name

    if duplicates:
        duplicate_desc = "; ".join(
            f"{stem}: {', '.join(names)}" for stem, names in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate image stems found in {directory}: {duplicate_desc}")

    return stem_to_name


def build_directory_cases(args) -> list[dict[str, str]]:
    if not args.content_dir or not args.style_dir:
        raise ValueError("Directory mode requires --content_dir and --style_dir.")

    content_map = build_stem_to_name_map(args.content_dir)
    style_map = build_stem_to_name_map(args.style_dir)
    paired_stems = sorted(set(content_map) & set(style_map))

    if not paired_stems:
        raise ValueError(f"No matching image names found between {args.content_dir} and {args.style_dir}")

    content_only = sorted(set(content_map) - set(style_map))
    style_only = sorted(set(style_map) - set(content_map))
    if content_only:
        print(f"Skipping {len(content_only)} content-only images.")
    if style_only:
        print(f"Skipping {len(style_only)} style-only images.")

    return [
        {
            "case_id": stem,
            "input_path": os.path.join(args.content_dir, content_map[stem]),
            "reference_path": os.path.join(args.style_dir, style_map[stem]),
            "output_path": os.path.join(args.output_dir, f"{stem}.{args.output_format}"),
        }
        for stem in paired_stems
    ]


# ============================================================================
# Models and inference
# ============================================================================

def load_models(style_model_path: str, lut_model_path: str, device, weight_dtype):
    """Load the photographic style encoder and the AdaIN/LUT model.

    The released style-branch checkpoint keeps the DINOv3 `mask_token` buffer of
    the pretraining recipe; it is unused at inference time and therefore skipped.
    """
    style_branch = StyleBranch(VIT_ARGS)
    state_dict = torch.load(style_model_path, map_location="cpu", weights_only=True)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    state_dict.pop("vit.mask_token", None)
    missing, unexpected = style_branch.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    style_branch.eval().to(device, dtype=weight_dtype)
    print("Photographic style encoder loaded.")

    adain_model = LUT()
    state_dict = torch.load(lut_model_path, map_location="cpu", weights_only=True)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = adain_model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    adain_model.eval().to(device, dtype=weight_dtype)
    param_count = sum(p.numel() for p in adain_model.parameters())
    print(f"AdaIN model loaded. Parameters: {param_count / 1e6:.2f}M")

    return style_branch, adain_model


@torch.inference_mode()
def inference(style_branch, adain_model, content_img, style_img, guidance_scale=1.0, style_input_size=512):
    """
    Run style transfer on a single content + style pair.

    Args:
        style_branch:  Style encoder (produces z embeddings).
        adain_model:   LUT / AdaIN model.
        content_img:   Content image tensor [B, 3, H, W] in [0, 1] RGB.
        style_img:     Style image  tensor [B, 3, H, W] in [0, 1] RGB.
        guidance_scale: Guidance scale for the AdaIN model.
        style_input_size: Resolution to resize images for style branch.

    Returns:
        pred_img: [B, 3, H, W] in [0, 1] RGB.
    """
    resolution = style_input_size

    content_lab = rgb2lab(torch.clamp(content_img, 0, 1))

    # Resize for style branch
    if content_img.shape[2] != resolution or content_img.shape[3] != resolution:
        content_resized = F.interpolate(content_img, size=(resolution, resolution),
                                        mode="bilinear", align_corners=False)
        content_lab_resized = rgb2lab(torch.clamp(content_resized, 0, 1))
    else:
        content_lab_resized = content_lab

    style_resized = F.interpolate(torch.clamp(style_img, 0, 1),
                                  size=(resolution, resolution),
                                  mode="bilinear", align_corners=False)
    style_lab = rgb2lab(style_resized)

    style_out = style_branch(torch.cat([style_lab, content_lab_resized], 0))
    z = style_out["z"]
    z_st, z_cont = z.chunk(2)

    # normalize z embeddings
    z_st = F.normalize(z_st, dim=-1)
    z_cont = F.normalize(z_cont, dim=-1)

    pred_img = adain_model(content_lab, z_st, z_cont, guidance_scale=guidance_scale)
    pred_img = torch.clamp(pred_img, 0, 1)
    pred_img = lab2rgb(pred_img)

    return torch.clamp(pred_img.detach(), 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Photo style transfer inference")

    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for results.")

    parser.add_argument("--content_dir", type=str, default=None, help="Directory of content images.")
    parser.add_argument("--style_dir", type=str, default=None, help="Directory of style images.")

    parser.add_argument("--lut_model_path", type=str, default="./ckpts/lut_model.pt", help="Path to LUT / AdaIN checkpoint.")
    parser.add_argument("--style_model_path", type=str, default="./ckpts/style_encoder.pt", help="Path to style encoder checkpoint.")

    # --- Inference settings ---
    parser.add_argument("--style_input_size", type=int, default=512, help="Image size to the style encoder")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for AdaIN.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_format", type=str, default="png", choices=["png", "jpg", "jpeg"], help="Output image format.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip cases whose output already exists.")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cases = build_directory_cases(args)
    print(f"Directory mode: {len(cases)} matched image pairs")
    if args.skip_existing:
        cases = [case for case in cases if not Path(case["output_path"]).exists()]
    if not cases:
        print("No pending cases.")
        return

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    style_branch, adain_model = load_models(
        args.style_model_path, args.lut_model_path, device, weight_dtype)

    # ---- Inference ----
    failures = []
    for case in tqdm(cases, desc="Inference", ncols=100):
        try:
            content = load_image_tensor(case["input_path"]).to(device, dtype=weight_dtype)
            style = load_image_tensor(case["reference_path"]).to(device, dtype=weight_dtype)
            pred = inference(style_branch, adain_model, content, style, args.guidance_scale, args.style_input_size)
            save_image_tensor(pred, case["output_path"], args.output_format)
        except Exception as exc:
            failures.append(case["case_id"])
            print(f"Failed case {case['case_id']}: {traceback.format_exception_only(type(exc), exc)[-1].strip()}")

    if failures:
        raise RuntimeError(f"{len(failures)} cases failed: {failures}")

    print("Done!")


if __name__ == "__main__":
    main()
