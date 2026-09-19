"""
PST50 paired benchmark for photographic style transfer.

PST50 (https://github.com/Ry3nG/SA-LUT, CC-BY-4.0) provides paired content,
style references and ground-truth stylizations:

    data/PST50/content_709/in{N}.png     content image (Rec.709)
    data/PST50/paired_style/tar{N}.png   style reference
    data/PST50/paired_gt/gt{N}.png       ground-truth stylization

The paired protocol transfers content through its paired style reference and
compares the result against the ground truth with the metrics:
    LPIPS (AlexNet), PSNR, SSIM and RGB histogram correlation
computed at the original image resolution.

Predictions must therefore have the same resolution as the ground truth; the
script raises an error when they differ. Pass `--allow_gt_resize` to resize the
ground truth to the prediction size instead, in which case the metrics are
computed at the prediction resolution and are flagged in the report.

Usage example:

uv run python benchmark_pst50.py \
    --pst50_root ./data/PST50 \
    --output_dir ./outputs/pst50_paired

To only re-compute the metrics of an existing prediction directory, pass
`--pred_dir` with predictions named `{N}.png`.
"""

import argparse
import os
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm.auto import tqdm

from infer import inference, load_image_tensor, load_models, save_image_tensor


METRIC_NAMES = ["lpips", "psnr", "ssim", "hist_corr"]


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def calculate_lpips(img1: np.ndarray, img2: np.ndarray, lpips_model) -> float:
    """LPIPS between two uint8 RGB images"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img1_tensor = transform(Image.fromarray(img1)).unsqueeze(0)
    img2_tensor = transform(Image.fromarray(img2)).unsqueeze(0)
    device = next(lpips_model.parameters()).device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    with torch.no_grad():
        dist = lpips_model(img1_tensor, img2_tensor)
    return dist.item()


def calculate_histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean RGB histogram correlation over the three channels."""
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    correlations = []
    for i in range(3):
        hist1 = cv2.calcHist([img1_bgr], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_bgr], [i], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        correlations.append(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    return float(np.mean(correlations))


def build_pairs(pst50_root: str, num: int) -> list[dict[str, str]]:
    """Build the paired PST50 cases: in{N}.png + tar{N}.png -> gt{N}.png."""
    root = Path(pst50_root)
    pairs = []
    for index in range(1, num + 1):
        content = root / "content_709" / f"in{index}.png"
        style = root / "paired_style" / f"tar{index}.png"
        gt = root / "paired_gt" / f"gt{index}.png"

        missing = [str(path) for path in (content, style, gt) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing PST50 files: {missing}")

        pairs.append({
            "index": f"{index:04d}",
            "content_path": str(content),
            "style_path": str(style),
            "gt_path": str(gt),
        })
    return pairs


def run_inference(pairs, args, pred_dir: str):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    style_branch, adain_model = load_models(
        args.style_model_path, args.lut_model_path, device, weight_dtype)

    for pair in tqdm(pairs, desc="Inference", ncols=100):
        output_path = os.path.join(pred_dir, f"{int(pair['index'])}.png")
        if args.skip_existing and os.path.exists(output_path):
            continue
        content = load_image_tensor(pair["content_path"]).to(device, dtype=weight_dtype)
        style = load_image_tensor(pair["style_path"]).to(device, dtype=weight_dtype)
        pred = inference(style_branch, adain_model, content, style,
                         guidance_scale=args.guidance_scale,
                         style_input_size=args.style_input_size)
        save_image_tensor(pred, output_path, "png")


def evaluate(pairs, pred_dir: str, allow_gt_resize: bool = False
             ) -> tuple[dict[str, list[float]], list[str]]:
    print("Initializing LPIPS model...")
    lpips_model = lpips.LPIPS(net="alex")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips_model.to(device)
    print(f"Computing LPIPS on {device}")

    results: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    resized: list[str] = []
    for pair in tqdm(pairs, desc="Metrics", ncols=100):
        index = f"{int(pair['index'])}"
        pred_img = load_image(os.path.join(pred_dir, f"{index}.png"))
        gt_img = load_image(pair["gt_path"])

        if pred_img.shape != gt_img.shape:
            if not allow_gt_resize:
                raise ValueError(
                    f"size mismatch for case {index}: prediction {pred_img.shape} vs ground truth "
                    f"{gt_img.shape}. Predictions must match the ground-truth resolution; pass "
                    f"--allow_gt_resize to resize the ground truth instead (metrics are then "
                    f"reported at the prediction resolution and are not comparable across "
                    f"prediction directories).")
            print(f"Warning: size mismatch for {index}: pred {pred_img.shape} vs gt {gt_img.shape}, "
                  f"resizing gt to pred")
            gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]),
                                interpolation=cv2.INTER_LANCZOS4)
            resized.append(index)

        results["lpips"].append(calculate_lpips(pred_img, gt_img, lpips_model))
        results["psnr"].append(psnr(pred_img, gt_img, data_range=255))
        results["ssim"].append(ssim(pred_img, gt_img, channel_axis=2, data_range=255))
        results["hist_corr"].append(calculate_histogram_correlation(pred_img, gt_img))

    return results, resized


def write_report(path: str, pairs, results: dict[str, list[float]], args, resized: list[str]):
    averages = {name: float(np.mean(values)) for name, values in results.items()}

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("PST50 paired evaluation\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"num_pairs: {len(pairs)}\n")
        if resized:
            handle.write("\nWARNING: ground truth was resized to the prediction size for "
                         f"{len(resized)} case(s): {', '.join(resized)}\n")
            handle.write("Metrics below are computed at the prediction resolution and are not "
                         "comparable with runs evaluated at the original resolution.\n")
        handle.write("\n")
        handle.write("Per-image results:\n")
        for i, pair in enumerate(pairs):
            handle.write(f"\n{pair['index']}.png:\n")
            handle.write(f"  LPIPS: {results['lpips'][i]:.4f}\n")
            handle.write(f"  PSNR: {results['psnr'][i]:.2f} dB\n")
            handle.write(f"  SSIM: {results['ssim'][i]:.4f}\n")
            handle.write(f"  Histogram Correlation (RGB): {results['hist_corr'][i]:.4f}\n")

        handle.write("\n" + "=" * 50 + "\n")
        handle.write("Averages:\n")
        handle.write(f"  LPIPS: {averages['lpips']:.4f} (lower is better)\n")
        handle.write(f"  PSNR: {averages['psnr']:.2f} dB (higher is better)\n")
        handle.write(f"  SSIM: {averages['ssim']:.4f} (higher is better)\n")
        handle.write(f"  Histogram Correlation (RGB): {averages['hist_corr']:.4f} (higher is better)\n")

    return averages


def parse_args():
    parser = argparse.ArgumentParser(description="PST50 paired benchmark")

    parser.add_argument("--pst50_root", type=str, default="./data/PST50",
                        help="Root of the PST50 dataset (with content_709/paired_style/paired_gt).")
    parser.add_argument("--num", type=int, default=50, help="Number of paired cases to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./outputs/pst50_paired",
                        help="Directory for the stylized predictions.")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Evaluate an existing prediction directory instead of running inference.")
    parser.add_argument("--report", type=str, default=None, help="Report path (default: <output_dir>/metrics.txt).")
    parser.add_argument("--allow_gt_resize", action="store_true",
                        help="Resize the ground truth to the prediction size when they differ. "
                             "Metrics are then computed at the prediction resolution and flagged "
                             "in the report.")

    parser.add_argument("--lut_model_path", type=str, default="./ckpts/lut_model.pt", help="Path to LUT / AdaIN checkpoint.")
    parser.add_argument("--style_model_path", type=str, default="./ckpts/style_encoder.pt", help="Path to style encoder checkpoint.")

    parser.add_argument("--guidance_scale", type=float, default=0.5, help="Guidance scale for AdaIN.")
    parser.add_argument("--style_input_size", type=int, default=512, help="Image size to the style encoder")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--skip_existing", action="store_true", help="Skip cases whose prediction already exists.")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    pairs = build_pairs(args.pst50_root, args.num)
    pred_dir = args.pred_dir or args.output_dir
    os.makedirs(pred_dir, exist_ok=True)

    print(f"PST50 paired: {len(pairs)} cases")
    if args.pred_dir is None:
        run_inference(pairs, args, pred_dir)
    else:
        print(f"Reusing predictions in {pred_dir}")

    results, resized_cases = evaluate(pairs, pred_dir, args.allow_gt_resize)

    report_path = args.report or os.path.join(pred_dir, "metrics.txt")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    averages = write_report(report_path, pairs, results, args, resized_cases)

    print("\n" + "=" * 50)
    print(f"PST50 paired averages over {len(pairs)} cases")
    print("=" * 50)
    print(f"LPIPS: {averages['lpips']:.4f}")
    print(f"PSNR: {averages['psnr']:.2f} dB")
    print(f"SSIM: {averages['ssim']:.4f}")
    print(f"Histogram Correlation (RGB): {averages['hist_corr']:.4f}")
    if resized_cases:
        print(f"\nWarning: ground truth was resized to the prediction size for "
              f"{len(resized_cases)} case(s).")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
