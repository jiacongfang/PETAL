# PST50: Benchmark for Photographic Style Transfer

This dataset is introduced in the paper:
**SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer**
[Project Page](https://github.com/Ry3nG/SA-LUT) • [Paper](https://huggingface.co/papers/2506.13465) • [Code](https://github.com/Ry3nG/SA-LUT)

Only the paired subset used by `benchmark_pst50.py` is kept in this directory.
The upstream release additionally ships `content_log/`, `unpaired_style/` and
`video/`, which are not needed for the paired protocol.

## Dataset Structure

```bash
PST50/
├── content_709/      # 50 content images in Rec.709 color space, in{N}.png
├── paired_style/     # 50 paired style references, tar{N}.png
├── paired_gt/        # 50 ground-truth stylizations, gt{N}.png
└── paired_outputs/   # PETAL predictions for the paired protocol, {N}.png (+ metrics.txt)
```

Images sharing the same index `N` (1 to 50) form one case.

## Evaluation Protocol

Paired: stylize `content_709/in{N}.png` with `paired_style/tar{N}.png` and
compare the result against `paired_gt/gt{N}.png` with LPIPS, PSNR, SSIM and
histogram correlation:

```bash
uv run python benchmark_pst50.py --pst50_root ./data/PST50
```

## PETAL Results

`paired_outputs/` contains the released PETAL predictions for the paired
protocol, produced at the original image resolution by the checkpoints hosted on
the [model repository](https://huggingface.co/JiacongFang/PETAL). It is not part
of the upstream PST50 release.

Recomputing the metrics for these predictions (no inference needed):

```bash
uv run python benchmark_pst50.py \
    --pst50_root ./data/PST50 \
    --pred_dir ./data/PST50/paired_outputs
```

| Method | LPIPS ↓ | PSNR ↑ | SSIM ↑ | H-corr (RGB) ↑ |
|---|---|---|---|---|
| PETAL | 0.0978 | 24.64 | 0.9332 | 0.5221 |

The same numbers are reported in the top-level `README.md`. Per-image values and
the averages are stored in `paired_outputs/metrics.txt`, which is what the
command above regenerates.

---
📜 License: cc-by-4.0

