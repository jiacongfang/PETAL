<div align="center">

# Enlightening Photographic Style Transfer with a Self-Supervised Photographic Embedding

[Chengxuan Zhu](https://freebutuselesssoul.github.io/)* · [Jiacong Fang](https://jiacongfang.github.io/)* ·
[Shuchen Weng](https://shuchenweng.github.io/) · [Youwei Lyu](https://youweilyu.github.io/) ·
[Jiajun Tang](https://me.jeffreet.com/) · [Qingnan Fan](https://fqnchina.github.io/) ·
Chao Xu · [Boxin Shi](https://ci.idm.pku.edu.cn/)

*: Equal Contribution

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://petal-pku.pages.dev/)
[![Paper](https://img.shields.io/badge/Paper-Springer-b31b1b.svg)](https://link.springer.com/chapter/10.1007/978-3-032-37490-5_2)
[![Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-yellow)](https://huggingface.co/JiacongFang/PETAL)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

</div>

Photographic style — the nuanced play of lightness, color, and tone a photographer crafts — is easy for the eye to read, yet invisible to most image embeddings. We present **PETAL** (**P**hotographic **E**mbedding for **T**ransfer with an **A**daptive **L**UT): we learn a **continuous photographic embedding** by self-supervision, and use it to drive a **lightweight adaptive neural LUT** that transfers style faithfully, with no test-time optimization.


## Quick Start

### 1. Environment Setup
```bash
# A CUDA GPU is strongly recommended; CPU inference also works but is much slower.

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync         # Create the virtual environment and install dependencies
```

### 2. Download Checkpoints

The released checkpoints (`style_encoder.pt`, `lut_model.pt`) are not stored in
this repository. Download them from the
[Hugging Face model repository](https://huggingface.co/JiacongFang/PETAL) into
`./ckpts`:

```bash
./download_weights.sh
```

You can also download the two files manually and place them in `./ckpts`.

### 3. Inference
Content images and style images are paired by **file stem** (filename without
extension): `content_dir/001.jpg` is paired with `style_dir/001.png`.

```bash
uv run python infer.py \
    --content_dir ./data/test_pairs/content \
    --style_dir ./data/test_pairs/reference \
    --output_dir ./outputs \
    --output_format png
```

| Argument | Default | Description |
|---|---|---|
| `--content_dir` | *(required)* | Directory of content images |
| `--style_dir` | *(required)* | Directory of style images |
| `--output_dir` | `./outputs` | Output directory |
| `--lut_model_path` | `./ckpts/lut_model.pt` | Path to LUT / AdaIN checkpoint |
| `--style_model_path` | `./ckpts/style_encoder.pt` | Path to style encoder checkpoint |
| `--style_input_size` | `512` | Resolution for the style encoder |
| `--guidance_scale` | `1.0` | Guidance scale for adain, suggested range [0.5, 1.2] |
| `--mixed_precision` | `no` | `no` / `fp16` / `bf16` |
| `--output_format` | `png` | `png` / `jpg` / `jpeg` (PNG is lossless; JPG/JPEG are lossy) |
| `--skip_existing` | off | Skip pairs whose output already exists |
| `--seed` | `42` | Random seed |

## Test Dataset

### 1. Qualitative Data

`data/test_pairs/` holds the images used for qualitative checks:

- `content/` and `reference/` — content/style pairs paired by file stem.
- `outputs/` - inference output of PETAL

### 2. PST50 Benchmark

[PST50](https://github.com/Ry3nG/SA-LUT) is a paired benchmark for
photorealistic style transfer (license: CC-BY-4.0). The expected layout is:

```bash
PST50/
├── content_709/in{N}.png     # content images (Rec.709)
├── paired_style/tar{N}.png   # paired style references
└── paired_gt/gt{N}.png       # ground-truth stylizations
```

The paired subset of the dataset is included under `./data/PST50`; see
`data/PST50/README.md` for details and the upstream project for the full
release.

The repository ships the released PETAL predictions for the paired protocol in
`./data/PST50/paired_outputs`, which reproduce the table below:

```bash
uv run python benchmark_pst50.py \
    --pst50_root ./data/PST50 \
    --pred_dir ./data/PST50/paired_outputs
```

| Method | LPIPS ↓ | PSNR ↑ | SSIM ↑ | H-corr (RGB) ↑ |
|---|---|---|---|---|
| PETAL | 0.0978 | 24.64 | 0.9332 | 0.5221 |

Omit `--pred_dir` to run the paired protocol (`content_709` + `paired_style`)
yourself at the original image resolution, writing predictions to
`--output_dir` (default `./outputs/pst50_paired`) first. Either way the script
reports LPIPS (AlexNet), PSNR, SSIM and RGB histogram correlation against
`paired_gt`, per image and averaged, in `<pred_dir>/metrics.txt`.

Predictions must have the same resolution as `paired_gt`; the script raises an
error otherwise. `--allow_gt_resize` opts into resizing the ground truth to the
prediction size, in which case the metrics are computed at the prediction
resolution and the report says so.

## Opensource Plan

- [x] Release the inference scripts
- [x] Release the model weights
- [ ] Release the training scripts

## Project Structure

```bash
./
├── infer.py                 # Inference entry point
├── benchmark_pst50.py       # PST50 paired benchmark
├── models_lut.py            # LUT / AdaIN model
├── models_style_encoder.py  # Photographic style encoder (ViT-L/16)
├── layers/                  # ViT building blocks and helpers (derived from DINOv3)
├── download_weights.sh      # Checkpoint download helper
├── ckpts/                   # Model checkpoints (not tracked)
├── data/
│   ├── PST50/               # PST50 paired benchmark subset
│   └── test_pairs/          # Test pairs (see below)
│       ├── content/         # Content images (1.jpg-24.jpg)
│       ├── reference/       # Matching style references, same file stems
│       └── outputs/         # Inference outputs (1.png-24.png)
├── outputs/                 # Inference / benchmark outputs (not tracked)
└── pyproject.toml           # uv / pip project config
```

## License

The PETAL code in this repository is released under the Apache-2.0 License (see
`LICENSE`).

The ViT building blocks and helpers in `layers/` are derived from
[DINOv3](https://github.com/facebookresearch/dinov3) and are distributed under
the terms of the DINOv3 License Agreement (see `LICENSE-DINOv3.md`). If you
publish results obtained with these components, please acknowledge the use of
DINOv3 as required by that agreement.

## Citation

```bibtex
@inproceedings{zhu2026photographic,
  title     = {Enlightening Photographic Style Transfer with a Self-Supervised Photographic Embedding},
  author    = {Zhu, Chengxuan and Fang, Jiacong and Weng, Shuchen and Lyu, Youwei and Tang, Jiajun and Fan, Qingnan and Xu, Chao and Shi, Boxin},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2026}
}
```
