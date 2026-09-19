#!/usr/bin/env bash
#
# Download the PETAL inference checkpoints into ./ckpts.
#
# The released weights are hosted on the Hugging Face Hub, inside the `ckpts/`
# folder of the model repository:
#
#   https://huggingface.co/JiacongFang/PETAL
#
# Usage:
#
#   ./download_weights.sh
#   CKPT_DIR=/data/petal_ckpts ./download_weights.sh
#
# Environment variables:
#
#   HF_REPO_ID      Repository hosting the weights   (default: JiacongFang/PETAL)
#   HF_REPO_SUBDIR  Folder inside the repository     (default: ckpts)
#   HF_REVISION     Revision to download             (default: main)
#   HF_TOKEN        Optional access token
#   CKPT_DIR        Destination directory            (default: <script dir>/ckpts)
set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-JiacongFang/PETAL}"
HF_REPO_SUBDIR="${HF_REPO_SUBDIR:-ckpts}"
HF_REVISION="${HF_REVISION:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${CKPT_DIR:-${SCRIPT_DIR}/ckpts}"

FILES=(
  "style_encoder.pt"
  "lut_model.pt"
)

mkdir -p "${CKPT_DIR}"

# ---------------------------------------------------------------------------
# Which checkpoints are still missing?
# ---------------------------------------------------------------------------
pending=()
for file in "${FILES[@]}"; do
  if [[ -s "${CKPT_DIR}/${file}" ]]; then
    echo "Already present: ${CKPT_DIR}/${file}"
  else
    pending+=("${file}")
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  echo "Checkpoints ready in ${CKPT_DIR}"
  exit 0
fi

echo "Downloading ${#pending[@]} file(s) from ${HF_REPO_ID}@${HF_REVISION} (${HF_REPO_SUBDIR}/)"

HF_CLI=""
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
fi

if [[ -n "${HF_CLI}" ]]; then
  # `--local-dir` mirrors the repository layout, so download into a staging
  # directory on the same filesystem and flatten `<subdir>/<file>` afterwards.
  STAGE_DIR="$(mktemp -d "${CKPT_DIR}/.hf-download.XXXXXX")"
  trap 'rm -rf "${STAGE_DIR}"' EXIT

  include_args=()
  for file in "${pending[@]}"; do
    include_args+=(--include "${HF_REPO_SUBDIR}/${file}")
  done

  if ! "${HF_CLI}" download "${HF_REPO_ID}" \
      --repo-type model \
      --revision "${HF_REVISION}" \
      --local-dir "${STAGE_DIR}" \
      "${include_args[@]}"; then
    echo "error: '${HF_CLI} download' failed." >&2
    echo "       Check the repository id and your network connection." >&2
    exit 1
  fi

  for file in "${pending[@]}"; do
    staged="${STAGE_DIR}/${HF_REPO_SUBDIR}/${file}"
    if [[ ! -s "${staged}" ]]; then
      echo "error: ${staged} was not downloaded; check HF_REPO_ID/HF_REPO_SUBDIR." >&2
      exit 1
    fi
    mv -f "${staged}" "${CKPT_DIR}/${file}"
  done
else
  echo "Neither 'hf' nor 'huggingface-cli' found; falling back to curl."

  curl_args=(-fL --retry 3 --retry-delay 2 --continue-at -)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi

  for file in "${pending[@]}"; do
    url="https://huggingface.co/${HF_REPO_ID}/resolve/${HF_REVISION}/${HF_REPO_SUBDIR}/${file}"
    echo "Downloading ${url}"
    curl "${curl_args[@]}" -o "${CKPT_DIR}/${file}.part" "${url}"
    mv -f "${CKPT_DIR}/${file}.part" "${CKPT_DIR}/${file}"
  done
fi

for file in "${FILES[@]}"; do
  if [[ ! -s "${CKPT_DIR}/${file}" ]]; then
    echo "error: ${CKPT_DIR}/${file} is missing after download." >&2
    exit 1
  fi
done

echo "Checkpoints ready in ${CKPT_DIR}"
