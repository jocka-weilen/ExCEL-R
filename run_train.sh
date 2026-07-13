#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-voc}"
shift || true

case "$dataset" in
  voc) script="scripts/train_voc.py" ;;
  coco) script="scripts/train_coco.py" ;;
  *) echo "Usage: bash run_train.sh {voc|coco} [CoSeR-CLIP arguments]" >&2; exit 2 ;;
esac

if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT:-29733}" "$script" "$@"
else
  python "$script" "$@"
fi
