#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:?Usage: bash infer_seg_voc.sh CHECKPOINT [extra arguments]}"
shift
python tools/evaluate_coser_clip.py --dataset_name pascal_voc --checkpoint "$checkpoint" "$@"
