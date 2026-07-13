#!/usr/bin/env bash
set -euo pipefail

bash run_train.sh coco \
  --data_folder "${COCO_ROOT:-datasets/MSCOCO2014}" \
  --ablation full \
  --log_tag aaai27_main \
  "$@"
