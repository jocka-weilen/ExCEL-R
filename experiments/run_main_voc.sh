#!/usr/bin/env bash
set -euo pipefail

bash run_train.sh voc \
  --data_folder "${VOC_ROOT:-datasets/VOC2012}" \
  --ablation full \
  --log_tag aaai27_main \
  "$@"
