#!/usr/bin/env bash
set -euo pipefail

data_root="${VOC_ROOT:-datasets/VOC2012}"
for ablation in deep_only no_confusion no_structure no_negative_routing full; do
  bash run_train.sh voc \
    --data_folder "$data_root" \
    --ablation "$ablation" \
    --log_tag "aaai27_ablation" \
    "$@"
done
