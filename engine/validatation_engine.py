"""Validation for CoSeR-CLIP CAMs and the visual-only segmentation branch."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import coco, voc
from model.losses import build_online_labels
from utils import evaluate
from utils.pyutils import format_tabs


def build_validation(
    model=None,
    par=None,
    val_loader=None,
    device="cuda",
    num_classes=21,
    foreground_threshold=0.55,
    background_threshold=0.20,
):
    del par  # CoSeR-CLIP uses its calibrated CAMs directly for online supervision.
    ground_truths, cam_predictions, seg_predictions = [], [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader), ncols=100, ascii=" >="):
            _, inputs, labels, class_labels = data
            inputs = F.interpolate(inputs, size=(320, 320), mode="bilinear", align_corners=False)
            inputs = inputs.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            outputs = model(inputs, class_labels=class_labels)

            resized_seg = F.interpolate(
                outputs["seg_logits"], size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            cam_labels = build_online_labels(
                outputs["cams"],
                class_labels,
                foreground_threshold=foreground_threshold,
                background_threshold=background_threshold,
                output_size=labels.shape[-2:],
            )
            cam_labels = cam_labels.masked_fill(cam_labels == 255, 0)
            cam_predictions.extend(cam_labels.cpu().numpy().astype(np.int16))
            seg_predictions.extend(
                resized_seg.argmax(dim=1).cpu().numpy().astype(np.int16)
            )
            ground_truths.extend(labels.cpu().numpy().astype(np.int16))

    cam_score = evaluate.scores(ground_truths, cam_predictions, num_classes=num_classes)
    seg_score = evaluate.scores(ground_truths, seg_predictions, num_classes=num_classes)
    if was_training:
        model.train()
    categories = voc.class_list if num_classes == 21 else coco.class_list
    return format_tabs(
        [cam_score, seg_score],
        name_list=["CoSeR_CAM", "CoSeR_Seg"],
        cat_list=categories,
    )
