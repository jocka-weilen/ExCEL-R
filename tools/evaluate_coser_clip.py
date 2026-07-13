"""Evaluate CoSeR-CLIP CAM and segmentation mIoU on VOC or COCO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import coco, voc
from model.coser_core import max_normalize
from model.losses import build_online_labels
from model.model_coser_clip import CoSeRCLIP, load_coser_checkpoint
from utils import evaluate
from utils.pyutils import format_tabs


def parser(default_dataset=None):
    p = argparse.ArgumentParser(description="Evaluate CoSeR-CLIP")
    p.add_argument("--checkpoint", "--model_path", dest="checkpoint", required=True)
    p.add_argument("--dataset_name", default=default_dataset, choices=["pascal_voc", "ms_coco"])
    p.add_argument("--data_folder", default=None)
    p.add_argument("--list_folder", default=None)
    p.add_argument("--infer_set", default=None)
    p.add_argument("--resize_size", default=320, type=int)
    p.add_argument("--scales", default="0.75,1.0,1.25,1.5")
    p.add_argument("--foreground_threshold", default=None, type=float)
    p.add_argument("--background_threshold", default=None, type=float)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--output", default=None, help="JSON result path")
    return p


def resolve_device(name):
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("auto", "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_config(checkpoint):
    raw = torch.load(checkpoint, map_location="cpu")
    return raw.get("args", {}) if isinstance(raw, dict) else {}


def build_model(config, dataset_name, device, resize_size):
    num_classes = 21 if dataset_name == "pascal_voc" else 81
    keys = {
        "clip_model": config.get("model", "CoSeR-CLIP_ViT-B/16"),
        "embedding_dim": config.get("embedding_dim", 256),
        "in_channels": config.get("in_channels", 768),
        "dataset_name": dataset_name,
        "num_classes": config.get("num_classes", num_classes),
        "img_size": resize_size,
        "mode": "val",
        "device": str(device),
        "shallow_layer": config.get("shallow_layer", 2),
        "middle_layer": config.get("middle_layer", 6),
        "deep_layer": config.get("deep_layer", 11),
        "region_dim": config.get("region_dim", 256),
        "num_region_queries": config.get("num_region_queries", 12),
        "topk_confusions": config.get("topk_confusions", 3),
        "topk_routing_negatives": config.get("topk_routing_negatives", 3),
        "topq_ratio": config.get("topq_ratio", 0.10),
        "warmup_iters": config.get("warmup_iters", 1000),
        "classifier_threshold": config.get("classifier_threshold", 0.35),
        "graph_temperature": config.get("graph_temperature", 0.20),
        "query_temperature": config.get("query_temperature", 0.10),
        "confusion_cue_temperature": config.get("confusion_cue_temperature", 0.20),
        "activation_temperature": config.get("activation_temperature", 0.20),
        "ownership_temperature": config.get("ownership_temperature", 0.10),
        "shallow_temperature": config.get("shallow_temperature", 0.10),
        "routing_temperature": config.get("routing_temperature", 0.20),
        "graph_spatial_weight": config.get("graph_spatial_weight", 1.0),
        "assignment_spatial_weight": config.get("assignment_spatial_weight", 1.0),
        "confusion_text_weight": config.get("confusion_text_weight", 1.0 / 3.0),
        "confusion_visual_weight": config.get("confusion_visual_weight", 1.0 / 3.0),
        "confusion_overlap_weight": config.get("confusion_overlap_weight", 1.0 / 3.0),
        "shallow_margin": config.get("shallow_gap_margin", 0.10),
        "structure_mix": config.get("structure_mix", 0.50),
        "ablation": config.get("ablation", "full"),
    }
    return CoSeRCLIP(**keys)


def build_dataset(args, config):
    if args.dataset_name == "pascal_voc":
        data_folder = args.data_folder or config.get("data_folder", "datasets/VOC2012")
        list_folder = args.list_folder or config.get("list_folder", "datasets/voc")
        split = args.infer_set or "val"
        dataset = voc.VOC12SegDataset(
            root_dir=data_folder, name_list_dir=list_folder, split=split, stage="val", aug=False
        )
        return dataset, 21, voc.class_list
    data_folder = args.data_folder or config.get("data_folder", "datasets/MSCOCO2014")
    list_folder = args.list_folder or config.get("list_folder", "datasets/coco")
    split = args.infer_set or "val_part"
    dataset = coco.CocoSegDataset(
        root_dir=data_folder, name_list_dir=list_folder, split=split, stage="val", aug=False
    )
    return dataset, 81, coco.class_list


def serializable_score(score):
    result = {}
    for metric, values in score.items():
        if isinstance(values, dict):
            result[metric] = {str(key): float(value) for key, value in values.items()}
        else:
            result[metric] = float(values)
    return result


def evaluate_model(args):
    config = checkpoint_config(args.checkpoint)
    if args.dataset_name is None:
        args.dataset_name = config.get("dataset_name", "pascal_voc")
    foreground_threshold = args.foreground_threshold
    if foreground_threshold is None:
        foreground_threshold = config.get("foreground_threshold", 0.55)
    background_threshold = args.background_threshold
    if background_threshold is None:
        background_threshold = config.get("background_threshold", 0.20)

    device = resolve_device(args.device)
    dataset, num_classes, categories = build_dataset(args, config)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    model = build_model(config, args.dataset_name, device, args.resize_size).to(device)
    _, incompatible = load_coser_checkpoint(model, args.checkpoint, strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if not name.startswith("encoder.")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model.eval()

    scales = [float(value) for value in args.scales.split(",") if value]
    ground_truths, cam_predictions, seg_predictions = [], [], []
    with torch.no_grad():
        for _, images, labels, class_labels in tqdm(loader, ncols=100, ascii=" >="):
            images = images.to(device).float()
            class_labels = class_labels.to(device).float()
            target_size = labels.shape[-2:]
            cam_scales, seg_scales = [], []
            for scale in scales:
                side = max(16, int(round(args.resize_size * scale / 16.0)) * 16)
                scaled = F.interpolate(images, size=(side, side), mode="bilinear", align_corners=False)
                paired = torch.cat((scaled, scaled.flip(-1)), dim=0)
                paired_labels = class_labels.repeat(2, 1)
                outputs = model(paired, class_labels=paired_labels)
                cams = outputs["cams"]
                cams = 0.5 * (cams[:1] + cams[1:].flip(-1))
                cams = F.interpolate(cams, size=target_size, mode="bilinear", align_corners=False)
                cam_scales.append(cams)
                seg = outputs["seg_logits"]
                seg = 0.5 * (seg[:1] + seg[1:].flip(-1))
                seg_scales.append(
                    F.interpolate(seg, size=target_size, mode="bilinear", align_corners=False)
                )
            cams = max_normalize(torch.stack(cam_scales).mean(0))
            seg_logits = torch.stack(seg_scales).mean(0)
            cam_labels = build_online_labels(
                cams,
                class_labels,
                foreground_threshold=foreground_threshold,
                background_threshold=background_threshold,
            )
            cam_labels.masked_fill_(cam_labels == 255, 0)
            cam_predictions.extend(cam_labels.cpu().numpy().astype(np.int16))
            seg_predictions.extend(seg_logits.argmax(1).cpu().numpy().astype(np.int16))
            ground_truths.extend(labels.cpu().numpy().astype(np.int16))

    cam_score = evaluate.scores(ground_truths, cam_predictions, num_classes=num_classes)
    seg_score = evaluate.scores(ground_truths, seg_predictions, num_classes=num_classes)
    print(format_tabs([cam_score, seg_score], ["CoSeR_CAM", "CoSeR_Seg"], categories))
    result = {
        "method": "CoSeR-CLIP",
        "version": "11.5-NT",
        "dataset": args.dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "scales": scales,
        "foreground_threshold": foreground_threshold,
        "background_threshold": background_threshold,
        "cam_miou": float(cam_score["miou"]),
        "seg_miou": float(seg_score["miou"]),
        "cam": serializable_score(cam_score),
        "segmentation": serializable_score(seg_score),
    }
    output = args.output or str(Path(args.checkpoint).with_suffix(".metrics.json"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {output}")
    return result


def main(default_dataset=None):
    evaluate_model(parser(default_dataset).parse_args())


if __name__ == "__main__":
    main()
