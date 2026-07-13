"""Unified VOC/COCO trainer for CoSeR-CLIP."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import coco, voc
from engine import build_network, build_optimizer, build_validation
from model.losses import coser_clip_loss
from model.model_coser_clip import load_coser_checkpoint
from utils.pyutils import AverageMeter, cal_eta, setup_logger


def build_parser(default_dataset: str = "pascal_voc") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CoSeR-CLIP")
    parser.add_argument("--method", default="CoSeR-CLIP", choices=["CoSeR-CLIP"])
    parser.add_argument("--dataset_name", default=default_dataset, choices=["pascal_voc", "ms_coco"])
    parser.add_argument("--model", default="CoSeR-CLIP_ViT-B/16")
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--in_channels", default=768, type=int)
    parser.add_argument("--crop_size", default=320, type=int)
    parser.add_argument("--num_classes", default=None, type=int)

    parser.add_argument("--shallow_layer", default=2, type=int)
    parser.add_argument("--middle_layer", default=6, type=int)
    parser.add_argument("--deep_layer", default=11, type=int)
    parser.add_argument("--region_dim", default=256, type=int)
    parser.add_argument("--num_region_queries", default=12, type=int)
    parser.add_argument("--topk_confusions", default=3, type=int)
    parser.add_argument("--topq_ratio", default=0.10, type=float)
    parser.add_argument("--classifier_threshold", default=0.35, type=float)
    parser.add_argument("--graph_temperature", default=0.20, type=float)
    parser.add_argument("--query_temperature", default=0.10, type=float)
    parser.add_argument("--confusion_cue_temperature", default=0.20, type=float)
    parser.add_argument("--negative_temperature", default=0.20, type=float)
    parser.add_argument("--activation_temperature", default=0.20, type=float)
    parser.add_argument("--ownership_temperature", default=0.10, type=float)
    parser.add_argument("--shallow_temperature", default=0.10, type=float)
    parser.add_argument("--routing_temperature", default=0.20, type=float)
    parser.add_argument("--graph_spatial_weight", default=1.0, type=float)
    parser.add_argument("--assignment_spatial_weight", default=1.0, type=float)
    parser.add_argument("--confusion_text_weight", default=1.0 / 3.0, type=float)
    parser.add_argument("--confusion_visual_weight", default=1.0 / 3.0, type=float)
    parser.add_argument("--confusion_overlap_weight", default=1.0 / 3.0, type=float)
    parser.add_argument("--structure_mix", default=0.50, type=float)
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full", "deep_only", "no_confusion", "no_structure", "no_negative_routing"],
    )

    parser.add_argument("--lambda_cam", default=1.0, type=float)
    parser.add_argument("--lambda_mask", default=1.0, type=float)
    parser.add_argument("--lambda_structure", default=0.10, type=float)
    parser.add_argument("--lambda_region", default=0.05, type=float)
    parser.add_argument("--foreground_threshold", default=0.55, type=float)
    parser.add_argument("--background_threshold", default=0.20, type=float)
    parser.add_argument("--structure_threshold", default=0.35, type=float)
    parser.add_argument("--positive_support_threshold", default=0.65, type=float)
    parser.add_argument("--low_ownership_threshold", default=0.25, type=float)
    parser.add_argument("--negative_support_threshold", default=0.30, type=float)
    parser.add_argument("--shallow_gap_margin", default=0.10, type=float)
    parser.add_argument("--structure_ranking_margin", default=0.10, type=float)
    parser.add_argument("--diversity_margin", default=0.80, type=float)
    parser.add_argument("--query_usage_margin", default=0.02, type=float)
    parser.add_argument("--query_usage_weight", default=1.0, type=float)

    parser.add_argument("--max_iters", default=None, type=int)
    parser.add_argument("--warmup_iters", default=1000, type=int)
    parser.add_argument("--log_iters", default=50, type=int)
    parser.add_argument("--eval_iters", default=None, type=int)
    parser.add_argument("--save_iters", default=None, type=int)
    parser.add_argument("--batch_size", "--spg", dest="spg", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--optimizer", default="PolyWarmupAdamW")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--warmup_lr", default=0.01, type=float)
    parser.add_argument("--wt_decay", default=1e-2, type=float)
    parser.add_argument("--betas", default=(0.9, 0.999))
    parser.add_argument("--power", default=1.0, type=float)
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--list_folder", default=None)
    parser.add_argument("--train_set", default=None)
    parser.add_argument("--val_set", default=None)
    parser.add_argument("--ignore_index", default=255, type=int)
    parser.add_argument("--work_dir", default="w_outputs/coser_clip")
    parser.add_argument("--log_tag", default="full")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--skip_validation", action="store_true")
    return parser


def apply_dataset_defaults(args):
    if args.dataset_name == "pascal_voc":
        defaults = {
            "num_classes": 21,
            "max_iters": 30000,
            "eval_iters": 2000,
            "save_iters": 2000,
            "data_folder": "datasets/VOC2012",
            "list_folder": "datasets/voc",
            "train_set": "train_aug",
            "val_set": "val",
        }
    else:
        defaults = {
            "num_classes": 81,
            "max_iters": 100000,
            "eval_iters": 5000,
            "save_iters": 5000,
            "data_folder": "datasets/MSCOCO2014",
            "list_folder": "datasets/coco",
            "train_set": "train",
            "val_set": "val_part",
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def resolve_runtime(args):
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.device in ("auto", "cuda") and torch.cuda.is_available():
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend=args.backend)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
    elif args.device in ("auto", "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        local_rank = -1
    else:
        device = torch.device("cpu")
        local_rank = -1
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    args.local_rank = local_rank
    args.model_device = str(device)
    return device, rank, world_size


def create_datasets(args):
    common = dict(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage="train",
        aug=True,
        rescale_range=[0.5, 2.0],
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    if args.dataset_name == "pascal_voc":
        train_dataset = voc.VOC12ClsDataset(**common)
        val_dataset = voc.VOC12SegDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.val_set,
            stage="val",
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    else:
        train_dataset = coco.CocoClsDataset(**common)
        val_dataset = coco.CocoSegDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.val_set,
            stage="val",
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    return train_dataset, val_dataset


def save_checkpoint(model, optimizer, iteration, args, path):
    unwrapped = model.module if hasattr(model, "module") else model
    trainable_state = {
        name: value.detach().cpu()
        for name, value in unwrapped.state_dict().items()
        if not name.startswith("encoder.")
    }
    torch.save(
        {
            "method": "CoSeR-CLIP",
            "iteration": iteration,
            "model": trainable_state,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def train(args):
    args = apply_dataset_defaults(args)
    setup_seed(args.seed)
    device, rank, world_size = resolve_runtime(args)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.dataset_name}_{args.log_tag}_{args.ablation}_{timestamp}"
    args.work_dir = os.path.join(args.work_dir, run_name)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        setup_logger(filename=os.path.join(args.work_dir, "train.log"))
        with open(os.path.join(args.work_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)
        logging.info("Method: CoSeR-CLIP | device=%s | world_size=%d", device, world_size)

    train_dataset, val_dataset = create_datasets(args)
    sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        sampler=sampler,
        shuffle=sampler is None,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(**loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model, parameter_groups = build_network(args)
    model.to(device)
    optimizer = build_optimizer(args, parameter_groups)
    start_iteration = 0
    if args.resume:
        checkpoint, incompatible = load_coser_checkpoint(model, args.resume, strict=False)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_iteration = int(checkpoint.get("iteration", 0))
        if rank == 0:
            logging.info("Resumed at iteration %d; incompatible=%s", start_iteration, incompatible)
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    meter = AverageMeter()
    start_time = datetime.datetime.now().replace(microsecond=0)
    epoch = 0
    iterator = iter(train_loader)

    for iteration in range(start_iteration, args.max_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        images = batch[1].to(device, non_blocking=True).float()
        class_labels = batch[2].to(device, non_blocking=True).float()

        optimizer.zero_grad()
        with autocast(amp_enabled):
            outputs = model(images, class_labels=class_labels, global_step=iteration)
            losses = coser_clip_loss(
                outputs,
                class_labels,
                lambda_cam=args.lambda_cam,
                lambda_mask=args.lambda_mask,
                lambda_structure=(
                    0.0 if args.ablation in ("deep_only", "no_structure")
                    else args.lambda_structure
                ),
                lambda_region=(0.0 if args.ablation == "deep_only" else args.lambda_region),
                foreground_threshold=args.foreground_threshold,
                background_threshold=args.background_threshold,
                structure_threshold=args.structure_threshold,
                positive_support_threshold=args.positive_support_threshold,
                low_ownership_threshold=args.low_ownership_threshold,
                negative_support_threshold=args.negative_support_threshold,
                shallow_gap_margin=args.shallow_gap_margin,
                structure_ranking_margin=args.structure_ranking_margin,
                diversity_margin=args.diversity_margin,
                query_usage_margin=args.query_usage_margin,
                query_usage_weight=args.query_usage_weight,
                ignore_index=args.ignore_index,
            )
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            args.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        meter.add({name: value.item() for name, value in losses.items() if name.endswith("loss")})
        current = iteration + 1
        if rank == 0 and current % args.log_iters == 0:
            elapsed, eta = cal_eta(start_time, current, args.max_iters)
            names = ["loss", "classification_loss", "cam_loss", "mask_loss", "structure_loss", "region_loss"]
            metrics = " | ".join(f"{name}={meter.pop(name):.4f}" for name in names)
            logging.info(
                "iter=%d/%d | lr=%.3e | %s | elapsed=%s | eta=%s",
                current, args.max_iters, optimizer.param_groups[2]["lr"], metrics, elapsed, eta,
            )

        if current % args.save_iters == 0 and rank == 0:
            save_checkpoint(
                model, optimizer, current, args,
                os.path.join(args.ckpt_dir, f"coser_clip_iter_{current}.pth"),
            )
        should_validate = not args.skip_validation and current % args.eval_iters == 0
        if should_validate:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                validation_model = model.module if hasattr(model, "module") else model
                table = build_validation(
                    model=validation_model,
                    val_loader=val_loader,
                    device=device,
                    num_classes=args.num_classes,
                    foreground_threshold=args.foreground_threshold,
                    background_threshold=args.background_threshold,
                )
                logging.info("\n%s", table)
            if dist.is_initialized():
                dist.barrier()

    if rank == 0:
        final_path = os.path.join(args.ckpt_dir, "coser_clip_final.pth")
        save_checkpoint(model, optimizer, args.max_iters, args, final_path)
        logging.info("Training complete: %s", final_path)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(default_dataset: str = "pascal_voc"):
    train(build_parser(default_dataset).parse_args())


if __name__ == "__main__":
    main()
