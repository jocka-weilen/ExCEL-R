import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred,bg_label.type(torch.long)).sum()/(bg_sum + 1e-6)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred,fg_label.type(torch.long)).sum()/(fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5

def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def build_online_labels(
    cams: torch.Tensor,
    class_labels: torch.Tensor,
    foreground_threshold: float = 0.55,
    background_threshold: float = 0.20,
    ignore_index: int = 255,
    output_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Build stopped-gradient CoSeR-CLIP online labels (Eq. 15)."""
    if not 0.0 < background_threshold < foreground_threshold < 1.0:
        raise ValueError("Expected 0 < background_threshold < foreground_threshold < 1")
    with torch.no_grad():
        cams = cams.detach()
        if output_size is not None and cams.shape[-2:] != output_size:
            cams = F.interpolate(cams, size=output_size, mode="bilinear", align_corners=False)
        valid_cams = cams * class_labels[:, :, None, None].to(cams.dtype)
        response, foreground_index = valid_cams.max(dim=1)
        online = torch.full_like(foreground_index, ignore_index, dtype=torch.long)
        online[response <= background_threshold] = 0
        foreground = response >= foreground_threshold
        online[foreground] = foreground_index[foreground] + 1
    return online


def _zero_like_graph(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _selected_bce(
    logits: torch.Tensor, mask: torch.Tensor, target: float
) -> torch.Tensor:
    if not bool(mask.any()):
        return _zero_like_graph(logits)
    selected = logits[mask]
    targets = torch.full_like(selected, target)
    return F.binary_cross_entropy_with_logits(selected, targets)


def coser_clip_loss(
    outputs: Dict[str, torch.Tensor],
    class_labels: torch.Tensor,
    lambda_cam: float = 1.0,
    lambda_mask: float = 1.0,
    lambda_structure: float = 0.1,
    lambda_region: float = 0.05,
    foreground_threshold: float = 0.55,
    background_threshold: float = 0.20,
    structure_threshold: float = 0.35,
    positive_support_threshold: float = 0.65,
    low_ownership_threshold: float = 0.25,
    negative_support_threshold: float = 0.30,
    shallow_gap_margin: float = 0.10,
    structure_ranking_margin: float = 0.10,
    query_overlap_margin: float = 0.20,
    query_usage_margin: float = 0.02,
    query_usage_weight: float = 1.0,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """Compute the complete CoSeR-CLIP objective (Eqs. 14--19)."""
    labels = class_labels.float()
    positive_mask = labels > 0.5
    hard_negative_mask = outputs["hard_negative_mask"].bool()

    classification_loss = F.binary_cross_entropy_with_logits(
        outputs["class_logits"], labels
    )
    routing_positive = _selected_bce(outputs["routing_logits"], positive_mask, 1.0)
    routing_negative = _selected_bce(
        outputs["routing_logits"], hard_negative_mask, 0.0
    )
    cam_loss = routing_positive + routing_negative

    seg_logits = outputs["seg_logits"]
    online_labels = build_online_labels(
        outputs["cams"],
        labels,
        foreground_threshold=foreground_threshold,
        background_threshold=background_threshold,
        ignore_index=ignore_index,
        output_size=seg_logits.shape[-2:],
    )
    valid = online_labels != ignore_index
    if bool(valid.any()):
        log_probability = F.log_softmax(seg_logits, dim=1)
        safe_labels = online_labels.masked_fill(~valid, 0)
        selected = torch.gather(log_probability, 1, safe_labels.unsqueeze(1)).squeeze(1)
        mask_loss = -(selected * valid.to(selected.dtype)).sum() / valid.sum().clamp_min(1)
    else:
        mask_loss = _zero_like_graph(seg_logits)

    structure = outputs["structure_map"]
    target_support = outputs["target_support"]
    middle_prior = outputs["middle_prior"]
    confusion_support = outputs["confusion_support"]
    shallow_gap = outputs["shallow_gap"]
    positive_evidence = outputs["positive_evidence"]
    negative_evidence = outputs["negative_evidence"]
    significant = structure >= structure_threshold
    positive_locations = (
        significant
        & (target_support >= positive_support_threshold)
        & (shallow_gap >= 0.0)
        & positive_mask[:, :, None, None]
    )
    negative_locations = (
        significant
        & (middle_prior <= low_ownership_threshold)
        & (confusion_support >= negative_support_threshold)
        & (shallow_gap <= -shallow_gap_margin)
        & positive_mask[:, :, None, None]
    )
    positive_ranking = F.relu(
        structure_ranking_margin + negative_evidence - positive_evidence
    )
    negative_ranking = F.relu(
        structure_ranking_margin + positive_evidence - negative_evidence
    )
    positive_count = positive_locations.flatten(2).sum(-1)
    negative_count = negative_locations.flatten(2).sum(-1)
    positive_term = (
        positive_ranking * positive_locations.to(positive_ranking.dtype)
    ).flatten(2).sum(-1) / positive_count.clamp_min(1)
    negative_term = (
        negative_ranking * negative_locations.to(negative_ranking.dtype)
    ).flatten(2).sum(-1) / negative_count.clamp_min(1)
    structure_loss = ((positive_term + negative_term) * positive_mask).sum()
    structure_loss = structure_loss / positive_mask.sum().clamp_min(1)

    spatial_assignments = outputs["region_assignment"].transpose(1, 2)
    normalized_assignments = F.normalize(
        spatial_assignments, p=2, dim=-1, eps=1e-6
    )
    assignment_overlap = torch.einsum(
        "bkl,bjl->bkj", normalized_assignments, normalized_assignments
    )
    query_count = assignment_overlap.shape[-1]
    off_diagonal = ~torch.eye(
        query_count, device=assignment_overlap.device, dtype=torch.bool
    ).unsqueeze(0)
    query_overlap_loss = F.relu(
        assignment_overlap - query_overlap_margin
    )[off_diagonal.expand_as(assignment_overlap)].mean()
    mean_assignment = outputs["region_assignment"].mean(dim=1)
    usage_loss = F.relu(query_usage_margin - mean_assignment).mean()
    region_loss = query_overlap_loss + query_usage_weight * usage_loss

    total = (
        classification_loss
        + lambda_cam * cam_loss
        + lambda_mask * mask_loss
        + lambda_structure * structure_loss
        + lambda_region * region_loss
    )
    return {
        "loss": total,
        "classification_loss": classification_loss,
        "cam_loss": cam_loss,
        "routing_positive_loss": routing_positive,
        "routing_negative_loss": routing_negative,
        "mask_loss": mask_loss,
        "structure_loss": structure_loss,
        "region_loss": region_loss,
        "query_overlap_loss": query_overlap_loss,
        "usage_loss": usage_loss,
        "online_labels": online_labels,
    }
