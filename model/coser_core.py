"""Trainable CoSeR-CLIP evidence construction and routing modules.

The implementation follows the equations in ``CoSeR_CLIP_Method_v11_CN_AAAI27``:
deep semantic anchors, confusion-aware middle-level region ownership, shallow
structure/semantic evidence, and pixel-wise signed evidence routing.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def max_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Maximum-normalize a non-negative spatial tensor with a detached scale."""
    maximum = x.flatten(-2).amax(dim=-1, keepdim=True).detach()
    return x / (maximum.unsqueeze(-1) + eps)


def _flat_max_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    maximum = x.amax(dim=-1, keepdim=True).detach()
    return x / (maximum + eps)


def _resize_flat(
    x: torch.Tensor,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    if source_size == target_size:
        return x
    shape = x.shape
    x = x.reshape(-1, 1, *source_size)
    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    return x.reshape(*shape[:-1], target_size[0] * target_size[1])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class ProjectionAdapter(nn.Module):
    """LayerNorm plus a linear projection, matching CLIP's final visual map."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x.float()))

    @torch.no_grad()
    def initialize_from_clip(self, norm: nn.Module, projection: torch.Tensor) -> None:
        self.norm.weight.copy_(norm.weight.float())
        self.norm.bias.copy_(norm.bias.float())
        self.proj.weight.copy_(projection.float().t())


class CoSeRCore(nn.Module):
    """CoSeR-CLIP's class-conditioned signed evidence router.

    This module is independent of the CLIP loader and segmentation decoder, which
    makes the method logic directly unit-testable with synthetic multi-layer tokens.
    """

    def __init__(
        self,
        feature_dim: int,
        text_dim: int,
        region_dim: int = 256,
        num_region_queries: int = 12,
        topk_confusions: int = 3,
        topq_ratio: float = 0.1,
        cls_temperature: float = 0.07,
        graph_temperature: float = 0.2,
        query_temperature: float = 0.1,
        confusion_cue_temperature: float = 0.2,
        negative_temperature: float = 0.2,
        activation_temperature: float = 0.2,
        ownership_temperature: float = 0.1,
        shallow_temperature: float = 0.1,
        routing_temperature: float = 0.2,
        graph_spatial_weight: float = 1.0,
        assignment_spatial_weight: float = 1.0,
        confusion_text_weight: float = 1.0 / 3.0,
        confusion_visual_weight: float = 1.0 / 3.0,
        confusion_overlap_weight: float = 1.0 / 3.0,
        shallow_margin: float = 0.1,
        structure_mix: float = 0.5,
        routing_scale_max: float = 10.0,
        classifier_threshold: float = 0.35,
        warmup_iters: int = 1000,
        ablation: str = "full",
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_region_queries < 2:
            raise ValueError("num_region_queries must be at least 2")
        if topk_confusions < 1:
            raise ValueError("topk_confusions must be positive")
        if not 0.0 < topq_ratio <= 1.0:
            raise ValueError("topq_ratio must be in (0, 1]")
        temperatures = {
            "cls_temperature": cls_temperature,
            "graph_temperature": graph_temperature,
            "query_temperature": query_temperature,
            "confusion_cue_temperature": confusion_cue_temperature,
            "negative_temperature": negative_temperature,
            "activation_temperature": activation_temperature,
            "ownership_temperature": ownership_temperature,
            "shallow_temperature": shallow_temperature,
            "routing_temperature": routing_temperature,
        }
        if any(value <= 0.0 for value in temperatures.values()):
            raise ValueError(f"All temperatures must be positive: {temperatures}")
        if not 0.0 <= structure_mix <= 1.0:
            raise ValueError("structure_mix must be in [0, 1]")
        if not 0.0 < classifier_threshold < 1.0:
            raise ValueError("classifier_threshold must be in (0, 1)")
        valid_ablations = {
            "full", "deep_only", "no_confusion", "no_structure", "no_negative_routing"
        }
        if ablation not in valid_ablations:
            raise ValueError(f"Unknown ablation '{ablation}'. Choose from {sorted(valid_ablations)}")

        confusion_sum = (
            confusion_text_weight + confusion_visual_weight + confusion_overlap_weight
        )
        if min(
            confusion_text_weight, confusion_visual_weight, confusion_overlap_weight
        ) < 0 or confusion_sum <= 0:
            raise ValueError("Confusion cue weights must be non-negative and not all zero")

        self.num_region_queries = num_region_queries
        self.topk_confusions = topk_confusions
        self.topq_ratio = topq_ratio
        self.cls_temperature = cls_temperature
        self.graph_temperature = graph_temperature
        self.query_temperature = query_temperature
        self.confusion_cue_temperature = confusion_cue_temperature
        self.negative_temperature = negative_temperature
        self.activation_temperature = activation_temperature
        self.ownership_temperature = ownership_temperature
        self.shallow_temperature = shallow_temperature
        self.routing_temperature = routing_temperature
        self.graph_spatial_weight = graph_spatial_weight
        self.assignment_spatial_weight = assignment_spatial_weight
        self.confusion_weights = (
            confusion_text_weight / confusion_sum,
            confusion_visual_weight / confusion_sum,
            confusion_overlap_weight / confusion_sum,
        )
        self.shallow_margin = shallow_margin
        self.structure_mix = structure_mix
        self.routing_scale_max = routing_scale_max
        self.classifier_threshold = classifier_threshold
        self.warmup_iters = warmup_iters
        self.ablation = ablation
        self.eps = eps

        self.deep_adapter = ProjectionAdapter(feature_dim, text_dim)
        self.global_adapter = ProjectionAdapter(feature_dim, text_dim)
        self.shallow_semantic_probe = ProjectionAdapter(feature_dim, text_dim)
        for parameter in self.shallow_semantic_probe.parameters():
            parameter.requires_grad = False

        self.shallow_structure_projection = nn.Linear(feature_dim, region_dim, bias=False)
        self.middle_projection = nn.Linear(feature_dim, region_dim, bias=False)
        self.region_queries = nn.Parameter(torch.empty(num_region_queries, region_dim))
        self.graph_projection = nn.Linear(region_dim, region_dim, bias=False)
        self.region_norm = nn.LayerNorm(region_dim)

        self.router = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 4, kernel_size=1),
        )

        self.raw_alpha_d = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_alpha_m = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_alpha_s = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_alpha_n = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.negative_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.raw_negative_scale = nn.Parameter(torch.tensor(-1.3862944))  # sigmoid = 0.2
        self.raw_routing_scale = nn.Parameter(torch.tensor(0.0))
        self.routing_bias = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.region_queries, std=region_dim ** -0.5)
        nn.init.xavier_uniform_(self.shallow_structure_projection.weight)
        nn.init.xavier_uniform_(self.middle_projection.weight)
        nn.init.xavier_uniform_(self.graph_projection.weight)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    @torch.no_grad()
    def initialize_visual_adapters(
        self, final_norm: nn.Module, visual_projection: torch.Tensor
    ) -> None:
        """Initialize trainable deep/global adapters and the frozen shallow probe."""
        self.deep_adapter.initialize_from_clip(final_norm, visual_projection)
        self.global_adapter.initialize_from_clip(final_norm, visual_projection)
        self.shallow_semantic_probe.initialize_from_clip(final_norm, visual_projection)

    @staticmethod
    def _coordinates(
        size: Tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        h, w = size
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:  # PyTorch 1.9 compatibility
            yy, xx = torch.meshgrid(ys, xs)
        return torch.stack((yy, xx), dim=-1).reshape(h * w, 2)

    def _structure_map(
        self, shallow_tokens: torch.Tensor, shallow_size: Tuple[int, int]
    ) -> torch.Tensor:
        b, length, _ = shallow_tokens.shape
        h, w = shallow_size
        if length != h * w:
            raise ValueError("shallow token count does not match shallow_size")
        projected = self.shallow_structure_projection(shallow_tokens.float())
        projected = projected.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        padded = F.pad(projected, (1, 1, 1, 1), mode="replicate")
        difference = torch.zeros((b, 1, h, w), device=projected.device, dtype=projected.dtype)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = padded[:, :, 1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            difference = difference + (projected - neighbour).pow(2).sum(1, keepdim=True).sqrt()
        difference = difference / 8.0
        return max_normalize(difference, self.eps).flatten(2)

    def _region_reasoning(
        self, middle_tokens: torch.Tensor, middle_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, length, _ = middle_tokens.shape
        coordinates = self._coordinates(middle_size, middle_tokens.device, torch.float32)
        projected = F.normalize(self.middle_projection(middle_tokens.float()), dim=-1)
        queries = F.normalize(self.region_queries.float(), dim=-1)
        initial_assignment = F.softmax(
            torch.einsum("bld,kd->blk", projected, queries) / self.query_temperature,
            dim=-1,
        )
        mass = initial_assignment.sum(dim=1).clamp_min(self.eps)
        nodes = torch.einsum("blk,bld->bkd", initial_assignment, projected) / mass.unsqueeze(-1)
        centroids = torch.einsum("blk,lp->bkp", initial_assignment, coordinates) / mass.unsqueeze(-1)

        normalized_nodes = F.normalize(nodes, dim=-1)
        appearance = torch.einsum("bkd,bjd->bkj", normalized_nodes, normalized_nodes)
        spatial_distance = (centroids[:, :, None] - centroids[:, None, :]).pow(2).sum(-1)
        graph_logits = appearance / self.graph_temperature - self.graph_spatial_weight * spatial_distance
        diagonal = torch.eye(
            self.num_region_queries, device=graph_logits.device, dtype=torch.bool
        ).unsqueeze(0)
        graph_logits = graph_logits.masked_fill(diagonal, torch.finfo(graph_logits.dtype).min)
        adjacency = F.softmax(graph_logits, dim=-1).masked_fill(diagonal, 0.0)
        message = torch.einsum("bkj,bjd->bkd", adjacency, self.graph_projection(nodes))
        updated_nodes = self.region_norm(nodes + message)

        content = torch.einsum(
            "bld,bkd->blk", projected, F.normalize(updated_nodes, dim=-1)
        )
        point_distance = (
            coordinates[None, :, None] - centroids[:, None, :]
        ).pow(2).sum(-1)
        refined_logits = content - self.assignment_spatial_weight * point_distance
        assignment = F.softmax(refined_logits / self.query_temperature, dim=-1)
        return assignment, updated_nodes, initial_assignment

    def _confusion_mining(
        self,
        deep_raw: torch.Tensor,
        deep_maps: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, classes, length = deep_raw.shape
        negative_count = min(self.topk_confusions, classes - 1)
        if negative_count < 1:
            raise ValueError("CoSeR-CLIP requires at least two foreground classes")

        text_similarity = text_features @ text_features.t()
        text_cue = (1.0 + text_similarity).clamp(0.0, 2.0) * 0.5

        topq = max(1, int(math.ceil(self.topq_ratio * length)))
        visual_cue = deep_raw.topk(topq, dim=-1).values.mean(dim=-1)
        overlap = torch.einsum("bcl,bdl->bcd", deep_maps, deep_maps)
        overlap = overlap / deep_maps.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        diagonal = torch.eye(classes, device=deep_raw.device, dtype=torch.bool)[None]
        text_logits = text_cue[None].expand(b, -1, -1)
        visual_logits = visual_cue[:, None].expand(-1, classes, -1)
        lowest = torch.finfo(deep_raw.dtype).min
        text_distribution = F.softmax(
            text_logits.masked_fill(diagonal, lowest) / self.confusion_cue_temperature,
            dim=-1,
        )
        visual_distribution = F.softmax(
            visual_logits.masked_fill(diagonal, lowest) / self.confusion_cue_temperature,
            dim=-1,
        )
        overlap_distribution = F.softmax(
            overlap.masked_fill(diagonal, lowest) / self.confusion_cue_temperature,
            dim=-1,
        )

        wt, wv, wo = self.confusion_weights
        confusion = (
            wt * text_distribution + wv * visual_distribution + wo * overlap_distribution
        ).detach()
        scores, indices = confusion.topk(negative_count, dim=-1)
        weights = F.softmax(scores / self.negative_temperature, dim=-1)
        return confusion, indices, weights

    @staticmethod
    def _gather_classes(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather a class tensor [B,C,...] for per-target indices [B,C,K]."""
        b, classes, negatives = indices.shape
        suffix = values.shape[2:]
        expanded = values[:, None].expand(b, classes, classes, *suffix)
        gather_index = indices.reshape(b, classes, negatives, *([1] * len(suffix)))
        gather_index = gather_index.expand(b, classes, negatives, *suffix)
        return torch.gather(expanded, dim=2, index=gather_index)

    def _routing_masks(
        self,
        class_logits: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        confusion_indices: torch.Tensor,
        competition_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, classes = class_logits.shape
        if class_labels is None:
            active = torch.sigmoid(class_logits) >= self.classifier_threshold
            empty = ~active.any(dim=1)
            if empty.any():
                active[empty, class_logits[empty].argmax(dim=1)] = True
            return active, torch.zeros_like(active)

        positive = class_labels > 0.5
        hard_negative = torch.zeros_like(positive)
        if competition_enabled:
            selected = confusion_indices[positive]
            if selected.numel() > 0:
                batch_index = (
                    torch.arange(b, device=positive.device)[:, None]
                    .expand(-1, classes)[positive]
                    .unsqueeze(-1)
                    .expand_as(selected)
                )
                hard_negative[batch_index.reshape(-1), selected.reshape(-1)] = True
            hard_negative &= ~positive
        return positive | hard_negative, hard_negative

    def forward(
        self,
        shallow_tokens: torch.Tensor,
        middle_tokens: torch.Tensor,
        deep_tokens: torch.Tensor,
        text_features: torch.Tensor,
        shallow_size: Tuple[int, int],
        middle_size: Tuple[int, int],
        deep_size: Tuple[int, int],
        class_labels: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Construct and route CoSeR-CLIP evidence for all foreground classes."""
        text_features = F.normalize(text_features.float(), dim=-1)
        shallow_patch = shallow_tokens[:, 1:]
        middle_patch = middle_tokens[:, 1:]
        deep_patch = deep_tokens[:, 1:]

        deep_embedding = F.normalize(self.deep_adapter(deep_patch), dim=-1)
        deep_raw = F.relu(torch.einsum("bld,cd->bcl", deep_embedding, text_features))
        deep_maps = _flat_max_normalize(deep_raw, self.eps)

        global_embedding = F.normalize(self.global_adapter(deep_tokens[:, 0]), dim=-1)
        class_logits = torch.einsum("bd,cd->bc", global_embedding, text_features)
        class_logits = class_logits / self.cls_temperature

        assignment, region_nodes, initial_assignment = self._region_reasoning(
            middle_patch, middle_size
        )
        confusion, confusion_indices, confusion_weights = self._confusion_mining(
            deep_raw, deep_maps, text_features
        )

        warmed_up = global_step is None or global_step >= self.warmup_iters
        competition_enabled = warmed_up and self.ablation != "no_confusion"
        route_mask, hard_negative_mask = self._routing_masks(
            class_logits, class_labels, confusion_indices, competition_enabled
        )

        deep_on_middle = _resize_flat(deep_maps, deep_size, middle_size)
        region_mass = assignment.sum(dim=1).clamp_min(self.eps)
        positive_support = torch.einsum(
            "blk,bcl->bck", assignment, deep_on_middle
        ) / region_mass[:, None]

        detached_support = torch.einsum(
            "blk,bcl->bck", assignment, deep_on_middle.detach()
        ) / region_mass[:, None]
        negative_support_by_class = self._gather_classes(
            detached_support, confusion_indices
        )
        negative_support = (
            negative_support_by_class * confusion_weights.unsqueeze(-1)
        ).sum(dim=2)

        activation = 1.0 - torch.exp(-positive_support / self.activation_temperature)
        if competition_enabled:
            ownership = activation * torch.sigmoid(
                (positive_support - negative_support) / self.ownership_temperature
            )
        else:
            ownership = activation
            negative_support = torch.zeros_like(negative_support)

        middle_prior = torch.einsum("blk,bck->bcl", assignment, ownership)
        confusion_support = torch.einsum("blk,bck->bcl", assignment, negative_support)

        structure_map = self._structure_map(shallow_patch, shallow_size)
        with torch.no_grad():
            shallow_embedding = F.normalize(
                self.shallow_semantic_probe(shallow_patch), dim=-1
            )
            shallow_response = torch.einsum(
                "bld,cd->bcl", shallow_embedding, text_features
            )
            if competition_enabled:
                negative_shallow = self._gather_classes(
                    shallow_response, confusion_indices
                )
                negative_shallow = (
                    negative_shallow * confusion_weights.unsqueeze(-1)
                ).sum(dim=2)
            else:
                negative_shallow = torch.zeros_like(shallow_response)
            shallow_gap = (
                (shallow_response - negative_shallow) / self.shallow_temperature
            ).detach()

        deep_on_shallow = _resize_flat(deep_maps, deep_size, shallow_size)
        middle_on_shallow = _resize_flat(middle_prior, middle_size, shallow_size)
        negative_on_shallow = _resize_flat(confusion_support, middle_size, shallow_size)
        target_support = (
            deep_on_shallow + middle_on_shallow - deep_on_shallow * middle_on_shallow
        )

        alpha_d = F.softplus(self.raw_alpha_d)
        alpha_m = F.softplus(self.raw_alpha_m)
        alpha_s = F.softplus(self.raw_alpha_s)
        alpha_n = F.softplus(self.raw_alpha_n)
        positive_gate = target_support * torch.sigmoid(
            alpha_d * deep_on_shallow
            + alpha_m * middle_on_shallow
            + alpha_s * shallow_gap
        )
        negative_gate = negative_on_shallow * (1.0 - middle_on_shallow) * torch.sigmoid(
            self.negative_gate_bias
            + alpha_n * F.relu(-shallow_gap - self.shallow_margin)
        )
        if not competition_enabled:
            negative_gate = torch.zeros_like(negative_gate)

        if self.ablation == "no_structure":
            structure_map = torch.zeros_like(structure_map)
        positive_evidence = positive_gate * structure_map
        negative_evidence = negative_gate * (
            self.structure_mix * structure_map + (1.0 - self.structure_mix)
        )
        if self.ablation == "no_negative_routing":
            negative_evidence = torch.zeros_like(negative_evidence)

        router_input = torch.stack(
            (deep_on_shallow, middle_on_shallow, positive_evidence, negative_evidence),
            dim=2,
        )
        b, classes, _, height_width = router_input.shape
        router_logits = self.router(
            router_input.reshape(b * classes, 4, *shallow_size)
        ).reshape(b, classes, 4, height_width)
        positive_weights = F.softmax(router_logits[:, :, :3], dim=2)
        negative_weight = torch.sigmoid(router_logits[:, :, 3])
        negative_scale = torch.sigmoid(self.raw_negative_scale)
        routed_score = (
            positive_weights[:, :, 0] * deep_on_shallow
            + positive_weights[:, :, 1] * middle_on_shallow
            + positive_weights[:, :, 2] * positive_evidence
            - negative_scale * negative_weight * negative_evidence
        )
        if self.ablation == "deep_only":
            routed_score = deep_on_shallow
        routed_score = torch.where(route_mask.unsqueeze(-1), routed_score, deep_on_shallow)
        cams = _flat_max_normalize(F.relu(routed_score), self.eps)

        spatial_count = routed_score.shape[-1]
        pooled_score = self.routing_temperature * (
            torch.logsumexp(routed_score / self.routing_temperature, dim=-1)
            - math.log(spatial_count)
        )
        routing_scale = self.routing_scale_max * torch.sigmoid(self.raw_routing_scale)
        routing_logits = routing_scale * pooled_score + self.routing_bias

        return {
            "cams": cams.reshape(b, classes, *shallow_size),
            "routed_scores": routed_score.reshape(b, classes, *shallow_size),
            "class_logits": class_logits,
            "routing_logits": routing_logits,
            "route_mask": route_mask,
            "hard_negative_mask": hard_negative_mask,
            "deep_maps": deep_on_shallow.reshape(b, classes, *shallow_size),
            "middle_prior": middle_on_shallow.reshape(b, classes, *shallow_size),
            "confusion_support": negative_on_shallow.reshape(b, classes, *shallow_size),
            "target_support": target_support.reshape(b, classes, *shallow_size),
            "positive_evidence": positive_evidence.reshape(b, classes, *shallow_size),
            "negative_evidence": negative_evidence.reshape(b, classes, *shallow_size),
            "structure_map": structure_map.reshape(b, 1, *shallow_size),
            "shallow_gap": shallow_gap.reshape(b, classes, *shallow_size),
            "region_assignment": assignment,
            "initial_region_assignment": initial_assignment,
            "region_nodes": region_nodes,
            "confusion_scores": confusion,
            "confusion_indices": confusion_indices,
            "confusion_weights": confusion_weights,
            "competition_enabled": torch.tensor(
                competition_enabled, device=deep_tokens.device, dtype=torch.bool
            ),
        }
