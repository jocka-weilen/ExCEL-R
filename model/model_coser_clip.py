"""End-to-end CoSeR-CLIP model built on the frozen ExCEL/CLIP baseline encoder."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

import clip
from datasets.clip_text import new_class_names, new_class_names_coco
from .coser_core import CoSeRCore
from .decoder.TransDecoder import DecoderTransformer
from .segformer_head import SegFormerHead


DEFAULT_PROMPTS = [
    "a photo of a {}.",
    "a clean photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "there is a {} in the scene.",
]


class CoSeRCLIP(nn.Module):
    """CoSeR-CLIP for weakly supervised semantic segmentation.

    The CLIP image/text encoders stay frozen. Only the projection adapters,
    region/evidence router, and visual-only segmentation decoder are optimized.
    """

    def __init__(
        self,
        clip_model: str = "CoSeR-CLIP_ViT-B/16",
        embedding_dim: int = 256,
        in_channels: int = 768,
        dataset_name: str = "pascal_voc",
        num_classes: int = 21,
        img_size: int = 320,
        mode: str = "train",
        device: str = "cuda",
        shallow_layer: int = 2,
        middle_layer: int = 6,
        deep_layer: int = 11,
        region_dim: int = 256,
        num_region_queries: int = 12,
        topk_confusions: int = 3,
        warmup_iters: int = 1000,
        classifier_threshold: float = 0.35,
        ablation: str = "full",
        prompt_templates=None,
        **core_kwargs,
    ):
        super().__init__()
        self.method_name = "CoSeR-CLIP"
        self.num_classes = num_classes
        self.num_foreground_classes = num_classes - 1
        self.dataset_name = dataset_name
        self.shallow_layer = shallow_layer
        self.middle_layer = middle_layer
        self.deep_layer = deep_layer

        self.encoder, _ = clip.load(clip_model, device=device)
        visual_layers = len(self.encoder.visual.transformer.resblocks)
        selected_layers = (shallow_layer, middle_layer, deep_layer)
        if min(selected_layers) < 0 or max(selected_layers) >= visual_layers:
            raise ValueError(
                f"Layer indices {selected_layers} are invalid for a {visual_layers}-layer encoder"
            )
        self.encoder.visual.reload_self_attn(
            layers=min(6, visual_layers), feat_size=img_size // 16, mode=mode
        )
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        feature_dim = self.encoder.visual.embed_dim
        text_dim = self.encoder.visual.proj.shape[1]
        if in_channels != feature_dim:
            raise ValueError(
                f"in_channels={in_channels}, but the selected CLIP encoder outputs {feature_dim}"
            )

        class_prompts = (
            new_class_names if self.num_foreground_classes == 20 else new_class_names_coco
        )
        if len(class_prompts) != self.num_foreground_classes:
            raise ValueError(
                f"No class prompt set for {self.num_foreground_classes} foreground classes"
            )
        with torch.no_grad():
            text_features = clip.encode_text_with_prompt_ensemble(
                self.encoder,
                class_prompts,
                device,
                prompt_templates=prompt_templates or DEFAULT_PROMPTS,
            ).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_features", text_features, persistent=True)

        self.coser = CoSeRCore(
            feature_dim=feature_dim,
            text_dim=text_dim,
            region_dim=region_dim,
            num_region_queries=num_region_queries,
            topk_confusions=topk_confusions,
            warmup_iters=warmup_iters,
            classifier_threshold=classifier_threshold,
            ablation=ablation,
            **core_kwargs,
        )
        self.coser.initialize_visual_adapters(
            self.encoder.visual.ln_post, self.encoder.visual.proj
        )

        self.decoder_fts_fuse = SegFormerHead(
            in_channels=feature_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            index=visual_layers,
        )
        self.decoder = DecoderTransformer(
            width=embedding_dim, layers=3, heads=8, output_dim=num_classes
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def get_param_groups(self):
        """Return optimizer groups: frozen backbone, adapters/router, segmentation."""
        groups = [[], [], [], []]
        groups[2].extend(
            parameter for parameter in self.coser.parameters() if parameter.requires_grad
        )
        groups[3].extend(self.decoder.parameters())
        groups[3].extend(self.decoder_fts_fuse.parameters())
        return groups

    def forward(
        self,
        images: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        _, _, height, width = images.shape
        patch_size = self.encoder.visual.conv1.kernel_size[0]
        grid_size = (height // patch_size, width // patch_size)
        if grid_size[0] * patch_size != height or grid_size[1] * patch_size != width:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by CLIP patch size {patch_size}"
            )

        self.encoder.eval()
        with torch.no_grad():
            _, attention_list, feature_list = self.encoder.encode_image(
                images, return_weights=True, ex_feats=None
            )
            all_features = torch.stack(feature_list, dim=0)
            attention_weights = torch.stack(attention_list, dim=0)

        patch_features = all_features[:, :, 1:].permute(0, 1, 3, 2)
        patch_features = patch_features.reshape(
            patch_features.shape[0],
            patch_features.shape[1],
            patch_features.shape[2],
            *grid_size,
        )
        fused_features = self.decoder_fts_fuse(patch_features)
        segmentation_logits, _ = self.decoder(fused_features)

        outputs = self.coser(
            shallow_tokens=all_features[self.shallow_layer],
            middle_tokens=all_features[self.middle_layer],
            deep_tokens=all_features[self.deep_layer],
            text_features=self.text_features,
            shallow_size=grid_size,
            middle_size=grid_size,
            deep_size=grid_size,
            class_labels=class_labels,
            global_step=global_step,
        )
        outputs["seg_logits"] = segmentation_logits
        outputs["fused_features"] = fused_features
        outputs["attention_weights"] = attention_weights
        return outputs


def load_coser_checkpoint(
    model: nn.Module, checkpoint_path: str, map_location: str = "cpu", strict: bool = False
):
    """Load plain, DDP, or trainer-format CoSeR-CLIP checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = OrderedDict()
    for name, value in state_dict.items():
        name = name.replace("module.", "", 1) if name.startswith("module.") else name
        if "encoder.visual.positional_embedding" not in name:
            cleaned[name] = value
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return checkpoint, incompatible
