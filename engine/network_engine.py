from model.model_coser_clip import CoSeRCLIP


def build_network(args):
    """Build CoSeR-CLIP from the shared training/evaluation arguments."""
    model = CoSeRCLIP(
        clip_model=args.model,
        embedding_dim=args.embedding_dim,
        in_channels=args.in_channels,
        dataset_name=args.dataset_name,
        num_classes=args.num_classes,
        img_size=args.crop_size,
        mode=args.train_set,
        device=args.model_device,
        shallow_layer=args.shallow_layer,
        middle_layer=args.middle_layer,
        deep_layer=args.deep_layer,
        region_dim=args.region_dim,
        num_region_queries=args.num_region_queries,
        topk_confusions=args.topk_confusions,
        topq_ratio=args.topq_ratio,
        warmup_iters=args.warmup_iters,
        classifier_threshold=args.classifier_threshold,
        graph_temperature=args.graph_temperature,
        query_temperature=args.query_temperature,
        confusion_cue_temperature=args.confusion_cue_temperature,
        negative_temperature=args.negative_temperature,
        activation_temperature=args.activation_temperature,
        ownership_temperature=args.ownership_temperature,
        shallow_temperature=args.shallow_temperature,
        routing_temperature=args.routing_temperature,
        graph_spatial_weight=args.graph_spatial_weight,
        assignment_spatial_weight=args.assignment_spatial_weight,
        confusion_text_weight=args.confusion_text_weight,
        confusion_visual_weight=args.confusion_visual_weight,
        confusion_overlap_weight=args.confusion_overlap_weight,
        shallow_margin=args.shallow_gap_margin,
        structure_mix=args.structure_mix,
        ablation=args.ablation,
    )
    return model, model.get_param_groups()
