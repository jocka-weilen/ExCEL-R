import unittest

import torch

from model.coser_core import CoSeRCore
from model.losses import build_online_labels, coser_clip_loss


class CoSeRCoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.batch = 2
        self.classes = 5
        self.feature_dim = 24
        self.text_dim = 12
        self.size = (4, 4)
        self.tokens = [
            torch.randn(self.batch, self.size[0] * self.size[1] + 1, self.feature_dim)
            for _ in range(3)
        ]
        self.text = torch.randn(self.classes, self.text_dim)
        self.labels = torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]], dtype=torch.float32
        )

    def make_core(self, **kwargs):
        return CoSeRCore(
            feature_dim=self.feature_dim,
            text_dim=self.text_dim,
            region_dim=16,
            num_region_queries=4,
            topk_confusions=2,
            warmup_iters=2,
            **kwargs,
        )

    def forward(self, core, step):
        return core(
            *self.tokens,
            text_features=self.text,
            shallow_size=self.size,
            middle_size=self.size,
            deep_size=self.size,
            class_labels=self.labels,
            global_step=step,
        )

    def test_full_route_shapes_and_hard_negatives(self):
        outputs = self.forward(self.make_core(), step=3)
        self.assertEqual(outputs["cams"].shape, (2, 5, 4, 4))
        self.assertEqual(outputs["routing_logits"].shape, (2, 5))
        self.assertEqual(outputs["region_assignment"].shape, (2, 16, 4))
        self.assertTrue(outputs["hard_negative_mask"].any())
        for value in outputs.values():
            if torch.is_tensor(value):
                self.assertTrue(torch.isfinite(value).all())
        assignment_sum = outputs["region_assignment"].sum(-1)
        self.assertTrue(torch.allclose(assignment_sum, torch.ones_like(assignment_sum), atol=1e-5))

    def test_warmup_disables_confusion_competition(self):
        outputs = self.forward(self.make_core(), step=0)
        self.assertFalse(outputs["competition_enabled"].item())
        self.assertFalse(outputs["hard_negative_mask"].any())
        self.assertTrue(torch.count_nonzero(outputs["confusion_support"]) == 0)
        self.assertTrue(torch.count_nonzero(outputs["negative_evidence"]) == 0)

    def test_complete_loss_is_finite_and_differentiable(self):
        core = self.make_core()
        outputs = self.forward(core, step=3)
        outputs["seg_logits"] = torch.randn(2, 6, 4, 4, requires_grad=True)
        losses = coser_clip_loss(outputs, self.labels)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        trainable = [parameter for parameter in core.parameters() if parameter.requires_grad]
        self.assertTrue(all(parameter.grad is not None for parameter in trainable))

    def test_online_label_thresholds(self):
        cams = torch.zeros(1, 2, 2, 2)
        cams[0, 0, 0, 0] = 0.8
        cams[0, 1, 1, 1] = 0.4
        labels = torch.tensor([[1.0, 0.0]])
        online = build_online_labels(
            cams, labels, foreground_threshold=0.6, background_threshold=0.2
        )
        self.assertEqual(online[0, 0, 0].item(), 1)
        self.assertEqual(online[0, 1, 1].item(), 0)
        self.assertEqual(online[0, 0, 1].item(), 0)


if __name__ == "__main__":
    unittest.main()
