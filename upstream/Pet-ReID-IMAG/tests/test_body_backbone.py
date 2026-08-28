import unittest
from unittest import mock

from torch import nn

from pet_id.body_backbone import FrozenSwinV2BodyBackbone


class BodyBackboneTest(unittest.TestCase):
    def test_removes_classifier_and_freezes_official_backbone(self):
        fake_model = mock.Mock()
        fake_model.features = nn.Identity()
        fake_model.norm = nn.Identity()
        fake_model.permute = nn.Identity()
        fake_model.avgpool = nn.AdaptiveAvgPool2d(1)
        fake_model.flatten = nn.Flatten(1)
        fake_model.head = nn.Linear(4, 2)

        with mock.patch("pet_id.body_backbone.swin_v2_b", return_value=fake_model):
            backbone = FrozenSwinV2BodyBackbone(pretrained=True, frozen=True)

        self.assertIsInstance(fake_model.head, nn.Identity)
        self.assertFalse(backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_train_keeps_frozen_backbone_in_eval_mode(self):
        fake_model = mock.Mock()
        fake_model.features = nn.Identity()
        fake_model.norm = nn.Identity()
        fake_model.permute = nn.Identity()
        fake_model.avgpool = nn.AdaptiveAvgPool2d(1)
        fake_model.flatten = nn.Flatten(1)
        fake_model.head = nn.Identity()

        with mock.patch("pet_id.body_backbone.swin_v2_b", return_value=fake_model):
            backbone = FrozenSwinV2BodyBackbone(pretrained=False, frozen=True)

        backbone.train(True)
        self.assertFalse(backbone.training)


if __name__ == "__main__":
    unittest.main()
