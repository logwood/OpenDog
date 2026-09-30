import torch

from fastreid.modeling.meta_arch.latent_workspace import identity_prototype_margin_loss


def test_identity_prototype_margin_loss_is_zero_when_prototypes_are_separated():
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, -0.01]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1])

    loss = identity_prototype_margin_loss(features, targets, margin=0.08)

    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_identity_prototype_margin_loss_backpropagates_for_a_hard_negative():
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.95, 0.05], [0.9, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1, 1])

    loss = identity_prototype_margin_loss(features, targets, margin=0.08)
    loss.backward()

    assert float(loss) > 0.0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
