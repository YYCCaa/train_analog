"""Smoke tests for the two autocal backbones."""

import torch

from autocal_utils import AutocalModelE, AutocalModelF


def test_model_e_forward_shape():
    model = AutocalModelE()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_model_f_forward_shape():
    model = AutocalModelF()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_model_e_backprop():
    model = AutocalModelE()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    out.sum().backward()
    # Stem is the deepest-back parameter; if it has gradient, the whole graph is alive.
    stem_w = model.stem[0].weight
    assert stem_w.grad is not None
    assert stem_w.grad.abs().sum() > 0


def test_model_f_backprop():
    model = AutocalModelF()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    out.sum().backward()
    enc1_w = model.enc1[0].weight  # F's first stem conv lives in enc1
    assert enc1_w.grad is not None
    assert enc1_w.grad.abs().sum() > 0
