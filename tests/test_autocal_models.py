"""Smoke tests for the two autocal backbones."""

import pytest
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
    # Spot-check one parameter has a gradient.
    found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            found = True
            break
    assert found


def test_model_f_backprop():
    model = AutocalModelF()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    out.sum().backward()
    found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            found = True
            break
    assert found
