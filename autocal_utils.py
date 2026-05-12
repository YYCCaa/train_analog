"""Autocalibration utilities: geometry math, parameter packing, loss, dataset, training loop.

Shared by `generate_synthetic_v3.py`, `train_autocal_e.py`, `train_autocal_f.py`,
and `infer_autocal.py`.
"""

import math

import numpy as np
import torch


IMG_SIZE = 256
N_MARKS = 10
DEG_PER_MARK = 360.0 / N_MARKS


# ── Geometry: canonical-circle → ellipse map ────────────────────────────

def canonical_to_image_np(t, r_c, cx, cy, a, b, theta):
    """Map a canonical-unit-circle point to ellipse image coords (numpy/float).

    For a canonical point at angle `t` and radius `r_c ∈ [0, 1]`, the image
    position is:
        x = cx + r_c · (a·cos(t)·cosθ - b·sin(t)·sinθ)
        y = cy + r_c · (a·cos(t)·sinθ + b·sin(t)·cosθ)

    Major axis is along (cosθ, sinθ); minor axis is perpendicular.
    """
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return x, y


def canonical_to_image_torch(t, r_c, cx, cy, a, b, theta):
    """Differentiable batched version of canonical_to_image.

    All inputs are 1-D tensors of shape (B,). Returns (B, 2).
    """
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    cos_o = torch.cos(theta)
    sin_o = torch.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return torch.stack([x, y], dim=-1)


def derive_marks_np(cx, cy, a, b, theta, phi_0):
    """Compute the 10 scale-mark positions in image coords.

    Mark i lies on the ellipse at canonical angle t_i = phi_0 + i · 36°.
    Returns array of shape (10, 2).
    """
    marks = np.empty((N_MARKS, 2), dtype=np.float64)
    for i in range(N_MARKS):
        t_i = phi_0 + i * math.radians(DEG_PER_MARK)
        x, y = canonical_to_image_np(t=t_i, r_c=1.0,
                                     cx=cx, cy=cy, a=a, b=b, theta=theta)
        marks[i, 0] = x
        marks[i, 1] = y
    return marks


def derive_marks_torch(cx, cy, a, b, theta, phi_0):
    """Differentiable batched mark derivation.

    All inputs are 1-D tensors of shape (B,). Returns (B, 10, 2).
    """
    device = cx.device
    dtype = cx.dtype
    i = torch.arange(N_MARKS, device=device, dtype=dtype)
    step = 2 * math.pi / N_MARKS
    # Broadcast: phi_0 (B,1) + i*(10,) -> (B, 10)
    t_i = phi_0.unsqueeze(-1) + i * step
    cos_t = torch.cos(t_i)
    sin_t = torch.sin(t_i)
    cos_o = torch.cos(theta).unsqueeze(-1)
    sin_o = torch.sin(theta).unsqueeze(-1)
    a_b = a.unsqueeze(-1)
    b_b = b.unsqueeze(-1)
    cx_b = cx.unsqueeze(-1)
    cy_b = cy.unsqueeze(-1)
    x = cx_b + a_b * cos_t * cos_o - b_b * sin_t * sin_o
    y = cy_b + a_b * cos_t * sin_o + b_b * sin_t * cos_o
    return torch.stack([x, y], dim=-1)
