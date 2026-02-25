"""
Color transfer via optimal transport.

Treats both images as unordered point clouds of pixel colors and solves the
discrete OT problem between them using the low-memory MDOT-TNT solver.  For
each source pixel the barycentric transport map gives a convex combination of
target colors, producing a smooth, globally-consistent color transfer.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch as th
from PIL import Image

from mdot_tnt.lowmem import mdot_lowmem, squared_euclidean

__all__ = ["transfer_colors"]

ImageLike = Union[str, Path, Image.Image]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open(img: ImageLike) -> Image.Image:
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    return img.convert("RGB")


def _resize_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale *img* so total pixel count ≤ *max_pixels* (aspect preserved)."""
    n = img.width * img.height
    if n <= max_pixels:
        return img
    scale = (max_pixels / n) ** 0.5
    return img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )


def _to_tensor(img: Image.Image, device: th.device) -> th.Tensor:
    """(H, W, 3) float32 tensor in [0, 1]."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    return th.from_numpy(arr).to(device)


def _srgb_to_lab(rgb: th.Tensor) -> th.Tensor:
    """sRGB [0, 1] → CIE L*a*b* (same device/dtype)."""
    linear = th.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp(min=0.0) ** 2.4,
    )
    M = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ M.T
    white = rgb.new_tensor([0.95047, 1.00000, 1.08883])
    t = xyz / white
    delta = 6.0 / 29.0
    f = th.where(
        t > delta**3,
        t.clamp(min=1e-8) ** (1.0 / 3.0),
        t / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return th.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        dim=-1,
    )


def _lab_to_srgb(lab: th.Tensor) -> th.Tensor:
    """CIE L*a*b* → sRGB [0, 1] (same device/dtype)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    delta = 6.0 / 29.0

    def f_inv(t: th.Tensor) -> th.Tensor:
        return th.where(t > delta, t**3, 3.0 * delta**2 * (t - 4.0 / 29.0))

    white = lab.new_tensor([0.95047, 1.00000, 1.08883])
    xyz = th.stack([f_inv(fx), f_inv(fy), f_inv(fz)], dim=-1) * white
    M_inv = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    linear = (xyz @ M_inv.T).clamp(0.0, 1.0)
    return th.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.clamp(min=1e-8) ** (1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transfer_colors(
    source: ImageLike,
    target: ImageLike,
    *,
    max_pixels: int = 8192,
    block_size: int = 512,
    gamma_f: float = 1024.0,
    color_space: Literal["lab", "rgb"] = "lab",
    device: str | th.device = "cpu",
) -> Image.Image:
    """Transfer the color palette of *target* onto the content of *source*.

    Both images are downscaled to at most ``max_pixels`` pixels, then treated
    as unordered point clouds in color space — every pixel is a support point
    with equal weight.  The OT plan P maps source pixels to target colors;
    the barycentric image ``T[i] = (P[i, :] / r[i]) @ Y`` is returned as the
    recolored source.

    Args:
        source: Content image — structure and geometry are preserved.
        target: Style image — color palette is transferred from here.
        max_pixels: Maximum pixel count per image after optional downscaling.
            Controls the size of the OT problem (plan is O(n·m) in memory).
        block_size: Column block size for the low-memory solver.
        gamma_f: OT temperature (inverse regularization strength).
        color_space: ``"lab"`` (perceptually uniform, default) or ``"rgb"``.
        device: Torch device for all tensor operations.

    Returns:
        Recolored source image as a :class:`~PIL.Image.Image`, at the
        (possibly downscaled) source resolution.
    """
    device = th.device(device)

    # -- load and optionally downscale ----------------------------------------
    src_img = _resize_to_max_pixels(_open(source), max_pixels)
    tgt_img = _resize_to_max_pixels(_open(target), max_pixels)

    src_rgb = _to_tensor(src_img, device)  # (H_s, W_s, 3)
    tgt_rgb = _to_tensor(tgt_img, device)  # (H_t, W_t, 3)

    # -- convert to working color space ----------------------------------------
    to_cs = _srgb_to_lab if color_space == "lab" else (lambda x: x)
    X = to_cs(src_rgb.reshape(-1, 3))  # (n_s, 3) — ALL source pixels
    Y = to_cs(tgt_rgb.reshape(-1, 3))  # (n_t, 3) — ALL target pixels
    n_s, n_t = X.shape[0], Y.shape[0]

    # -- normalise to [0, 1] for well-conditioned squared-Euclidean costs ------
    if color_space == "lab":
        # L ∈ [0, 100] → /100;  a, b ∈ [-128, 127] → +128 then /255
        offset = X.new_tensor([0.0, 128.0, 128.0])
        scale = X.new_tensor([100.0, 255.0, 255.0])
        X_n = (X + offset) / scale
        Y_n = (Y + offset) / scale
    else:
        # sRGB pixels are already in [0, 1]
        X_n, Y_n = X, Y

    # -- uniform marginals — promote to float64 if needed ---------------------
    dtype = X.dtype
    if gamma_f > 2**10 and dtype != th.float64:
        warnings.warn(f"Switching to float64 for gamma_f > 2^10 (output will be {dtype}).")
        X, Y, X_n, Y_n = X.double(), Y.double(), X_n.double(), Y_n.double()

    r = X.new_full((n_s,), 1.0 / n_s)
    c = Y.new_full((n_t,), 1.0 / n_t)

    # -- solve OT — every pixel is a support point ----------------------------
    blk = min(block_size, n_t)
    cost_block_fn = lambda s, e: squared_euclidean(X_n, Y_n[s:e])

    u, v, gamma_final, _, _ = mdot_lowmem(r, c, cost_block_fn, gamma_f, blk)

    # -- blocked rounding (mirrors blocked_rounded_cost, log-space only) ------
    def col_blocks():
        for s in range(0, n_t, blk):
            yield s, min(s + blk, n_t)

    def blocked_lse_r(u_, v_):
        acc = None
        for s, e in col_blocks():
            part = th.logsumexp(v_[s:e].unsqueeze(-2) - gamma_final * cost_block_fn(s, e), dim=-1)
            acc = part if acc is None else th.logaddexp(acc, part)
        return acc

    def blocked_lse_c(u_, v_):
        return th.cat(
            [
                th.logsumexp(u_.unsqueeze(-1) - gamma_final * cost_block_fn(s, e), dim=-2)
                for s, e in col_blocks()
            ],
            dim=-1,
        )

    # row rounding
    delta_u = th.min(r.log() - (u + blocked_lse_r(u, v)), th.zeros_like(r))
    u = u + delta_u
    # column rounding
    delta_v = th.min(c.log() - (v + blocked_lse_c(u, v)), th.zeros_like(v))
    v = v + delta_v

    # residual errors for rank-1 correction
    err_r = r - (u + blocked_lse_r(u, v)).exp()
    err_r_norm = err_r / (err_r.norm(p=1) + 1e-30)
    err_c = c - (v + blocked_lse_c(u, v)).exp()

    # -- blocked barycentric map: T[i] = (P_rounded[i,:] / r[i]) @ Y ---------
    # P_rounded[i,j] = exp(u[i]+v[j]-γ C[i,j]) + err_r_norm[i]*err_c[j]
    err_c_Y = err_c @ Y  # (3,) — rank-1 term
    T = Y.new_zeros(n_s, Y.shape[-1])
    for s, e in col_blocks():
        P_b = th.exp(u[:, None] + v[s:e][None, :] - gamma_final * cost_block_fn(s, e))
        T += P_b @ Y[s:e]
    T += err_r_norm[:, None] * err_c_Y[None, :]  # rank-1 correction
    T = T / r[:, None]  # normalise by marginal
    T = T.to(dtype)

    # -- convert back to sRGB and render --------------------------------------
    from_cs = (
        _lab_to_srgb
        if color_space == "lab"
        else (lambda x: x / x.amax(dim=-1, keepdim=True).clamp(min=1.0))
    )
    new_rgb = from_cs(T).reshape(src_img.height, src_img.width, 3)
    out = new_rgb.mul(255.0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(out, mode="RGB")
