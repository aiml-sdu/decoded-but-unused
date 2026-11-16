"""Activation interventions: addition, projection-removal, subspace-removal.

Each intervention is a hook factory that returns a `(name, fn)` pair suitable
for `HookedModel.run_with_hooks`.
"""

from __future__ import annotations

import torch


def addition_hook(layer: int, d: torch.Tensor, alpha: float, position: int | str = "final"):
    """Add `alpha * d` to `hook_resid_post` at the given position."""
    name = f"blocks.{layer}.hook_resid_post"
    d_t = d.detach()

    def hook(t, _n):
        new = t.clone()
        if position == "final":
            new[:, -1] = new[:, -1] + alpha * d_t.to(new.device, new.dtype)
        elif position == "all":
            new = new + alpha * d_t.to(new.device, new.dtype)
        else:
            new[:, int(position)] = new[:, int(position)] + alpha * d_t.to(new.device, new.dtype)
        return new

    return name, hook


def projection_removal_hook(layer: int, d: torch.Tensor, position: int | str = "final"):
    """Remove the component of `t` along `d_hat` from the residual stream."""
    name = f"blocks.{layer}.hook_resid_post"
    d_hat = (d / (d.norm() + 1e-8)).detach()

    def hook(t, _n):
        new = t.clone()
        if position == "final":
            v = new[:, -1].float()
            proj = (v @ d_hat.to(v.device, v.dtype)).unsqueeze(-1) * d_hat.to(v.device, v.dtype)
            new[:, -1] = (v - proj).to(new.dtype)
        elif position == "all":
            v = new.float()
            proj = (v @ d_hat.to(v.device, v.dtype)).unsqueeze(-1) * d_hat.to(v.device, v.dtype)
            new = (v - proj).to(new.dtype)
        else:
            i = int(position)
            v = new[:, i].float()
            proj = (v @ d_hat.to(v.device, v.dtype)).unsqueeze(-1) * d_hat.to(v.device, v.dtype)
            new[:, i] = (v - proj).to(new.dtype)
        return new

    return name, hook


def subspace_removal_hook(layer: int, basis: torch.Tensor, position: int | str = "final"):
    """Remove the projection onto a `[k, D]` orthonormal basis."""
    name = f"blocks.{layer}.hook_resid_post"
    Q, _ = torch.linalg.qr(basis.float().T)  # [D, k] orthonormal
    P = (Q @ Q.T).detach()  # [D, D]

    def hook(t, _n):
        new = t.clone()
        Pl = P.to(new.device, torch.float32)
        if position == "final":
            v = new[:, -1].float()
            new[:, -1] = (v - v @ Pl).to(new.dtype)
        elif position == "all":
            v = new.float()
            new = (v - v @ Pl).to(new.dtype)
        else:
            i = int(position)
            v = new[:, i].float()
            new[:, i] = (v - v @ Pl).to(new.dtype)
        return new

    return name, hook
