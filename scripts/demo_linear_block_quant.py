#!/usr/bin/env python3
"""CPU-only demo: Linear-only MLP + dynamic NVFP4 / HiF4 fake quantization.

No static calibration: NVFP4 global_amax is the per-tensor max of the current
activation / weight; HiF4 scales are fully dynamic by design.

Run from repo root:
  PYTHONPATH=. python scripts/demo_linear_block_quant.py
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_quantization.tensor_quant import fake_hif4_quant, fake_nvfp4_quant


class LinearMLP(nn.Module):
    """Tiny model with Linear layers only (no Conv / Norm / attention)."""

    def __init__(self, d_in=64, d_hidden=128, d_out=16):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.fc3 = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _amax(t: torch.Tensor) -> torch.Tensor:
    """Dynamic per-tensor amax from the tensor itself (no calib cache)."""
    return t.detach().abs().amax().reshape(())


def linear_nvfp4(x, weight, bias, block_axis=-1):
    """Dynamic NVFP4 fake-quant on A/W, then FP32 linear."""
    x_q = fake_nvfp4_quant(x, _amax(x), block_axis=block_axis)
    w_q = fake_nvfp4_quant(weight, _amax(weight), block_axis=block_axis)
    return F.linear(x_q, w_q, bias)


def linear_hif4(x, weight, bias, block_axis=-1):
    """HiF4 fake-quant on A/W (all scales dynamic), then FP32 linear."""
    x_q = fake_hif4_quant(x, block_axis=block_axis)
    w_q = fake_hif4_quant(weight, block_axis=block_axis)
    return F.linear(x_q, w_q, bias)


@torch.no_grad()
def forward_fp32(model: LinearMLP, x: torch.Tensor) -> torch.Tensor:
    return model(x)


@torch.no_grad()
def forward_nvfp4(model: LinearMLP, x: torch.Tensor) -> torch.Tensor:
    h = x
    h = F.relu(linear_nvfp4(h, model.fc1.weight, model.fc1.bias))
    h = F.relu(linear_nvfp4(h, model.fc2.weight, model.fc2.bias))
    return linear_nvfp4(h, model.fc3.weight, model.fc3.bias)


@torch.no_grad()
def forward_hif4(model: LinearMLP, x: torch.Tensor) -> torch.Tensor:
    h = x
    h = F.relu(linear_hif4(h, model.fc1.weight, model.fc1.bias))
    h = F.relu(linear_hif4(h, model.fc2.weight, model.fc2.bias))
    return linear_hif4(h, model.fc3.weight, model.fc3.bias)


def _err(ref: torch.Tensor, pred: torch.Tensor) -> dict:
    diff = (ref - pred).float()
    return {
        "mse": (diff * diff).mean().item(),
        "max_abs": diff.abs().max().item(),
        "rel_l2": (diff.norm() / (ref.float().norm() + 1e-12)).item(),
    }


def _print_tensor_stats(name: str, t: torch.Tensor):
    print(f"  {name}: shape={tuple(t.shape)}  "
          f"min={t.min().item():.4g}  max={t.max().item():.4g}  "
          f"amax={t.abs().max().item():.4g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--d-in", type=int, default=64)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--d-out", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    model = LinearMLP(args.d_in, args.d_hidden, args.d_out).to(device).eval()
    x_eval = torch.randn(args.batch, args.d_in, device=device)

    print("=== Linear-only MLP (dynamic fake quant only) ===")
    print(model)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters())}")
    print("NVFP4: global_amax = absmax(tensor) at call time; HiF4: no global amax.")

    print("\n=== Step 1: one Linear under dynamic NVFP4 (fc1) ===")
    w = model.fc1.weight
    x0 = x_eval
    amax_x, amax_w = _amax(x0), _amax(w)
    print(f"  dynamic amax_x={amax_x.item():.6g}  amax_w={amax_w.item():.6g}")
    x0_q = fake_nvfp4_quant(x0, amax_x)
    w_q = fake_nvfp4_quant(w, amax_w)
    _print_tensor_stats("x", x0)
    _print_tensor_stats("x_nvfp4", x0_q)
    _print_tensor_stats("W", w)
    _print_tensor_stats("W_nvfp4", w_q)
    y_fp = F.linear(x0, w, model.fc1.bias)
    y_nv = F.linear(x0_q, w_q, model.fc1.bias)
    print(f"  fc1 output err vs FP32: {_err(y_fp, y_nv)}")

    print("\n=== Step 2: same Linear under HiF4 (fully dynamic) ===")
    x0_h = fake_hif4_quant(x0)
    w_h = fake_hif4_quant(w)
    _print_tensor_stats("x_hif4", x0_h)
    _print_tensor_stats("W_hif4", w_h)
    y_hi = F.linear(x0_h, w_h, model.fc1.bias)
    print(f"  fc1 output err vs FP32: {_err(y_fp, y_hi)}")

    print("\n=== Step 3: full MLP end-to-end (all dynamic) ===")
    y_ref = forward_fp32(model, x_eval)
    y_nvfp4 = forward_nvfp4(model, x_eval)
    y_hif4 = forward_hif4(model, x_eval)
    print(f"  FP32   out[0,:4]={y_ref[0, :4].tolist()}")
    print(f"  NVFP4  out[0,:4]={y_nvfp4[0, :4].tolist()}  err={_err(y_ref, y_nvfp4)}")
    print(f"  HiF4   out[0,:4]={y_hif4[0, :4].tolist()}  err={_err(y_ref, y_hif4)}")
    print("\nDone. No static calib; matmul stayed FP32; only A/W were fake-quantized.")


if __name__ == "__main__":
    main()
