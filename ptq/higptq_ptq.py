"""HiGPTQ: GPTQ with HiFloat4 weight quantizer (arXiv:2602.11287 §IV-A).

v1 scope:
  - QuantLinear only (Conv keeps HiF4 cast if enabled by set_block_quant)
  - Strict sequential Hessian: each layer's H is accumulated from activations
    produced by already-GPTQ'd predecessors (Frantar et al., GPTQ Setup)
"""
from __future__ import annotations

import logging
import math

import torch
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import (
    HIF4_BLOCK,
    fake_hif4_quant,
    hif4_apply_scales,
    hif4_compute_scales,
)

from .base_ptq import PTQBase

logger = logging.getLogger("my_app")


def _freeze_hif4_group_scales(group_fp: torch.Tensor):
    """Freeze HiF4 scales from a (out, g) weight slice (g ≤ 64, padded)."""
    g = group_fp.shape[1]
    if g < HIF4_BLOCK:
        pad = HIF4_BLOCK - g
        group_fp = torch.nn.functional.pad(group_fp, (0, pad))
    # (out, 1, 64) so scale helpers see a single block dim
    blocks = group_fp.unsqueeze(1)
    return hif4_compute_scales(blocks)


def _quant_column_frozen(w_col: torch.Tensor, scales, local_j: int) -> torch.Tensor:
    """Quantize one column with frozen group scales. w_col: (out,)."""
    e6m2, e1_8, e1_16, zero = scales
    # Build a dummy block and only fill column local_j — apply_scales is elementwise
    # given frozen scales, so other columns can be zero.
    block = w_col.new_zeros(w_col.shape[0], 1, HIF4_BLOCK)
    block[:, 0, local_j] = w_col
    q_block = hif4_apply_scales(block, e6m2, e1_8, e1_16, zero)
    return q_block[:, 0, local_j]


@torch.no_grad()
def gptq_hif4(
    weight: torch.Tensor,
    H: torch.Tensor,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """GPTQ on a weight matrix with HiF4 as the column quantizer.

    HiF4 scales are frozen at the start of each 64-column group from the then-
    current (error-compensated) weights — same pattern as GPTQ group quantization.

    Args:
        weight: (out_features, in_features) FP weights
        H: (in_features, in_features) Hessian ≈ 2 X Xᵀ accumulated over calib
        blocksize: GPTQ lazy-update block (independent of HiF4's 64)
        percdamp: diagonal dampening as a fraction of mean(diag(H))

    Returns:
        Quantized weight on the HiF4 grid (same shape/dtype device as ``weight``).
    """
    W = weight.detach().float().clone()
    columns = W.shape[1]
    H = H.float().clone()
    device = W.device

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp

    try:
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
    except Exception as exc:  # noqa: BLE001 — fall back to RTN HiF4
        logger.warning("HiGPTQ Cholesky failed (%s); falling back to HiF4 cast", exc)
        return fake_hif4_quant(weight).to(dtype=weight.dtype)

    Hinv = H
    Q = torch.zeros_like(W)
    frozen_scales = None
    frozen_g0 = -1

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            global_j = i1 + i
            g0 = (global_j // HIF4_BLOCK) * HIF4_BLOCK
            if g0 != frozen_g0:
                g1 = min(g0 + HIF4_BLOCK, columns)
                # Live group: already-quantized prefix in Q, remainder from W
                group = W[:, g0:g1].clone()
                if g0 < i1:
                    group[:, : i1 - g0] = Q[:, g0:i1]
                # Inside current GPTQ block, use W1 for not-yet-finalized cols
                for k in range(max(i1, g0), i2):
                    if k < g1:
                        group[:, k - g0] = W1[:, k - i1]
                frozen_scales = _freeze_hif4_group_scales(group)
                frozen_g0 = g0

            w = W1[:, i]
            d = Hinv1[i, i]
            q = _quant_column_frozen(w, frozen_scales, global_j - g0)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return Q.to(dtype=weight.dtype)


def _update_hessian(H: torch.Tensor, n_samples: int, inp: torch.Tensor):
    """Incremental H ← 2 X Xᵀ average (GPTQ / ModelOpt convention)."""
    x = inp.reshape(-1, inp.shape[-1]).float()
    if x.shape[0] == 0:
        return H, n_samples
    # (d, N)
    x = x.T
    ns = x.shape[1]
    H *= n_samples / (n_samples + ns)
    n_samples += ns
    x = math.sqrt(2.0 / n_samples) * x
    H += x @ x.T
    return H, n_samples


class HiGPTQPTQ(PTQBase):
    """Sequential HiGPTQ over QuantLinear layers.

    calib_fn: zero-argument callable that runs ``model`` forward over the whole
    calibration set once, with the same preprocessing as inference. It is called
    once per QuantLinear, so it must not cache outputs.
    """

    def __init__(
        self,
        model,
        calib_fn,
        blocksize: int = 128,
        percdamp: float = 0.01,
        device: str = "cuda",
        **kwargs,
    ):
        self.model = model
        self.calib_fn = calib_fn
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

    def _iter_quant_linears(self):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.QuantLinear):
                # Skip layers whose weight quantizer was disabled by coverage=gemm
                # only when *input* is also disabled (fully FP). If weight is off
                # but we still want GPTQ — v1: GPTQ every QuantLinear that is
                # meant to be quantized. After set_block_quant(gemm), non-gemm
                # have both disabled; Linear is always gemm so both stay on.
                if module.weight_quantizer._disabled and module.input_quantizer._disabled:
                    continue
                yield name, module

    def _accumulate_hessian(self, module: quant_nn.QuantLinear):
        """Returns the Hessian, or None if the layer saw no calibration input."""
        columns = module.in_features
        H = torch.zeros(columns, columns, device=self.device, dtype=torch.float32)
        n_samples = 0
        state = {"H": H, "n": n_samples}

        def pre_hook(mod, inputs):
            x = inputs[0].detach()
            if not mod.input_quantizer._disabled:
                x = mod.input_quantizer(x)
            state["H"], state["n"] = _update_hessian(state["H"], state["n"], x)

        handle = module.register_forward_pre_hook(pre_hook)
        try:
            with torch.no_grad():
                self.calib_fn()
        finally:
            handle.remove()

        if state["n"] == 0:
            return None
        return state["H"]

    def run(self):
        self.model.to(self.device)
        self.model.eval()

        layers = list(self._iter_quant_linears())
        logger.info("HiGPTQ: %d QuantLinear layers, sequential", len(layers))

        # Unprocessed Linears use FP weights (GPTQ input); keep act HiF4 on.
        for _, module in layers:
            module.weight_quantizer.disable()

        for idx, (name, module) in enumerate(tqdm(layers, desc="HiGPTQ")):
            logger.info("HiGPTQ [%d/%d] %s  (%d×%d)",
                        idx + 1, len(layers), name,
                        module.out_features, module.in_features)

            H = self._accumulate_hessian(module)
            if H is None:
                logger.warning("HiGPTQ: %s got no calibration input, keeping HiF4 cast", name)
                module.weight_quantizer.enable()
                continue
            q_weight = gptq_hif4(
                module.weight.data,
                H,
                blocksize=self.blocksize,
                percdamp=self.percdamp,
            )
            module.weight.data.copy_(q_weight)
            # Keep weight_quantizer disabled: W is already on the HiF4 grid used during
            # GPTQ (scales frozen per 64-group). Re-enabling would recompute scales from
            # Q and slightly move values. Activations stay HiF4 via input_quantizer.
            # module.weight_quantizer remains disabled

        return self.model
