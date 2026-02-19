import pytorch_quantization
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .base_ptq import PTQBase

from models_ptq.mpsenet.model import FeedForwardModule, AttentionModule, ConformerConvModule
from models_ptq.conformer.model import FeedForwardBlock
from timm.models.vision_transformer import Block

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        if isinstance(fc, nn.Linear):
            assert ln.weight.numel() == fc.in_features == act_scales.numel()
        elif isinstance(fc, nn.MultiheadAttention):
            assert ln.weight.numel() == fc.embed_dim == act_scales.numel()

    if isinstance(fc, nn.Linear):
        device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    elif isinstance(fc, nn.MultiheadAttention):
        device, dtype = fcs[0].in_proj_weight.device, fcs[0].in_proj_weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    if isinstance(fcs[0], nn.Linear):
        weight_scales = torch.cat(
            [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
        )
    elif isinstance(fcs[0], nn.MultiheadAttention):
        weight_scales = torch.cat(
            [fc.in_proj_weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
        )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.mul_(scales) ##seems incoherent but ok perform better, incoherent with original implementation
    ln.bias.mul_(scales)

    for fc in fcs:
        if isinstance(fc, nn.Linear):
            fc.weight.div_(scales.view(1, -1))
        elif isinstance(fc, nn.MultiheadAttention):
            fc.in_proj_weight.div_(scales.view(1, -1))

class SmoothQuantPTQ(PTQBase):
    def __init__(self, model, alpha=0.5):
        self.model = model
        self.alpha = alpha
    
    def run(self):
        # 1. Calibrate or search clipping points per layer
        for name, module in self.model.named_modules():
            if isinstance(module, FeedForwardModule):
                ln = module.ffm._modules['0']
                fc1 = module.ffm._modules['1']
                fc1_input_scales = fc1._input_quantizer._amax.flatten()
                smooth_ln_fcs(ln, fc1, fc1_input_scales, self.alpha)
            if isinstance(module, AttentionModule):
                ln = module.layernorm
                qkv = module.attn
                qkv_input_scales = qkv._input_quantizer._amax.flatten()
                smooth_ln_fcs(ln, qkv, qkv_input_scales, self.alpha)
            if isinstance(module, FeedForwardBlock) and int(name.split('.')[-2]) < 1: #to evaluate
                ln = module.module._modules['0']
                fc1 = module.module._modules['1']
                fc1_input_scales = fc1._input_quantizer._amax.flatten()
                smooth_ln_fcs(ln, fc1, fc1_input_scales, self.alpha)
            if isinstance(module, Block) and int(name.split('.')[-1]) < 1:
                ln_1 = module.norm1
                qkv = module.attn.qkv
                qkv_input_scales = qkv._input_quantizer._amax.flatten()
                smooth_ln_fcs(ln_1, qkv, qkv_input_scales, self.alpha)

                ln_2 = module.norm2
                mlp = module.mlp.fc1
                mlp_input_scales = mlp._input_quantizer._amax.flatten()
                smooth_ln_fcs(ln_2, mlp, mlp_input_scales, self.alpha)
        return self.model