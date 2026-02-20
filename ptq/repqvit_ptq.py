import pytorch_quantization
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .base_ptq import PTQBase

class RepQViTPTQ(PTQBase):
    def __init__(self, model):
        self.model = model
    
    def run(self):
        # 1. Calibrate or search clipping points per layer
        module_dict={}
        for name, module in self.model.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if isinstance(module, pytorch_quantization.nn.QuantLayerNorm):
                scaling = True
                if hasattr(father_module, 'attn') and name.split('.')[-1] == "norm1":
                    next_module = father_module.attn.qkv
                elif hasattr(father_module, 'mlp') and name.split('.')[-1] == "norm2":
                    next_module = father_module.mlp.fc1
                elif hasattr(father_module, 'attn'):
                    next_module = father_module.attn
                elif hasattr(father_module, 'mlp'):
                    next_module = father_module.mlp.fc1
                elif hasattr(father_module, 'reduction'):
                    next_module = father_module.reduction
                elif hasattr(father_module, 'fc'):
                    next_module = father_module.fc
                elif hasattr(father_module, 'conv1d_2') and 'layer_norm_1' in name:
                    next_module = father_module.conv1d_2.conv
                elif name.split('.')[-1] == 'layer_norm_2':
                    scaling = False
                elif isinstance(father_module, torch.nn.Sequential):
                    suffix = name.split('.')[-1]
                    next_module = father_module._modules[str(int(suffix) + 1)]
                else:
                    scaling = False
                
                suffix = name.split('.')[-1]
                if suffix == "post_norm":
                    scaling = False

                if scaling:
                    act_delta = next_module._input_quantizer._get_scale().reshape(-1)
                    act_zero_point = torch.zeros_like(act_delta) #replace one day if exist zero_point
                    act_min = -act_zero_point * act_delta

                    target_delta = torch.mean(act_delta)
                    target_zero_point = torch.mean(act_zero_point)
                    target_min = -target_zero_point * target_delta

                    r = act_delta / target_delta
                    b = act_min / r - target_min

                    module.weight.data = module.weight.data / r
                    module.bias.data = module.bias.data / r - b


                    if hasattr(father_module, 'attn'):
                        try:
                            next_module.in_proj_weight.data = next_module.in_proj_weight.data * r
                        except:
                            next_module.weight.data = next_module.weight.data * r
                    else:
                        next_module.weight.data = next_module.weight.data * r
                    
                    max_bound = torch.tensor((2.0**(next_module._input_quantizer._num_bits - 1 + int(next_module._input_quantizer._unsigned))) - 1.0, device=target_delta.device)
                    amax = max_bound / target_delta
                    next_module._input_quantizer._amax = amax
                    a = 1

        return self.model