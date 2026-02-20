import pytorch_quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .base_ptq import PTQBase

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param      rank_ratio: rank_of_decomposed_matrix
    :return: L, R
    """

    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    reduced_rank = int(reduced_rank)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R

class DiTASPTQ(PTQBase):
    def __init__(self, model, layers_input=None, iter_rank = 10, reduced_rank = 32, grid = 21, device = "cuda"):
        self.model = model
        self.layers_input = layers_input
        self.iter_rank = iter_rank
        self.reduced_rank = reduced_rank
        self.grid = grid
        self.device = device
    
    def run(self):
        allow_name = ['proj', 'qkv', 'fc1', 'fc2', 'ffm', 'linear_layer', 'linear_proj', 'fc6', 'linear', 'mlp_head']
        forbid_name = ['mel_linear', 'module.v.blocks']

        for name, module in self.model.named_modules():
            if type(module) == pytorch_quantization.nn.QuantLinear and any(an in name for an in allow_name) and not any(fn in name for fn in forbid_name):
                L, R = 0, 0
                for i in range(self.iter_rank):
                    low_rank_product = L @ R if torch.is_tensor(L) else 0
                    residual = module.weight - low_rank_product
                    quant_w = module._weight_quantizer(residual)
                    output = low_rank_decomposition(module.weight  - quant_w, reduced_rank=self.reduced_rank)
                    L, R = output[0], output[1]
                
                linear_loras = pytorch_quantization.nn.QuantLoRA(module.in_features, module.out_features, reduced_rank=int(self.reduced_rank))
                
                linear_loras.quant_weight = nn.Parameter(quant_w, requires_grad=False)
                linear_loras.ln_weight = nn.Parameter(module.weight, requires_grad=False)
                linear_loras.bias = nn.Parameter(module.bias, requires_grad=False)
                linear_loras.left_weight = nn.Parameter(L, requires_grad=False)
                linear_loras.right_weight = nn.Parameter(R, requires_grad=False)
                linear_loras._input_quantizer = module._input_quantizer
                linear_loras._weight_quantizer = module._weight_quantizer

                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = self.model.get_submodule(parent_name) if parent_name else self.model

                setattr(parent_module, child_name, linear_loras)
        
        keys_list = list(self.layers_input.keys())
        for name, module in self.model.named_modules():
            if name in keys_list and any(an in name for an in allow_name) and not any(fn in name for fn in forbid_name):
                print(name)
                best_alpha = None
                best_loss = float("inf")
                loss_total = None
                for grid in range(self.grid):
                    loss_total = 0
                    alpha = 0.05 * grid
                    weight_scale = module.quant_weight.abs().max(dim=0)[0].clamp(min=1e-4)
                    scales = (
                            ((module._input_quantizer._get_scale()).pow(alpha)/ weight_scale.pow(1 - alpha))
                            .clamp(min=1e-4).view(-1)
                            .to(self.device)
                        )
                    scales = 1 / scales
                    for x in self.layers_input[name]:
                        with torch.no_grad():
                            x = x.to(self.device)
                            if module.bias.numel() == 0:
                                HX_original = F.linear(x, module.ln_weight)
                            else:
                                HX_original = F.linear(x, module.ln_weight) + module.bias

                            x_quant = module._input_quantizer(x)
                            w_right = module.right_weight * scales.view(1,-1)
                            if module.bias.numel() == 0:
                                HX_quant = F.linear(x_quant, module.quant_weight * scales.view(1, -1)) + F.linear(F.linear(x_quant, w_right), module.left_weight)
                            else:
                                HX_quant = F.linear(x_quant, module.quant_weight * scales.view(1, -1)) + F.linear(F.linear(x_quant, w_right), module.left_weight) + module.bias

                            loss = mean_flat((HX_quant - HX_original) ** 2).mean().requires_grad_()
                            loss_total = loss_total + loss.item()
                    
                    if loss_total < best_loss:
                        best_loss = loss_total
                        best_alpha = alpha
                    
                scales = (
                            ((module._input_quantizer._get_scale()).pow(best_alpha)/ weight_scale.pow(1 - best_alpha))
                            .clamp(min=1e-4).view(-1)
                            .to(self.device)
                        )
                scales = 1 / scales
                
                module.quant_weight.mul_(scales.view(1, -1).detach())
                module.right_weight.mul_(scales.view(1, -1).detach())

        return self.model
