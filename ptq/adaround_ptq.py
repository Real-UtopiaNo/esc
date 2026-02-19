import pytorch_quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ptq import PTQBase
from utils import collect_layer_inputs

class AdaroundPTQ(PTQBase):
    def __init__(self, model, layers_input=None, max_iter=2, reg_lambda = 0.01, device = "cuda"):
        self.model = model
        self.layers_input = layers_input
        self.max_iter = max_iter #1000 since we have 10 input for each
        self.reg_lambda = reg_lambda #seems ok
        self.device = device
    
    def run(self):
        # 1. Calibrate or search clipping points per layer
        keys_list = list(self.layers_input.keys())
        for name, module in self.model.named_modules():
            if name in keys_list:
                print(name)
                alpha = nn.Parameter(torch.zeros_like(module.weight))
                module._weight_quantizer.init_ptq("adaround", alpha=alpha)
                module._weight_quantizer.train()
                optimizer = torch.optim.Adam([module._weight_quantizer.alpha], lr=1e-2)

                for it in range(self.max_iter):
                    total_loss = 0.0
                    for x in self.layers_input[name]:
                        x = x.to(self.device)

                        y_quantized = module(x)

                        module._weight_quantizer.disable()
                        module._input_quantizer.disable()
                        y_ref = module(x)
                        module._weight_quantizer.enable()
                        module._input_quantizer.enable()

                        # reconstruction loss + rounding regularizer
                        recon_loss = F.mse_loss(y_quantized, y_ref)
                        round_loss = (torch.sigmoid(alpha) - 0.5).abs().mean()
                        loss = recon_loss + self.reg_lambda * round_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    # if (it + 1) % 100 == 0:
                    #     print(f"[Iter {it+1}] Loss: {total_loss / len(layers_input[name]):.6f}")
                module._weight_quantizer.disable_calib_ptq()
        return self.model