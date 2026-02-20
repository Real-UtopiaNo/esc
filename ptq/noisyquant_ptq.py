import pytorch_quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .base_ptq import PTQBase

class NoisyQuantPTQ(PTQBase):
    def __init__(self, model, layers_input=None, search_space_mean=200, search_space_range=1000, device = "cuda"):
        self.model = model
        self.layers_input = layers_input
        self.search_space_mean = search_space_mean
        self.search_space_range = search_space_range
        self.device = device
    
    def run(self):
        # 1. Calibrate or search clipping points per layer
        keys_list = list(self.layers_input.keys())
        for name, module in self.model.named_modules():
            if name in keys_list:
                print(name)
                criterion = torch.nn.MSELoss()
                scale = module._input_quantizer._get_scale()
                if len(self.layers_input[name][0].shape) >= 3:
                    for x in self.layers_input[name]:
                        x = x.to(self.device)
                        noisy_bias = (torch.randn_like(x[:1,:1,:])*2-1) / scale
                        break

                    loss_min = 1e6
                    best_candidate = torch.tensor([0.0])
                    pbar = tqdm(range(-self.search_space_mean,self.search_space_mean),desc=f"noisy mean: {name}")
                    for ii in pbar:
                        loss = 0
                        candidate = (1 / scale) * ii/self.search_space_mean
                        module._input_quantizer.init_ptq("noisyquant", noisy_bias=candidate)
                        for x in self.layers_input[name]:
                            x = x.to(self.device)
                            y_quantized = module(x)
                            
                            module._weight_quantizer.disable()
                            module._input_quantizer.disable()
                            y_ref = module(x)
                            module._weight_quantizer.enable()
                            module._input_quantizer.enable()

                            loss += criterion(y_quantized, y_ref)
                        pbar.set_postfix(
                            loss=f"{loss.item():.2e}", 
                            loss_min=f"{loss_min:.2e}",
                            best_mean=f"{best_candidate.item():.2e}"
                        )
                        if loss < loss_min:
                            loss_min = loss
                            best_candidate = candidate
                    best_noisy_mean = best_candidate

                    loss_min = 1e6
                    best_candidate_scale = 0
                    pbar = tqdm(range(0,self.search_space_range*2),desc=f"noisy range: {name}")
                    for ii in pbar:
                        loss = 0
                        candidate = best_noisy_mean + noisy_bias * ii/self.search_space_range
                        module._input_quantizer.init_ptq("noisyquant", noisy_bias=candidate)
                        for x in self.layers_input[name]:
                            x = x.to(self.device)
                            y_quantized = module(x)
                            
                            module._weight_quantizer.disable()
                            module._input_quantizer.disable()
                            y_ref = module(x)
                            module._weight_quantizer.enable()
                            module._input_quantizer.enable()

                            loss += criterion(y_quantized, y_ref)
                        pbar.set_postfix(
                            loss=f"{loss.item():.2e}", 
                            loss_min=f"{loss_min:.2e}",
                            best_range=f"{best_candidate_scale:.2e}"
                        )
                        if loss < loss_min:
                            loss_min = loss
                            best_candidate = candidate
                            best_candidate_scale = ii/self.search_space_range
                    
                    module._input_quantizer.init_ptq("noisyquant", noisy_bias=best_candidate)
                    module._input_quantizer.disable_calib_ptq()
        return self.model
