import pytorch_quantization
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .base_ptq import PTQBase
from collections import defaultdict

class BRECQPTQ(PTQBase):
    def __init__(self, model, layers_input=None, max_iter=2, lr=0.01, p=2.4, device = "cuda"):
        self.model = model
        self.layers_input = layers_input
        self.max_iter = max_iter
        self.lr = lr
        self.p = p
        self.device = device
    
    def run(self):
        layers_output = defaultdict(list)

        keys_list = list(self.layers_input.keys())
        self.model.train()
        for name, module in self.model.named_modules():
            if name in keys_list:
                print(name)
                optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)

                for _, submodule in module.named_modules():
                    if hasattr(submodule, "_input_quantizer"):
                        submodule._input_quantizer.disable()
                    if hasattr(submodule, "_weight_quantizer"):
                        submodule._weight_quantizer.disable()
                with torch.no_grad():
                    for x in self.layers_input[name]:
                        x = x.to(self.device)
                        out = module(x)
                        layers_output[name].append(out.detach().cpu())
                
                for _, submodule in module.named_modules():
                    if hasattr(submodule, "_input_quantizer"):
                        submodule._input_quantizer.enable()
                    if hasattr(submodule, "_weight_quantizer"):
                        submodule._weight_quantizer.enable()

                with torch.enable_grad():
                    for it in range(self.max_iter):
                        total_loss = 0.0
                        for i in range(len(self.layers_input[name])):
                            module.zero_grad()
                            x = self.layers_input[name][i]
                            x = x.to(self.device)
                            y_quant = module(x)

                            y_ref = layers_output[name][i]
                            y_ref = y_ref.to(self.device)
                            
                            loss = lp_loss(y_quant, y_ref, p=self.p)
                            loss.backward()
                            total_loss += loss.item()
                            optimizer.step()
                        if it % 50 == 0: 
                            print(f"Iteration {it}, Loss: {total_loss}")
        return self.model

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()