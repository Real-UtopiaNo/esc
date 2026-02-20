import pytorch_quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ptq import PTQBase

class BCPTQ(PTQBase):
    def __init__(self, model, layers_input=None, device = "cuda"):
        self.model = model
        self.layers_input = layers_input
        self.device = device
    
    def run(self):
        keys_list = list(self.layers_input.keys())
        for name, module in self.model.named_modules():
            if name in keys_list:
                reduce_shape=[]
                
                for i in range(self.layers_input[name][0].dim() - 1):
                    reduce_shape.append(i)
                
                with torch.no_grad():
                    if type(module) == pytorch_quantization.nn.QuantLayerNorm:
                        bias = torch.nn.Parameter(torch.zeros(self.layers_input[name][0].shape[-1]))
                    elif type(module) == pytorch_quantization.nn.QuantLinear:
                        bias = torch.nn.Parameter(torch.zeros(module.out_features))
                    bias = bias.to(self.device)
                    for i, x in enumerate(self.layers_input[name]):
                        x = x.to(self.device)
                        y_quantized = module(x)
                        
                        module._weight_quantizer.disable()
                        module._input_quantizer.disable()
                        y_ref = module(x)
                        module._weight_quantizer.enable()
                        module._input_quantizer.enable()
                        bias_update = y_ref - y_quantized #you can add +bias to test
                        bias_update = torch.mean(bias_update,tuple(reduce_shape))
                        bias = (bias*(i)+bias_update.view(-1))/(i + 1)
                        
                    module.bias_bc = bias

        for name, module in self.model.named_modules():
            if hasattr(module, "bias_bc"):
                original_forward = module.forward

                # Define the new forward
                def modified_forward(x, original_forward=original_forward, module=module):
                    out = original_forward(x)
                    return out + module.bias_bc  # add your custom bias

                module.forward = modified_forward

        return self.model