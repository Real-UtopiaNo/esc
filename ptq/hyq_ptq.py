import pytorch_quantization
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .base_ptq import PTQBase
import logging

logger = logging.getLogger("my_app")

class HyQPTQ(PTQBase):
    def __init__(self, model, layers_input=None, max_iter=2, lr = 0.01, int_softmax = False, device = "cuda"):
        self.model = model
        self.layers_input = layers_input
        self.max_iter = max_iter
        self.lr = lr
        self.int_softmax = int_softmax
        self.device = device
    
    def run(self):
        # 1. Calibrate or search clipping points per layer
        if self.int_softmax:
            for name, module in self.model.named_modules():
                if type(module) == pytorch_quantization.nn.QuantSoftmax:
                    print(name)
                    q_softmax = pytorch_quantization.nn.QuantIntSoftmax(module.dim)

                    q_softmax._input_quantizer = module._input_quantizer
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.model.get_submodule(parent_name) if parent_name else self.model

                    setattr(parent_module, child_name, q_softmax)

        keys_list = list(self.layers_input.keys())
        for name, module in self.model.named_modules():
            if name in keys_list:
                print(name)
                if isinstance(module, pytorch_quantization.nn.QuantConv1d):
                    alpha_shape = (1, self.layers_input[name][0].shape[1], 1)
                elif isinstance(module, pytorch_quantization.nn.QuantConv2d):
                    alpha_shape = (1, self.layers_input[name][0].shape[1], 1, 1)
                
                alpha = nn.Parameter(torch.ones(alpha_shape, device=self.device))

                optimizer = optim.Adam([alpha], lr=self.lr)
                criterion = nn.MSELoss()
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter)

                # Initialize variables to store the minimum loss and best alpha
                min_loss = float('inf')
                best_alpha = None
                counter_freeze = 0

                with torch.enable_grad():
                    for it in range(self.max_iter):
                        total_loss = 0.0
                        optimizer.zero_grad()
                        for x in self.layers_input[name]:
                            x.requires_grad_()
                            x = x.to(self.device)

                            if isinstance(module, pytorch_quantization.nn.QuantConv1d):
                                y_ref = F.conv1d(x, module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
                            elif isinstance(module, pytorch_quantization.nn.QuantConv2d):
                                y_ref = F.conv2d(x, module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)

                            if module.groups == 1:
                                smoothed_w = module.weight * alpha
                            else:
                                if isinstance(module, pytorch_quantization.nn.QuantConv1d):
                                    smoothed_w = module.weight * alpha.reshape(-1, 1, 1)
                                elif isinstance(module, pytorch_quantization.nn.QuantConv2d):
                                    smoothed_w = module.weight * alpha.reshape(-1, 1, 1, 1)
                            
                            smoothed_x = x / alpha
                            x_q = module._input_quantizer(smoothed_x)
                            w_q = module._weight_quantizer(smoothed_w)

                            if isinstance(module, pytorch_quantization.nn.QuantConv1d):
                                y_q = F.conv1d(x_q, w_q, module.bias, module.stride, module.padding, module.dilation, module.groups)
                            elif isinstance(module, pytorch_quantization.nn.QuantConv2d):
                                y_q = F.conv2d(x_q, w_q, module.bias, module.stride, module.padding, module.dilation, module.groups)
                            
                            loss = criterion(y_ref, y_q)
                            loss = loss / len(self.layers_input[name])
                            total_loss += loss.item()

                            loss.backward()


                        if total_loss < min_loss:
                            min_loss = total_loss
                            best_alpha = alpha.clone()
                            counter_freeze = 0
                        else:
                            counter_freeze +=1
                            if counter_freeze > 50:
                                logger.info('50 iterations without improving: break')
                                break

                        if it % 50 == 0:
                            logger.info(f"Iteration {it}, Loss: {total_loss}, Best loss: {min_loss}")

                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                if module.groups == 1:
                    module._weight_quantizer.init_ptq("hyq", alpha=best_alpha)
                else:
                    if isinstance(module, pytorch_quantization.nn.QuantConv1d):
                        module._weight_quantizer.init_ptq("hyq", alpha=best_alpha.reshape(-1, 1, 1))
                    elif isinstance(module, pytorch_quantization.nn.QuantConv2d):
                        module._weight_quantizer.init_ptq("hyq", alpha=best_alpha.reshape(-1, 1, 1, 1))

                module._input_quantizer.init_ptq("hyq", alpha=(1 / best_alpha))

        return self.model