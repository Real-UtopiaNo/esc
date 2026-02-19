from models.mpsenet.model import FeedForwardModule, AttentionModule
from models.conformer.model import FeedForwardBlock
from timm.models.vision_transformer import Block
import pytorch_quantization
import torch

class BaseModel:
    def __init__(self):
        self.model = None

    def load(self):
        raise NotImplementedError("Load method must be implemented in subclass.")

    def evaluate(self):
        """Evaluate model on dataset."""
        raise NotImplementedError("Evaluate method must be implemented in subclass.")
    
    def objective_function(self):
        """Objective function for CMA-ES."""
        raise NotImplementedError("Objective function method must be implemented in subclass.")

    def export(self, **kwargs):
        """Export model to this model."""
        raise NotImplementedError("Evaluate method must be implemented in subclass.")
    
    def collect_stats(self, num_calib_batch, **kwargs):
        """Collect stats for PTQ."""
        raise NotImplementedError("Collect stat method must be implemented in subclass.")
    
    def collect_layer_inputs(self, ptq, layers_to_quantize, num_batches):
        """Collect layers inputs for PTQ."""
        raise NotImplementedError("Collect layers inputs method must be implemented in subclass.")
    
    def init_dataloader(self):
        """Initialize dataloader"""
        raise NotImplementedError("Initialize dataloader method must be implemented in subclass.")
    
    def pre_calibration(self, ptq):
        """Operation before PTQ"""
        if ptq == "smoothquant":
            for name, module in self.model.named_modules():
                if isinstance(module, FeedForwardModule):
                    fc1 = module.ffm._modules['1']
                    fc1._input_quantizer._calibrator._axis = 2
                if isinstance(module, AttentionModule):
                    module.attn._input_quantizer._calibrator._axis = 2
                if isinstance(module, FeedForwardBlock):
                    fc1 = module.module._modules['1']
                    fc1._input_quantizer._calibrator._axis = 2
                if isinstance(module, Block):
                    module.attn.qkv._input_quantizer._calibrator._axis = 2
                    module.mlp.fc1._input_quantizer._calibrator._axis = 2
        elif ptq == "repqvit":
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
                    if name.split('.')[-1] == "post_norm":
                        pass
                    elif name.split('.')[-1] == 'layer_norm_2' or name.split('.')[-1] == 'layer_norm_1':
                        pass
                    elif hasattr(father_module, 'attn') and name.split('.')[-1] == "norm1":
                        next_module = father_module.attn.qkv
                        next_module._input_quantizer._calibrator._axis = 2
                    elif hasattr(father_module, 'mlp') and name.split('.')[-1] == "norm2":
                        next_module = father_module.mlp.fc1
                        next_module._input_quantizer._calibrator._axis = 2
                    elif hasattr(father_module, 'attn'):
                        next_module = father_module.attn
                        next_module._input_quantizer._calibrator._axis = 2
                    elif hasattr(father_module, 'mlp'):
                        next_module = father_module.mlp.fc1
                        next_module._input_quantizer._calibrator._axis = 2
                    elif hasattr(father_module, 'reduction'):
                        next_module = father_module.reduction
                        next_module._input_quantizer._calibrator._axis = 2
                    elif hasattr(father_module, 'fc'):
                        next_module = father_module.fc
                        next_module._input_quantizer._calibrator._axis = 2
                    elif isinstance(father_module, torch.nn.Sequential):
                        suffix = name.split('.')[-1]
                        next_module = father_module._modules[str(int(suffix) + 1)]
                        next_module._input_quantizer._calibrator._axis = 2
        else:
            pass
    
    def export(self):
        """Export model"""
        raise NotImplementedError("Export method must be implemented in subclass.")
    
    def benchmark(self, mode):
        """Benchmark inference speed."""
        raise NotImplementedError("Benchmarking method must be implemented in subclass.")