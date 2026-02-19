from ..base_model import BaseModel
from .model import FastSpeech2

from .utils import FastSpeech2_Dataset, FastSpeech2_Loss, to_device

import torch
from torch.utils.data import DataLoader

from pytorch_quantization import nn as quant_nn

from collections import defaultdict
import logging

import torch_tensorrt
import time
import os
from utils.utils import _load_or_compile
import numpy as np

logger = logging.getLogger("my_app")

ckpt_path= "checkpoints/fastspeech2.pth.tar"

compile_spec_encoder = {
    "inputs": [
        torch_tensorrt.Input([16, 105, 256]),
    ],
    "enabled_precisions": {torch.int8},
    "debug": True,
    "truncate_long_and_double": True,
    "require_full_compilation": True,
    "ir": "torchscript"
}

compile_spec_decoder = {
    "inputs": [
        torch_tensorrt.Input([16, 795, 256]),
    ],
    "enabled_precisions": {torch.int8},
    "debug": True,
    "truncate_long_and_double": True,
    "require_full_compilation": True,
    "ir": "torchscript"
}

nwarmup=50
nruns=1000

class FastSpeech2Model(BaseModel):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        self.model = FastSpeech2()
        self.device = "cuda"
        self.model = self.model.cuda()
        self.init_dataloader()
        self.loss = FastSpeech2_Loss().to(self.device)
    
    def evaluate(self):
        self.model.eval()
        self.model.requires_grad_ = False
        loss_sums = [0 for _ in range(6)]
        for batchs in self.val_loader:
            for batch in batchs:
                batch = to_device(batch, self.device)
                with torch.no_grad():
                    # Forward
                    output = self.model(*(batch[2:]))

                    # Cal Loss
                    losses = self.loss(batch, output)

                    for i in range(len(losses)):
                        loss_sums[i] += losses[i].item() * len(batch[0])

        loss_means = [loss_sum / self.len_dataset for loss_sum in loss_sums]

        message = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            *([l for l in loss_means])
        )
        logger.info(message)
    
    def objective_function(self):
        self.model.eval()
        self.model.requires_grad_ = False
        loss_sums = [0 for _ in range(6)]
        j = 0
        for batchs in self.train_loader:
            for batch in batchs:
                batch = to_device(batch, self.device)
                with torch.no_grad():
                    # Forward
                    output = self.model(*(batch[2:]))

                    # Cal Loss
                    losses = self.loss(batch, output)

                    for i in range(len(losses)):
                        loss_sums[i] += losses[i].item() * len(batch[0])
                    j += 1
                    if j >= 100:
                        break
                if j >= 100:
                    break
            if j >= 100:
                break

        loss_means = [loss_sum / self.len_dataset for loss_sum in loss_sums]
        loss = loss_means[2]

        return loss
    
    def load(self):
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
    
    def init_dataloader(self):
        """Initialize dataloader"""
        batch_size = 16

        val_dataset = FastSpeech2_Dataset(filename='val.txt')
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )
        self.len_dataset = len(val_dataset)

        train_dataset = FastSpeech2_Dataset(filename='train.txt', sort=True, drop_last=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=train_dataset.collate_fn,
        )
    
    def collect_stats(self, num_calib_batch):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        i = 0
        for batchs in self.train_loader:
            for batch in batchs:
                batch = to_device(batch, "cuda")
                with torch.no_grad():
                    self.model(*(batch[2:]))
                    i += 1
                if i >= num_calib_batch:
                    break
            if i >= num_calib_batch:
                    break

        # Disable calibrators
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
    
    def collect_layer_inputs(self, ptq, layers_to_quantize, num_batches):
        """Collect stats for PTQ."""
        list_method = ["adaround", "noisyquant", "ditas", "hyq", "brecq", "bc"]
        if ptq in list_method:
            self.model.eval()
            layer_inputs = defaultdict(list)

            hooks = []

            match_by_name = all(isinstance(x, str) for x in layers_to_quantize)
            match_by_type = all(isinstance(x, type) for x in layers_to_quantize)

            # Register forward hooks to capture inputs
            def get_hook(name):
                def hook(module, inp, out):
                    # inp is a tuple
                    layer_inputs[name].append(inp[0].detach().cpu())
                return hook

            for name, module in self.model.named_modules():
                if match_by_type:
                    if any(isinstance(module, t) for t in layers_to_quantize):
                        hooks.append(module.register_forward_hook(get_hook(name)))
                elif match_by_name:
                    last_name = name.split(".")[-1]
                    if any(substring in last_name for substring in layers_to_quantize):
                        hooks.append(module.register_forward_hook(get_hook(name)))
                else:
                    raise ValueError("layers_to_quantize should contain only layer types OR only names.")

            # Run a few batches to collect activations
            i = 0
            for batchs in self.train_loader:
                for batch in batchs:
                    batch = to_device(batch, "cuda")
                    with torch.no_grad():
                        self.model(*(batch[2:]))
                        i += 1
                    if i >= num_batches:
                        break
                if i >= num_batches:
                        break

            # Remove hooks
            for h in hooks:
                h.remove()
            return layer_inputs
        else:
            return None
        
    def export(self):
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        jit_path_encoder = "torchscript_file/fastspeech2/fastspeech2_model_encoder.jit.pt"
        jit_path_decoder = "torchscript_file/fastspeech2/fastspeech2_model_decoder.jit.pt"

        if not os.path.exists(jit_path_encoder) or not os.path.exists(jit_path_decoder):
            self.model.eval()
            input_data_encoder = torch.randn(16, 105, 256, device="cuda")
            mask_encoder = torch.zeros(16, 105, dtype=torch.bool, device="cuda")
            attn_mask_encoder = torch.zeros(16, 105, 105, dtype=torch.bool, device="cuda")

            input_data_decoder = torch.randn(16, 795, 256, device="cuda")
            mask_decoder = torch.zeros(16, 795, dtype=torch.bool, device="cuda")
            attn_mask_decoder = torch.zeros(16, 795, 795, dtype=torch.bool, device="cuda")

            jit_model_encoder = torch.jit.trace(self.model.encoder.layer_stack._modules['0'], (input_data_encoder, mask_encoder, attn_mask_encoder))
            jit_model_decoder = torch.jit.trace(self.model.decoder.layer_stack._modules['0'], (input_data_decoder, mask_decoder, attn_mask_decoder))
           
            torch.jit.save(jit_model_encoder, jit_path_encoder)
            torch.jit.save(jit_model_decoder, jit_path_decoder)
        
        self.jit_model_encoder = torch.jit.load(jit_path_encoder).eval()
        self.jit_model_decoder = torch.jit.load(jit_path_decoder).eval()

        self.trt_fastspeech2_encoder_linear = _load_or_compile(
            self.jit_model_encoder.slf_attn.w_qs,
            compile_spec_encoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_encoder_linear.ts"
        )
        self.trt_fastspeech2_encoder_fc = _load_or_compile(
            self.jit_model_encoder.slf_attn.fc,
            compile_spec_encoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_encoder_fc.ts"
        )
        self.trt_fastspeech2_encoder_layernorm = _load_or_compile(
            self.jit_model_encoder.slf_attn.layer_norm,
            compile_spec_encoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_encoder_layernorm.ts"
        )
        self.trt_fastspeech2_encoder_pos = _load_or_compile(
            self.jit_model_encoder.pos_ffn,
            compile_spec_encoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_encoder_pos.ts"
        )

        self.trt_fastspeech2_decoder_linear = _load_or_compile(
            self.jit_model_decoder.slf_attn.w_qs,
            compile_spec_decoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_decoder_linear.ts"
        )
        self.trt_fastspeech2_decoder_fc = _load_or_compile(
            self.jit_model_decoder.slf_attn.fc,
            compile_spec_decoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_decoder_fc.ts"
        )
        self.trt_fastspeech2_decoder_layernorm = _load_or_compile(
            self.jit_model_decoder.slf_attn.layer_norm,
            compile_spec_decoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_decoder_layernorm.ts"
        )
        self.trt_fastspeech2_decoder_pos = _load_or_compile(
            self.jit_model_decoder.pos_ffn,
            compile_spec_decoder,
            "torchscript_file/fastspeech2/trt_fastspeech2_decoder_pos.ts"
        )
    
    def benchmark(self, mode="int"):

        assert mode in ["int", "fp32"], "mode must be 'int' or 'fp32'"
        self.model = self.model.cuda()
        self.model.eval()

        print(f"Benchmark mode: {mode.upper()}")

        def run_int(x_enc, x_dec):
            for _ in range(4):
                for _ in range(3):
                    x_enc = self.trt_fastspeech2_encoder_linear(x_enc)
                x_enc = self.trt_fastspeech2_encoder_fc(x_enc)
                x_enc = self.trt_fastspeech2_encoder_layernorm(x_enc)
                x_enc = self.trt_fastspeech2_encoder_pos(x_enc)
            for _ in range(6):
                for _ in range(3):
                    x_dec = self.trt_fastspeech2_decoder_linear(x_dec)
                x_dec = self.trt_fastspeech2_decoder_fc(x_dec)
                x_dec = self.trt_fastspeech2_decoder_layernorm(x_dec)
                x_dec = self.trt_fastspeech2_decoder_pos(x_dec)
            return x_enc, x_dec

        def run_fp32(x_enc, x_dec):
            for _ in range(4):
                for _ in range(3):
                    x_enc = self.model.encoder.layer_stack._modules['0'].slf_attn.w_qs(x_enc)
                x_enc = self.model.encoder.layer_stack._modules['0'].slf_attn.fc(x_enc)
                x_enc = self.model.encoder.layer_stack._modules['0'].slf_attn.layer_norm(x_enc)
                x_enc = self.model.encoder.layer_stack._modules['0'].pos_ffn(x_enc)
            for _ in range(6):
                for _ in range(3):
                    x_dec = self.model.decoder.layer_stack._modules['0'].slf_attn.w_qs(x_dec)
                x_dec = self.model.decoder.layer_stack._modules['0'].slf_attn.fc(x_dec)
                x_dec = self.model.decoder.layer_stack._modules['0'].slf_attn.layer_norm(x_dec)
                x_dec = self.model.decoder.layer_stack._modules['0'].pos_ffn(x_dec)
            return x_enc, x_dec

        # ---------------- Warmup ----------------
        print("Warmup...")
        with torch.no_grad():
            input_data_encoder = torch.randn(16, 105, 256, device="cuda")
            input_data_decoder = torch.randn(16, 795, 256, device="cuda")
            for _ in range(nwarmup):
                if mode == "int":
                    _ = run_int(input_data_encoder, input_data_decoder)
                else:
                    _ = run_fp32(input_data_encoder, input_data_decoder)

        torch.cuda.synchronize()

        # ---------------- Timing ----------------
        print("Start timing...")
        timings = []

        with torch.no_grad():
            input_data_encoder = torch.randn(16, 105, 256, device="cuda")
            input_data_decoder = torch.randn(16, 795, 256, device="cuda")
            for i in range(1, nruns + 1):
                start = time.time()
                if mode == "int":
                    out = run_int(input_data_encoder, input_data_decoder)
                else:
                    out = run_fp32(input_data_encoder, input_data_decoder)
                torch.cuda.synchronize()
                end = time.time()

                timings.append(end - start)

                if i % 100 == 0:
                    print(f"Iter {i}/{nruns}, avg {np.mean(timings)*1000:.2f} ms")

        print(f"Last input shape: {input_data_encoder.shape}")
        if isinstance(out, tuple):
            print(f"Last output shape: {out[0].shape}")
        else:
            print(f"Last output shape: {out.shape}")

        print(f"Average {mode.upper()} latency: {np.mean(timings)*1000:.2f} ms")


    