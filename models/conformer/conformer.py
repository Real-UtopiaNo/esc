from ..base_model import BaseModel
from .model import Conformer

from .utils import GreedyCTCDecoder, load_featurizer, fix_length
from utils.utils import _load_or_compile

import torch
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Subset, DataLoader
from pytorch_quantization import nn as quant_nn
from jiwer import wer, cer
from tqdm import tqdm

from collections import OrderedDict, defaultdict
import logging

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset
import os
import torch_tensorrt
import time
import numpy as np


logger = logging.getLogger("my_app")

ckpt_path = "checkpoints/conformer.ckpt"
token_path = "dataset/conformer/tokens.txt"
data_path = "/data/librispeech"

compile_spec_encoder_conv = {"inputs": [torch_tensorrt.Input([1, 734, 80])],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                #"allow_shape_tensors": True,
                "require_full_compilation": True,
                "ir": "torchscript"
                }

compile_spec_encoder_linear = {"inputs": [torch_tensorrt.Input([1, 182, 2736])],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                #"allow_shape_tensors": True,
                "require_full_compilation": True,
                "ir": "torchscript"
                }

compile_spec_encoder_block = {"inputs": [torch_tensorrt.Input([1, 182, 144])],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                #"allow_shape_tensors": True,
                "require_full_compilation": True,
                "ir": "torchscript"
                }

compile_spec_encoder_block_input = {"inputs": [torch_tensorrt.Input([1, 182, 144]), torch_tensorrt.Input([1, 182, 144])],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                #"allow_shape_tensors": True,
                "require_full_compilation": True,
                "ir": "torchscript"
                }

nwarmup=50
nruns=1000

#FP32
# Average WER: 0.1594
# Average CER: 0.0484

class ConformerModel(BaseModel):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        self.model = Conformer()
        with open(token_path, "r") as f:
            tokens = f.read().splitlines()
        self.decoder = GreedyCTCDecoder(tokens)
        self.init_dataloader()
    
    def evaluate(self):
        self.model = self.model.cuda()
        self.model.eval()

        total_wer, total_cer = 0.0, 0.0
        sample_count = 0

        iterable = enumerate(self.val_dataset)

        for idx, (waveform, sample_rate, reference, *_ ) in tqdm(iterable, total=len(self.val_dataset), desc="Processing"):
            try:
                waveform = load_featurizer()(waveform)
                waveform = waveform.to("cuda")
                waveform = waveform.permute(0, 2, 1)

                emission = self.model(waveform)

                greedy_result = self.decoder(emission[0])
                prediction = " ".join(greedy_result)

                reference = reference.lower()
                prediction = prediction.lower()

                sample_wer = wer(reference, prediction)
                sample_cer = cer(reference, prediction)
                total_wer += sample_wer
                total_cer += sample_cer
                sample_count += 1

            except Exception as e:
                tqdm.write(f"[{idx}] ⚠️ Error during transcription: {e}")
                continue

        if sample_count == 0:
            logger.info("❌ No samples processed. Evaluation aborted.")
        else:
            avg_wer = total_wer / sample_count
            avg_cer = total_cer / sample_count

            logger.info(f"Average WER: {avg_wer:.4f}")
            logger.info(f"Average CER: {avg_cer:.4f}")
    
    def objective_function(self, validate=False):
        self.model = self.model.cuda()
        self.model.eval()

        dataset = self.train_dataset
        n = 100

        if validate:
            indices = list(range(n-20, n))
        else:
            indices = list(range(0, 100))

        layers_to_quantize=[nn.Conv1d, nn.Conv2d, nn.Linear, nn.LayerNorm]
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=1, num_workers=4)

        total_cer = 0.0
        sample_count = 0

        with torch.no_grad():
            for waveform, sample_rate, reference, *_ in loader:
                waveform = waveform.squeeze(0)
                waveform = load_featurizer()(waveform)
                waveform = waveform.cuda()
                waveform = waveform.permute(0, 2, 1)

                emission = self.model(waveform)

                greedy_result = self.decoder(emission[0])
                prediction = " ".join(greedy_result)

                reference = reference[0].lower()
                prediction = prediction.lower()

                total_cer += cer(reference, prediction)
                sample_count += 1

                if not validate:
                    for name, module in self.model.named_modules():
                        if hasattr(module, "input_quantizer") and any(isinstance(module, t) for t in layers_to_quantize):
                            module.input_quantizer.disable()
                    
                    emission_ori = self.model(waveform)

                    for name, module in self.model.named_modules():
                        if hasattr(module, "input_quantizer") and any(isinstance(module, t) for t in layers_to_quantize):
                            module.input_quantizer.enable()

        avg_cer = total_cer / sample_count
        return avg_cer
    
    def load(self):
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        model_state_dict = checkpoint['state_dict']

        def remap_conv_block_keys(name):
            """
            Remap keys for conv_block.module after adding ConstantPad1d at index 2.
            Shifts the indices of layers after the padding by +1.
            """
            if "conv_block.module." in name:
                parts = name.split("conv_block.module.")
                idx_and_rest = parts[1].split('.', 1)
                idx = int(idx_and_rest[0])
                rest = idx_and_rest[1] if len(idx_and_rest) > 1 else ''

                # Shift indices for layers after the padding (which is at index 2)
                if idx >= 2:
                    idx += 1

                name = parts[0] + f"conv_block.module.{idx}" + ('.' + rest if rest else '')
            return name

        # Initialize state dictionaries
        encoder_state_dict = OrderedDict()
        decoder_state_dict = OrderedDict()

        # Separate encoder and decoder state dictionaries
        for k, v in model_state_dict.items():
            k = remap_conv_block_keys(k)
            # For Optimized model
            if k.startswith('model._orig_mod.encoder.'):
                name = k.replace('model._orig_mod.encoder.', '')
                encoder_state_dict[name] = v
            elif k.startswith('model._orig_mod.decoder.'):
                name = k.replace('model._orig_mod.decoder.', '')
                decoder_state_dict[name] = v
            # For Normal model
            elif k.startswith('model.encoder.'):
                name = k.replace('model.encoder.', '')
                encoder_state_dict[name] = v
            elif k.startswith('model.decoder.'):
                name = k.replace('model.decoder.', '')
                decoder_state_dict[name] = v

        # Load state dictionaries into the model
        self.model.encoder.load_state_dict(encoder_state_dict)
        self.model.decoder.load_state_dict(decoder_state_dict)
    
    def init_dataloader(self):
        """Initialize dataloader"""
        self.val_dataset = LIBRISPEECH(data_path, url="test-clean", download=False)
        self.train_dataset = LIBRISPEECH(data_path, url="train-clean-100", download=False)
    
    def collect_stats(self, num_calib_batch):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        iterable = enumerate(self.train_dataset)
        
        for i, (waveform, sample_rate, reference, *_ ) in tqdm(iterable, total=len(self.train_dataset), desc="Processing"):
            waveform = load_featurizer()(waveform)
            waveform = waveform.to("cuda")
            waveform = waveform.permute(0, 2, 1)
            self.model(waveform)
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
            
            iterable = enumerate(self.train_dataset)
        
            for i, (waveform, sample_rate, reference, *_ ) in tqdm(iterable, total=len(self.train_dataset), desc="Processing"):
                if i >= num_batches:
                    break
                waveform = load_featurizer()(waveform)
                waveform = waveform.to("cuda")
                waveform = waveform.permute(0, 2, 1)
                _ = self.model(waveform)

            # Remove hooks
            for h in hooks:
                h.remove()
            return layer_inputs
        else:
            return None
    
    def export(self):
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        jit_path = "torchscript_file/conformer/conformer_model.jit.pt"
        os.makedirs(os.path.dirname(jit_path), exist_ok=True)

        if not os.path.exists(jit_path):
            self.model.eval()
            iterable = enumerate(self.train_dataset)
            with torch.no_grad():
                for i, (waveform, sample_rate, reference, *_ ) in iterable:
                    waveform = load_featurizer()(waveform)
                    waveform = waveform.to("cuda")
                    waveform = waveform.permute(0, 2, 1)
                    waveform = fix_length(waveform, 734)
                    jit_model = torch.jit.trace(self.model, waveform)
                    torch.jit.save(jit_model, jit_path)
                    if i  > 0:
                        break

        self.jit_model = torch.jit.load(jit_path).eval()

        self.trt_encoder_conv = _load_or_compile(
            self.jit_model.encoder.conv_subsample,
            compile_spec_encoder_conv,
            "torchscript_file/conformer/trt_model_conv.ts"
        )

        self.trt_encoder_linear = _load_or_compile(
            self.jit_model.encoder.linear_proj,
            compile_spec_encoder_linear,
            "torchscript_file/conformer/trt_model_linear.ts"
        )

        self.trt_encoder_layers = []

        for i in range(16):
            if i == 0:
                trt_layer = _load_or_compile(
                    self.jit_model.encoder.layers._modules[str(i)],
                    compile_spec_encoder_block,
                    f"torchscript_file/conformer/trt_model_layer{i}.ts"
                )
            else:
                trt_layer = _load_or_compile(
                    self.jit_model.encoder.layers._modules[str(i)],
                    compile_spec_encoder_block_input,
                    f"torchscript_file/conformer/trt_model_layer{i}.ts"
                )
                
            self.trt_encoder_layers.append(trt_layer)
            
    def benchmark(self, mode="int"):

        assert mode in ["int", "fp32"], "mode must be 'int' or 'fp32'"
        self.model = self.model.cuda()
        self.model.eval()

        print(f"Benchmark mode: {mode.upper()}")

        def run_int(x):
            x = self.trt_encoder_conv(x)
        
            x = self.trt_encoder_linear(x)

            for i, layer in enumerate(self.trt_encoder_layers):
                if i == 0:
                    x = layer(x)                
                else:
                    x = layer(x,x)
            return x

        def run_fp32(input_data):
            return self.model.encoder(input_data)

        # ---------------- Warmup ----------------
        print("Warmup...")
        with torch.no_grad():
            input_data = torch.randn(1, 734, 80, device="cuda")
            for _ in range(nwarmup):
                if mode == "int":
                    _ = run_int(input_data)
                else:
                    _ = run_fp32(input_data)

        torch.cuda.synchronize()

        # ---------------- Timing ----------------
        print("Start timing...")
        timings = []

        with torch.no_grad():
            input_data = torch.randn(1, 734, 80, device="cuda")
            for i in range(1, nruns + 1):
                start = time.time()
                if mode == "int":
                    out = run_int(input_data)
                else:
                    out = run_fp32(input_data)
                torch.cuda.synchronize()
                end = time.time()

                timings.append(end - start)

                if i % 100 == 0:
                    print(f"Iter {i}/{nruns}, avg {np.mean(timings)*1000:.2f} ms")

        print(f"Last input shape: {input_data.shape}")
        if isinstance(out, tuple):
            print(f"Last output shape: {out[0].shape}")
        else:
            print(f"Last output shape: {out.shape}")

        print(f"Average {mode.upper()} latency: {np.mean(timings)*1000:.2f} ms")