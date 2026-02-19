from ..base_model import BaseModel
from .model import AST
from .utils import AudiosetDataset, calculate_stats

import torch
import numpy as np

from pytorch_quantization import nn as quant_nn
from collections import defaultdict
import logging

import torch_tensorrt
import time
import torch.nn as nn

from utils.utils import _load_or_compile
import os

logger = logging.getLogger("my_app")

ckpt_path = '/home/S3G-LABS/u1004/Documents/audio/git_task/ast/pretrained_models/speechcommands_10_10_0.9812.pth'
data_train = '/home/S3G-LABS/u1004/Documents/audio/git_task/ast/data/datafiles/speechcommand_train_data.json'
data_val = '/home/S3G-LABS/u1004/Documents/audio/git_task/ast/data/datafiles/speechcommand_eval_data.json'
label_csv = '/home/S3G-LABS/u1004/Documents/audio/git_task/ast/data/speechcommands_class_labels_indices.csv'
audio_conf = {'num_mel_bins': 128, 'target_length': 128, 'freqm': 48, 'timem': 48, 'mixup': 0, 'dataset': 'speechcommands', 'mode': 'train', 'mean': -6.845978, 'std': 5.5654526, 'noise': True}
val_audio_conf = {'num_mel_bins': 128, 'target_length': 128, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'speechcommands', 'mode': 'evaluation', 'mean': -6.845978, 'std': 5.5654526, 'noise': False}

compile_spec = {"inputs": [torch_tensorrt.Input([12, 128, 128])],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                #"allow_shape_tensors": True,
                "require_full_compilation": True,
                "ir": "torchscript"
                }

nwarmup=50
nruns=1000

class ASTModel(BaseModel):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        self.model = AST()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.init_dataloader()
        a = 1
    
    def evaluate(self):
        self.model = self.model.to(self.device)
        # switch to evaluate mode
        self.model.eval()

        A_predictions = []
        A_targets = []
        A_loss = []
        with torch.no_grad():
            for i, (audio_input, labels) in enumerate(self.val_loader):
                audio_input = audio_input.to(self.device)
                # compute output
                audio_output = self.model(audio_input)
                audio_output = torch.sigmoid(audio_output)
                predictions = audio_output.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)

                labels = labels.to(self.device)

            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)
            loss = 0
            stats = calculate_stats(audio_output, target)


        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))

        logger.info("Accuracy: {:.6f}".format(eval_acc))
        logger.info("AUC: {:.6f}".format(eval_mAUC))

        self.model = self.model.cuda()
        self.model.eval()
    
    def train_calib(self):
        self.model = self.model.to(self.device)
        # switch to evaluate mode
        self.model.eval()

        A_predictions = []
        A_targets = []
        A_loss = []
        with torch.no_grad():
            for i, (audio_input, labels) in enumerate(self.train_loader):
                audio_input = audio_input.to(self.device)
                # compute output
                audio_output = self.model(audio_input)
                audio_output = torch.sigmoid(audio_output)
                predictions = audio_output.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)

                labels = labels.to(self.device)

                if i >= 100:
                    break

            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)
            loss = 0
            stats = calculate_stats(audio_output, target)


        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])

        return 1 - eval_acc, 0
    
    def load(self):
        sd = torch.load(ckpt_path, map_location=self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(sd)
    
    def init_dataloader(self):
        """Initialize dataloader"""
        train_dataset = AudiosetDataset(data_train, label_csv=label_csv, audio_conf=audio_conf)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, pin_memory=True)

        val_dataset = AudiosetDataset(data_val, label_csv=label_csv, audio_conf=val_audio_conf)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=24, shuffle=False, pin_memory=True)
    
    def collect_stats(self, num_calib_batch):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        self.model.module.v.head = torch.nn.Identity()
        self.model.module.v.head_dist = torch.nn.Identity()
        for i, (audio_input, labels) in enumerate(self.train_loader):
            audio_input = audio_input.to(self.device)
            self.model(audio_input)
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
                    layer_inputs[name].append(inp[0].detach().cpu().float())
                return hook

            for name, module in self.model.named_modules():
                if match_by_type:
                    if any(isinstance(module, t) for t in layers_to_quantize):
                        hooks.append(module.register_forward_hook(get_hook(name)))
                elif match_by_name:
                    last_name = name.split(".")[-1]
                    if any(substring == last_name for substring in layers_to_quantize):
                        hooks.append(module.register_forward_hook(get_hook(name)))
                else:
                    raise ValueError("layers_to_quantize should contain only layer types OR only names.")
            
            for i, (audio_input, labels) in enumerate(self.train_loader):
                if i >= num_batches:
                    break
                audio_input = audio_input.to(self.device)
                # compute output
                _ = self.model(audio_input)

            # Remove hooks
            for h in hooks:
                h.remove()
            return layer_inputs
        else:
            return None
    
    def export(self):
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        jit_path = "torchscript_file/ast/ast_model.jit.pt"

        model_to_trace = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        model_to_trace.eval()

        if not os.path.exists(jit_path):
            with torch.no_grad():
                for i, (audio_input, labels) in enumerate(self.train_loader):
                    audio_input = audio_input.to(self.device).float()
                    jit_model = torch.jit.trace(model_to_trace, audio_input)
                    torch.jit.save(jit_model, jit_path)
                    if i  > 0:
                        break

        self.jit_model = torch.jit.load(jit_path).eval()

        self.trt_ast = _load_or_compile(
            self.jit_model,
            compile_spec,
            "torchscript_file/ast/trt_ast.ts"
        )

        a = 1
    
    def benchmark(self, mode="int"):

        assert mode in ["int", "fp32"], "mode must be 'int' or 'fp32'"
        self.model = self.model.cuda()
        self.model.eval()

        print(f"Benchmark mode: {mode.upper()}")

        def run_int(x):
            x = self.trt_ast(x)
            return x

        def run_fp32(x):
            return self.model.module(x)

        # ---------------- Warmup ----------------
        print("Warmup...")
        with torch.no_grad():
            input_data = torch.randn(12, 128, 128, device="cuda")
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
            input_data = torch.randn(12, 128, 128, device="cuda")
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