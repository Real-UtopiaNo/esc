from ..base_model import BaseModel
from .model import ECAPA
from .utils import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf, sample_with_pairs, generate_pairs, fix_length_2d
from utils.utils import _load_or_compile

import torch
import numpy as np
import os
import tqdm
import soundfile
from pathlib import Path

from pytorch_quantization import nn as quant_nn
from collections import defaultdict, OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import logging
import random
import copy

import torch_tensorrt
import time

logger = logging.getLogger("my_app")

ckpt_path = 'checkpoints/ecapa.model'
eval_list = 'dataset/voxceleb/veri_test2.txt'
eval_path = 'dataset/voxceleb/voxceleb1'

train_list = 'dataset/voxceleb/train_list.txt'
train_path = 'dataset/voxceleb/voxceleb2'

calib_list = 'dataset/voxceleb/veri_train.txt'

#min, max in valid 397 and 1302
compile_spec = {"inputs": [torch_tensorrt.Input(min_shape=[1, 80, 397],
                            opt_shape=[1, 80, 536],
                            max_shape=[1, 80, 675],
                            )],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                "require_full_compilation": True,
                "ir": "torchscript"
                }

size_range = (397, 675)
nwarmup=50
nruns=1000

class ECAPAModel(BaseModel):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        self.model = ECAPA()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.init_dataloader()
    
    def evaluate(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        lines = open(eval_list).read().splitlines()
        embeddings = {}
        for idx, file in tqdm.tqdm(enumerate(self.val_loader), total = len(self.val_loader)):
            audio, _  = soundfile.read(file)
            # Full utterance
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = np.stack(feats, axis = 0).astype(float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.model(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.model(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels  = [], []

        for line in lines:			
            embedding_11, embedding_12 = embeddings[os.path.join(eval_path,line.split()[1])]
            embedding_21, embedding_22 = embeddings[os.path.join(eval_path,line.split()[2])]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))
            
        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        logger.info("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
        #EER 0.97%, minDCF 0.0717%
    
    def objective_function(self, validate=False):
        self.model = self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        layers_to_quantize=[nn.Conv1d, nn.Conv2d, nn.Linear, nn.LayerNorm]

        lines = open(calib_list).read().splitlines()
        embeddings = {}

        if not validate:
            loader = self.calib_loader[:80]
        else:
            loader = self.calib_loader[-60:]

        for idx, file in enumerate(loader):
            audio, _  = soundfile.read(file)
            # Full utterance
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = np.stack(feats, axis = 0).astype(float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.model(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.model(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

            if not validate:
                for name, module in self.model.named_modules():
                    if hasattr(module, "input_quantizer") and any(isinstance(module, t) for t in layers_to_quantize):
                        module.input_quantizer.disable()
                
                with torch.no_grad():
                    embedding_1_ori = self.model(data_1)
                    embedding_1_ori = F.normalize(embedding_1_ori, p=2, dim=1)
                    embedding_2_ori = self.model(data_2)
                    embedding_2_ori = F.normalize(embedding_2_ori, p=2, dim=1)

                for name, module in self.model.named_modules():
                    if hasattr(module, "input_quantizer") and any(isinstance(module, t) for t in layers_to_quantize):
                        module.input_quantizer.enable()
                
                loss = F.mse_loss(embedding_1_ori, embedding_1).item() + F.mse_loss(embedding_2_ori, embedding_2).item()
                total_loss += loss
            
        scores, labels  = [], []

        for line in lines:
            if os.path.join(train_path,line.split()[1]) in embeddings.keys() and os.path.join(train_path,line.split()[2]) in embeddings.keys():
                embedding_11, embedding_12 = embeddings[os.path.join(train_path,line.split()[1])]
                embedding_21, embedding_22 = embeddings[os.path.join(train_path,line.split()[2])]
                # Compute the scores
                score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
                score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
                score = (score_1 + score_2) / 2
                score = score.detach().cpu().numpy()
                scores.append(score)
                labels.append(int(line.split()[0]))
            
        # Coumpute EER and minDCF
        avg_loss = total_loss / len(scores)
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        return EER / 100, avg_loss
    
    def load(self):
        checkpoint = torch.load(ckpt_path)
        checkpoint_new = OrderedDict()

        old_prefix = "speaker_encoder."

        for k, v in checkpoint.items():
            if k.startswith(old_prefix):
                k = k[len(old_prefix):]

            # remap to your model structure
            if k.startswith("torchfbank"):
                new_key = "feature_extractor." + k
            else:
                new_key = "ecapa_compute." + k
            
            if new_key == "ecapa_compute.speaker_loss.weight":
                continue

            checkpoint_new[new_key] = v
            
        self.model.load_state_dict(checkpoint_new)
    
    def init_dataloader(self):
        """Initialize dataloader"""
        files = []
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(os.path.join(eval_path,line.split()[1]))
            files.append(os.path.join(eval_path,line.split()[2]))
        val_setfiles = list(set(files))
        val_setfiles.sort()
        self.val_loader = val_setfiles

        files = []
        lines = open(train_list).read().splitlines()
        for line in lines:
            filename = os.path.join(train_path,line.split()[1])
            if os.path.exists(filename):
                files.append(filename)
        train_setfiles = list(set(files))
        train_setfiles.sort()
        self.train_loader = train_setfiles

        self.calib_loader = sample_with_pairs(self.train_loader, n=100)
        generate_pairs(self.calib_loader, output_file=calib_list)

    def collect_stats(self, num_calib_batch):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()
        
        self.model.eval()
        for i, file in enumerate(self.calib_loader):
            audio, _  = soundfile.read(file)
            # Full utterance
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = np.stack(feats, axis = 0).astype(float)
            # Speaker embeddings
            self.model(data_1)
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
            
            self.model.eval()
            for i, file in enumerate(self.calib_loader):
                if i >= num_batches:
                    break
                audio, _  = soundfile.read(file)
                # Full utterance
                data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()

                # Spliited utterance matrix
                max_audio = 300 * 160 + 240
                if audio.shape[0] <= max_audio:
                    shortage = max_audio - audio.shape[0]
                    audio = np.pad(audio, (0, shortage), 'wrap')
                feats = []
                startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
                for asf in startframe:
                    feats.append(audio[int(asf):int(asf)+max_audio])
                feats = np.stack(feats, axis = 0).astype(float)
                # Speaker embeddings
                _ = self.model(data_1)

            # Remove hooks
            for h in hooks:
                h.remove()
            return layer_inputs
        else:
            return None
    
    def export(self):
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        jit_path = "torchscript_file/ecapa/ecapa_model.jit.pt"
        os.makedirs(os.path.dirname(jit_path), exist_ok=True)

        if not os.path.exists(jit_path):
            self.model.eval()
            for i, file in enumerate(self.calib_loader):
                audio, _  = soundfile.read(file)
                # Full utterance
                data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
                data_1 = fix_length_2d(data_1, 74752)
                jit_model = torch.jit.trace(self.model, data_1)
                torch.jit.save(jit_model, jit_path)
                if i  > 0:
                    break

        self.jit_model = torch.jit.load(jit_path).eval()
        
        self.trt_ecapa_compute = _load_or_compile(
            self.jit_model.ecapa_compute,
            compile_spec,
            "torchscript_file/ecapa/trt_ecapa_compute.ts"
        )

        a = 1
    
    def benchmark(self, mode="int"):

        assert mode in ["int", "fp32"], "mode must be 'int' or 'fp32'"
        self.model = self.model.cuda()
        self.model.eval()

        print(f"Benchmark mode: {mode.upper()}")

        def run_int(x):
            x = self.trt_ecapa_compute(x)
            return x

        def run_fp32(x):
            return self.model.ecapa_compute(x)

        # ---------------- Warmup ----------------
        print("Warmup...")
        with torch.no_grad():
            D = random.randint(*size_range)
            input_data = torch.randn(1, 80, D, device="cuda")
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
            D = random.randint(*size_range)
            input_data = torch.randn(1, 80, D, device="cuda")
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