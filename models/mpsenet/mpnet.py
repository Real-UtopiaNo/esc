from ..base_model import BaseModel
from .model import MPNet

from .utils import mag_pha_stft, mag_pha_istft, sisdr, load_checkpoint, get_dataset_filelist, MPNet_Dataset
from utils.utils import _load_or_compile

from pytorch_quantization import nn as quant_nn

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from rich.progress import track
from collections import defaultdict
from pesq import pesq
import pysepm
import random
import logging
import torch_tensorrt
import time
import shutil

logger = logging.getLogger("my_app")

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sampling_rate = 16000
n_fft = 400
hop_size = 100
win_size = 400
compress_factor = 0.3
segment_size = 32000
batch_size = 2
num_workers = 4
num_gpus = 0

ckpt_path = "checkpoints/mpnet.ckpt"
training_path = "dataset/VoiceBank+DEMAND/training.txt"
input_clean_wavs_dir = "dataset/VoiceBank+DEMAND/wavs_clean/wav_clean"
input_noisy_wavs_dir = "dataset/VoiceBank+DEMAND/wavs_noisy/wav_noisy"

component=2
channels=201
size_range=(232, 544)
middle=386
nwarmup=50
nruns=500

compile_spec_encoder = {"inputs": [torch_tensorrt.Input(min_shape=[1, component, size_range[0], channels],
                            opt_shape=[1, component, middle, channels],
                            max_shape=[1, component, size_range[1], channels],
                            )],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                "allow_shape_tensors": True,
                "require_full_compilation": True,
                "ir": "torchscript" #bad torch_compile
                }
        
compile_spec_freq = {"inputs": [torch_tensorrt.Input(min_shape=[size_range[0], 100, 64],
                            opt_shape=[middle, 100, 64],
                            max_shape=[size_range[1], 100, 64],
                            )],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                #"allow_shape_tensors": True,
                "require_full_compilation": True
                }

compile_spec_time = {"inputs": [torch_tensorrt.Input(min_shape=[100, size_range[0], 64],
                            opt_shape=[100, middle, 64],
                            max_shape=[100, size_range[1], 64],
                            )],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                "allow_shape_tensors": True, #False for norm2,
                "require_full_compilation": True
                }

compile_spec_mask_decoder = {"inputs": [torch_tensorrt.Input(min_shape=[1, 64, size_range[0], 100],
                            opt_shape=[1, 64, middle, 100],
                            max_shape=[1, 64, size_range[1], 100],
                            )],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                "allow_shape_tensors": True, #False for norm2
                "require_full_compilation": True
                }

compile_spec_phase_decoder = {"inputs": [torch_tensorrt.Input(min_shape=[1, 64, size_range[0], 100],
                            opt_shape=[1, 64, middle, 100],
                            max_shape=[1, 64, size_range[1], 100],
                            )],
                "enabled_precisions": {torch.int8},
                "debug": True,
                "truncate_long_and_double": True,
                "allow_shape_tensors": True, #False for norm2
                "require_full_compilation": True
                }

class MPNetModel(BaseModel):
    def __init__(self):
        # Initialize parent class
        super().__init__()
        self.model = MPNet()
        self.init_dataloader()
        a = 1
    
    def evaluate(self, input_dir="dataset/VoiceBank+DEMAND/noisy_testset_wav/", output_dir="garbage"):        
        test_indexes = os.listdir(input_dir)
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for i, index in tqdm(enumerate(test_indexes)):
                self.model = self.model.cuda()
                self.model.eval()
                noisy_wav, _ = librosa.load(os.path.join(input_dir, index), sr=sampling_rate)
                noisy_wav = torch.FloatTensor(noisy_wav).to("cuda")
                norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to("cuda")
                noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
                noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
                x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)
                amp_g, pha_g = self.model(x)

                noisy_amp = noisy_amp.unsqueeze(-1).permute(0, 3, 2, 1)
                amp_g = (noisy_amp * amp_g).permute(0, 3, 2, 1).squeeze(-1)
                pha_g = pha_g.permute(0, 3, 2, 1).squeeze(-1)
                com_g = torch.stack((amp_g*torch.cos(pha_g),
                                        amp_g*torch.sin(pha_g)), dim=-1)

                audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
                audio_g = audio_g / norm_factor
                sf.write(os.path.join(output_dir, index), audio_g.squeeze().detach().cpu().numpy(), sampling_rate, 'PCM_16')
        
        indexes = os.listdir(output_dir)
        metrics = {'pesq_wb':[], 'pesq_nb':[], 'stoi':[], 'sisdr': [], 'apd': []}

        for index in track(indexes):
            
            noisy_wav = os.path.join(input_dir, index)
            clean_wav = os.path.join(output_dir,index)

            clean, _ = librosa.load(clean_wav, sr=sampling_rate)
            noisy, _ = librosa.load(noisy_wav, sr=sampling_rate)
            length = min(len(clean), len(noisy))
            clean = clean[0: length]
            noisy = noisy[0: length]

            pesq_wb_score = pesq(fs=sampling_rate, ref=clean, deg=noisy, mode="wb")
            pesq_nb_score = pesq(fs=sampling_rate, ref=clean, deg=noisy, mode="nb")
            stoi_score = pysepm.stoi(clean, noisy, sampling_rate)
            sisdr_score = sisdr(noisy, clean)

            metrics['pesq_wb'].append(pesq_wb_score)
            metrics['pesq_nb'].append(pesq_nb_score)
            metrics['stoi'].append(stoi_score)
            metrics['sisdr'].append(sisdr_score)

        pesq_wb_mean = np.mean(metrics['pesq_wb'])
        pesq_nb_mean = np.mean(metrics['pesq_nb'])
        stoi_mean = np.mean(metrics['stoi'])
        sisdr_mean = np.mean(metrics['sisdr'])

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        logger.info('PESQ_WB: {:.3f}'.format(pesq_wb_mean))
        logger.info('PESQ_NB: {:.3f}'.format(pesq_nb_mean))
        logger.info('STOI: {:.3f}'.format(stoi_mean*100))
        logger.info('SI-SDR: {:.3f}'.format(sisdr_mean))
    
    def objective_function(self):
        metrics = {'pesq_wb':[], 'pesq_nb':[], 'stoi':[], 'sisdr': [], 'apd': []}
        with torch.no_grad():
            for i, (clean_audio, noisy_audio) in enumerate(self.train_loader):
                noisy_audio = torch.autograd.Variable(noisy_audio.to("cuda", non_blocking=True))
                norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0)).to("cuda")
                noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, n_fft, hop_size, win_size, compress_factor)
                x = torch.stack((noisy_mag, noisy_pha), dim=-1).permute(0, 3, 2, 1)

                amp_g, pha_g = self.model(x)

                noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)
                amp_g = (noisy_mag * amp_g).permute(0, 3, 2, 1).squeeze(-1)
                pha_g = pha_g.permute(0, 3, 2, 1).squeeze(-1)
                com_g = torch.stack((amp_g*torch.cos(pha_g),
                                        amp_g*torch.sin(pha_g)), dim=-1)

                audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
                audio_g = audio_g / norm_factor
                audio_g = audio_g.squeeze()

                clean = clean_audio.detach().cpu().numpy()[0]
                noisy = audio_g.detach().cpu().numpy()[0]
                length = min(len(clean), len(noisy))
                clean = clean[0: length]
                noisy = noisy[0: length]

                pesq_wb_score = pesq(fs=sampling_rate, ref=clean, deg=noisy, mode="wb")
                pesq_nb_score = pesq(fs=sampling_rate, ref=clean, deg=noisy, mode="nb")
                stoi_score = pysepm.stoi(clean, noisy, sampling_rate)
                sisdr_score = sisdr(noisy, clean)

                metrics['pesq_wb'].append(pesq_wb_score)
                metrics['pesq_nb'].append(pesq_nb_score)
                metrics['stoi'].append(stoi_score)
                metrics['sisdr'].append(sisdr_score)

                if i >= 100:
                    break

        pesq_wb_mean = np.mean(metrics['pesq_wb'])
        pesq_nb_mean = np.mean(metrics['pesq_nb'])
        stoi_mean = np.mean(metrics['stoi'])
        sisdr_mean = np.mean(metrics['sisdr'])

        return 5 - pesq_wb_mean - stoi_mean

    def load(self):
        state_dict = load_checkpoint(ckpt_path, "cpu")
        self.model.load_state_dict(state_dict['state_dict'])
    
    def init_dataloader(self):
        """Initialize dataloader"""
        training_indexes = get_dataset_filelist(training_path)

        trainset = MPNet_Dataset(training_indexes, input_clean_wavs_dir, input_noisy_wavs_dir, segment_size, sampling_rate,
                        split=True, n_cache_reuse=0, shuffle=False, device="cuda")
        
        self.train_loader = DataLoader(trainset, num_workers=num_workers, shuffle=False,
                                batch_size=batch_size,
                                pin_memory=True,
                                drop_last=True)
    
    def collect_stats(self, num_calib_batch):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, (_, noisy_audio) in tqdm(enumerate(self.train_loader), total=num_calib_batch):
            noisy_audio = torch.autograd.Variable(noisy_audio.to("cuda", non_blocking=True))
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, n_fft, hop_size, win_size, compress_factor)
            x = torch.stack((noisy_mag, noisy_pha), dim=-1).permute(0, 3, 2, 1)
            self.model(x)
            if i >= num_calib_batch:
                break

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
            with torch.no_grad():
                for i, (_, noisy_audio) in tqdm(enumerate(self.train_loader), total=num_batches):
                    if i >= num_batches:
                        break
                    noisy_audio = torch.autograd.Variable(noisy_audio.to("cuda", non_blocking=True))
                    noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, n_fft, hop_size, win_size, compress_factor)
                    x = torch.stack((noisy_mag, noisy_pha), dim=-1).permute(0, 3, 2, 1)
                    _ = self.model(x)

            # Remove hooks
            for h in hooks:
                h.remove()
            return layer_inputs
        else:
            return None
    
    def export(self):
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        jit_path = "torchscript_file/mpnet/mpnet_model.jit.pt"
        os.makedirs(os.path.dirname(jit_path), exist_ok=True)

        if not os.path.exists(jit_path):
            with torch.no_grad():
                for i in range(2):
                    data = iter(self.train_loader)
                    _, noisy_audio = next(data)
                    noisy_audio = noisy_audio[:1, :]
                    noisy_audio = noisy_audio.to("cuda", non_blocking=True)
                    noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
                        noisy_audio, n_fft, hop_size, win_size, compress_factor
                    )
                    x = torch.stack((noisy_mag, noisy_pha), dim=-1).permute(0, 3, 2, 1)
                    jit_model = torch.jit.trace(self.model, x)
                    torch.jit.save(jit_model, jit_path)

        self.jit_model = torch.jit.load(jit_path).eval()

        # ---------------- TensorRT modules ----------------
        self.trt_encoder = _load_or_compile(
            self.jit_model.dense_encoder,
            compile_spec_encoder,
            "torchscript_file/mpnet/trt_model_encoder.ts"
        )

        self.trt_mask_decoder = _load_or_compile(
            self.jit_model.mask_decoder,
            compile_spec_mask_decoder,
            "torchscript_file/mpnet/trt_model_mask_decoder.ts"
        )

        self.trt_phase_decoder = _load_or_compile(
            self.jit_model.phase_decoder,
            compile_spec_phase_decoder,
            "torchscript_file/mpnet/trt_model_phase_decoder.ts"
        )

        # TSConformer blocks
        self.trt_c0_time = _load_or_compile(
            self.jit_model.TSConformer._modules['0'].time_conformer,
            compile_spec_time,
            "torchscript_file/mpnet/trt_model_c0_time.ts"
        )

        self.trt_c0_freq = _load_or_compile(
            self.jit_model.TSConformer._modules['0'].freq_conformer,
            compile_spec_freq,
            "torchscript_file/mpnet/trt_model_c0_freq.ts"
        )

        self.trt_c1_time = _load_or_compile(
            self.jit_model.TSConformer._modules['1'].time_conformer,
            compile_spec_time,
            "torchscript_file/mpnet/trt_model_c1_time.ts"
        )

        self.trt_c1_freq = _load_or_compile(
            self.jit_model.TSConformer._modules['1'].freq_conformer,
            compile_spec_freq,
            "torchscript_file/mpnet/trt_model_c1_freq.ts"
        )

        self.trt_c2_time = _load_or_compile(
            self.jit_model.TSConformer._modules['2'].time_conformer,
            compile_spec_time,
            "torchscript_file/mpnet/trt_model_c2_time.ts"
        )

        self.trt_c2_freq = _load_or_compile(
            self.jit_model.TSConformer._modules['2'].freq_conformer,
            compile_spec_freq,
            "torchscript_file/mpnet/trt_model_c2_freq.ts"
        )

        self.trt_c3_time = _load_or_compile(
            self.jit_model.TSConformer._modules['3'].time_conformer,
            compile_spec_time,
            "torchscript_file/mpnet/trt_model_c3_time.ts"
        )

        self.trt_c3_freq = _load_or_compile(
            self.jit_model.TSConformer._modules['3'].freq_conformer,
            compile_spec_freq,
            "torchscript_file/mpnet/trt_model_c3_freq.ts"
        )

    

    def benchmark(self, mode="int"):

        assert mode in ["int", "fp32"], "mode must be 'int' or 'fp32'"
        self.model = self.model.cuda()
        self.model.eval()

        print(f"Benchmark mode: {mode.upper()}")

        def run_int(input_data):
            x = self.trt_encoder(input_data)
            
            b, c, t, f = x.size()
            x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
            x = self.trt_c0_time(x) + x
            x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
            x = self.trt_c0_freq(x) + x
            x = x.view(b, t, f, c).permute(0, 3, 1, 2)

            x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
            x = self.trt_c1_time(x) + x
            x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
            x = self.trt_c1_freq(x) + x
            x = x.view(b, t, f, c).permute(0, 3, 1, 2)

            x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
            x = self.trt_c2_time(x) + x
            x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
            x = self.trt_c2_freq(x) + x
            x = x.view(b, t, f, c).permute(0, 3, 1, 2)

            x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
            x = self.trt_c3_time(x) + x
            x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
            x = self.trt_c3_freq(x) + x
            x = x.view(b, t, f, c).permute(0, 3, 1, 2)

            denoised_amp = self.trt_mask_decoder(x)
            denoised_pha = self.trt_phase_decoder(x)
            return denoised_amp, denoised_pha

        def run_fp32(input_data):
            return self.model(input_data)

        # ---------------- Warmup ----------------
        print("Warmup...")
        with torch.no_grad():
            for _ in range(nwarmup):
                D = random.randint(*size_range)
                input_data = torch.randn(1, component, D, channels, device="cuda")
                if mode == "int":
                    _ = run_int(input_data)
                else:
                    _ = run_fp32(input_data)

        torch.cuda.synchronize()

        # ---------------- Timing ----------------
        print("Start timing...")
        timings = []

        with torch.no_grad():
            for i in range(1, nruns + 1):
                D = random.randint(*size_range)
                input_data = torch.randn(1, component, D, channels, device="cuda")

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

