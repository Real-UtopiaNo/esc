import torch
from collections import defaultdict
from tqdm import tqdm
from .utils import mag_pha_stft
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
import os

def collect_layer_inputs(model, data_loader, layers_to_quantize, num_batches=10, device="cuda"):
    """
    Runs the model for a few batches and captures each layer's input activations.
    Returns a dictionary: {layer_name: [list of input tensors]}.
    """
    model.eval()
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

    for name, module in model.named_modules():
        if match_by_type:
            if any(isinstance(module, t) for t in layers_to_quantize):
                hooks.append(module.register_forward_hook(get_hook(name)))
        elif match_by_name:
            last_name = name.split(".")[-1]
            if any(substring in last_name for substring in layers_to_quantize):
                hooks.append(module.register_forward_hook(get_hook(name)))
        else:
            raise ValueError("layers_to_quantize should contain only layer types OR only names.")

    model.collect_layer_inputs(num_batches)

    # Run a few batches to collect activations
    with torch.no_grad():
        for i, (_, noisy_audio) in tqdm(enumerate(data_loader), total=num_batches):
            if i >= num_batches:
                break
            noisy_audio = torch.autograd.Variable(noisy_audio.to("cuda", non_blocking=True))
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            x = torch.stack((noisy_mag, noisy_pha), dim=-1).permute(0, 3, 2, 1)
            _ = model(x)

    # Remove hooks
    for h in hooks:
        h.remove()
    return layer_inputs

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()

def calibrate_model(model_ptq, model_name, num_calib_batch, method="max", percentile=99.9):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """
    model_ptq.model.eval()
    model_ptq.model = model_ptq.model.cuda()
    if num_calib_batch > 0:
        with torch.no_grad():
            model_ptq.collect_stats(num_calib_batch)
        
        if method == "max":
            compute_amax(model_ptq.model, method="max")
        elif method == "percentile":
            compute_amax(model_ptq.model, method="percentile", percentile=percentile)
        elif method == "mse":
            compute_amax(model_ptq.model, method="mse")
        elif method == "mae":
            compute_amax(model_ptq.model, method="mae")
        elif method == "sqnr":
            compute_amax(model_ptq.model, method="sqnr")
        elif method == "wmse":
            compute_amax(model_ptq.model, method="wmse")
        elif method == "entropy":
            compute_amax(model_ptq.model, method="entropy")

def set_bits(model, bits):
    if bits == 8:
        pass
    else:
        for name, module in model.model.named_modules():
            #if 'encoder.conv_subsample' not in name:
            if isinstance(module, quant_nn.TensorQuantizer):
                module.num_bits = bits

def set_calibrators(model):
    for name, module in model.model.named_modules():
        if hasattr(module, "input_quantizer"):
            module.input_quantizer._calibrator = calib.HistogramCalibrator(num_bits=module.input_quantizer._num_bits, axis=module.input_quantizer._axis, unsigned=module.input_quantizer._unsigned)