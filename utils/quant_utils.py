import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
import pickle

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

def set_scale(model, scale_list):
    i = 0
    for name, module in model.model.named_modules():
        if hasattr(module, "input_quantizer"):
            module.input_quantizer.amax = scale_list[i]
            i += 1

def save_scale(model, path):
    dict_amax = {}
    for name, module in model.model.named_modules():
        if hasattr(module, "input_quantizer"):
            dict_amax[name] = module.input_quantizer.amax
    
    with open(path, "wb") as f:
        pickle.dump(dict_amax, f)

def objective_function(model, theta, list_amax):
    scale_list = [x * y for x, y in zip(theta, list_amax)]
    set_scale(model, scale_list)
    metric = model.objective_function()
    return metric