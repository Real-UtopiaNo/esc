import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
import pickle

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None and not module._disabled:
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

def _is_gemm(module):
    """Layers that run as a GEMM: Linear and dense convs (pointwise Conv1d, im2col Conv2d).
    Depthwise convs and the single-input-channel first Conv2d are excluded."""
    if isinstance(module, quant_nn.QuantLinear):
        return True
    if isinstance(module, (quant_nn.QuantConv1d, quant_nn.QuantConv2d)):
        return module.groups == 1 and module.in_channels > 1
    return False

def _block_input_axis(module):
    """Blocks follow each layer's reduction dimension."""
    if isinstance(module, (quant_nn.QuantConv1d, quant_nn.QuantConv2d)):
        return -1 if module.groups > 1 or module.in_channels == 1 else 1
    if isinstance(module, quant_nn.QuantGLU):
        return module.dim
    if isinstance(module, quant_nn.QuantBatchNorm1d):
        return 1
    return -1

def set_block_quant(model, coverage="all", element="e2m1"):
    """Switch quantizers to block floating-point formats.
    element "e2m1"/"int4": NVFP4-style 16-element blocks with E4M3 scales (+ static global amax).
    element "hif4": HiFloat4 (arXiv:2602.11287) 64-element blocks with E6M2/E1/E1 scales (all dynamic).
    element "hif8": HiFloat8 cast (arXiv:2409.16626) per-element TA, no PTS.
    coverage="gemm" leaves non-GEMM layers in full precision."""
    for name, module in model.model.named_modules():
        if not hasattr(module, "input_quantizer"):
            continue
        quantizers = [module.input_quantizer]
        if hasattr(module, "_weight_quantizer"):
            quantizers.append(module.weight_quantizer)
        if coverage == "gemm" and not _is_gemm(module):
            for quantizer in quantizers:
                quantizer.disable()
            continue
        is_conv = isinstance(module, (quant_nn.QuantConv1d, quant_nn.QuantConv2d))
        in_axis = _block_input_axis(module)
        w_axis = None if is_conv else -1
        if element == "hif4":
            module.input_quantizer.use_hif4(in_axis)
            if hasattr(module, "_weight_quantizer"):
                module.weight_quantizer.use_hif4(w_axis)
        elif element == "hif8":
            module.input_quantizer.use_hif8()
            if hasattr(module, "_weight_quantizer"):
                module.weight_quantizer.use_hif8()
        else:
            module.input_quantizer.use_block_quant(in_axis, element)
            if hasattr(module, "_weight_quantizer"):
                module.weight_quantizer.use_block_quant(w_axis, element)

def _frame_reduce_dims(module):
    """Dims holding one time frame: channels for [B, C, T] inputs, channels and frequency for
    [B, C, T, F] inputs, the feature dim for [..., T, C] inputs."""
    if isinstance(module, quant_nn.QuantConv2d):
        return (1, 3)
    if isinstance(module, (quant_nn.QuantConv1d, quant_nn.QuantGLU, quant_nn.QuantBatchNorm1d)):
        return 1
    return -1

def set_dynamic_token_act(model):
    """Quantize activations with one scale per time frame computed at run time.
    Disabled quantizers are skipped: collect_stats re-enables modules that have no calibrator."""
    for name, module in model.model.named_modules():
        if hasattr(module, "input_quantizer") and not module.input_quantizer._disabled:
            module.input_quantizer.use_dynamic_amax(_frame_reduce_dims(module))

def set_gemm_only(model):
    """Leave only GEMM layers quantized; disable input/weight quantizers on all other layers."""
    for name, module in model.model.named_modules():
        if not hasattr(module, "input_quantizer"):
            continue
        if _is_gemm(module):
            continue
        module.input_quantizer.disable()
        if hasattr(module, "_weight_quantizer"):
            module.weight_quantizer.disable()

def set_dynamic_tensor_act(model):
    """Take the per-tensor activation amax from each input at run time instead of calibration.
    Disabled quantizers are skipped: collect_stats re-enables quantizers that have no calibrator."""
    for name, module in model.model.named_modules():
        if hasattr(module, "input_quantizer") and not module.input_quantizer._disabled:
            module.input_quantizer.use_dynamic_amax()

def set_weight_only(model):
    for name, module in model.model.named_modules():
        if hasattr(module, "input_quantizer"):
            module.input_quantizer.disable()

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