#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Basic tensor quantization functions"""
import numpy as np
import yaml

from absl import logging

import torch
from torch.autograd import Function

class ScaledQuantDescriptor():
    """Supportive descriptor of quantization

    Describe how a tensor should be quantized. A QuantDescriptor and a tensor defines a quantized tensor.

    Args:
        num_bits: An integer. Number of bits of quantization. It is used to calculate scaling factor. Default 8.
        name: Seems a nice thing to have

    Keyword Arguments:
        fake_quant: A boolean. If True, use fake quantization mode. Default True.
        axis: None, int or tuple of int. axes which will have its own max for computing scaling factor.
            If None (the default), use per tensor scale.
            Must be in the range [-rank(input_tensor), rank(input_tensor)).
            e.g. For a KCRS weight tensor, quant_axis=(0) will yield per channel scaling.
            Default None.
        amax: A float or list/ndarray of floats of user specified absolute max range. If supplied,
            ignore quant_axis and use this to quantize. If learn_amax is True, will be used to initialize
            learnable amax. Default None.
        learn_amax: A boolean. If True, learn amax. Default False.
        scale_amax: A float. If supplied, multiply amax by scale_amax. Default None. It is useful for some
            quick experiment.
        calib_method: A string. One of ["max", "histogram"] indicates which calibration to use. Except the simple
            max calibration, other methods are all hisogram based. Default "max".
        unsigned: A Boolean. If True, use unsigned. Default False.

    Raises:
        TypeError: If unsupported type is passed in.

    Read-only properties:
        - fake_quant:
        - name:
        - learn_amax:
        - scale_amax:
        - axis:
        - calib_method:
        - num_bits:
        - amax:
        - unsigned:
    """

    def __init__(self, num_bits=8, name=None, **kwargs):
        if not isinstance(num_bits, int):
            raise TypeError("num_bits must be an integer, not {}.".format(type(num_bits)))
        if num_bits < 0:
            raise ValueError("num_bits must be >= 0, not {}.".format(num_bits))
        if num_bits == 0:
            logging.error("num_bits is 0. This will result in the tensor being quantized to all zeros."
                          " This mode should only be used for debugging purposes.")
        self._num_bits = num_bits
        if not isinstance(name, str) and name is not None:
            raise TypeError("name must be a string or None, not {}.".format(type(name)))
        self._name = name

        self._fake_quant = kwargs.pop('fake_quant', True)
        self._axis = kwargs.pop('axis', None)
        if self._axis is not None:
            logging.debug("Meaning of axis has changed since v2.0. Make sure to update.")
        self._learn_amax = kwargs.pop('learn_amax', False)
        if self._learn_amax and self._axis is not None:
            raise TypeError(
                "axis is ignored and must be None when learn_amax is true, got {}.".format(type(self._axis)))
        amax = kwargs.pop('amax', None)
        if amax is not None:
            if not isinstance(amax, float) and not isinstance(
                    amax, list) and not isinstance(amax, np.ndarray):
                raise TypeError("amax must be float, list or ndarray, not {}".format(type(amax)))
            # Make it single precision array
            self._amax = np.array(amax, dtype=np.float32)
        else:
            self._amax = amax

        self._scale_amax = kwargs.pop('scale_amax', None)
        self._calib_method = kwargs.pop('calib_method', "max")
        self._unsigned = kwargs.pop('unsigned', False)
        self._narrow_range = kwargs.pop('narrow_range', False)

        if kwargs:
            raise TypeError("Unused keys: {}".format(kwargs.keys()))

    # pylint:disable=missing-docstring
    @property
    def num_bits(self):
        return self._num_bits

    @property
    def fake_quant(self):
        return self._fake_quant

    @property
    def axis(self):
        return self._axis

    @property
    def amax(self):
        return self._amax

    @property
    def learn_amax(self):
        return self._learn_amax

    @property
    def scale_amax(self):
        return self._scale_amax

    @property
    def name(self):
        return self._name

    @property
    def calib_method(self):
        return self._calib_method

    @property
    def unsigned(self):
        return self._unsigned

    @property
    def narrow_range(self):
        return self._narrow_range
    # pylint:enable=missing-docstring

    def __str__(self):
        s = (self._name + ': ') if self._name is not None else 'QuantDescriptor'
        s += "({}{}bit".format("unsigned " if self._unsigned else "", self._num_bits)
        s += " fake" if self._fake_quant else " real"
        s += " axis={}".format(self._axis if self._axis is not None else " per-tensor")
        if isinstance(self._amax, torch.Tensor):
            s += " amax={}".format(np.array2string(self._amax.cpu().numpy().flatten(), edgeitems=1,
                                                   formatter={'all': "{:.2e}".format}))
        elif self._amax is not None:
            s += " amax={_amax}"
            s += " full_range"
        if self._learn_amax:
            s += " learn_amax"
        if self._scale_amax:
            s += " scale_amax={_scale_amax}"
        s += ")"
        return s.format(**self.__dict__)

    def __eq__(self, rhs):
        """Compare 2 descriptors"""
        return self.__dict__ == rhs.__dict__

    def dict(self):
        """Serialize to dict

        The build-in __dict__ method returns all the attributes, which includes those have default value and have
        protected prefix "_". This method only returns those have values other than the default one and don't have _ in
        key. Construct a instance by dict returned by this method should get exactly the same instance.
        """
        obj_dict = {}
        obj_dict['num_bits'] = self._num_bits
        obj_dict['name'] = self._name

        if not self._fake_quant:
            obj_dict['fake_quant'] = self._fake_quant
        if self._axis is not None:
            obj_dict['axis'] = self._axis
        if self._amax is not None:
            obj_dict['amax'] = self._amax.tolist()
        if self._scale_amax is not None:
            obj_dict['scale_amax'] = self._scale_amax
        if self._learn_amax:
            obj_dict['learn_amax'] = self._learn_amax
        if self._unsigned:
            obj_dict['unsigned'] = self._unsigned

        return obj_dict

    def to_yaml(self):
        """Create yaml serialization
        Some attributes need special treatment to have human readable form, including amax, axis.
        """
        obj_dict = self.dict()

        if "axis" in obj_dict:
            obj_dict['axis'] = list(obj_dict['axis'])
        return yaml.dump(obj_dict, width=120)

    @classmethod
    def from_yaml(cls, yaml_str):
        """Create descriptor from yaml str"""
        obj_dict = yaml.safe_load(yaml_str)

        if 'axis' in obj_dict:
            obj_dict['axis'] = tuple(obj_dict['axis'])

        quant_desc = cls(**obj_dict)

        return quant_desc

QuantDescriptor = ScaledQuantDescriptor

# Predefined descriptors
QUANT_DESC_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8)
QUANT_DESC_4BIT_PER_TENSOR = QuantDescriptor(num_bits=4)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_INSTANCENORM_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))


class TensorQuantFunction(Function):
    """A universal tensor quantization function

    Take an input tensor, output an quantized tensor. The granularity of scale can be interpreted from the
    shape of amax.
    output_dtype indicates whether the quantized value will be stored in integer or float. The reason we want to store
    it in float is the pytorch function takes the quantized value may not accept integer input, e.g. Conv2D.

    It uses 2^num_bits -1 values instead of 2^num_bits. e.g., for num_bits=8, it uses [-127, 127] instead of [-128, 127]
    """

    @staticmethod
    def forward(ctx, inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
        """

        Follow tensorflow convention, max value is passed in and used to decide scale, instead of inputing scale
        directly. Though inputing scale directly may be more natural to use.

        Args:
            ctx: A Context object to store tensors for backward.
            inputs: A Tensor of type float32.
            amax: A Tensor of type float32. Inputs will be quantized within range [-amax, amax]
                amax will be broadcasted to inputs tensor.
            num_bits: A integer used to calculate scaling factor, scale = (2^(num_bits-1) - 1) / max
                Effectively, it indicates how many integer bits is used to represent the value. Default 8.
            output_dtype: A type of Tensor. torch.int32 or torch.float32.
            unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
            narrow_range: A boolean. Use symmetric integer range for signed quantization
                E.g. [-127,127] instead of [-128,127] for num_bits=8. Default True.

        Returns:
            outputs: A Tensor of type output_dtype.
            scale: A Tensor of type float32. outputs / scale will dequantize outputs tensor.

        Raises:
            ValueError:
        """
        ctx.save_for_backward(inputs, amax)
        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        # Check if scale overflows FP16
        if outputs.dtype == torch.half and scale.max() > 65504:
            raise ValueError("scale is too large for FP16 with amax={}".format(amax))
        return outputs, scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs, grad_scale):
        """
        Implements straight through estimation with clipping. For -amax <= input <= amax
        the gradient passes straight through, otherwise the gradient is zero.

        Args:
            ctx: A Context object with saved tensors from forward.
            grad_outputs: A tensor of gradient of outputs.
            grad_scale: A tensor of gradient of scale.

        Returns:
            grad_inputs: A tensor of gradient.
        """
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1) # create a zero tensor with the same type and device
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None


class FakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction
    See comments of TensorQuantFunction, arguments are the same.
    """

    @staticmethod
    def forward(ctx, inputs, amax, num_bits=8, unsigned=False, narrow_range=True, ptq=None, dict=None):
        ctx.save_for_backward(inputs, amax)
        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range, ptq, dict)
        if ptq == "noisyquant":
            outputs = outputs / scale.to(inputs.dtype)
            outputs -= dict['noisy_bias']
            return outputs
        return outputs / scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None, None, None, None

def _tensor_quant(inputs, amax, num_bits=8, unsigned=False, narrow_range=True, ptq=None, dict=None):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
        # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
        logging.debug("amax %s has different shape than inputs %s. Make sure broadcast works as expected!",
                      amax.size(), inputs.size())

    logging.debug("{} bits quantization on shape {} tensor.".format(num_bits, inputs.size()))

    if unsigned:
        if inputs.min() < 0.:
            raise TypeError("Negative values encountered in unsigned quantization.")

    # Computation must be in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    min_amax = amax.min()
    #if min_amax < 0:
    #    raise ValueError("Negative values in amax")
    
    max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1 #Non uniform but default

    scale = max_bound / amax

    epsilon = 1. / (1<<24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = (amax <= epsilon)
        scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

    if ptq == "adaround":
        hard_rounded = torch.floor(inputs * scale) + (torch.sigmoid(dict['alpha']) >= 0.5).float()
        outputs = torch.clamp(hard_rounded, min_bound, max_bound)
    elif ptq == "noisyquant":
        outputs = torch.clamp(((inputs + dict['noisy_bias']) * scale).round_(), min_bound, max_bound)
    elif ptq == "hyq":
        outputs = inputs * dict['alpha']
        outputs = torch.clamp((outputs * scale).round_(), min_bound, max_bound)
    else:
        outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if min_amax <= epsilon:
        scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

    if input_dtype == torch.half:
        outputs = outputs.half()
   
    return outputs, scale

class SoftRounding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Save alpha for backward
        ctx.save_for_backward(alpha)
        # "soft" rounding
        return torch.floor(x) + torch.sigmoid(alpha)

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        # gradient only through alpha
        grad_x = grad_output * 0  # no gradient through x
        grad_alpha = grad_output * torch.sigmoid(alpha) * (1 - torch.sigmoid(alpha))
        return grad_x, grad_alpha

class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    def forward(ctx, inputs, min_range, max_range, num_bits=8):
        """

        As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        logging.debug("{} bits quantization on shape {} tensor.".format(num_bits, inputs.size()))
        ctx.save_for_backward(inputs, min_range, max_range)

        step_size = (max_range - min_range) / (2.0**num_bits - 1)

        min_bound = -2.0**(num_bits - 1)
        max_bound = 2.0**(num_bits - 1) - 1

        quant_zero = torch.round(min_range / step_size) - min_bound
        quantized = torch.round(inputs / step_size) - quant_zero
        quantized = torch.clamp(quantized, min_bound, max_bound)

        outputs = (quantized + quant_zero) * step_size

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs

        Returns:
            grad_inputs: A tensor of gradient
        """

        inputs, min_range, max_range = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where((inputs <= max_range)*(inputs >= min_range), grad_outputs, zero)
        return grad_inputs, None, None, None


tensor_quant = TensorQuantFunction.apply
fake_tensor_quant = FakeTensorQuantFunction.apply
fake_affine_tensor_quant = FakeAffineTensorQuantFunction.apply

E2M1_MAX = 6.0
E4M3_MAX = 448.0
_E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoints whose upper neighbour has an even mantissa; ties there round up
_E2M1_TIES_UP = [0.75, 1.75, 3.5]


def _round_to_e2m1(x):
    """Round magnitudes to the E2M1 grid with round-half-to-even, saturating at 6."""
    mag = x.abs()
    bounds = torch.tensor(_E2M1_BOUNDS, device=x.device, dtype=mag.dtype)
    idx = torch.searchsorted(bounds, mag)
    ties_up = torch.tensor(_E2M1_TIES_UP, device=x.device, dtype=mag.dtype)
    idx = idx + torch.isin(mag, ties_up).long()
    values = torch.tensor(_E2M1_VALUES, device=x.device, dtype=mag.dtype)
    return torch.sign(x) * values[idx]


def _round_to_int4(x):
    """Round to the INT4 grid [-8, 7] with round-half-to-even."""
    return x.round().clamp(-8, 7)


# element format -> (largest magnitude a block amax maps to, rounding function)
BLOCK_ELEMENTS = {
    "e2m1": (E2M1_MAX, _round_to_e2m1),
    "int4": (7.0, _round_to_int4),
}


def fake_block_quant(inputs, global_amax, element="e2m1", block_axis=-1, block_size=16):
    """NVFP4-style two-level fake quantization: one E4M3 scale per block of `block_size`
    elements along `block_axis` and an FP32 per-tensor scale derived from `global_amax`.
    `element` picks the 4-bit element grid: "e2m1" (NVFP4) or "int4".

    block_axis=None blocks along all dims after dim 0 flattened together (the GEMM reduction
    dim of a conv weight). A partial last block is zero-padded.
    """
    if block_axis is None:
        return fake_block_quant(inputs.reshape(inputs.shape[0], -1), global_amax, element,
                                block_size=block_size).reshape(inputs.shape)
    if block_axis not in (-1, inputs.dim() - 1):
        x = inputs.movedim(block_axis, -1)
        return fake_block_quant(x, global_amax, element, block_size=block_size).movedim(-1, block_axis)

    element_max, round_to_grid = BLOCK_ELEMENTS[element]
    x = inputs.float()
    global_scale = global_amax.float().to(x.device).reshape(()) / (element_max * E4M3_MAX)

    length = x.shape[-1]
    pad = -length % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    blocks = x.reshape(*x.shape[:-1], -1, block_size)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = block_amax / (element_max * global_scale)
    block_scale = torch.where(block_scale == 0, torch.ones_like(block_scale), block_scale)
    block_scale = block_scale.clamp(min=2**-9, max=E4M3_MAX).to(torch.float8_e4m3fn).float()

    scale = block_scale * global_scale
    out = round_to_grid(blocks / scale) * scale
    out = out.reshape(x.shape)[..., :length]
    return out.to(inputs.dtype)


def fake_nvfp4_quant(inputs, global_amax, block_axis=-1, block_size=16):
    """NVFP4 fake quantization: E2M1 elements with E4M3 block scales, see fake_block_quant."""
    return fake_block_quant(inputs, global_amax, "e2m1", block_axis, block_size)


# ---- HiFloat4 (arXiv:2602.11287 Algorithm 1) ----
HIF4_BLOCK = 64
E6M2_QMIN = 2.0 ** -48
E6M2_QMAX = 1.5 * (2.0 ** 15)  # max finite: 2^15 * 1.50
S1P2_MAX = 1.75
# Intra-group peak the hierarchy can represent: 2^(1+1) * 1.75 = 7
HIF4_INTRAGROUP_MAX = 7.0


def _to_e6m2(x):
    """Unsigned E6M2 used by HiF4: X = 2^E * 1.M (no subnormals; NaN is 111111_11).

    Mantissa field 11 (=1.75) is valid except at the max exponent, where it encodes NaN
    and the max finite value is 1.50 * 2^15.
    """
    x = x.float().clamp(min=E6M2_QMIN, max=E6M2_QMAX)
    e = torch.floor(torch.log2(x))
    # Integer with 2 fraction bits of the significand: round(x / 2^(e-2)) ∈ {4,5,6,7}
    m_int = torch.round(x / torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), e - 2))
    overflow = m_int >= 8
    e = torch.where(overflow, e + 1, e)
    m_int = torch.where(overflow, torch.full_like(m_int, 4.0), m_int)
    # Clamp to max finite E6M2
    too_big = (e > 15) | ((e == 15) & (m_int > 6))
    e = torch.where(too_big, torch.full_like(e, 15.0), e)
    m_int = torch.where(too_big, torch.full_like(m_int, 6.0), m_int)
    # Exponent 15 + mantissa 7 would be NaN encoding → clamp to 1.50
    m_int = torch.where((e == 15) & (m_int >= 7), torch.full_like(m_int, 6.0), m_int)
    return m_int * torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), e - 2)


def _round_to_s1p2(x):
    """Round to S1P2 / E1M2 grid with round-half-to-even, saturating at ±1.75."""
    mag = x.abs().clamp(max=S1P2_MAX)
    q = (mag / 0.25).round() * 0.25
    return torch.sign(x) * q


def fake_hif4_quant(inputs, block_axis=-1, block_size=HIF4_BLOCK):
    """HiF4 fake quantization (arXiv:2602.11287): 64-element blocks with E6M2 / E1_8 / E1_16
    hierarchical scales and S1P2 elements. All scales are computed from the tensor itself
    (no per-tensor static amax).

    block_axis=None flattens dims after dim 0 (conv weight reduction dim). Partial last
    blocks are zero-padded.
    """
    if block_axis is None:
        return fake_hif4_quant(inputs.reshape(inputs.shape[0], -1),
                               block_size=block_size).reshape(inputs.shape)
    if block_axis not in (-1, inputs.dim() - 1):
        x = inputs.movedim(block_axis, -1)
        return fake_hif4_quant(x, block_size=block_size).movedim(-1, block_axis)

    x = inputs.float()
    length = x.shape[-1]
    pad = -length % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    blocks = x.reshape(*x.shape[:-1], -1, block_size)  # (..., B, 64)
    abs_b = blocks.abs()
    v16 = abs_b.reshape(*abs_b.shape[:-1], 16, 4).amax(dim=-1)       # (..., B, 16)
    v8 = v16.reshape(*v16.shape[:-1], 8, 2).amax(dim=-1)             # (..., B, 8)
    vmax = v8.amax(dim=-1, keepdim=True)                              # (..., B, 1)
    zero = vmax == 0

    sf = (vmax / HIF4_INTRAGROUP_MAX).clamp(min=E6M2_QMIN)
    e6m2 = _to_e6m2(sf)
    rec = torch.where(zero, torch.ones_like(e6m2), 1.0 / e6m2)

    e1_8 = (v8 * rec >= 4).to(x.dtype)                                # (..., B, 8)
    e1_8_for16 = e1_8.repeat_interleave(2, dim=-1)                    # (..., B, 16)
    e1_16 = (v16 * rec * torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype),
                                   -e1_8_for16) >= 2).to(x.dtype)

    e1_8_el = e1_8.repeat_interleave(8, dim=-1)                       # (..., B, 64)
    e1_16_el = e1_16.repeat_interleave(4, dim=-1)
    two = torch.tensor(2.0, device=x.device, dtype=x.dtype)
    scaled = blocks * rec * torch.pow(two, -e1_8_el) * torch.pow(two, -e1_16_el)
    s1p2 = _round_to_s1p2(scaled)
    out = e6m2 * torch.pow(two, e1_8_el + e1_16_el) * s1p2
    out = torch.where(zero, torch.zeros_like(out), out)
    out = out.reshape(x.shape)[..., :length]
    return out.to(inputs.dtype)
