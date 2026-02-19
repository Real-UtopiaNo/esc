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


"""Quantized Linear"""
from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Function, Variable

from pytorch_quantization import tensor_quant

from . import _utils

__all__ = ["Linear", "QuantLinear", "MultiheadAttention", "QuantMultiheadAttention", "LayerNorm", "QuantLayerNorm", "QuantGLU", "QuantLoRA", "QuantSoftmax", "QuantIntSoftmax"]

class QuantLinear(nn.Linear, _utils.QuantMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    #default_quant_desc_input = tensor_quant.QUANT_DESC_4BIT_PER_TENSOR #here to 4 bits, how to generalize it?
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        output = F.linear(quant_input, quant_weight, bias=self.bias)

        #print('Linear')
        #print(self._weight_quantizer.extra_repr())

        num_unique_input = torch.unique(quant_input).numel()
        num_unique_weight = torch.unique(quant_weight).numel()
        num_unique_weight2 = torch.unique(self.weight).numel()
        num_unique_output = torch.unique(output).numel()

        return output

Linear = QuantLinear

class QuantMultiheadAttention(nn.MultiheadAttention, _utils.QuantMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_query = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_key = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_value = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super(QuantMultiheadAttention, self).__init__(embed_dim, num_heads, dropout=0.0)
        #quant_desc_query, quant_desc_key, quant_desc_value, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        #self.init_quantizer(quant_desc_query, quant_desc_key, quant_desc_value, quant_desc_weight)

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        quant_query = self._input_quantizer(query)
        quant_key = self._input_quantizer(key)
        quant_value = self._input_quantizer(value)

        quant_weight = self._weight_quantizer(self.in_proj_weight)

        '''output_query, output_key = F.multi_head_attention_forward(
                quant_query, quant_key, quant_value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=None, need_weights=True,
                attn_mask=None)'''
        
        tgt_len, bsz, embed_dim = quant_query.size()

        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        output = F.linear(query, self.in_proj_weight, self.in_proj_bias)

        split_size = output.size(2) // 3
        #split_size = 64
        q = output[:,:,:split_size]
        k = output[:,:,split_size:2*split_size]
        v = output[:,:,2*split_size:]

        #q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim)
        #q = q.contiguous().view(tgt_len, bsz, embed_dim)
        #return q,q

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)

        #return q, q

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, attn_output_weights

        #return output, output

        '''output_query = F.multi_head_attention_forward_minimal(
                quant_query, quant_key, quant_value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=None, need_weights=True,
                attn_mask=None)

        #return quant_query, quant_key
        return output_query, output_query'''

        

        '''output = F.linear(quant_input, quant_weight, bias=self.bias)

        #print('Linear')
        #print(self._weight_quantizer.extra_repr())

        num_unique_input = torch.unique(quant_input).numel()
        num_unique_weight = torch.unique(quant_weight).numel()
        num_unique_weight2 = torch.unique(self.weight).numel()
        num_unique_output = torch.unique(output).numel()

        return output'''

MultiheadAttention = QuantMultiheadAttention

class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


'''QUANT_DESC_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantDescriptor(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))
QUANT_DESC_8BIT_INSTANCENORM_WEIGHT_PER_CHANNEL = QuantDescriptor(num_bits=8, axis=(0))'''

class QuantLayerNorm(nn.LayerNorm, _utils.QuantMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    #default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW #worked befire
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, normalized_shape, **kwargs):
        super(QuantLayerNorm, self).__init__(normalized_shape)
        if 'eps' in kwargs.keys():
            self.eps = kwargs['eps']
            kwargs.pop('eps', None)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        dims = tuple(range(-len(self.normalized_shape), 0))  # e.g., (-1,) for last dim

        mean = quant_input.mean(dim=dims, keepdim=True)
        var = quant_input.var(dim=dims, keepdim=True, unbiased=False)  # match C++ default
        normalized = (quant_input - mean) / torch.sqrt(var + self.eps)

        #return normalized

        #weight = self.weight.view(*([1] * (quant_input.dim() - len(self.normalized_shape))), *self.normalized_shape)
        weight = quant_weight.view(*([1] * (quant_input.dim() - len(self.normalized_shape))), *self.normalized_shape)
        normalized = normalized * weight

        #return normalized

        bias = self.bias.view(*([1] * (quant_input.dim() - len(self.normalized_shape))), *self.normalized_shape)
        normalized = normalized + bias

        return normalized
    
        '''def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        dims = tuple(range(-len(self.normalized_shape), 0))  # e.g., (-1,) for last dim

        mean = quant_input.mean(dim=dims, keepdim=True)
        var = quant_input.var(dim=dims, keepdim=True, unbiased=False)  # match C++ default
        normalized = (quant_input - mean) / torch.sqrt(var + self.eps)

        weight = quant_weight.view(*([1] * (quant_input.dim() - len(self.normalized_shape))), *self.normalized_shape)
        normalized = normalized * weight

        bias = self.bias.view(*([1] * (quant_input.dim() - len(self.normalized_shape))), *self.normalized_shape)
        normalized = normalized + bias

        return normalized'''


        #output = quant_input

        #output = F.layer_norm(quant_input, self.normalized_shape, quant_weight, self.bias, self.eps)

        # Normalization: computes mean and variance(std)
        '''x_int = quant_input
        mean_int = x_int.mean(axis=2, keepdim=True)
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = (k + var_int/k)/2
            k = k_1
        std_int = k

        factor = (2 ** 31-1) / std_int
        y_int = y_int * factor / 2

        y_int = y_int + self.bias
        x = y_int * quant_weight

        return x'''


        #print('Linear')
        #print(self._weight_quantizer.extra_repr())

        #return output

LayerNorm = QuantLayerNorm

class QuantGLU(nn.GLU, _utils.QuantInputMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, dim, **kwargs):
        super(QuantGLU, self).__init__(dim)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        dim_split = int(quant_input.shape[self.dim] / 2) #i can stat that once i want to export, not robust to change of dim
        a = quant_input[:,:dim_split,:]
        b = quant_input[:,dim_split:,:]
        output = a * torch.sigmoid(b)
        return output

GLU = QuantGLU

class QuantLoRA(nn.Module, _utils.QuantMixin):
    """Quantized version of nn.Linear

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, in_features, out_features, reduced_rank, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.quant_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.ln_weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.bias = nn.Parameter(torch.zeros(out_features, requires_grad=False))

        self.right_weight = nn.Parameter(torch.zeros(in_features, reduced_rank))
        self.left_weight = nn.Parameter(torch.zeros(reduced_rank, out_features))

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.quant_weight)
        #right_weight = self._weight_quantizer(self.right_weight)
        #left_weight = self._weight_quantizer(self.left_weight)

        HX = F.linear(quant_input, quant_weight)
        right_output = F.linear(quant_input, self.right_weight)
        LRX = F.linear(right_output, self.left_weight)

        if self.bias.numel() == 0:
            output = HX + LRX
        else:
            output = HX + LRX + self.bias
        return output

LoRA = QuantLoRA

class QuantSoftmax(nn.Softmax, _utils.QuantInputMixin):
    """Quantized version of nn.Softmax

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, dim, **kwargs):
        super(QuantSoftmax, self).__init__(dim)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        output = F.softmax(quant_input, dim=self.dim)
        return output

Softmax = QuantSoftmax

def int_poly_8(x_int, scaling_factor):
    coef = [0.08947606, 0.12121466]
    coef[1] /= coef[0]
    b_int = torch.floor(coef[1] / scaling_factor) # Alg 8
    z = x_int + b_int # Alg 10
    scaling_factor = coef[0] * scaling_factor # Alg 9
    #z = z/8
    
    return z, scaling_factor

def int_exp_first_8(x_int, scaling_factor):
    x0 = -0.6931  # -ln2
    n = 30  # sufficiently large integer
    x0_int = torch.floor(x0 / scaling_factor) # Alg 4
                
    x_int = torch.max(x_int, n * x0_int) # Alg 5
    q = torch.floor(x_int / x0_int)# Alg 6            
    r = x_int - x0_int * q # Alg 7
    exp_int, exp_scaling_factor = int_poly_8(r, scaling_factor) # Alg 8~10

    exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0) # Alg 11
    
    scaling_factor = exp_scaling_factor / 2**n # Alg 11
    return exp_int, scaling_factor

def log_round(x):
    x_log_floor = x.log2().floor()
    big = x_log_floor
    extra_mask = (x - 2**big) >= 2**(big - 1)
    big[extra_mask] = big[extra_mask] + 1
    return big

class QuantIntSoftmax(nn.Softmax, _utils.QuantInputMixin):
    """Quantized version of nn.Softmax, used for HyQ, results seems strange

    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, dim, **kwargs):
        super(QuantIntSoftmax, self).__init__(dim)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        output = F.softmax(quant_input, dim=self.dim)
        scaling_factor = 1 / self._input_quantizer._get_scale()

        x = (quant_input / scaling_factor)
        x_int = (quant_input / scaling_factor).round()
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, exp_scaling_factor = int_exp_first_8(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        softmax_out = torch.round(exp_int_sum / exp_int)

        rounds = log_round(softmax_out)
        max_bound = (2.0**(self._input_quantizer._num_bits - 1 + int(self._input_quantizer._unsigned))) - 1.0
        min_bound = -max_bound - 1
        qlog = torch.clamp(rounds, min_bound, max_bound)

        deq_softmax = 2 ** (-qlog)
        return deq_softmax

IntSoftmax = QuantIntSoftmax
