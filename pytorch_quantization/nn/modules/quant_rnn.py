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


"""RNN implementation in python
originally copied from https://github.com/pytorch/pytorch/blob/v0.4.1/torch/nn/modules/rnn.py
    backend is changed to _functions/rnn.py
"""
import math
import torch
import warnings
import itertools
import numbers

from torch import nn
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch._VF as _VF  # Internal PyTorch ops
from torch.nn import functional as F

from pytorch_quantization import tensor_quant
from pytorch_quantization.nn._functions import quant_rnn

#from torch.func import scan
from . import _utils

__all__ = ["QuantLSTM", "QuantLSTMCell", "LSTM", "LSTMCell", "QuantGRU", "QuantGRU2", "QuantGRUCell", "GRU", "GRUCell"]

class QuantRNNBase(nn.Module, _utils.QuantMixin):

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, proj_size=0, **kwargs):
        super(QuantRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.dyna_first = batch_first
        #self.forward_fn = self.forward_dyna_first if self.dyna_first else self.forward_dyna_second
        #self.forward_fn = self.forward_dyna_second

        self.batch_first = False
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size > 0:
            raise ValueError("proj_size is not supported in pytorch-quantization yet")

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.init_quantizer(quant_desc_input, quant_desc_weight, num_layers=num_layers * (1 + bidirectional))

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            weight_arr = list(itertools.chain.from_iterable(self.all_weights))
            weight_stride0 = len(self.all_weights[0])

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on weight_arr, that's why the
                # no_grad() is necessary.
                weight_buf = torch._cudnn_rnn_flatten_weight(weight_arr, weight_stride0, self.input_size,
                                                             rnn.get_cudnn_mode(self.mode), self.hidden_size,
                                                             self.proj_size, self.num_layers, self.batch_first,
                                                             bool(self.bidirectional))

            self._param_buf_size = weight_buf.size(0)
            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(QuantRNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def forward(self, input, hx=None):
        #if dyna_first:
        #return self.forward_fn(input, hx)
        if input.shape[1] != 101:
            return self.forward_dyna_first(input, hx)
        else:
            return self.forward_dyna_second(input, hx)



    def forward_dyna_first(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        #max_batch_size = 550
        '''num_directions = 2 if self.bidirectional else 1'''
        '''hx = input.new_zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                requires_grad=False)'''
        '''hx = torch.zeros(self.num_layers * num_directions,
                max_batch_size, self.hidden_size,
                device=input.device, dtype=input.dtype,
                requires_grad=False)
        hx = hx[:,:input.shape[1]]'''

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                requires_grad=False)
            '''hx = torch.zeros(self.num_layers * num_directions,
                 max_batch_size, self.hidden_size,
                 device=input.device, dtype=input.dtype,
                 requires_grad=False)
            hx = hx[:,:input.shape[1]]'''
            if self.mode == 'LSTM':
                hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        #self.check_forward_args(input, hx, batch_sizes)
        func = quant_rnn.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            dropout_state=self.dropout_state,
            variable_length=is_packed,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx, batch_sizes, self._input_quantizers, self._weight_quantizers)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden
    
    def forward_dyna_second(self, input, hx):
        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        T, B, _ = input.size()
        split_size = 128  # hidden_size

        # Quantize inputs and initial hidden states
        input_f = self._input_quantizers[0](input)
        input_b = self._input_quantizers[1](input)
        h_f = self._input_quantizers[0](hx)[0]
        h_b = self._input_quantizers[1](hx)[1]

        # Quantize weights
        weight_ih_l0 = self._weight_quantizers[0](self.weight_ih_l0)
        weight_hh_l0 = self._weight_quantizers[0](self.weight_hh_l0)
        weight_ih_l0_reverse = self._weight_quantizers[1](self.weight_ih_l0_reverse)
        weight_hh_l0_reverse = self._weight_quantizers[1](self.weight_hh_l0_reverse)

        # Biases
        bias_ih_l0 = self.bias_ih_l0
        bias_hh_l0 = self.bias_hh_l0
        bias_ih_l0_reverse = self.bias_ih_l0_reverse
        bias_hh_l0_reverse = self.bias_hh_l0_reverse

        # Allocate output tensors in advance
        out_f = torch.cat([input_f, input_f], dim=2)
        out_b = torch.cat([input_b, input_b], dim=2)

        # Precompute input linear projections
        gi_all_f = F.linear(input_f, weight_ih_l0, bias_ih_l0)
        gi_all_b = F.linear(input_b, weight_ih_l0_reverse, bias_ih_l0_reverse)

        # Forward GRU
        for t in range(T):
            gi = gi_all_f[t]
            gh = F.linear(h_f, weight_hh_l0, bias_hh_l0)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_f = newgate + inputgate * (h_f - newgate)

            out_f[t] = h_f

        # Backward GRU
        for t in reversed(range(T)):
            gi = gi_all_b[t]
            gh = F.linear(h_b, weight_hh_l0_reverse, bias_hh_l0_reverse)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_b = newgate + inputgate * (h_b - newgate)

            out_b[t] = h_b  # Write directly to time step t (not reversed index)

        # Concatenate forward and backward outputs at each timestep
        out = torch.cat([out_f, out_b], dim=2)  # (T, B, 2*hidden_size)

        return out, 1
    
    def forward_(self, input, hx):
        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        
        T, B, _ = input.shape
        T, B, _ = input.size()

        #h_f = torch.zeros(B, self.hidden_size, device=input.device)
        #h_b = torch.zeros(B, self.hidden_size, device=input.device)

        input_f = self._input_quantizers[0](input)
        input_b = self._input_quantizers[1](input)
        h_f = self._input_quantizers[0](hx)[0]
        h_b = self._input_quantizers[1](hx)[1]

        weight_hh_l0 = self._weight_quantizers[0](self.weight_hh_l0)
        weight_hh_l0_reverse = self._weight_quantizers[1](self.weight_hh_l0_reverse)

        weight_ih_l0 = self._weight_quantizers[0](self.weight_ih_l0)
        weight_ih_l0_reverse = self._weight_quantizers[1](self.weight_ih_l0_reverse)

        out_f = torch.cat([input_f, input_f], dim=2)
        #out_b = torch.cat([input_b, input_b], dim=2)

        #out_f = []
        #out_b = []

        #out_f = torch.zeros(T, B, self.hidden_size, device=input.device)
        #out_b = torch.zeros(T, B, self.hidden_size, device=input.device)

        #return torch.cat([input_f, input_b], dim=2), 1

        #return out_f, 1
        '''x_t = input_f[0]
        hy = self.GRUCellCustom(x_t, h_f, weight_ih_l0, weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)

        #h_f = _VF.gru_cell(x_t, h_f, weight_ih_l0, weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)
        x_expanded = hy[0].unsqueeze(0)             # Shape: (1, 535, 64)
        x_repeated = x_expanded.repeat(101, 1, 1)

        return x_repeated, 1'''

        gi_all = F.linear(input_f, weight_ih_l0, self.bias_ih_l0)  # shape: (T, batch, 3*hidden_size)
        split_size = 128

        for t in range(T):
            gh = F.linear(h_f, weight_hh_l0, self.bias_hh_l0)  # (batch, 3*hidden_size)
            
            # Slice gi and gh at t-th timestep
            gi = gi_all[t]  # (batch, 3*hidden_size)

            # Follow your gate split and calculations using gi and gh
            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_f = newgate + inputgate * (h_f - newgate)

            out_f[t] = h_f

        # Forward pass
        '''for t in range(5):
            x_t = input_f[t]
            #h_f = _VF.gru_cell(x_t, h_f, weight_ih_l0, weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)
            h_f = self.GRUCellCustom(x_t, h_f, weight_ih_l0, weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)
            #out_f.append(h_f)
            out_f[t] = h_f
            #out_f.append(h_f.unsqueeze(0))'''
        
        return out_f, 1

        # Backward pass
        #for t in reversed(range(T)):
        '''for t in range(T - 1, -1, -1):
            x_t = input_b[t]
            h_b = _VF.gru_cell(x_t, h_b, weight_ih_l0_reverse, weight_hh_l0_reverse, self.bias_ih_l0_reverse, self.bias_hh_l0_reverse)
            out_b[t] = h_b
            #out_b.insert(0, h_b.unsqueeze(0))'''
        
        #return torch.cat([input_f, input_b], dim=2), 1
        # Concatenate outputs
        #out_f = torch.cat(out_f, dim=0)  # (B, T, H)
        #out_b = torch.cat(out_b, dim=0)  # (B, T, H)
        return torch.stack(out_f, dim=0), 1
        return torch.cat([out_f, out_b], dim=2), 1  # (B, T, 2H)
    
    def GRUCellCustom(self, x_t, h_f, weight_ih, weight_hh, bias_ih, bias_hh):
        gi = F.linear(x_t, weight_ih, bias_ih)
        gh = F.linear(h_f, weight_hh, bias_hh)

        split_size = 128
        #split_size = gi.size(1) // 3
        i_r = gi[:, :split_size]
        i_i = gi[:, split_size:2*split_size]
        i_n = gi[:, 2*split_size:]

        h_r = gh[:, :split_size]
        h_i = gh[:, split_size:2*split_size]
        h_n = gh[:, 2*split_size:]

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (h_f - newgate)
        return hy

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(QuantRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class QuantRNN(QuantRNNBase):
    r"""Applies a multi-layer Elman RNN with `tanh` or `ReLU` non-linearity to an
    input sequence.
    """

    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(QuantRNN, self).__init__(mode, *args, **kwargs)


class QuantLSTM(QuantRNNBase):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.
    """

    def __init__(self, *args, **kwargs):
        super(QuantLSTM, self).__init__('LSTM', *args, **kwargs)


'''class QuantGRU(QuantRNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    """

    def __init__(self, *args, **kwargs):
        super(QuantGRU, self).__init__('GRU', *args, **kwargs)
    
    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)
        
        hx = torch.zeros(self.num_layers * 2,
                 600, self.hidden_size,
                 device=input.device, dtype=input.dtype,
                 requires_grad=False)
        hx = hx[:,:input.shape[1]]

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        #self.check_forward_args(input, hx, batch_sizes)
        func = quant_rnn.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            dropout_state=self.dropout_state,
            variable_length=is_packed,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx, batch_sizes, self._input_quantizers, self._weight_quantizers)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden'''

'''class QuantGRU2(QuantRNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    """

    def __init__(self, *args, **kwargs):
        super(QuantGRU2, self).__init__('GRU', *args, **kwargs)
    
    def forward(self, input, hx):
        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        T, B, _ = input.size()
        split_size = 128  # hidden_size

        # Quantize inputs and initial hidden states
        input_f = self._input_quantizers[0](input)
        input_b = self._input_quantizers[1](input)
        h_f = self._input_quantizers[0](hx)[0]
        h_b = self._input_quantizers[1](hx)[1]

        # Quantize weights
        weight_ih_l0 = self._weight_quantizers[0](self.weight_ih_l0)
        weight_hh_l0 = self._weight_quantizers[0](self.weight_hh_l0)
        weight_ih_l0_reverse = self._weight_quantizers[1](self.weight_ih_l0_reverse)
        weight_hh_l0_reverse = self._weight_quantizers[1](self.weight_hh_l0_reverse)

        # Biases
        bias_ih_l0 = self.bias_ih_l0
        bias_hh_l0 = self.bias_hh_l0
        bias_ih_l0_reverse = self.bias_ih_l0_reverse
        bias_hh_l0_reverse = self.bias_hh_l0_reverse

        # Allocate output tensors in advance
        out_f = torch.cat([input_f, input_f], dim=2)
        out_b = torch.cat([input_b, input_b], dim=2)

        # Precompute input linear projections
        gi_all_f = F.linear(input_f, weight_ih_l0, bias_ih_l0)
        gi_all_b = F.linear(input_b, weight_ih_l0_reverse, bias_ih_l0_reverse)

        # Forward GRU
        for t in range(T):
            gi = gi_all_f[t]
            gh = F.linear(h_f, weight_hh_l0, bias_hh_l0)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_f = newgate + inputgate * (h_f - newgate)

            out_f[t] = h_f

        # Backward GRU
        for t in reversed(range(T)):
            gi = gi_all_b[t]
            gh = F.linear(h_b, weight_hh_l0_reverse, bias_hh_l0_reverse)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_b = newgate + inputgate * (h_b - newgate)

            out_b[t] = h_b  # Write directly to time step t (not reversed index)

        # Concatenate forward and backward outputs at each timestep
        out = torch.cat([out_f, out_b], dim=2)  # (T, B, 2*hidden_size)

        return out, 1'''

class QuantGRU(nn.Module, _utils.QuantMixin):

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, proj_size=0, **kwargs):
        super(QuantGRU, self).__init__()
        mode = 'GRU'
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.dyna_first = batch_first
        #self.forward_fn = self.forward_dyna_first if self.dyna_first else self.forward_dyna_second
        #self.forward_fn = self.forward_dyna_second

        self.batch_first = False
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size > 0:
            raise ValueError("proj_size is not supported in pytorch-quantization yet")

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.init_quantizer(quant_desc_input, quant_desc_weight, num_layers=num_layers * (1 + bidirectional))

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            weight_arr = list(itertools.chain.from_iterable(self.all_weights))
            weight_stride0 = len(self.all_weights[0])

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on weight_arr, that's why the
                # no_grad() is necessary.
                weight_buf = torch._cudnn_rnn_flatten_weight(weight_arr, weight_stride0, self.input_size,
                                                             rnn.get_cudnn_mode(self.mode), self.hidden_size,
                                                             self.proj_size, self.num_layers, self.batch_first,
                                                             bool(self.bidirectional))

            self._param_buf_size = weight_buf.size(0)
            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(QuantGRU, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward_test(self, input, hx):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        T, B, _ = input.size()
        split_size = 128  # hidden_size

        # Quantize weights
        weight_ih_l0 = self._weight_quantizers[0](self.weight_ih_l0)
        weight_hh_l0 = self._weight_quantizers[0](self.weight_hh_l0)
        weight_ih_l0_reverse = self._weight_quantizers[1](self.weight_ih_l0_reverse)
        weight_hh_l0_reverse = self._weight_quantizers[1](self.weight_hh_l0_reverse)

        # Biases
        bias_ih_l0 = self.bias_ih_l0
        bias_hh_l0 = self.bias_hh_l0
        bias_ih_l0_reverse = self.bias_ih_l0_reverse
        bias_hh_l0_reverse = self.bias_hh_l0_reverse

        params = (
            weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
            weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse
        )

        out, h_n = _VF.gru(
            input,           # input (T, B, input_size)
            hx,          # initial hidden
            tuple(self.all_weights),      # tuple of weights/biases
            True,        # has_biases
            1,  # num_layers
            0.0,         # dropout
            True,       # train
            True,        # bidirectional
            True        # batch_first
        )
        return out, 1

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        '''if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)'''
        
        '''hx = torch.zeros(self.num_layers * 2,
                 600, self.hidden_size,
                 device=input.device, dtype=input.dtype,
                 requires_grad=False)
        hx = hx[:,:input.shape[1]]'''

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        #self.check_forward_args(input, hx, batch_sizes)
        func = quant_rnn.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            dropout_state=self.dropout_state,
            variable_length=is_packed,
            flat_weight=flat_weight
        )
        output, hidden = func(input, self.all_weights, hx, batch_sizes, self._input_quantizers, self._weight_quantizers)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(QuantRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

class QuantGRU2(nn.Module, _utils.QuantMixin):

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, proj_size=0, **kwargs):
        super(QuantGRU2, self).__init__()
        mode = 'GRU'
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.dyna_first = batch_first
        #self.forward_fn = self.forward_dyna_first if self.dyna_first else self.forward_dyna_second
        #self.forward_fn = self.forward_dyna_second

        self.batch_first = False
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        num_directions = 2 if bidirectional else 1
        self.zeros = torch.zeros(2,101, hidden_size, device="cuda", requires_grad=False)

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size > 0:
            raise ValueError("proj_size is not supported in pytorch-quantization yet")

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.init_quantizer(quant_desc_input, quant_desc_weight, num_layers=num_layers * (1 + bidirectional))

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            weight_arr = list(itertools.chain.from_iterable(self.all_weights))
            weight_stride0 = len(self.all_weights[0])

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on weight_arr, that's why the
                # no_grad() is necessary.
                weight_buf = torch._cudnn_rnn_flatten_weight(weight_arr, weight_stride0, self.input_size,
                                                             rnn.get_cudnn_mode(self.mode), self.hidden_size,
                                                             self.proj_size, self.num_layers, self.batch_first,
                                                             bool(self.bidirectional))

            self._param_buf_size = weight_buf.size(0)
            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(QuantGRU2, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def gru_step(h, gi_t):
        # recurrent contribution
        gh = F.linear(h, weight_hh_l0, bias_hh_l0)

        # input splits
        i_r, i_i, i_n = torch.split(gi_t, split_size, dim=1)
        h_r, h_i, h_n = torch.split(gh, split_size, dim=1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + resetgate * h_n)

        h_new = newgate + inputgate * (h - newgate)
        return h_new, h_new   # carry, output at this step

    def forward(self, input):
    #def forward(self, input, hx):
        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None

        hx = self.zeros

        T, B, _ = input.size()
        split_size = 128  # hidden_size

        # Quantize inputs and initial hidden states
        input_f = self._input_quantizers[0](input)
        input_b = self._input_quantizers[1](input)
        h_f = self._input_quantizers[0](hx)[0]
        h_b = self._input_quantizers[1](hx)[1]

        # Quantize weights
        weight_ih_l0 = self._weight_quantizers[0](self.weight_ih_l0)
        weight_hh_l0 = self._weight_quantizers[0](self.weight_hh_l0)
        weight_ih_l0_reverse = self._weight_quantizers[1](self.weight_ih_l0_reverse)
        weight_hh_l0_reverse = self._weight_quantizers[1](self.weight_hh_l0_reverse)

        # Biases
        bias_ih_l0 = self.bias_ih_l0
        bias_hh_l0 = self.bias_hh_l0
        bias_ih_l0_reverse = self.bias_ih_l0_reverse
        bias_hh_l0_reverse = self.bias_hh_l0_reverse

        self.flatten_parameters()
        params = (
            weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
            weight_ih_l0_reverse, weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse
        )

        out, h_n = _VF.gru(
            input,           # input (T, B, input_size)
            hx,          # initial hidden
            params,      # tuple of weights/biases
            True,        # has_biases
            1,  # num_layers
            0.0,         # dropout
            True,       # train
            True,        # bidirectional
            False        # batch_first
        )
        return out, 1

        # Allocate output tensors in advance
        out_f = torch.cat([input_f, input_f], dim=2)
        out_b = torch.cat([input_b, input_b], dim=2)
        

        #out_f_list, out_b_list = [], []

        # Precompute input linear projections
        gi_all_f = F.linear(input_f, weight_ih_l0, bias_ih_l0)
        gi_all_b = F.linear(input_b, weight_ih_l0_reverse, bias_ih_l0_reverse)

        # Forward GRU
        for t in range(T):
            gi = gi_all_f[t]
            #gi_all_f = gi_all_f[1:]
            #gi = gi_all_f[t]
            #gi = F.linear(input_f[t], weight_ih_l0, bias_ih_l0)
            gh = F.linear(h_f, weight_hh_l0, bias_hh_l0)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_f = newgate + inputgate * (h_f - newgate)
            #h_f = newgate + inputgate * ((h_f.clone()) - newgate) #was before

            out_f[t] = h_f
            #out_f[t].copy_(h_f)
            #out_f_list.append(h_f)

        #outputs_b = []
        # Backward GRU
        for t in reversed(range(T)):
            gi = gi_all_b[t]
            #gi = F.linear(input_b[t], weight_ih_l0_reverse, bias_ih_l0_reverse)
            gh = F.linear(h_b, weight_hh_l0_reverse, bias_hh_l0_reverse)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_b = newgate + inputgate * (h_b - newgate)

            out_b[t] = h_b  # Write directly to time step t (not reversed index)

        #out_f = torch.stack(out_f_list, dim=0)
        #out_b = torch.stack(list(reversed(out_b_list)), dim=0)

        # Concatenate forward and backward outputs at each timestep
        #out_b = torch.cat(outputs_b, dim=0)
        #out_b = torch.stack(outputs_b[::-1], dim=0)
        out = torch.cat([out_f, out_b], dim=2)  # (T, B, 2*hidden_size)

        return out, 1
    
    def forward_ref_old(self, input):
    #def forward(self, input, hx):
        '''has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None'''

        hx = self.zeros

        T, B, _ = input.size()
        split_size = 128  # hidden_size

        # Quantize inputs and initial hidden states
        input_f = self._input_quantizers[0](input)
        input_b = self._input_quantizers[1](input)
        h_f = self._input_quantizers[0](hx)[0]
        h_b = self._input_quantizers[1](hx)[1]

        # Quantize weights
        weight_ih_l0 = self._weight_quantizers[0](self.weight_ih_l0)
        weight_hh_l0 = self._weight_quantizers[0](self.weight_hh_l0)
        weight_ih_l0_reverse = self._weight_quantizers[1](self.weight_ih_l0_reverse)
        weight_hh_l0_reverse = self._weight_quantizers[1](self.weight_hh_l0_reverse)

        # Biases
        bias_ih_l0 = self.bias_ih_l0
        bias_hh_l0 = self.bias_hh_l0
        bias_ih_l0_reverse = self.bias_ih_l0_reverse
        bias_hh_l0_reverse = self.bias_hh_l0_reverse

        # Allocate output tensors in advance
        out_f = torch.cat([input_f, input_f], dim=2)
        out_b = torch.cat([input_b, input_b], dim=2)

        #out_f_list, out_b_list = [], []

        # Precompute input linear projections
        gi_all_f = F.linear(input_f, weight_ih_l0, bias_ih_l0)
        gi_all_b = F.linear(input_b, weight_ih_l0_reverse, bias_ih_l0_reverse)

        # Forward GRU
        for t in range(T):
            gi = gi_all_f[t]
            #gi_all_f = gi_all_f[1:]
            #gi = gi_all_f[t]
            #gi = F.linear(input_f[t], weight_ih_l0, bias_ih_l0)
            gh = F.linear(h_f, weight_hh_l0, bias_hh_l0)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_f = newgate + inputgate * (h_f - newgate)
            #h_f = newgate + inputgate * ((h_f.clone()) - newgate) #was before

            out_f[t] = h_f
            #out_f_list.append(h_f)

        #outputs_b = []
        # Backward GRU
        for t in reversed(range(T)):
            gi = gi_all_b[t]
            #gi = F.linear(input_b[t], weight_ih_l0_reverse, bias_ih_l0_reverse)
            gh = F.linear(h_b, weight_hh_l0_reverse, bias_hh_l0_reverse)

            i_r = gi[:, :split_size]
            i_i = gi[:, split_size:2*split_size]
            i_n = gi[:, 2*split_size:]

            h_r = gh[:, :split_size]
            h_i = gh[:, split_size:2*split_size]
            h_n = gh[:, 2*split_size:]

            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h_b = newgate + inputgate * (h_b - newgate)
            '''prev_h_b = h_b.clone()
            h_b = newgate + inputgate * (prev_h_b - newgate)'''

            #h_b = newgate + inputgate * ((h_b.clone()) - newgate)

            out_b[t] = h_b  # Write directly to time step t (not reversed index)
            #out_b_list.append(h_b)

        #out_f = torch.stack(out_f_list, dim=0)
        #out_b = torch.stack(list(reversed(out_b_list)), dim=0)

        # Concatenate forward and backward outputs at each timestep
        #out_b = torch.cat(outputs_b, dim=0)
        #out_b = torch.stack(outputs_b[::-1], dim=0)
        out = torch.cat([out_f, out_b], dim=2)  # (T, B, 2*hidden_size)

        return out, 1
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(QuantRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

class QuantRNNCellBase(nn.Module, _utils.QuantMixin):

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class QuantRNNCell(QuantRNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(QuantRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        if self.nonlinearity == "tanh":
            func = quant_rnn.RNNTanhCell
        elif self.nonlinearity == "relu":
            func = quant_rnn.RNNReLUCell
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        return func(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class QuantLSTMCell(QuantRNNCellBase):
    r"""A long short-term memory (LSTM) cell.
    """

    def __init__(self, input_size, hidden_size, bias=True, **kwargs):
        super(QuantLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return quant_rnn.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self._input_quantizer, self._weight_quantizer
        )


class QuantGRUCell(QuantRNNCellBase):
    r"""A gated recurrent unit (GRU) cell
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(QuantGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        return quant_rnn.GRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

LSTM = QuantLSTM
LSTMCell = QuantLSTMCell
GRU = QuantGRU
GRUCell = QuantGRUCell
