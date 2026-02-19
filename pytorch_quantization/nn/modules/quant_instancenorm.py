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

"""Quantized instance normalization module
   Base code is from nn.InstanceNorm, details of the module can be found from the offical repo.
"""

from torch.nn.modules.batchnorm import _NormBase
import torch.nn.functional as F
from torch.nn.modules import instancenorm
from torch.nn.modules import batchnorm
from torch import nn
import torch
from torch.nn import Parameter


from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization import tensor_quant
from . import _utils

__all__ = [
    "QuantInstanceNorm1d", "QuantInstanceNorm2d", "QuantInstanceNorm3d", "QuantBatchNorm1d"
]

class QuantInstanceNorm1d(instancenorm.InstanceNorm1d, _utils.QuantInputMixin):
    r"""Applies Quantized Instance Normalization over a 3D input
    """
    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False,
            track_running_stats: bool = False, **kwargs):
        super(QuantInstanceNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantInstanceNorm1d, self).forward(quant_input)

#class QuantInstanceNorm2d(instancenorm.InstanceNorm2d, _utils.QuantMixin):
class QuantInstanceNorm2d(nn.InstanceNorm2d, _utils.QuantMixin):
#class QuantInstanceNorm2d(instancenorm.InstanceNorm2d, _utils.QuantInputMixin):
    r"""Applies Quantized Instance Normalization over a 4D input
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    #default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_INSTANCENORM_WEIGHT_PER_CHANNEL

    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False,
            track_running_stats: bool = False, **kwargs):
        super(QuantInstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if affine:
            quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            #self.quant_weight = self._weight_quantizer(self.weight)
            #self.quant_weight = self.quant_weight.cuda()
            a = 1
            self.calib = False
        else:
            quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
            self.init_quantizer(quant_desc_input)
    
    def calibration(self):
        quant_weight = self._weight_quantizer(self.weight)
        with torch.no_grad():
            self.weight.copy_(quant_weight)
        self.calib = True
        a = 2


    def forward(self, input):
        if self.affine:
            quant_input = self._input_quantizer(input)
            if self.calib:
                output = super(QuantInstanceNorm2d, self).forward(quant_input)
            else:
                quant_weight = self._weight_quantizer(self.weight)
                quant_weight = quant_weight.detach()  # avoid gradient
                output = F.instance_norm(quant_input, weight=self.weight)
            
            #print('Instance')
            #print(self._weight_quantizer.extra_repr())

            '''print('quant')
            print(quant_weight)

            print('original')
            print(self.weight)
            print('________________')'''
            #output = F.instance_norm(quant_input, weight=self.weight)
            #output = super(QuantInstanceNorm2d, self).forward(quant_input)

            '''num_unique_input = torch.unique(quant_input).numel()
            num_unique_weight = torch.unique(quant_weight).numel()
            num_unique_weight2 = torch.unique(self.weight).numel()
            num_unique_output = torch.unique(output).numel()'''
            return output
        else:
            quant_input = self._input_quantizer(input)
            output = super(QuantInstanceNorm2d, self).forward(quant_input)
            num_unique_output = torch.unique(output).numel()
            return output
    
    '''@property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer'''

'''class QuantInstanceNorm2d(nn.InstanceNorm2d, _utils.QuantInputMixin):
    r"""Applies Quantized Instance Normalization over a 4D input
    """
    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False,
            track_running_stats: bool = False, **kwargs):
        super(QuantInstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantInstanceNorm2d, self).forward(quant_input)'''


class QuantInstanceNorm3d(instancenorm.InstanceNorm3d, _utils.QuantInputMixin):
    r"""Applies Quantized Instance Normalization over a 5D input
    """
    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False,
            track_running_stats: bool = False, **kwargs):
        super(QuantInstanceNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantInstanceNorm3d, self).forward(quant_input)


class QuantBatchNorm1d(batchnorm.BatchNorm1d, _utils.QuantInputMixin):
    r"""Applies Quantized Batch Normalization over a 1D input
    """
    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
            track_running_stats: bool = True, **kwargs):
        super(QuantBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantBatchNorm1d, self).forward(quant_input)