import types
import unittest

import torch
import torch.nn.functional as F

from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import fake_block_quant, fake_hif4_quant, fake_nvfp4_quant, fake_tensor_quant
from utils.quant_utils import (set_bits, set_block_quant, set_dynamic_tensor_act, set_dynamic_token_act,
                               set_gemm_only, set_weight_only)

GEMM_LAYERS = [
    "encoder.conv_subsample.module.2",
    "encoder.linear_proj",
    "encoder.layers.0.ff1.module.1",
    "encoder.layers.0.attention.W_q",
    "encoder.layers.0.conv_block.module.0",
    "encoder.layers.0.conv_block.module.6",
    "decoder.linear",
]
NON_GEMM_LAYERS = [
    "encoder.conv_subsample.module.0",
    "encoder.layers.0.ff1.module.0",
    "encoder.layers.0.conv_block.module.1",
    "encoder.layers.0.conv_block.module.3",
    "encoder.layers.0.conv_block.module.4",
    "encoder.layers.0.layer_norm",
]


def _build_quant_conformer():
    from models.conformer.model import Conformer

    quant_modules.initialize()
    try:
        return types.SimpleNamespace(model=Conformer().eval())
    finally:
        quant_modules.deactivate()


def _quantizers(model_ptq):
    for name, module in model_ptq.model.named_modules():
        if hasattr(module, "input_quantizer"):
            yield name, "input", module.input_quantizer
        if hasattr(module, "_weight_quantizer"):
            yield name, "weight", module.weight_quantizer


def _set_static_amax(quantizer, value):
    quantizer.amax = value


def _acts_as_nvfp4(quantizer, x, block_axis):
    _set_static_amax(quantizer, x.abs().max().item())
    return torch.equal(quantizer(x), fake_nvfp4_quant(x, x.abs().max(), block_axis=block_axis))


class SetBlockQuantTest(unittest.TestCase):
    def test_all_coverage_quantizes_every_quantizer(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="all")
        counts = {"input": 0, "weight": 0}
        x = torch.randn(2, 32, 48)
        for name, kind, quantizer in _quantizers(model_ptq):
            with self.subTest(name=name, kind=kind):
                self.assertFalse(torch.equal(quantizer(x), x))
            counts[kind] += 1
        self.assertEqual(counts, {"input": 308, "weight": 276})

    def test_gemm_coverage_leaves_non_gemm_layers_in_fp32(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="gemm")
        modules = dict(model_ptq.model.named_modules())
        x = torch.randn(2, 32, 48)
        for name in GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertFalse(torch.equal(modules[name].input_quantizer(x), x))
                self.assertFalse(torch.equal(modules[name].weight_quantizer(x), x))
        for name in NON_GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertTrue(torch.equal(modules[name].input_quantizer(x), x))
                if hasattr(modules[name], "_weight_quantizer"):
                    self.assertTrue(torch.equal(modules[name].weight_quantizer(x), x))

    def test_weight_only_keeps_activations_in_fp32(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="all")
        set_weight_only(model_ptq)
        x = torch.randn(2, 32, 48)
        for name, kind, quantizer in _quantizers(model_ptq):
            with self.subTest(name=name, kind=kind):
                self.assertEqual(torch.equal(quantizer(x), x), kind == "input")

    def test_linear_forward_uses_nvfp4_inputs_and_weights(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="all")
        linear = model_ptq.model.encoder.layers[0].ff1.module[1]
        torch.manual_seed(0)
        x = torch.randn(2, 10, 144) * 3
        _set_static_amax(linear.input_quantizer, 4.0)
        w_amax = linear.weight.abs().max()
        _set_static_amax(linear.weight_quantizer, w_amax.item())
        expected = F.linear(fake_nvfp4_quant(x, torch.tensor(4.0)),
                            fake_nvfp4_quant(linear.weight, w_amax), linear.bias)
        torch.testing.assert_close(linear(x), expected, rtol=0, atol=0)

    def test_linear_forward_uses_int4_block_inputs_and_weights(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="all", element="int4")
        linear = model_ptq.model.encoder.layers[0].ff1.module[1]
        torch.manual_seed(0)
        x = torch.randn(2, 10, 144) * 3
        _set_static_amax(linear.input_quantizer, 4.0)
        w_amax = linear.weight.abs().max()
        _set_static_amax(linear.weight_quantizer, w_amax.item())
        expected = F.linear(fake_block_quant(x, torch.tensor(4.0), element="int4"),
                            fake_block_quant(linear.weight, w_amax, element="int4"), linear.bias)
        torch.testing.assert_close(linear(x), expected, rtol=0, atol=0)

    def test_pointwise_conv_blocks_inputs_along_channels(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="gemm")
        conv = model_ptq.model.encoder.layers[0].conv_block.module[0]
        torch.manual_seed(0)
        x = torch.randn(2, 144, 20)
        self.assertTrue(_acts_as_nvfp4(conv.input_quantizer, x, block_axis=1))


class DynamicTensorActTest(unittest.TestCase):
    """Block-quantized activations whose global amax is the max of each input, not calibrated."""

    def _build(self, element):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="gemm", element=element)
        set_dynamic_tensor_act(model_ptq)
        return dict(model_ptq.model.named_modules())

    def test_global_amax_follows_each_input(self):
        torch.manual_seed(0)
        for element in ["e2m1", "int4"]:
            modules = self._build(element)
            quantizer = modules["encoder.layers.0.ff1.module.1"].input_quantizer
            for gain in [0.01, 1.0, 100.0]:
                x = torch.randn(1, 10, 144) * gain
                with self.subTest(element=element, gain=gain):
                    expected = fake_block_quant(x, x.abs().max(), element=element)
                    torch.testing.assert_close(quantizer(x), expected, rtol=0, atol=0)

    def test_pointwise_conv_keeps_channel_blocks(self):
        modules = self._build("e2m1")
        x = torch.randn(1, 144, 20)
        expected = fake_nvfp4_quant(x, x.abs().max(), block_axis=1)
        torch.testing.assert_close(modules["encoder.layers.0.conv_block.module.0"].input_quantizer(x),
                                   expected, rtol=0, atol=0)

    def test_non_gemm_layers_stay_in_fp32(self):
        modules = self._build("e2m1")
        x = torch.randn(2, 32, 48)
        for name in NON_GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertTrue(torch.equal(modules[name].input_quantizer(x), x))


class SetHiF4Test(unittest.TestCase):
    def test_linear_forward_uses_hif4(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="all", element="hif4")
        linear = model_ptq.model.encoder.layers[0].ff1.module[1]
        torch.manual_seed(0)
        x = torch.randn(2, 10, 144) * 3
        expected = F.linear(fake_hif4_quant(x), fake_hif4_quant(linear.weight), linear.bias)
        torch.testing.assert_close(linear(x), expected, rtol=0, atol=0)

    def test_gemm_coverage_leaves_non_gemm_in_fp32(self):
        model_ptq = _build_quant_conformer()
        set_block_quant(model_ptq, coverage="gemm", element="hif4")
        modules = dict(model_ptq.model.named_modules())
        x = torch.randn(2, 32, 48)
        for name in NON_GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertTrue(torch.equal(modules[name].input_quantizer(x), x))


class IntDynamicTensorActTest(unittest.TestCase):
    """Paper-style INT4: per-channel weights, dynamic per-tensor activations."""

    def test_all_coverage_uses_one_scale_per_tensor(self):
        model_ptq = _build_quant_conformer()
        set_bits(model_ptq, 4)
        set_dynamic_tensor_act(model_ptq)
        linear = dict(model_ptq.model.named_modules())["encoder.layers.0.ff1.module.1"]
        torch.manual_seed(0)
        x = torch.randn(2, 10, 144) * 3
        expected = fake_tensor_quant(x, x.abs().max(), 4)
        torch.testing.assert_close(linear.input_quantizer(x), expected, rtol=0, atol=0)

    def test_gemm_coverage_disables_non_gemm_layers(self):
        model_ptq = _build_quant_conformer()
        set_bits(model_ptq, 4)
        set_gemm_only(model_ptq)
        set_dynamic_tensor_act(model_ptq)
        modules = dict(model_ptq.model.named_modules())
        x = torch.randn(2, 32, 48)
        for name in GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertFalse(torch.equal(modules[name].input_quantizer(x), x))
        for name in NON_GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertTrue(torch.equal(modules[name].input_quantizer(x), x))


class DynamicTokenActTest(unittest.TestCase):
    """One INT4 scale per time frame, computed from the frame itself at run time."""

    @classmethod
    def setUpClass(cls):
        cls.model_ptq = _build_quant_conformer()
        set_bits(cls.model_ptq, 4)
        set_dynamic_token_act(cls.model_ptq)
        cls.modules = dict(cls.model_ptq.model.named_modules())
        torch.manual_seed(0)

    def test_gemm_coverage_keeps_non_gemm_in_fp32(self):
        model_ptq = _build_quant_conformer()
        set_bits(model_ptq, 4)
        set_gemm_only(model_ptq)
        set_dynamic_token_act(model_ptq)
        modules = dict(model_ptq.model.named_modules())
        x = torch.randn(2, 32, 48)
        for name in NON_GEMM_LAYERS:
            with self.subTest(name=name):
                self.assertTrue(torch.equal(modules[name].input_quantizer(x), x))
                self.assertTrue(modules[name].input_quantizer._disabled)

    def _assert_per_frame(self, name, x, reduce_dims):
        amax = x.abs().amax(dim=reduce_dims, keepdim=True)
        expected = fake_tensor_quant(x, amax, 4)
        torch.testing.assert_close(self.modules[name].input_quantizer(x), expected, rtol=0, atol=0)

    def test_linear_and_layernorm_inputs_use_one_scale_per_frame(self):
        x = torch.randn(2, 10, 144) * torch.logspace(-2, 2, 10).view(1, 10, 1)
        for name in ["encoder.layers.0.ff1.module.1", "encoder.layers.0.ff1.module.0"]:
            with self.subTest(name=name):
                self._assert_per_frame(name, x, -1)

    def test_conv1d_glu_and_batchnorm_inputs_reduce_over_channels(self):
        x = torch.randn(2, 288, 12) * torch.logspace(-2, 2, 12).view(1, 1, 12)
        for name in ["encoder.layers.0.conv_block.module.1", "encoder.layers.0.conv_block.module.4"]:
            with self.subTest(name=name):
                self._assert_per_frame(name, x, 1)

    def test_conv2d_inputs_reduce_over_channels_and_frequency(self):
        x = torch.randn(2, 144, 12, 19) * torch.logspace(-2, 2, 12).view(1, 1, 12, 1)
        self._assert_per_frame("encoder.conv_subsample.module.2", x, (1, 3))

    def test_weights_stay_per_channel_static(self):
        linear = self.modules["encoder.layers.0.ff1.module.1"]
        w_amax = linear.weight.abs().amax(dim=1, keepdim=True)
        linear.weight_quantizer.amax = w_amax
        expected = fake_tensor_quant(linear.weight, w_amax, 4)
        torch.testing.assert_close(linear.weight_quantizer(linear.weight), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
