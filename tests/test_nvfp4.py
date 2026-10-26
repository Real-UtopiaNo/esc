import unittest

import torch

from pytorch_quantization.tensor_quant import fake_block_quant, fake_nvfp4_quant

E2M1_BLOCK = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]
# global_amax = 6 * 448 makes the per-tensor scale 1, so a block whose amax is 6 has block scale 1
UNIT_GLOBAL_AMAX = torch.tensor(6.0 * 448.0)


class FakeNVFP4QuantTest(unittest.TestCase):
    def test_e2m1_grid_values_are_unchanged(self):
        x = torch.tensor([E2M1_BLOCK])
        out = fake_nvfp4_quant(x, UNIT_GLOBAL_AMAX)
        torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_rounds_half_to_even_and_to_nearest(self):
        # 6.0 pins the block amax so the block scale stays exactly 1
        x = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75,
                           0.3, 0.7, 2.6, 4.9, 5.1, -3.4, 0.1, 6.0]])
        expected = torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0,
                                  0.5, 0.5, 3.0, 4.0, 6.0, -3.0, 0.0, 6.0]])
        out = fake_nvfp4_quant(x, UNIT_GLOBAL_AMAX)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_matches_modelopt_reference(self):
        torch.manual_seed(0)
        row_scales = torch.logspace(-3, 3, 8).unsqueeze(1)
        x = torch.randn(8, 64) * row_scales
        x[3, 16:32] = 0.0
        cases = {
            "dynamic global amax": x.abs().max(),
            "static global amax below data max": x.abs().max() * 0.25,
        }
        for name, global_amax in cases.items():
            with self.subTest(name):
                out = fake_nvfp4_quant(x, global_amax)
                torch.testing.assert_close(out, _modelopt_nvfp4(x, global_amax), rtol=0, atol=0)

    def test_static_global_amax_saturates_values(self):
        x = torch.zeros(1, 16)
        x[0, 0] = 10.0
        out = fake_nvfp4_quant(x, torch.tensor(5.0))
        self.assertEqual(out[0, 0].item(), 5.0)

    def test_blocks_along_requested_axis(self):
        # [batch, channel, time]: blocks run along channels, so the outlier at t=1 must not
        # share a block with the t=0 values
        x = torch.zeros(1, 16, 2)
        x[0, :, 0] = 0.5 * torch.tensor(E2M1_BLOCK).roll(-1)
        x[0, 0, 1] = 1000.0
        out = fake_nvfp4_quant(x, UNIT_GLOBAL_AMAX, block_axis=1)
        torch.testing.assert_close(out[0, :, 0], x[0, :, 0], rtol=0, atol=0)

    def test_pads_partial_last_block(self):
        x = torch.tensor([E2M1_BLOCK + [0.5 * v for v in E2M1_BLOCK[:15]]])
        self.assertEqual(x.shape[-1], 31)
        out = fake_nvfp4_quant(x, UNIT_GLOBAL_AMAX)
        torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_flattened_blocks_for_conv_weights(self):
        torch.manual_seed(0)
        w = torch.randn(4, 3, 31)
        amax = w.abs().max()
        out = fake_nvfp4_quant(w, amax, block_axis=None)
        expected = fake_nvfp4_quant(w.reshape(4, -1), amax).reshape(w.shape)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


INT4_BLOCK = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 0.0]
# global_amax = 7 * 448 makes the per-tensor scale 1, so a block whose amax is 7 has block scale 1
INT4_UNIT_GLOBAL_AMAX = torch.tensor(7.0 * 448.0)


class FakeInt4BlockQuantTest(unittest.TestCase):
    def test_int4_grid_values_are_unchanged(self):
        x = torch.tensor([INT4_BLOCK])
        out = fake_block_quant(x, INT4_UNIT_GLOBAL_AMAX, element="int4")
        torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_rounds_half_to_even_and_to_nearest(self):
        # 7.0 pins the block amax so the block scale stays exactly 1
        x = torch.tensor([[0.5, 1.5, 2.5, -2.5, 0.4, 3.6, 6.5, -6.6,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]])
        expected = torch.tensor([[0.0, 2.0, 2.0, -2.0, 0.0, 4.0, 6.0, -7.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]])
        out = fake_block_quant(x, INT4_UNIT_GLOBAL_AMAX, element="int4")
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_outlier_only_affects_its_own_block(self):
        x = torch.tensor([INT4_BLOCK + [1000.0] + [0.0] * 15])
        out = fake_block_quant(x, INT4_UNIT_GLOBAL_AMAX, element="int4")
        torch.testing.assert_close(out[:, :16], x[:, :16], rtol=0, atol=0)

    def test_e2m1_element_is_nvfp4(self):
        torch.manual_seed(0)
        x = torch.randn(3, 2, 40)
        amax = x.abs().max()
        torch.testing.assert_close(fake_block_quant(x, amax, element="e2m1", block_axis=1),
                                   fake_nvfp4_quant(x, amax, block_axis=1), rtol=0, atol=0)


def _modelopt_nvfp4(x, global_amax):
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    global_scale = global_amax.float() / (6.0 * 448.0)
    qtensor, block_scale, global_scale = NVFP4QTensor.quantize(
        x.clone(), 16, weights_scaling_factor_2=global_scale)
    return qtensor.dequantize(dtype=torch.float32, scale=block_scale,
                              double_scale=global_scale, block_sizes={-1: 16})


if __name__ == "__main__":
    unittest.main()
