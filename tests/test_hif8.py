"""Unit tests for HiF8 cast fake quantization (arXiv:2409.16626)."""
import math
import unittest

import torch

from pytorch_quantization.tensor_quant import (
    _hif8_decode_byte, fake_hif8_quant, hif8_decode, hif8_encode,
)


class HiF8DecodeTableTest(unittest.TestCase):
    def test_specials_from_paper_table4(self):
        self.assertEqual(_hif8_decode_byte(0x00), 0.0)
        self.assertTrue(math.isnan(_hif8_decode_byte(0x80)))
        self.assertEqual(_hif8_decode_byte(0x6F), float("inf"))
        self.assertEqual(_hif8_decode_byte(0xEF), float("-inf"))
        self.assertEqual(_hif8_decode_byte(0x6E), 2.0 ** 15)
        self.assertEqual(_hif8_decode_byte(0x7E), 2.0 ** -15)
        self.assertEqual(_hif8_decode_byte(0x07), 2.0 ** -16)
        self.assertEqual(_hif8_decode_byte(0x01), 2.0 ** -22)

    def test_roundtrip_all_finite_codes(self):
        for code in range(256):
            v = _hif8_decode_byte(code)
            if math.isnan(v) or math.isinf(v):
                continue
            out = fake_hif8_quant(torch.tensor([v]))
            self.assertEqual(out.item(), v)

    def test_signed_mirror(self):
        self.assertEqual(_hif8_decode_byte(0x6E | 0x80), -(2.0 ** 15))
        self.assertEqual(_hif8_decode_byte(0x01 | 0x80), -(2.0 ** -22))


class HiF8EncodeTATest(unittest.TestCase):
    def test_exact_grid_unchanged(self):
        vals = torch.tensor([0.0, 2.0 ** -22, 2.0 ** -16, 1.0, 2.0, 2.0 ** 15, -1.0, -2.0])
        torch.testing.assert_close(fake_hif8_quant(vals), vals, rtol=0, atol=0)

    def test_round_half_away_from_zero(self):
        # Midpoint between 0 and min positive finite 2^-22 → away from 0 → 2^-22
        mid0 = 0.5 * (2.0 ** -22)
        out = fake_hif8_quant(torch.tensor([mid0, -mid0]))
        torch.testing.assert_close(out, torch.tensor([2.0 ** -22, -(2.0 ** -22)]), rtol=0, atol=0)

    def test_overflow_to_inf(self):
        out = fake_hif8_quant(torch.tensor([2.0 ** 16, -(2.0 ** 20)]))
        self.assertTrue(math.isinf(out[0].item()) and out[0].item() > 0)
        self.assertTrue(math.isinf(out[1].item()) and out[1].item() < 0)

    def test_nan_stays_nan(self):
        out = fake_hif8_quant(torch.tensor([float("nan")]))
        self.assertTrue(math.isnan(out.item()))

    def test_neg_zero_collapses_to_pos_zero_code(self):
        codes = hif8_encode(torch.tensor([-0.0, 0.0]))
        self.assertEqual(int(codes[0]), 0)
        self.assertEqual(int(codes[1]), 0)

    def test_tiny_negative_flush_to_zero_not_nan(self):
        # |x| << 2^-22 rounds to 0; must be code 0x00, not 0x80 (NaN)
        x = torch.tensor([-3.14e-8, 3.14e-8])
        codes = hif8_encode(x)
        self.assertEqual(int(codes[0]), 0)
        self.assertEqual(int(codes[1]), 0)
        out = fake_hif8_quant(x)
        self.assertFalse(torch.isnan(out).any().item())
        torch.testing.assert_close(out, torch.zeros(2), rtol=0, atol=0)

    def test_encode_decode_tensor_api(self):
        x = torch.randn(4, 8)
        codes = hif8_encode(x)
        self.assertEqual(codes.dtype, torch.uint8)
        recon = hif8_decode(codes)
        torch.testing.assert_close(recon, fake_hif8_quant(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
