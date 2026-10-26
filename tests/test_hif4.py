"""Unit tests for HiF4 fake quantization (arXiv:2602.11287 Algorithm 1)."""
import unittest

import torch

from pytorch_quantization.tensor_quant import fake_hif4_quant, _to_e6m2, _round_to_s1p2

S1P2_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


class E6M2AndS1P2Test(unittest.TestCase):
    def test_e6m2_exact_powers(self):
        for e in [-10, 0, 5]:
            x = torch.tensor([2.0 ** e])
            out = _to_e6m2(x)
            torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_e6m2_rounds_mantissa(self):
        # 1.3 * 2^3 ≈ 10.4 → nearest E6M2 is 1.25*8=10 or 1.5*8=12
        x = torch.tensor([1.3 * 8.0])
        out = _to_e6m2(x)
        self.assertTrue(out.item() in (10.0, 12.0))

    def test_s1p2_grid_unchanged(self):
        vals = torch.tensor(S1P2_GRID + [-v for v in S1P2_GRID[1:]])
        torch.testing.assert_close(_round_to_s1p2(vals), vals, rtol=0, atol=0)

    def test_s1p2_saturates_at_1_75(self):
        x = torch.tensor([2.0, -3.0, 1.8])
        out = _round_to_s1p2(x)
        torch.testing.assert_close(out, torch.tensor([1.75, -1.75, 1.75]), rtol=0, atol=0)


class FakeHiF4QuantTest(unittest.TestCase):
    def test_zero_block_stays_zero(self):
        x = torch.zeros(1, 64)
        torch.testing.assert_close(fake_hif4_quant(x), x, rtol=0, atol=0)

    def test_uniform_block_stays_nearly_constant(self):
        # Flat block: reconstruction should be a single S1P2 level for all 64 elements
        x = torch.full((1, 64), 0.5)
        out = fake_hif4_quant(x)
        self.assertEqual(out.unique().numel(), 1)
        self.assertLess((out - x).abs().max().item(), 0.1)

    def test_outlier_in_one_microblock_does_not_force_others(self):
        # 64 elems: first 4 are huge, rest are 0.25. After hierarchical scales, the
        # quiet micro-blocks should keep fine resolution near 0.25.
        x = torch.full((1, 64), 0.25)
        x[0, :4] = 7.0
        out = fake_hif4_quant(x)
        quiet = out[0, 4:]
        self.assertTrue((quiet.abs() - 0.25).abs().max().item() < 0.26)

    def test_matches_algorithm1_reference(self):
        torch.manual_seed(0)
        x = torch.randn(3, 128) * torch.logspace(-2, 1, 3).unsqueeze(1)
        x[1, 64:96] = 0.0
        out = fake_hif4_quant(x)
        expected = _algorithm1_reference(x)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_pads_partial_last_block(self):
        x = torch.randn(1, 70)
        out = fake_hif4_quant(x)
        self.assertEqual(out.shape, x.shape)
        # First 64 should match a lone block quant of that slice
        torch.testing.assert_close(out[:, :64], fake_hif4_quant(x[:, :64]), rtol=0, atol=0)

    def test_blocks_along_requested_axis(self):
        x = torch.randn(2, 64, 3)
        out = fake_hif4_quant(x, block_axis=1)
        expected = fake_hif4_quant(x.movedim(1, -1)).movedim(-1, 1)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_flattened_blocks_for_conv_weights(self):
        w = torch.randn(4, 3, 31)
        out = fake_hif4_quant(w, block_axis=None)
        expected = fake_hif4_quant(w.reshape(4, -1)).reshape(w.shape)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _algorithm1_reference(x):
    """Literal per-block Algorithm 1 (arXiv:2602.11287) for last-axis blocks of 64."""
    from pytorch_quantization.tensor_quant import _to_e6m2, _round_to_s1p2

    length = x.shape[-1]
    pad = -length % 64
    xp = torch.nn.functional.pad(x.float(), (0, pad)) if pad else x.float()
    blocks = xp.reshape(*xp.shape[:-1], -1, 64)
    out_blocks = torch.zeros_like(blocks)
    for b in range(blocks.shape[-2]):
        v64 = blocks[..., b, :]
        v16 = v64.abs().reshape(*v64.shape[:-1], 16, 4).amax(-1)
        v8 = v16.reshape(*v16.shape[:-1], 8, 2).amax(-1)
        vmax = v8.amax(-1)
        zero = vmax == 0
        sf = vmax / 7.0
        e6m2 = _to_e6m2(sf.clamp(min=2.0 ** -48))
        rec = 1.0 / e6m2
        e1_8 = (v8 * rec.unsqueeze(-1) >= 4).to(v64.dtype)
        e1_8_for16 = e1_8.repeat_interleave(2, dim=-1)
        e1_16 = (v16 * rec.unsqueeze(-1) * (2.0 ** (-e1_8_for16)) >= 2).to(v64.dtype)
        e1_8_el = e1_8.repeat_interleave(8, dim=-1)
        e1_16_el = e1_16.repeat_interleave(4, dim=-1)
        scaled = v64 * rec.unsqueeze(-1) * (2.0 ** (-e1_8_el)) * (2.0 ** (-e1_16_el))
        s1p2 = _round_to_s1p2(scaled)
        recon = e6m2.unsqueeze(-1) * (2.0 ** (e1_8_el + e1_16_el)) * s1p2
        recon = torch.where(zero.unsqueeze(-1), torch.zeros_like(recon), recon)
        out_blocks[..., b, :] = recon
    return out_blocks.reshape(*xp.shape)[..., :length].to(x.dtype)


if __name__ == "__main__":
    unittest.main()
