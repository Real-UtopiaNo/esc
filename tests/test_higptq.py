"""Unit tests for HiGPTQ core (GPTQ + HiF4 quantizer)."""
import unittest

import torch

from pytorch_quantization.tensor_quant import fake_hif4_quant
from ptq.higptq_ptq import gptq_hif4, _update_hessian


class HessianUpdateTest(unittest.TestCase):
    def test_matches_batch_outer_product(self):
        torch.manual_seed(0)
        d, n = 32, 64
        X = torch.randn(n, d)
        H = torch.zeros(d, d)
        H, ns = _update_hessian(H, 0, X)
        self.assertEqual(ns, n)
        # After one batch: H = 2/n * X.T @ X  (with the incremental formula)
        expected = (2.0 / n) * (X.T @ X)
        torch.testing.assert_close(H, expected, rtol=1e-5, atol=1e-5)


class GptqHif4Test(unittest.TestCase):
    def test_frozen_scale_column_is_idempotent(self):
        """A column quantized with frozen group scales is unchanged by re-apply."""
        from ptq.higptq_ptq import _freeze_hif4_group_scales, _quant_column_frozen

        torch.manual_seed(0)
        group = torch.randn(16, 64)
        scales = _freeze_hif4_group_scales(group)
        q0 = _quant_column_frozen(group[:, 3], scales, 3)
        q1 = _quant_column_frozen(q0, scales, 3)
        torch.testing.assert_close(q0, q1, rtol=0, atol=0)

    def test_gptq_output_finite_same_shape(self):
        torch.manual_seed(0)
        d_out, d_in, n = 32, 128, 256
        W = torch.randn(d_out, d_in)
        X = torch.randn(n, d_in)
        H = 2.0 * (X.T @ X) / n
        Q = gptq_hif4(W, H, blocksize=64, percdamp=0.01)
        self.assertEqual(Q.shape, W.shape)
        self.assertTrue(torch.isfinite(Q).all())

    def test_beats_plain_cast_on_reconstruction(self):
        torch.manual_seed(1)
        d_out, d_in, n = 64, 128, 512
        W = torch.randn(d_out, d_in)
        X = torch.randn(n, d_in)
        H = 2.0 * (X.T @ X) / n

        W_cast = fake_hif4_quant(W)
        W_gptq = gptq_hif4(W, H, blocksize=128, percdamp=0.01)

        err_cast = (W_cast @ X.T - W @ X.T).pow(2).sum().item()
        err_gptq = (W_gptq @ X.T - W @ X.T).pow(2).sum().item()
        self.assertLessEqual(err_gptq, err_cast * 1.05)


if __name__ == "__main__":
    unittest.main()
