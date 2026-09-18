# Owner(s): ["oncall: quantization"]

import time

import torch
from torch.ao.quantization.nfloat import (
    dequantize_nfloat4,
    nfloat20_quantize_ste,
    NFloat20FakeQuantize,
    NFloat20Linear,
    NFloat4Linear,
    pack_uint4,
    quantize_nfloat20,
    quantize_nfloat4,
    unpack_uint4,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestNFloatPrecision(TestCase):
    def test_nfloat20_precision_accuracy(self):
        """Test NFloat20 precision provides higher accuracy than standard BF16."""
        x = torch.randn(1000, 1000, dtype=torch.float32)

        # NFloat20 (11 mantissa bits)
        x_nf20 = quantize_nfloat20(x)
        err_nf20 = torch.mean(torch.abs(x - x_nf20)).item()

        # Standard BF16 (7 mantissa bits)
        x_bf16 = x.to(torch.bfloat16).to(torch.float32)
        err_bf16 = torch.mean(torch.abs(x - x_bf16)).item()

        # NFloat20 error must be lower than BF16 error
        self.assertLess(err_nf20, err_bf16)

    def test_nfloat20_negative_values(self):
        """Test NFloat20 preserves sign bit on negative numbers."""
        x = torch.tensor([-1.5, -0.125, -100.0, -1e-5], dtype=torch.float32)
        x_nf20 = quantize_nfloat20(x)

        # All negative values must stay negative
        self.assertTrue(torch.all(x_nf20 < 0))
        self.assertLess(torch.max(torch.abs(x - x_nf20)).item(), 1e-3)

    def test_nfloat20_autograd_ste(self):
        """Test NFloat20 Straight-Through Estimator passes gradients in autograd."""
        x = torch.randn(10, 10, requires_grad=True)
        out = nfloat20_quantize_ste(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_nfloat20_fake_quantize_and_linear(self):
        """Test NFloat20FakeQuantize module and NFloat20Linear layer."""
        fq = NFloat20FakeQuantize()
        x = torch.randn(5, 10)
        x_fq = fq(x)
        self.assertEqual(x_fq.shape, x.shape)

        linear = NFloat20Linear(10, 20)
        x_in = torch.randn(4, 10, requires_grad=True)
        out = linear(x_in)
        self.assertEqual(out.shape, (4, 20))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x_in.grad)
        self.assertIsNotNone(linear.weight.grad)

    def test_pack_unpack_uint4(self):
        """Test exact bitwise roundtrip for packing and unpacking 4-bit values (0..15)."""
        values = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8
        )
        packed = pack_uint4(values)
        self.assertEqual(packed.shape, (8,))

        unpacked = unpack_uint4(packed)
        self.assertTrue(torch.equal(values, unpacked))

    def test_nfloat4_quantization_roundtrip(self):
        """Test NFloat4 block-wise quantization and dequantization on normal distribution tensors."""
        torch.manual_seed(42)
        x = torch.randn(128, 256)  # Simulated LLM weight matrix

        packed, scales, orig_shape = quantize_nfloat4(x, block_size=64)
        x_rec = dequantize_nfloat4(packed, scales, orig_shape, block_size=64)

        self.assertEqual(x_rec.shape, x.shape)
        # Cosine similarity on reconstructed LLM weights should be > 0.99
        cos_sim = torch.nn.functional.cosine_similarity(
            x.flatten(), x_rec.flatten(), dim=0
        ).item()
        self.assertGreater(cos_sim, 0.99)

    def test_nfloat4_autograd_ste_and_linear(self):
        """Test NFloat4 autograd STE and NFloat4Linear layer."""
        linear = NFloat4Linear(128, 64, block_size=64)
        self.assertIsNotNone(linear.packed_weight)
        self.assertIsNotNone(linear.weight_scales)

        linear.eval()
        x_in = torch.randn(16, 128, requires_grad=True)
        out = linear(x_in)
        self.assertEqual(out.shape, (16, 64))

        linear.train()
        out_tr = linear(x_in)
        loss = out_tr.pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x_in.grad)
        self.assertIsNotNone(linear.weight.grad)

    def test_nfloat_performance_benchmark(self):
        """Benchmark speed and efficiency gains of NFloat20 and NFloat4."""
        x = torch.randn(1024, 1024)

        # Benchmark NFloat20 quantization speed
        t0 = time.perf_counter()
        for _ in range(20):
            _ = quantize_nfloat20(x)
        t_nf20 = (time.perf_counter() - t0) / 20.0

        # Benchmark NFloat4 quantization speed
        t0 = time.perf_counter()
        for _ in range(20):
            packed, scales, orig_shape = quantize_nfloat4(x, block_size=64)
        t_nf4 = (time.perf_counter() - t0) / 20.0

        # Verify operations execute quickly (< 50ms per call on CPU)
        self.assertLess(t_nf20, 0.05)
        self.assertLess(t_nf4, 0.05)


if __name__ == "__main__":
    run_tests()
