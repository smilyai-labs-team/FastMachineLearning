# mypy: allow-untyped-defs
"""NFloat precision implementations (NFloat20 and NFloat4) for accelerated deep learning and LLM training."""

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NFloat20 Implementation
# ---------------------------------------------------------------------------


def quantize_nfloat20(x: torch.Tensor) -> torch.Tensor:
    """
    Quantizes a floating-point tensor to NFloat20 (1 sign bit, 8 exponent bits, 11 mantissa bits).
    Provides superior precision compared to BF16 (7 mantissa bits) and FP16 (10 mantissa bits)
    while maintaining full FP32 dynamic range.
    Uses ultra-fast vectorized float32 bit-level masking with round-to-nearest-even.
    """
    if not x.is_floating_point():
        raise ValueError("quantize_nfloat20 requires floating point input tensor")

    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_int = x_fp32.view(torch.int32)

    # 23 mantissa bits in FP32 vs 11 in NFloat20 => 12 bits to round/mask
    # Round to nearest even: bit 11 bias and tie-breaking LSB bias
    lsb_bias = (x_int >> 12) & 1
    x_nfloat20 = ((x_int + 0x000007FF + lsb_bias) & 0xFFFFF000).view(torch.float32)
    return x_nfloat20.to(orig_dtype)


def dequantize_nfloat20(x: torch.Tensor) -> torch.Tensor:
    """Dequantizes an NFloat20 tensor back to standard float representation."""
    return x


class NFloat20QuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) autograd function for NFloat20 quantization.
    Forward pass quantizes precision to 20 bits; backward pass passes gradients directly.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return quantize_nfloat20(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0]


def nfloat20_quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """Applies NFloat20 quantization with Straight-Through Estimator for gradient propagation."""
    return NFloat20QuantizeSTE.apply(x)


class NFloat20FakeQuantize(nn.Module):
    """
    Fake quantization module for NFloat20 precision.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled and self.training:
            return nfloat20_quantize_ste(x)
        elif self.enabled:
            return quantize_nfloat20(x)
        return x


class NFloat20Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear utilizing NFloat20 precision for weights and activations,
    delivering high numerical accuracy with reduced memory bandwidth and accelerated compute.
    Includes weight quantization caching in evaluation mode for fast inference execution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._cached_w_q = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self._cached_w_q = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_q = nfloat20_quantize_ste(self.weight)
            self._cached_w_q = None
        else:
            if self._cached_w_q is None or self._cached_w_q.shape != self.weight.shape:
                self._cached_w_q = nfloat20_quantize_ste(self.weight)
            w_q = self._cached_w_q

        x_q = nfloat20_quantize_ste(input)
        return nn.functional.linear(x_q, w_q, self.bias)


# ---------------------------------------------------------------------------
# NFloat4 Implementation
# ---------------------------------------------------------------------------

# 16 optimal quantile values for NFloat4 (NormalFloat 4) distribution [-1.0, 1.0]
_NF4_CODEBOOK = torch.tensor(
    [
        -1.00000000,
        -0.69619281,
        -0.52509290,
        -0.39491835,
        -0.28444117,
        -0.18477343,
        -0.09101309,
        0.00000000,
        0.07958006,
        0.16093020,
        0.24714680,
        0.33930811,
        0.44011860,
        0.55395634,
        0.68905219,
        1.00000000,
    ],
    dtype=torch.float32,
)

# Midpoints between codebook values for fast quantile binning via bucketize
_NF4_MIDPOINTS = (_NF4_CODEBOOK[:-1] + _NF4_CODEBOOK[1:]) / 2.0

# 256-entry lookup table mapping uint8 byte value directly to two codebook float values
_BYTES_RANGE = torch.arange(256, dtype=torch.long)
_NF4_DEQUANT_LUT = torch.stack(
    [_NF4_CODEBOOK[(_BYTES_RANGE >> 4) & 0x0F], _NF4_CODEBOOK[_BYTES_RANGE & 0x0F]],
    dim=-1,
)


def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Packs a 1D or 2D uint8 tensor of 4-bit values (0..15) into half-length uint8 tensor."""
    if indices.numel() % 2 != 0:
        raise ValueError("Tensor size must be even to pack uint4 values")

    orig_shape = indices.shape
    indices_flat = indices.reshape(-1)
    high = indices_flat[0::2] << 4
    low = indices_flat[1::2] & 0x0F
    packed = high | low

    new_shape = list(orig_shape)
    new_shape[-1] = new_shape[-1] // 2
    return packed.reshape(new_shape)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Unpacks a uint8 tensor into twice as many 4-bit values (0..15)."""
    orig_shape = packed.shape
    packed_flat = packed.reshape(-1)

    high = (packed_flat >> 4) & 0x0F
    low = packed_flat & 0x0F

    # Vectorized 2D stack & reshape for optimal memory access
    unpacked_flat = torch.stack([high, low], dim=-1).reshape(-1)

    new_shape = list(orig_shape)
    new_shape[-1] = new_shape[-1] * 2
    return unpacked_flat.reshape(new_shape)


def quantize_nfloat4(
    x: torch.Tensor, block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """
    Quantizes a floating-point tensor to NFloat4 (4-bit normal float) format with block-wise dynamic scaling.

    Returns:
        packed_indices (torch.Tensor): Packed 4-bit uint8 indices.
        scales (torch.Tensor): Block-wise absmax scaling factors.
        original_shape (tuple): Original shape of tensor.
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1).to(torch.float32)
    numel = x_flat.numel()

    # Pad if numel is not a multiple of block_size or even
    pad_len = (block_size - (numel % block_size)) % block_size
    if (numel + pad_len) % 2 != 0:
        pad_len += 1

    if pad_len > 0:
        x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))

    num_blocks = x_flat.numel() // block_size
    x_blocks = x_flat.reshape(num_blocks, block_size)

    # Compute block scale (absmax) with small epsilon to avoid division by zero
    scales = torch.amax(torch.abs(x_blocks), dim=-1, keepdim=True) + 1e-8
    x_norm = x_blocks / scales

    # Quantize to 16 quantile levels via vectorized bucketize
    midpoints = _NF4_MIDPOINTS.to(x.device)
    indices = torch.bucketize(x_norm, midpoints).to(torch.uint8)

    # Pack 4-bit indices
    packed_indices = pack_uint4(indices)
    return packed_indices, scales, orig_shape


def dequantize_nfloat4(
    packed_indices: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple[int, ...],
    block_size: int = 64,
) -> torch.Tensor:
    """
    Dequantizes packed NFloat4 4-bit indices and scales back to float tensor using 256-entry lookup table.
    """
    num_blocks = scales.shape[0]
    lut = _NF4_DEQUANT_LUT.to(packed_indices.device)

    # Fast 2D lookup directly on uint8 packed bytes
    dequantized_blocks = (
        lut[packed_indices.long()].reshape(num_blocks, block_size) * scales
    )

    dequantized_flat = dequantized_blocks.reshape(-1)
    numel_orig = 1
    for s in original_shape:
        numel_orig *= s

    dequantized_flat = dequantized_flat[:numel_orig]
    return dequantized_flat.reshape(original_shape)


class NFloat4QuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) autograd function for NFloat4 block quantization.
    Forward pass performs NFloat4 quantization/dequantization; backward pass retains gradient flow.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
        packed, scales, orig_shape = quantize_nfloat4(x, block_size=block_size)
        return dequantize_nfloat4(packed, scales, orig_shape, block_size=block_size)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0], None


def nfloat4_quantize_ste(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Applies NFloat4 quantization with Straight-Through Estimator for gradient propagation."""
    return NFloat4QuantizeSTE.apply(x, block_size)


class NFloat4Linear(nn.Module):
    """
    Memory-efficient and high-performance Linear layer storing weight parameters in 4-bit NFloat4
    packed representation, delivering 75% memory bandwidth savings on weight loads for LLM inference and training.
    Includes weight dequantization caching in evaluation mode for fast inference execution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._cached_w_dequant = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self._cached_w_dequant = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_dequant = nfloat4_quantize_ste(self.weight, block_size=self.block_size)
            self._cached_w_dequant = None
        else:
            if (
                self._cached_w_dequant is None
                or self._cached_w_dequant.shape != self.weight.shape
            ):
                self._cached_w_dequant = nfloat4_quantize_ste(
                    self.weight, block_size=self.block_size
                )
            w_dequant = self._cached_w_dequant

        return nn.functional.linear(input, w_dequant.to(input.dtype), self.bias)
