# mypy: allow-untyped-defs
"""NFloat neural network modules."""

from torch.ao.quantization.nfloat import NFloat20Linear, NFloat4Linear


__all__ = [
    "NFloat20Linear",
    "NFloat4Linear",
]
