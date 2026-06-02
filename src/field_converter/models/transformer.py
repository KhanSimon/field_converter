from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import nn

from field_converter.models.mlp import ActivationStr


PositionalEncodingStr = Literal["learned", "sinusoidal"]


class TemporalPositionalEncoding(nn.Module):
    """Add learned or sinusoidal positional encodings to (B,T,C) tensors."""

    def __init__(
        self,
        *,
        d_model: int,
        max_window_size: int,
        encoding: PositionalEncodingStr = "learned",
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if max_window_size <= 0:
            raise ValueError("max_window_size must be > 0")
        if encoding not in {"learned", "sinusoidal"}:
            raise ValueError("encoding must be one of: learned, sinusoidal")

        self.d_model = int(d_model)
        self.max_window_size = int(max_window_size)
        self.encoding = encoding

        if encoding == "learned":
            self.embedding = nn.Embedding(self.max_window_size, self.d_model)
            self.register_buffer("sinusoidal", torch.empty(0), persistent=False)
        else:
            self.embedding = None
            self.register_buffer(
                "sinusoidal",
                self._build_sinusoidal(self.max_window_size, self.d_model),
                persistent=False,
            )

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,C), got {x.shape}")
        T = int(x.shape[1])
        if T > self.max_window_size:
            raise ValueError(
                f"Input sequence length T={T} exceeds max_window_size={self.max_window_size}"
            )

        if self.encoding == "learned":
            assert self.embedding is not None
            pos = torch.arange(T, device=x.device)
            pe = self.embedding(pos).unsqueeze(0)
        else:
            pe = self.sinusoidal[:T].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        return x + pe


class TransformerEncoderBackbone(nn.Module):
    """Batch-first Transformer encoder over per-frame embeddings."""

    supports_valid_mask = True

    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: ActivationStr,
        norm_first: bool,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be > 0")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout must be in [0,1)")
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported activation: {activation}")

        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    @staticmethod
    def _padding_mask_from_valid(valid_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if valid_mask is None:
            return None
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must have shape (B,T), got {valid_mask.shape}")

        padding_mask = ~valid_mask.bool()
        if padding_mask.numel() == 0:
            return padding_mask

        # MultiheadAttention returns NaNs when every token in a row is masked.
        all_padded = padding_mask.all(dim=1)
        if bool(all_padded.any()):
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False
        return padding_mask

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,C), got {x.shape}")
        padding_mask = self._padding_mask_from_valid(valid_mask)
        return self.encoder(x, src_key_padding_mask=padding_mask)
