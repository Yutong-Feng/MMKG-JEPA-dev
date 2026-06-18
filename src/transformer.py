import math
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from .norm import RMSNorm


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to queries and keys.
    Dynamically computes frequencies based on individual sequence lengths
    to support cross-attention and inference caching where len(q) != len(k).
    """
    B, H, L_q, D = q.shape
    _, _, L_k, _ = k.shape
    device = q.device

    def _get_complex_freqs(seq_len: int) -> torch.Tensor:
        # Generate position indices for the given sequence length
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device) * -(math.log(10000.0) / D)
        )
        freqs = position * div_term
        return torch.polar(torch.ones_like(freqs), freqs)

    q_complex = torch.view_as_complex(q.float().reshape(B, H, L_q, -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(B, H, L_k, -1, 2))

    # Apply rotations using matching sequence lengths
    q_out = (
        torch.view_as_real(q_complex * _get_complex_freqs(L_q)).flatten(3).type_as(q)
    )
    k_out = (
        torch.view_as_real(k_complex * _get_complex_freqs(L_k)).flatten(3).type_as(k)
    )

    return q_out, k_out


class GateAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nhead: int,
    ):
        super().__init__()
        assert (
            in_dim % nhead == 0
        ), f"dim {in_dim} should be divided by num_heads {nhead}."

        self.nhead = nhead
        self.head_dim = in_dim // nhead

        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.kv = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.gate = nn.Linear(in_dim, in_dim, bias=False)
        self.mixer = nn.Linear(in_dim, in_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights for stable training start
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.kv.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.mixer.weight)

    def forward(
        self,
        target: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:

        query = self.q(target)
        # Use memory for key/value projection if provided (Cross-Attention)
        key_value = self.kv(target) if memory is None else self.kv(memory)

        query = rearrange(query, "b l (h f) -> b h l f", h=self.nhead)
        key_value = rearrange(key_value, "b l (h f) -> b h l f", h=self.nhead)
        key, value = key_value.chunk(2, dim=-1)

        if is_causal:
            query, key = apply_rotary_pos_emb(query, key)

        use_causal_flag = is_causal if mask is None else False

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=use_causal_flag
        )

        result = rearrange(result, "b h l f -> b l (h f)")

        # Apply gating mechanism. Replaced deprecated F.sigmoid with torch.sigmoid
        result = result * torch.sigmoid(self.gate(target))
        result = self.mixer(result)

        return result


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nhead: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = GateAttention(
            in_dim=in_dim,
            nhead=nhead,
        )
        self.norm1 = RMSNorm(in_dim)

        # Standard 2-layer MLP projection
        self.ffn_dim = ffn_dim or in_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, in_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = RMSNorm(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:

        attn_out = self.attention(self.norm1(x), memory)
        x = attn_out + x

        x = self.ffn(self.norm2(x)) + x
        return x
