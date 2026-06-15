"""
gnn.py
------
Standalone, vectorised GNN layer definitions.
All Python loops are eliminated in favour of native PyTorch segmented operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
# 0. Core Geometric Normalization
# ===========================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ===========================================================================
# 1. Vectorised GCN Layer (Loop-Free)
# ===========================================================================

class GCNLayer(nn.Module):
    """
    Relation-aware GCN layer powered by native PyTorch parallel index operations.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(embed_dim, embed_dim)
        self.upd_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.norm     = RMSNorm(embed_dim)
        self.dropout  = nn.Dropout(dropout)
        self.act      = nn.GELU()

    def forward(
        self,
        center_embed:   Tensor,   # [B, D]
        neighbor_embed: Tensor,   # [E, D]
        edge_embed:     Tensor,   # [E, D]
        ptr:            Tensor,   # [B+1]
    ) -> Tensor:                  # [B, D]
        B, D = center_embed.shape
        device = center_embed.device

        # --- multiplicative message composition ---
        messages = self.msg_proj(neighbor_embed * edge_embed)   # [E, D]
        messages = self.act(messages)

        # --- vectorized segment index generation ---
        degrees = ptr[1:] - ptr[:-1]  # [B]
        # Maps each edge to its corresponding center node ID: [E]
        edge_to_node_idx = torch.repeat_interleave(
            torch.arange(B, device=device), degrees
        )

        # --- parallel sum aggregation ---
        agg = torch.zeros(B, D, device=device)
        agg.index_add_(0, edge_to_node_idx, messages)

        # --- variance-preserving scaling ---
        deg_scaled = degrees.clamp(min=1).float().unsqueeze(-1)  # [B, 1]
        agg = agg / (deg_scaled ** 0.5)

        # --- update ---
        combined = torch.cat([center_embed, agg], dim=-1)       # [B, 2D]
        updated  = self.upd_proj(combined)                       # [B, D]
        updated  = self.dropout(updated)
        return self.norm(updated + center_embed)                 


# ===========================================================================
# 2. Vectorised GAT Layer (Loop-Free)
# ===========================================================================

class GATLayer(nn.Module):
    """
    Relation-aware GAT layer with fully-vectorised segmented softmax and scaling.
    """

    def __init__(
        self,
        embed_dim:  int,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.upd_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm      = RMSNorm(embed_dim)
        self.dropout   = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def _split_heads(self, x: Tensor) -> Tensor:
        """[N, D] -> [N, H, d]"""
        N, D = x.shape
        return x.view(N, self.num_heads, self.head_dim)

    def forward(
        self,
        center_embed:   Tensor,   # [B, D]
        neighbor_embed: Tensor,   # [E, D]
        edge_embed:     Tensor,   # [E, D]
        ptr:            Tensor,   # [B+1]
    ) -> Tensor:                  # [B, D]
        B, D = center_embed.shape
        device = center_embed.device

        # --- feature projection and transformation ---
        q = self._split_heads(self.q_proj(center_embed))  # [B, H, d]
        comp_features = neighbor_embed * edge_embed       # [E, D]
        k = self._split_heads(self.k_proj(comp_features)) # [E, H, d]
        vi_h = comp_features.view(-1, self.num_heads, self.head_dim) # [E, H, d]

        # --- dynamic segment index generation ---
        degrees = ptr[1:] - ptr[:-1]                      # [B]
        edge_to_node_idx = torch.repeat_interleave(
            torch.arange(B, device=device), degrees
        ) # [E]

        # --- fully-vectorised segmented attention ---
        # Expand queries to match edge dimensions
        q_expanded = q[edge_to_node_idx]                  # [E, H, d]
        scores = (q_expanded * k).sum(dim=-1) * self.scale # [E, H]

        # Stable segmented softmax using native scatter primitives
        idx_expanded = edge_to_node_idx.unsqueeze(-1).expand(-1, self.num_heads) # [E, H]
        
        max_scores = torch.full((B, self.num_heads), float('-inf'), device=device)
        max_scores.scatter_reduce_(0, idx_expanded, scores, reduce='amax', include_self=False)
        max_scores = torch.where(max_scores == float('-inf'), torch.zeros_like(max_scores), max_scores)

        scores_shifted = scores - max_scores[edge_to_node_idx]
        scores_exp = torch.exp(scores_shifted)            # [E, H]

        sum_exp = torch.zeros(B, self.num_heads, device=device)
        sum_exp.index_add_(0, edge_to_node_idx, scores_exp) # [B, H]
        
        alpha = scores_exp / (sum_exp[edge_to_node_idx] + 1e-9) # [E, H]
        alpha = self.attn_drop(alpha)

        # --- vectorised attention variance stabilization ---
        attn_square_sum = torch.zeros(B, self.num_heads, device=device)
        attn_square_sum.index_add_(0, edge_to_node_idx, alpha ** 2) # [B, H]
        scale_factor = 1.0 / torch.sqrt(attn_square_sum + 1e-9)     # [B, H]

        # --- parallel weighted aggregation ---
        weighted_vi = alpha.unsqueeze(-1) * vi_h          # [E, H, d]
        agg_h = torch.zeros(B, self.num_heads, self.head_dim, device=device)
        agg_h.index_add_(0, edge_to_node_idx, weighted_vi) # [B, H, d]

        # Rescale heads and flatten
        agg_h = agg_h * scale_factor.unsqueeze(-1)        # [B, H, d]
        agg = agg_h.reshape(B, D)                         # [B, D]

        # --- update ---
        combined = torch.cat([center_embed, agg], dim=-1)   # [B, 2D]
        updated  = self.upd_proj(combined)                   # [B, D]
        updated  = self.dropout(updated)
        return self.norm(updated + center_embed)


# ===========================================================================
# 3. Layer Factory
# ===========================================================================

_LAYER_REGISTRY: dict[str, type] = {
    "gcn": GCNLayer,
    "gat": GATLayer,
}

def build_gnn_layer(layer_type: str, embed_dim: int, **kwargs) -> nn.Module:
    """
    Instantiate a vectorised geometry-optimized GNN layer by name.
    """
    key = layer_type.lower()
    if key not in _LAYER_REGISTRY:
        raise ValueError(f"Unknown layer_type '{layer_type}'.")
    return _LAYER_REGISTRY[key](embed_dim, **kwargs)