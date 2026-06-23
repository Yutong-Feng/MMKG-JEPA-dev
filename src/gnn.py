"""
gnn.py
------
Standalone, vectorised GNN layer definitions.
All Python loops are eliminated in favour of native PyTorch segmented operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .norm import RMSNorm

# ===========================================================================
# 1. Vectorised GCN Layer (Loop-Free)
# ===========================================================================


class GCNLayer(nn.Module):
    """
    Relation-aware GCN layer powered by native PyTorch parallel index operations.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.upd_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.act = nn.GELU()
        self.norm = RMSNorm(embed_dim)

    def forward(
        self,
        center_embed: Tensor,  # [B, D]
        neighbor_embed: Tensor,  # [E, D]
        edge_embed: Tensor,  # [E, D]
        ptr: Tensor,  # [B+1]
    ) -> Tensor:  # [B, D]
        B, D = center_embed.shape
        device = center_embed.device

        # --- multiplicative message composition ---
        messages = self.msg_proj(
            torch.cat([neighbor_embed, edge_embed], dim=-1)
        )  # [E, D]
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
        agg = agg * torch.rsqrt(deg_scaled)  # [B, D]

        # --- update ---
        combined = torch.cat([center_embed, agg], dim=-1)  # [B, 2D]
        updated = self.upd_proj(combined)  # [B, D]
        
        return self.norm(updated + center_embed)


# ===========================================================================
# 2. Vectorised GAT Layer (Loop-Free)
# ===========================================================================


class GATLayer(nn.Module):
    """
    Relation-aware Multi-Head GAT layer using Q, K, V attention mechanism.
    Powered by native PyTorch parallel index operations.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.context_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        
        self.attn_vec = nn.Parameter(torch.Tensor(1, self.num_heads, 2 * self.head_dim))

        self.upd_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.act = nn.GELU()
        self.norm = RMSNorm(embed_dim)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.upd_proj.weight)
        nn.init.xavier_uniform_(self.attn_vec)

    def forward(
        self,
        center_embed: Tensor,  # [B, D]
        neighbor_embed: Tensor,  # [E, D]
        edge_embed: Tensor,  # [E, D]
        ptr: Tensor,  # [B+1]
    ) -> Tensor:  # [B, D]
        B = center_embed.shape[0]
        H = self.num_heads
        device = center_embed.device
        
        contextual_feat = self.context_proj(
            torch.cat([neighbor_embed, edge_embed], dim=-1)
        ) # [E, D]
        contextual_feat = self.act(contextual_feat)

        # --- 1. Q, K, V generation & Multi-Head Reshape ---
        # Explicitly split embedding dimension into (heads * head_dim)
        q = rearrange(self.q_proj(center_embed), 'b (h d) -> b h d', h=H)  # [B, H, Head_Dim]
        k = rearrange(self.k_proj(contextual_feat), 'e (h d) -> e h d', h=H)    # [E, H, Head_Dim]
        v = rearrange(self.v_proj(contextual_feat), 'e (h d) -> e h d', h=H)  # [E, H, Head_Dim]

        # --- 2. Vectorized segment index generation ---
        degrees = ptr[1:] - ptr[:-1]  # [B]
        # Maps each edge to its corresponding center node ID: [E]
        edge_to_node_idx = torch.repeat_interleave(
            torch.arange(B, device=device), degrees
        )

        # --- 3. Compute attention scores per head ---
        q_gathered = q[edge_to_node_idx]  # [E, H, Head_Dim]
        
        # attention
        cat_qk = torch.cat([q_gathered, k], dim=-1)
        scores = self.act((cat_qk * self.attn_vec).sum(dim=-1))
        # scores = (q_gathered * k).sum(dim=-1) * self.scale  # [E, H]

        # --- 4. Segmented Softmax (Multi-Head) ---
        # Find max score per node per head for numerical stability
        node_max = torch.full((B, H), -float('inf'), device=device)
        idx_expanded = edge_to_node_idx.unsqueeze(-1).expand(-1, H)
        node_max.scatter_reduce_(0, idx_expanded, scores, reduce="amax", include_self=False)
        
        # Subtract max score before exponential
        scores = scores - node_max[edge_to_node_idx]  # [E, H]
        exp_scores = torch.exp(scores)  # [E, H]
        
        # Sum exponentials per node per head
        node_sum = torch.zeros(B, H, device=device)
        node_sum.index_add_(0, edge_to_node_idx, exp_scores)
        
        # Normalize to get final attention weights
        attn_weights = exp_scores / (node_sum[edge_to_node_idx] + 1e-9)  # [E, H]

        # --- 5. Parallel sum aggregation ---
        # Weight values by attention scores
        weighted_v = v * attn_weights.unsqueeze(-1)  # [E, H, Head_Dim]
        
        agg = torch.zeros(B, H, self.head_dim, device=device)
        agg.index_add_(0, edge_to_node_idx, weighted_v)
        
        # --- 6. Recombine heads & Update ---
        # Concatenate all heads back to original embed_dim
        agg = rearrange(agg, 'b h d -> b (h d)')  # [B, D]

        combined = torch.cat([center_embed, agg], dim=-1)  # [B, 2D]
        updated = self.upd_proj(combined)  # [B, D]
        
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
