"""
model.py
--------
LeJEPA-style representation model for knowledge graphs.

Architecture overview
---------------------

  Training
  --------
                         ┌─────────────────────────────────────┐
  batch["triples"]       │           KGJEPAModel                │
  batch["head_neighbor"] │                                      │
  batch["tail_neighbor"] │  EntityEncoder  ──►  entity_embed    │
         │               │  EdgeEncoder    ──►  edge_embed      │
         │               │                                      │
         │               │  NeighborhoodEncoder (online, GNN)   │
         │               │    head  ──►  head_ctx_embed         │
         │               │    tail  ──►  tail_ctx_embed         │
         │               │                                      │
         │               │  Predictor                           │
         │               │    (head_ctx_embed + rel_embed)      │
         │               │    ──►  pred_embed                   │
         │               │                                      │
         └───────────────┴──────────────────────────────────────┘

  Note: LeJEPA does NOT use a momentum / EMA target encoder.
  Representation collapse is prevented by an external regularisation
  term (SIGReg) applied to the training loss outside this module.

  Inference
  ---------
  1. build_lut(entity_loader) — iterate EntityLoader, fill embedding LUT
  2. retrieve(triples)        — look up head embedding in LUT, run Predictor,
                                 rank all entities by L2 distance, return Top-K

  Forward output (training)
  -------------------------
  pred_embed      : Tensor [B, D]  – predicted tail-entity embedding
  tail_ctx_embed  : Tensor [B, D]  – true tail-entity context embedding

  Multimodal extension point
  --------------------------
  EntityEncoder.fuse_modality(entity_ids, entity_embed, modal_embeds) is the
  designated hook for MMKG.  Override or extend it when text / image
  representations become available.  The rest of the model requires no changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor

from .gnn import build_gnn_layer
from .norm import RMSNorm
from .transformer import TransformerBlock

# ===========================================================================
# 1. Entity Encoder
# ===========================================================================


class EntityEncoder(nn.Module):
    """Multimodal Entity Encoder with Text and Vision Fusion."""

    def __init__(
        self, num_entities: int, embed_dim: int, nhead: int, dropout: float
    ) -> None:
        super().__init__()
        self.entity_embed = nn.Embedding(num_entities, embed_dim)

        # Load pre-trained codebooks: text [30522, 768], vision [8192, 32]
        text_cb = torch.load("tokens/textual.pth")
        vis_cb = torch.load("tokens/visual.pth")

        self.text_embed = nn.Embedding.from_pretrained(text_cb, freeze=True)
        self.vis_embed = nn.Embedding.from_pretrained(vis_cb, freeze=True)

        # Project each codebook's native dim to shared embed_dim
        self.text_proj = nn.Linear(768, embed_dim)
        self.vis_proj = nn.Linear(32, embed_dim)

        # Unimodal self-attention encoders
        self.text_encoder = TransformerBlock(
            in_dim=embed_dim, nhead=nhead, dropout=dropout
        )
        self.vis_encoder = TransformerBlock(
            in_dim=embed_dim, nhead=nhead, dropout=dropout
        )

        # Cross-attention: entity embedding attends over multimodal context
        self.fusion_decoder = TransformerBlock(
            in_dim=embed_dim, nhead=nhead, dropout=dropout
        )

    def _encode_vis(self, vis_tokens: Tensor) -> tuple[Tensor, Tensor]:
        pad_mask = vis_tokens != -1  # [B, L_v]
        tokens = vis_tokens.masked_fill(~pad_mask, 0)
        emb = self.vis_proj(self.vis_embed(tokens))  # [B, L_v, D]

        valid_samples = pad_mask.any(dim=1)  # [B]

        if valid_samples.all():
            return self.vis_encoder(emb, is_causal=False, mask=pad_mask), pad_mask

        out = torch.zeros_like(emb)  # [B, L_v, D]
        if valid_samples.any():
            out[valid_samples] = self.vis_encoder(
                emb[valid_samples],
                is_causal=False,
                mask=pad_mask[valid_samples],
            )

        return out, pad_mask

    def _encode_text(self, text_tokens: Tensor) -> tuple[Tensor, Tensor]:
        pad_mask = text_tokens != -1
        tokens = text_tokens.masked_fill(~pad_mask, 0)
        emb = self.text_proj(self.text_embed(tokens))

        valid_samples = pad_mask.any(dim=1)

        if valid_samples.all():
            return self.text_encoder(emb, is_causal=False, mask=pad_mask), pad_mask

        out = torch.zeros_like(emb)
        if valid_samples.any():
            out[valid_samples] = self.text_encoder(
                emb[valid_samples],
                is_causal=False,
                mask=pad_mask[valid_samples],
            )

        return out, pad_mask

    def forward(
        self,
        entity_ids: Tensor,
        text_tokens: Optional[Tensor] = None,
        vis_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        # 1. Entity embedding — serves as cross-attention query
        entity_emb = self.entity_embed(entity_ids)  # [B, D]

        # 2. Return early when no modality tokens are provided
        if text_tokens is None and vis_tokens is None:
            return entity_emb

        # 3. Encode each available modality and collect results
        ctx_parts, mask_parts = [], []

        if text_tokens is not None:
            t_emb, t_mask = self._encode_text(text_tokens)
            ctx_parts.append(t_emb)
            mask_parts.append(t_mask)

        if vis_tokens is not None:
            v_emb, v_mask = self._encode_vis(vis_tokens)
            ctx_parts.append(v_emb)
            mask_parts.append(v_mask)

        # 4. Concatenate modalities into a single context sequence
        context = torch.cat(ctx_parts, dim=1)  # [B, L_t+L_v, D]
        ctx_mask = torch.cat(mask_parts, dim=1)  # [B, L_t+L_v]


        # 5. Cross-attention: entity query attends over multimodal context
        query = entity_emb.unsqueeze(1)  # [B, 1, D]
        fused = self.fusion_decoder(query, context, mask=ctx_mask)  # [B, 1, D]

        return fused.squeeze(1)  # [B, D]


# ===========================================================================
# 2. Edge Encoder
# ===========================================================================


class EdgeEncoder(nn.Module):
    """
    Assigns a learnable embedding to every relation / edge type.

    Parameters
    ----------
    num_relations : total number of relation types in the KG
    embed_dim     : embedding dimension
    """

    def __init__(self, num_relations: int, embed_dim: int) -> None:
        super().__init__()
        self.edge_embed = nn.Embedding(num_relations, embed_dim)

    def forward(self, edge_ids: Tensor) -> Tensor:  # [E] -> [E, D]
        return self.edge_embed(edge_ids)


# ===========================================================================
# 3. Neighborhood Encoder  (GNN stack)
# ===========================================================================


class NeighborhoodEncoder(nn.Module):
    """
    Computes context embeddings by aggregating 1-hop neighbourhood information
    through a stack of GNN layers.

    Accepts the packed-edge format produced by the dataloader:

      neighbor_graph = {
          "edge_index": Tensor [2, E],   # global entity IDs (src, dst)
          "edge_attr" : Tensor [E],      # relation IDs
          "ptr"       : Tensor [B+1],    # slice ptr[i]:ptr[i+1] for sample i
      }

    GNN layer contract
    ------------------
    Every layer returned by ``build_gnn_layer`` must accept the signature:
        layer(ctx_embed, neighbor_embed, edge_embed, ptr) -> Tensor [B, D]
    where ``ptr`` is the packed-edge pointer tensor described above.

    Parameters
    ----------
    embed_dim     : feature dimension (shared across entities and edges)
    layer_types   : sequence of layer type strings, one per layer.
                    Length must equal ``num_layers``.
                    Supported values: ``"gcn"``, ``"gat"``.
                    Example: ``["gcn", "gcn", "gat"]``
    layer_kwargs  : extra kwargs forwarded to every layer constructor
                    (e.g. ``{"num_heads": 4, "dropout": 0.1}``)
    """

    def __init__(
        self,
        embed_dim: int,
        layer_types: str,
        layer_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        layer_kwargs = layer_kwargs or {}
        self.layer = build_gnn_layer(layer_types, embed_dim, **layer_kwargs)

    def forward(
        self,
        center_ids: Tensor,  # [B]
        neighbor_graph: Dict[str, Tensor],  # packed neighbor graph
        entity_encoder: Callable[[Tensor], Tensor],
        edge_encoder: EdgeEncoder,
    ) -> Tensor:  # [B, D]
        """
        Parameters
        ----------
        center_ids     : global entity IDs of the B center nodes
        neighbor_graph : packed neighbor graph dict from the dataloader
        entity_encoder : used to look up neighbor entity embeddings
        edge_encoder   : used to look up edge embeddings
        """
        edge_index = neighbor_graph["edge_index"]  # [2, E]
        edge_attr = neighbor_graph["edge_attr"]  # [E]
        ptr = neighbor_graph["ptr"]  # [B+1]

        # Initial center embeddings
        ctx_embed = entity_encoder(center_ids)  # [B, D]

        if edge_index.size(1) == 0:
            # No neighbours in this batch — skip GNN layers
            return ctx_embed

        neighbor_ids = edge_index[1]  # [E]
        neighbor_embed = entity_encoder(neighbor_ids)  # [E, D]
        edge_embed = edge_encoder(edge_attr)  # [E, D]

        ctx_embed = self.layer(ctx_embed, neighbor_embed, edge_embed, ptr)

        return ctx_embed


# ===========================================================================
# 4. Predictor
# ===========================================================================


class Predictor(nn.Module):
    """
    Predicts the target entity context embedding conditioned on the source context,
    the relation, and a sampled latent variable 'z'.

    Architecture improvements:
      - Uses Addition-based injection (instead of concatenation) for the latent variable,
        allowing a learned, smooth semantic shift in the hidden space.
      - Injects 'z' before the activation function to enable complex non-linear interactions.
      - Standardized Dropout placement (Linear -> GELU -> Dropout).
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.positional_embedding = nn.Parameter(
            torch.randn(2, embed_dim)
        )  # Learnable positional embedding
        self.query_norm = RMSNorm(embed_dim)
        self.encoder = TransformerBlock(in_dim=embed_dim, nhead=nhead, dropout=dropout)
        self.decoder = TransformerBlock(in_dim=embed_dim, nhead=nhead, dropout=dropout)

    def forward(
        self,
        head_ctx_embed: Tensor,  # [B, D]
        rel_embed: Tensor,  # [B, D]
        num_samples: int = 8,  # K samples for diversity
    ) -> Tensor:  # [B, K, D] or [B, D]
        batch_size, device = head_ctx_embed.size(0), head_ctx_embed.device

        # Process context -> [B, 2, hidden_dim]
        ctx = torch.stack([head_ctx_embed, rel_embed], dim=1)
        # ctx = ctx + self.positional_embedding.unsqueeze(0)
        ctx = self.encoder(ctx)

        # Sample and process latent variable 'z' -> [B, K, hidden_dim]
        z = torch.randn(batch_size, num_samples, self.embed_dim, device=device)

        query = self.query_norm(rel_embed.unsqueeze(1) + z)  # [B, K, D]

        # Pass through decoder block -> [B, K, D]
        pred_embed = self.decoder(query, ctx)

        return pred_embed


# class Predictor(nn.Module):
#     """
#     JEPA Predictor leveraging a continuous latent variable 'z' to handle
#     one-to-many (1-to-N) relationships by mapping to a prediction manifold.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         rank: int,
#         dropout: float,
#         latent_dim: Optional[int] = None,
#     ) -> None:
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.rank = rank
#         self.latent_dim = (
#             embed_dim if latent_dim is None else latent_dim
#         )  # default latent dimension

#         # Relation-conditioned LoRA transformations
#         self.rel_to_A = nn.Linear(2 * embed_dim, embed_dim * rank)
#         self.rel_to_B = nn.Linear(2 * embed_dim, rank * embed_dim)
#         self.rel_to_b = nn.Linear(2 * embed_dim, embed_dim)
#         self.out_norm = RMSNorm(embed_dim)

#         self.scale = embed_dim**-0.5
#         self.dropout = nn.Dropout(dropout)
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.kaiming_uniform_(self.rel_to_A.weight)
#         nn.init.zeros_(self.rel_to_A.bias)
#         nn.init.zeros_(self.rel_to_B.weight)
#         nn.init.zeros_(self.rel_to_B.bias)
#         nn.init.xavier_uniform_(self.rel_to_b.weight)
#         nn.init.zeros_(self.rel_to_b.bias)
#         # nn.init.xavier_uniform_(self.z_to_bottleneck.weight)
#         # nn.init.zeros_(self.z_to_bottleneck.bias)

#     def forward(
#         self, head_ctx_embed: Tensor, rel_embed: Tensor, num_samples: int = 8
#     ) -> Tensor:
#         """
#         Forward pass that samples 'num_samples' latent noises to generate multiple predictions.

#         Parameters
#         ----------
#         head_ctx_embed : Tensor [B, D] - Head entity context embedding
#         rel_embed      : Tensor [B, D] - Relation embedding
#         num_samples    : int           - Number of Monte Carlo samples for latent z

#         Returns
#         -------
#         Tensor [B, K, D] - K distinct predicted tail embeddings (where K = num_samples)
#         """
#         B, D = head_ctx_embed.shape
#         r = self.rank
#         K = num_samples

#         # 1. Compute dynamic relation-specific LoRA weights
#         A_flat = self.rel_to_A(
#             torch.concat([head_ctx_embed, rel_embed], dim=-1)
#         )  # [B, D*r]
#         B_flat = self.rel_to_B(
#             torch.concat([head_ctx_embed, rel_embed], dim=-1)
#         )  # [B, r*D]
#         b = self.rel_to_b(torch.concat([head_ctx_embed, rel_embed], dim=-1))  # [B, D]

#         lora_A = A_flat.view(B, D, r)  # [B, D, r]
#         lora_B = B_flat.view(B, r, D)  # [B, r, D]

#         # 2. Project head context to bottleneck space
#         Bh = torch.bmm(lora_B, head_ctx_embed.unsqueeze(-1)).squeeze(-1)  # [B, r]
#         Bh = self.dropout(Bh)

#         # 3. Sample latent variable z ~ N(0, I) and project to bottleneck space
#         z = torch.randn(
#             size=(B, K, self.rank), device=head_ctx_embed.device
#         )  # [B, K, latent_dim]
#         # z_feat = self.z_to_bottleneck(z)  # [B, K, r]

#         # 4. Fuse deterministic context with stochastic latent features
#         Bh_latent = Bh.unsqueeze(1) + z  # [B, K, r]

#         # 5. Project back to embedding dimension and add relation bias
#         ABh = torch.bmm(lora_A, Bh_latent.transpose(1, 2))  # [B, D, K]
#         ABh = ABh.permute(0, 2, 1)  # [B, K, D]

#         out = ABh + b.unsqueeze(1)  # [B, K, D]
#         out = self.out_norm(out)
#         return out


# ===========================================================================
# 5. KG-JEPA  (top-level model)
# ===========================================================================


class KGJEPAModel(nn.Module):
    """
    LeJEPA-style representation model for knowledge graphs.

    The model follows the JEPA principle:
      *predict the representation of the target (tail entity) from the
       representation of the context (head entity + relation), entirely
       in embedding space — never in input space.*

    Unlike I-JEPA / JEPA variants that use a momentum target encoder,
    LeJEPA uses a single shared online encoder for both head and tail.
    Representation collapse is prevented by an external regularisation
    term (SIGReg) computed outside this module.

    Components
    ----------
    entity_encoder       : EntityEncoder          – lookup + optional modal fusion
    edge_encoder         : EdgeEncoder            – relation lookup
    neighborhood_encoder : NeighborhoodEncoder    – GNN context encoder (online)
    predictor            : Predictor              – head_ctx + rel -> pred_tail

    Training: forward()
    -------------------
    Returns (pred_embed, tail_ctx_embed) — both Tensor [B, D].
    The caller computes the JEPA loss (e.g. MSE / cosine) plus SIGReg.

    Inference: build_lut() + retrieve()
    ------------------------------------
    1. build_lut(entity_loader)  — pre-compute every entity's embedding once
       and store it in a look-up table (LUT) on the given device.
    2. retrieve(triples, top_k)  — for each (head, rel, tail_query) triple,
       look up head in the LUT, run the Predictor, then return the top_k
       nearest entity IDs by L2 distance for external metric computation.

    Parameters
    ----------
    num_entities   : vocabulary size for entities
    num_relations  : vocabulary size for relations
    embed_dim      : unified embedding dimension
    nhead          : number of attention heads for the predictor
    layer_types    : one layer-type string per GNN layer, e.g. ``["gcn","gat"]``
    layer_kwargs   : extra kwargs for GNN layers (dropout, num_heads, …)
    pred_dropout   : dropout in the predictor
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        nhead: int,
        pred_dropout: float,
        layer_types: Optional[List[str]] = None,
        layer_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations

        # ── shared encoders ───────────────────────────────────────────
        self.entity_encoder = EntityEncoder(
            num_entities=num_entities,
            embed_dim=embed_dim,
            nhead=nhead,
            dropout=pred_dropout,
        )
        self.edge_encoder = EdgeEncoder(num_relations * 2, embed_dim)

        # ── single online neighborhood encoder (updated by backprop) ──
        # LeJEPA does NOT use a momentum / EMA target encoder.
        self.neighborhood_encoder = NeighborhoodEncoder(
            embed_dim=embed_dim,
            layer_types=layer_types,
            layer_kwargs=layer_kwargs,
        )

        # ── predictor ─────────────────────────────────────────────────
        self.predictor = Predictor(
            embed_dim=embed_dim,
            nhead=nhead,
            dropout=pred_dropout,
        )

        self.embed_dim = embed_dim
        self._lut: Optional[Tensor] = None  # filled by build_lut()

    # ==================================================================
    # Training forward
    # ==================================================================

    def forward(
        self,
        batch: Dict[str, object],
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        batch : dict produced by ``TrainKGLoader``
            - ``triples``       : LongTensor [B, 3]  (head, relation, tail)
            - ``head_neighbor`` : packed neighbor graph for head entities
            - ``tail_neighbor`` : packed neighbor graph for tail entities

        Returns
        -------
        pred_embed     : Tensor [B, D]  – predicted tail-entity context embedding
        tail_ctx_embed : Tensor [B, D]  – true    tail-entity context embedding
        """
        triples: Tensor = batch["triples"]  # [B, 3]
        head_neighbor: Dict[str, Tensor] = batch["head_neighbor"]
        tail_neighbor: Dict[str, Tensor] = batch["tail_neighbor"]

        # --- Multimodal Fusion for the whole batch ---
        batch_entities = batch["batch_entities"]
        text_tokens = batch["text_tokens"]
        vis_tokens = batch["vis_tokens"]
        device = batch_entities.device

        # [Num_Batch_Entities, D]
        fused_batch_embeds = self.entity_encoder(
            batch_entities,
            text_tokens=text_tokens,
            vis_tokens=vis_tokens,
        )

        # Create a local fast-lookup function for NeighborhoodEncoder
        id_to_idx = torch.full(
            (self.num_entities,), -1, device=device, dtype=torch.long
        )
        id_to_idx[batch_entities] = torch.arange(len(batch_entities), device=device)

        def entity_lookup(ids: Tensor) -> Tensor:
            # assert (id_to_idx[ids] >= 0).all(), f"OOB entity ids: {ids[id_to_idx[ids] < 0]}"
            return fused_batch_embeds[id_to_idx[ids]]

        head_ids = triples[:, 0]  # [B]
        rel_ids = triples[:, 1]  # [B]
        tail_ids = triples[:, 2]  # [B]

        # ── relation embedding ────────────────────────────────────────
        rel_embed: Tensor = self.edge_encoder(rel_ids)  # [B, D]

        # ── head context embedding ────────────────────────────────────
        head_ctx_embed: Tensor = self.neighborhood_encoder(
            center_ids=head_ids,
            neighbor_graph=head_neighbor,
            entity_encoder=entity_lookup,
            edge_encoder=self.edge_encoder,
        )

        # ── predict tail context embedding ────────────────────────────
        pred_embed: Tensor = self.predictor(head_ctx_embed, rel_embed)  # [B, D]

        # ── true tail context embedding (same online encoder) ─────────
        tail_ctx_embed: Tensor = self.neighborhood_encoder(
            center_ids=tail_ids,
            neighbor_graph=tail_neighbor,
            entity_encoder=entity_lookup,
            edge_encoder=self.edge_encoder,
        )  # [B, D]

        return pred_embed, tail_ctx_embed

    # ==================================================================
    # Inference helpers
    # ==================================================================

    @torch.no_grad()
    def build_lut(self, entity_loader, device: Optional[torch.device] = None) -> None:
        """
        Pre-compute context embeddings for every entity and store them in an
        in-memory look-up table (LUT).

        Must be called before ``retrieve()``.  Call again whenever the model
        weights change (e.g. after each training epoch during evaluation).

        Parameters
        ----------
        entity_loader : EntityLoader
            Yields batches with keys ``entity`` [B, 1] and ``neighbor`` graph.
            Iterates over every entity exactly once per epoch.
        device : torch.device, optional
            Target device for the LUT.  Defaults to the device of the entity
            embedding weights.
        """
        if device is None:
            device = next(self.parameters()).device

        lut = torch.zeros(self.num_entities, self.embed_dim, device=device)

        self.eval()
        for batch in entity_loader:
            entity_ids: Tensor = batch["entity"].squeeze(1).to(device)
            neighbor_graph = {k: v.to(device) for k, v in batch["neighbor"].items()}

            # --- Load modal tokens and compute local batch fusion ---
            batch_entities = batch["batch_entities"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            vis_tokens = batch["vis_tokens"].to(device)

            fused_batch_embeds = self.entity_encoder(
                batch_entities,
                text_tokens=text_tokens,
                vis_tokens=vis_tokens,
            )

            id_to_idx = torch.full(
                (self.num_entities,), -1, device=device, dtype=torch.long
            )
            id_to_idx[batch_entities] = torch.arange(len(batch_entities), device=device)

            def entity_lookup(ids: Tensor) -> Tensor:
                # assert (id_to_idx[ids] >= 0).all(), f"OOB entity ids: {ids[id_to_idx[ids] < 0]}"
                return fused_batch_embeds[id_to_idx[ids]]

            # --------------------------------------------------------

            ctx_embed = self.neighborhood_encoder(
                center_ids=entity_ids,
                neighbor_graph=neighbor_graph,
                entity_encoder=entity_lookup,
                edge_encoder=self.edge_encoder,
            )
            lut[entity_ids] = ctx_embed

        self._lut = lut

    @torch.no_grad()
    def retrieve(
        self,
        batch: Dict[str, object],
        top_k: Optional[int] = None,  # Allow None for full-graph retrieval
        device: Optional[torch.device] = None,
        return_scores: bool = False,
        num_samples: int = 16,  # Control Monte Carlo samples during inference
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Retrieve candidate entity IDs by evaluating K latent samples
        and routing via the minimum distance (maximum similarity) strategy.
        If top_k is None, returns the sorted indices for all entities in the graph.
        """
        if self._lut is None:
            raise RuntimeError(
                "LUT is empty. Call build_lut(entity_loader) before retrieve()."
            )
        if device is None:
            device = self._lut.device

        triples: Tensor = batch["triples"].to(device)
        head_ids = triples[:, 0]
        rel_ids = triples[:, 1]

        head_ctx_embed = self._lut[head_ids]
        rel_embed = self.edge_encoder(rel_ids)

        # Predict K candidate embeddings using latent variable z: [B, K, D]
        pred_embed = self.predictor(head_ctx_embed, rel_embed, num_samples=num_samples)

        # # Normalize both predictions and LUT for cosine similarity
        # pred_norm = torch.nn.functional.normalize(pred_embed, p=2, dim=-1)  # [B, K, D]
        # lut_norm = torch.nn.functional.normalize(self._lut, p=2, dim=-1)    # [num_entities, D]

        # # Pairwise dot product: [B, K, D] x [D, num_entities] -> [B, K, num_entities]
        # cos_sim = torch.matmul(pred_norm, lut_norm.T)

        # # Route via max similarity (min distance) across all K latent samples: [B, num_entities]
        # mean_cos_sim = torch.mean(cos_sim, dim=1)

        # # Convert to distance score (smaller means closer)
        # dists = 1.0 - mean_cos_sim

        # # Handle full-graph evaluation (Standard KGC protocol) vs Top-K search
        # if top_k is None:
        #     candidates = torch.argsort(dists, dim=1)
        # else:
        #     candidates = dists.topk(top_k, dim=1, largest=False).indices

        # if return_scores:
        #     return candidates, dists
        # return candidates

        # dists_all = torch.cdist(pred_embed, self._lut.unsqueeze(0), p=2)

        # # Route via minimum distance across all K latent samples: [B, num_entities]
        # # This finds the closest prediction among the K samples for each entity
        # dists = torch.min(dists_all, dim=1).values

        # # Handle full-graph evaluation (Standard KGC protocol) vs Top-K search
        # if top_k is None:
        #     # Default behavior of argsort is ascending (smallest distance first)
        #     candidates = torch.argsort(dists, dim=1)
        # else:
        #     # Retrieve the indices of the smallest distances
        #     candidates = dists.topk(top_k, dim=1, largest=False).indices

        # if return_scores:
        #     return candidates, dists
        # return candidates

        # Pairwise dot product without normalization
        # pred_embed: [B, K, D], lut: [num_entities, D] -> Output: [B, K, num_entities]
        dot_product = torch.matmul(pred_embed, self._lut.T)

        # Route via maximum dot product (higher is more similar)
        max_dot = torch.max(dot_product, dim=1).values

        # Convert similarity to a descending metric (or use descending=True in argsort)
        dists = -max_dot

        if return_scores:
            return dists
        else:
            # Handle full-graph evaluation (Standard KGC protocol) vs Top-K search
            if top_k is None:
                candidates = torch.argsort(dists, dim=1)
            else:
                candidates = dists.topk(top_k, dim=1, largest=False).indices
            return candidates

        # pred_embed_norm = torch.nn.functional.normalize(pred_embed, p=2, dim=-1)
        # lut_norm = torch.nn.functional.normalize(self._lut, p=2, dim=-1)

        # # Compute pairwise cosine similarity matrix [B, num_entities]
        # cos_sim = torch.matmul(pred_embed_norm, lut_norm.T)

        # # Convert similarity to distance (smaller is closer, range [0, 2])
        # dists = 1.0 - cos_sim

        # # Retrieve the closest Top-K candidates
        # candidates = dists.topk(top_k, dim=1, largest=False).indices  # [B, top_k]

        # if return_scores:
        #     return candidates, dists
        # return candidates
