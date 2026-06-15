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

File layout
-----------
  gnn.py   – standalone GNN layer definitions (GCNLayer, GATLayer, factory)
  model.py – this file; imports from gnn.py only
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.gnn import build_gnn_layer, RMSNorm

# ===========================================================================
# 1. Entity Encoder
# ===========================================================================


class EntityEncoder(nn.Module):
    """
    Assigns a learnable embedding to every entity.

    The method ``fuse_modality`` is the designated extension point for MMKG:
    when text or image representations become available, override (or call
    super and add to) that method to blend structural and modal embeddings.
    Currently the method is a no-op pass-through.

    Parameters
    ----------
    num_entities : total number of entities in the KG
    embed_dim    : embedding dimension
    """

    def __init__(self, num_entities: int, embed_dim: int) -> None:
        super().__init__()
        self.entity_embed = nn.Embedding(num_entities, embed_dim)

    # ------------------------------------------------------------------
    # Multimodal extension point
    # ------------------------------------------------------------------

    def fuse_modality(
        self,
        entity_ids: Tensor,  # [N]
        entity_embed: Tensor,  # [N, D]
        modal_embeds: Optional[Tensor] = None,  # [N, D]  text / image
    ) -> Tensor:  # [N, D]
        """
        Fuse structural entity embeddings with optional modal representations.

        Current behaviour (no modality): identity — returns ``entity_embed``
        unchanged.

        Override this method (or replace this module) to implement multimodal
        fusion strategies, for example:
          - simple addition or gating
          - cross-attention between structural and modal tokens
          - MLP-based projection and fusion

        Parameters
        ----------
        entity_ids   : entity indices (may be used to look up per-entity
                       cached modal embeddings in a future implementation)
        entity_embed : structural embeddings from ``self.entity_embed``
        modal_embeds : pre-computed text or image representations;
                       ``None`` when running in unimodal mode
        """
        if modal_embeds is None:
            return entity_embed
        # ----------------------------------------------------------------
        # TODO (MMKG): implement fusion, e.g.
        #   gate = torch.sigmoid(self.fusion_gate(entity_embed))
        #   return gate * entity_embed + (1 - gate) * modal_embeds
        # ----------------------------------------------------------------
        raise NotImplementedError(
            "fuse_modality: modal_embeds supplied but fusion is not yet "
            "implemented.  Override this method to add multimodal support."
        )

    def forward(
        self,
        entity_ids: Tensor,  # [N]
        modal_embeds: Optional[Tensor] = None,  # [N, D] or None
    ) -> Tensor:  # [N, D]
        entity_embed = self.entity_embed(entity_ids)
        return self.fuse_modality(entity_ids, entity_embed, modal_embeds)


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
        entity_encoder: EntityEncoder,
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

        raw_identity = entity_encoder.entity_embed(center_ids)

        return ctx_embed + raw_identity * 0.5


# ===========================================================================
# 4. Predictor
# ===========================================================================

# class Predictor(nn.Module):
#     """
#     Predicts the tail-entity context embedding from:
#       - the head entity's context embedding  (GNN output)
#       - the relation embedding connecting head and tail

#     Architecture: a 2-layer MLP with residual connection.

#     Parameters
#     ----------
#     embed_dim  : input and output dimension
#     hidden_dim : hidden layer width (defaults to 2 * embed_dim)
#     dropout    : dropout probability
#     """

#     def __init__(
#         self,
#         embed_dim:  int,
#         hidden_dim: Optional[int] = None,
#         dropout:    float = 0.1,
#     ) -> None:
#         super().__init__()
#         hidden_dim = hidden_dim or (2 * embed_dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(2 * embed_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#         )

#     def forward(
#         self,
#         head_ctx_embed: Tensor,   # [B, D]
#         rel_embed:      Tensor,   # [B, D]
#     ) -> Tensor:                  # [B, D]
#         x = torch.cat([head_ctx_embed, rel_embed], dim=-1)   # [B, 2D]
#         pred_embed = self.mlp(x)                             # [B, D]
#         # residual: prediction stays close to the head context
#         return pred_embed + head_ctx_embed


class Predictor(nn.Module):
    """
    LoRA-style predictor.
    """

    def __init__(self, embed_dim: int, rank: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank

        self.rel_to_A = nn.Linear(embed_dim, embed_dim * rank)
        self.rel_to_B = nn.Linear(embed_dim, rank * embed_dim)
        self.rel_to_b = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.rel_to_A.weight)
        nn.init.zeros_(self.rel_to_A.bias)

        nn.init.zeros_(self.rel_to_B.weight)
        nn.init.zeros_(self.rel_to_B.bias)

        nn.init.xavier_uniform_(self.rel_to_b.weight)
        nn.init.zeros_(self.rel_to_b.bias)

    def forward(self, head_ctx_embed: Tensor, rel_embed: Tensor) -> Tensor:
        B, D = head_ctx_embed.shape
        r = self.rank

        A_flat = self.rel_to_A(rel_embed)  # [B, D*r]
        B_flat = self.rel_to_B(rel_embed)  # [B, r*D]
        b = self.rel_to_b(rel_embed)  # [B, D]

        A = A_flat.view(B, D, r)  # [B, D, r]
        B = B_flat.view(B, r, D)  # [B, r, D]

        # Bottleneck: project down to r-dim
        Bh = torch.bmm(B, head_ctx_embed.unsqueeze(-1))  # [B, r, 1]
        Bh = self.dropout(Bh)  # dropout on bottleneck

        # Project back up to D-dim
        ABh = torch.bmm(A, Bh).squeeze(-1)  # [B, D]

        out = ABh * self.scale + b
        return out


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
    rank           : rank for the LoRA-style predictor
    layer_types    : one layer-type string per GNN layer, e.g. ``["gcn","gat"]``
    layer_kwargs   : extra kwargs for GNN layers (dropout, num_heads, …)
    pred_dropout   : dropout in the predictor
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        rank: int,
        pred_dropout: float,
        layer_types: Optional[List[str]] = None,
        layer_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations


        # ── shared encoders ───────────────────────────────────────────
        self.entity_encoder = EntityEncoder(num_entities, embed_dim)
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
            rank=rank,
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
        head_modal_embeds: Optional[Tensor] = None,  # [B, D] MMKG, TBD
        tail_modal_embeds: Optional[Tensor] = None,  # [B, D] MMKG, TBD
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        batch : dict produced by ``TrainKGLoader``
            - ``triples``       : LongTensor [B, 3]  (head, relation, tail)
            - ``head_neighbor`` : packed neighbor graph for head entities
            - ``tail_neighbor`` : packed neighbor graph for tail entities
        head_modal_embeds : optional modal representation for heads (MMKG, TBD)
        tail_modal_embeds : optional modal representation for tails (MMKG, TBD)

        Returns
        -------
        pred_embed     : Tensor [B, D]  – predicted tail-entity context embedding
        tail_ctx_embed : Tensor [B, D]  – true    tail-entity context embedding
        """
        triples: Tensor = batch["triples"]  # [B, 3]
        head_neighbor: Dict[str, Tensor] = batch["head_neighbor"]
        tail_neighbor: Dict[str, Tensor] = batch["tail_neighbor"]

        head_ids = triples[:, 0]  # [B]
        rel_ids = triples[:, 1]  # [B]
        tail_ids = triples[:, 2]  # [B]

        # ── relation embedding ────────────────────────────────────────
        rel_embed: Tensor = self.edge_encoder(rel_ids)  # [B, D]

        # ── head context embedding ────────────────────────────────────
        head_ctx_embed: Tensor = self.neighborhood_encoder(
            center_ids=head_ids,
            neighbor_graph=head_neighbor,
            entity_encoder=self.entity_encoder,
            edge_encoder=self.edge_encoder,
        )  # [B, D]

        # ── predict tail context embedding ────────────────────────────
        pred_embed: Tensor = self.predictor(head_ctx_embed, rel_embed)  # [B, D]

        # ── true tail context embedding (same online encoder) ─────────
        tail_ctx_embed: Tensor = self.neighborhood_encoder(
            center_ids=tail_ids,
            neighbor_graph=tail_neighbor,
            entity_encoder=self.entity_encoder,
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
            entity_ids: Tensor = batch["entity"].squeeze(1).to(device)  # [B]
            neighbor_graph = {k: v.to(device) for k, v in batch["neighbor"].items()}
            ctx_embed = self.neighborhood_encoder(
                center_ids=entity_ids,
                neighbor_graph=neighbor_graph,
                entity_encoder=self.entity_encoder,
                edge_encoder=self.edge_encoder,
            )  # [B, D]
            lut[entity_ids] = ctx_embed

        self._lut = lut  # [num_entities, D]

    @torch.no_grad()
    def retrieve(
        self,
        batch: Dict[str, object],
        top_k: int,
        device: Optional[torch.device] = None,
        return_scores: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Retrieve top-K candidate entity IDs for each query triple.

        Parameters
        ----------
        batch        : dict from EvalLoader, must contain key 'triples' [B, 3].
        top_k        : number of top candidates to return.
        device       : device for intermediate tensors; defaults to LUT device.
        return_scores: if True, also return the full [B, num_entities] distance
                    matrix needed for filtered evaluation.

        Returns
        -------
        candidates : LongTensor [B, top_k]  (always returned)
        scores     : FloatTensor [B, num_entities]  (only when return_scores=True)
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
        pred_embed = self.predictor(head_ctx_embed, rel_embed)

        pred_norm = torch.nn.functional.normalize(pred_embed, p=2, dim=-1)
        lut_norm = torch.nn.functional.normalize(self._lut, p=2, dim=-1)

        # Pairwise dot product on normalized vectors equals cosine similarity
        cos_sim = torch.matmul(pred_norm, lut_norm.T)

        # Cosine distance: smaller means closer angular proximity
        dists = 1.0 - cos_sim

        candidates = dists.topk(top_k, dim=1, largest=False).indices

        if return_scores:
            return candidates, dists
        return candidates

        # # Full pairwise L2 distances [B, num_entities]
        # dists = torch.cdist(pred_embed, self._lut, p=2)

        # candidates = dists.topk(top_k, dim=1, largest=False).indices  # [B, top_k]

        # if return_scores:
        #     return candidates, dists
        # return candidates

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
