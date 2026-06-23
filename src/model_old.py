from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .gnn import build_gnn_layer
from .norm import RMSNorm
from .transformer import TransformerBlock

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

        # raw_identity = entity_encoder.entity_embed(center_ids)

        return ctx_embed


# ===========================================================================
# 4. Predictor
# ===========================================================================


class Predictor(nn.Module):
    """
    MLP predictor: maps a (head context, relation) pair to a predicted
    tail-entity context embedding, conditioned on a sampled latent 'z'.

    For one (head, relation) pair we draw K independent latent vectors
    'z ~ N(0, I)', so the predictor produces K diverse candidate embeddings.

    Design:
      - head context and relation are concatenated and encoded by a 'trunk'
        MLP into an intermediate hidden representation;
      - the latent 'z' is injected by ADDITION into that INTERMEDIATE hidden
        layer (before an activation), NOT at the input, so the model first
        builds a stable condition representation and the latent then perturbs
        it deeper in the network;
      - a 'head' MLP maps the perturbed hidden state back to embed_dim.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,  # defaults to embed_dim // 2
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or embed_dim // 2
        self.hidden_dim = hidden_dim

        # Trunk: encode the [head_ctx ; relation] condition into hidden space.
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Normalize the condition to zero-mean / unit-var BEFORE adding z,
        self.cond_norm = nn.LayerNorm(hidden_dim)

        # Head: applied AFTER the latent has been injected at the hidden layer.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        self.output_norm = RMSNorm(embed_dim)
        
        self.init_weights()
        
    def init_weights(self):
        # nn.init.zeros_(self.head[-1].weight)
        # nn.init.zeros_(self.head[-1].bias)
        pass

    def forward(
        self,
        head_ctx_embed: Tensor,  # [B, D]
        rel_embed: Tensor,  # [B, D]
        num_samples: int = 8,  # K latent samples for diversity
    ) -> Tensor:  # [B, K, D]
        batch_size, device = head_ctx_embed.size(0), head_ctx_embed.device

        # Encode the condition once, then broadcast over the K samples: [B, 1, H].
        cond = self.trunk(torch.cat([head_ctx_embed, rel_embed], dim=-1))
        cond = self.cond_norm(cond).unsqueeze(1)  # [B, 1, H], unit scale

        # K latent samples projected into hidden space: [B, K, H].
        z = torch.randn(batch_size, num_samples, self.hidden_dim, device=device)

        # Inject the latent at the intermediate hidden layer (before activation),
        # then map back to embed_dim -> [B, K, D].
        pred_embed = self.head(cond + z)
        return self.output_norm(pred_embed)


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

        return head_ctx_embed, pred_embed, tail_ctx_embed

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
        self._valid_mask = lut.abs().sum(dim=-1) > 0

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

        # Pairwise dot product without normalization
        # pred_embed: [B, K, D], lut: [num_entities, D] -> Output: [B, K, num_entities]
        dot_product = torch.matmul(pred_embed, self._lut.T)
        max_dot = torch.max(dot_product, dim=1).values
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
