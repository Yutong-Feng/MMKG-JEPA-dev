import json
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Triple = Tuple[int, int, int]
# Mapping: entity_id -> (neighbor_entity_ids, edge_type_ids)
NeighborIndex = Dict[int, Tuple[Tensor, Tensor]]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = {"DB15K", "MKG-W", "MKG-Y"}


def _check_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {SUPPORTED_DATASETS}."
        )


def load_triples(path: str) -> List[Triple]:
    """Load triples from a tab/space-separated txt file."""
    triples: List[Triple] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return triples


def load_split(dataset_dir: str, split: str) -> List[Triple]:
    return load_triples(os.path.join(dataset_dir, f"{split}.txt"))


def load_splits(dataset_dir: str, splits: List[str]) -> List[Triple]:
    triples: List[Triple] = []
    for s in splits:
        fpath = os.path.join(dataset_dir, f"{s}.txt")
        if os.path.exists(fpath):
            triples.extend(load_triples(fpath))
    return triples


def load_modal_features(
    dataset: str, modality: str, top_n: int, pad_val: int = -1
) -> Dict[int, Tensor]:
    """Load token IDs from JSON, retaining the top-N most frequent tokens per entity.
    Entities with fewer than top_n tokens are right-padded with -1."""
    path = os.path.join("tokens", f"{dataset}-{modality}.json")
    with open(path, "r") as f:
        data = json.load(f)

    result = {}
    for k, v in data.items():
        if not v:
            # All-padding tensor for entities with no tokens
            result[int(k)] = torch.full((top_n,), pad_val, dtype=torch.long)
            continue
        tokens = torch.tensor(v, dtype=torch.long)
        unique_tokens, counts = tokens.unique(return_counts=True)
        top_n_actual = min(top_n, len(unique_tokens))
        top_indices = counts.topk(top_n_actual).indices
        selected = unique_tokens[top_indices]

        # Pad to fixed length top_n with pad_val
        padded = torch.full((top_n,), pad_val, dtype=torch.long)
        padded[:top_n_actual] = selected
        result[int(k)] = padded
    return result


def _get_padded_tokens(
    entities: Tensor, token_dict: Dict[int, Tensor], pad_val: int = -1
) -> Tensor:
    """Fetch and pad tokens for a given 1D tensor of entity IDs."""
    # Use empty tensor for missing entities to let pad_sequence handle it gracefully
    tokens = [
        token_dict.get(int(e), torch.empty(0, dtype=torch.long)) for e in entities
    ]
    # Pad variable-length sequences to the max length in this batch
    return pad_sequence(tokens, batch_first=True, padding_value=pad_val)


# ---------------------------------------------------------------------------
# Neighbor index
# ---------------------------------------------------------------------------


def build_neighbor_index(triples: List[Triple], num_relations: int) -> NeighborIndex:
    """
    Build an undirected per-entity neighbor index.
    Forward edges use relation `r`; inverse edges use `r + num_relations`.
    """
    adj: Dict[int, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for h, r, t in triples:
        adj[h][0].append(t)
        adj[h][1].append(r)

        adj[t][0].append(h)
        adj[t][1].append(r + num_relations)

    return {
        e: (torch.tensor(nb, dtype=torch.long), torch.tensor(et, dtype=torch.long))
        for e, (nb, et) in adj.items()
    }


# ---------------------------------------------------------------------------
# Shared edge-packing utilities
# ---------------------------------------------------------------------------


def _neighbor_edges(
    entity: int,
    neighbor_index: NeighborIndex,
    exclude_entity: Optional[int] = None,
    exclude_relation: Optional[int] = None,
    drop_rate: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Extract 1-hop neighborhood for a given entity.
    Optionally masks a specific target edge (exact match on entity and relation).
    Optionally applies random edge dropout for data augmentation.
    """
    if entity not in neighbor_index:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)

    neighbors, edge_types = neighbor_index[entity]

    if exclude_entity is not None and exclude_relation is not None:
        mask = ~((neighbors == exclude_entity) & (edge_types == exclude_relation))
        neighbors, edge_types = neighbors[mask], edge_types[mask]

    if drop_rate > 0.0 and neighbors.numel() > 0:
        keep_mask = torch.rand(neighbors.size(0)) >= drop_rate
        neighbors, edge_types = neighbors[keep_mask], edge_types[keep_mask]

    src = torch.full((neighbors.size(0),), entity, dtype=torch.long)
    return torch.stack([src, neighbors], dim=0), edge_types


def _pack_edges(
    edge_indices: List[Tensor], edge_attrs: List[Tensor]
) -> Dict[str, Tensor]:
    """
    Concatenate edge tensors for a batch and build a PyG-style pointer.

    Returns:
        edge_index: [2, total_edges]
        edge_attr:  [total_edges]
        ptr:        [B + 1] boundary indices for O(1) slicing
    """
    counts = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return {
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "ptr": ptr,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class KGTripleDataset(Dataset):
    """Dataset wrapper for a flat list of (head, relation, tail) triples."""

    def __init__(self, triples: List[Triple]) -> None:
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Triple:
        return self.triples[idx]


class KGEntityDataset(Dataset):
    """Dataset wrapper streaming unique entities from the neighbor index."""

    def __init__(self, neighbor_index: NeighborIndex) -> None:
        self.entities = torch.tensor(sorted(neighbor_index.keys()), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.entities)

    def __getitem__(self, idx: int) -> int:
        return int(self.entities[idx])


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def _collate_train(
    batch: List[Triple],
    neighbor_index: NeighborIndex,
    num_relations: int,
    text_dict: Dict[int, Tensor],
    vis_dict: Dict[int, Tensor],
    top_n: int,
    drop_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Collate function for training. Returns triples and their contextual subgraphs
    with the direct target edge masked out to prevent data leakage.
    """
    heads, rels, tails = zip(*batch)
    triples_t = torch.tensor(list(zip(heads, rels, tails)), dtype=torch.long)

    h_ei, h_ea, t_ei, t_ea = [], [], [], []
    for h, r, t in zip(heads, rels, tails):
        # Mask direct forward edge
        ei, ea = _neighbor_edges(
            h, neighbor_index, exclude_entity=t, exclude_relation=r, drop_rate=drop_rate
        )
        h_ei.append(ei)
        h_ea.append(ea)

        # Mask direct inverse edge (modulo ensures safety for augmented triples)
        r_inv = (r + num_relations) % (2 * num_relations)
        ei, ea = _neighbor_edges(
            t,
            neighbor_index,
            exclude_entity=h,
            exclude_relation=r_inv,
            drop_rate=drop_rate,
        )
        t_ei.append(ei)
        t_ea.append(ea)

    head_neighbor = _pack_edges(h_ei, h_ea)
    tail_neighbor = _pack_edges(t_ei, t_ea)

    # Extract all unique entities involved in this batch for efficient feature fetching
    batch_entities = torch.cat(
        [
            triples_t[:, 0],
            triples_t[:, 2],
            head_neighbor["edge_index"][1],
            tail_neighbor["edge_index"][1],
        ]
    ).unique()

    return {
        "triples": triples_t,
        "head_neighbor": head_neighbor,
        "tail_neighbor": tail_neighbor,
        "batch_entities": batch_entities,  # To map embeddings back to triples/edges downstream
        "text_tokens": torch.stack(
            [
                text_dict.get(int(e), torch.full((top_n,), -1, dtype=torch.long))
                for e in batch_entities
            ]
        ),
        "vis_tokens": torch.stack(
            [
                vis_dict.get(int(e), torch.full((top_n,), -1, dtype=torch.long))
                for e in batch_entities
            ]
        ),
    }


def _collate_entity(
    batch: List[int],
    neighbor_index: NeighborIndex,
    text_dict: Dict[int, Tensor],
    vis_dict: Dict[int, Tensor],
    top_n: int,
) -> Dict[str, Any]:
    """Collate function yielding entity IDs and their full 1-hop neighborhoods."""
    ei_list, ea_list = [], []
    for e in batch:
        ei, ea = _neighbor_edges(e, neighbor_index)
        ei_list.append(ei)
        ea_list.append(ea)

    neighbor = _pack_edges(ei_list, ea_list)
    batch_t = torch.tensor(batch, dtype=torch.long)

    batch_entities = torch.cat([batch_t, neighbor["edge_index"][1]]).unique()

    return {
        "entity": batch_t.unsqueeze(1),
        "neighbor": neighbor,
        "batch_entities": batch_entities,
        "text_tokens": torch.stack(
            [
                text_dict.get(int(e), torch.full((top_n,), -1, dtype=torch.long))
                for e in batch_entities
            ]
        ),
        "vis_tokens": torch.stack(
            [
                vis_dict.get(int(e), torch.full((top_n,), -1, dtype=torch.long))
                for e in batch_entities
            ]
        ),
    }


def _collate_eval(
    batch: List[Triple],
    text_dict: Dict[int, Tensor],
    vis_dict: Dict[int, Tensor],
    top_n: int,
) -> Dict[str, Tensor]:
    """Collate function for evaluation yielding only batched triples."""

    triples = torch.tensor(list(batch), dtype=torch.long)
    batch_entities = torch.cat([triples[:, 0], triples[:, 2]]).unique()

    return {
        "triples": triples,
        "batch_entities": batch_entities,
        "text_tokens": torch.stack(
            [
                text_dict.get(int(e), torch.full((top_n,), -1, dtype=torch.long))
                for e in batch_entities
            ]
        ),
        "vis_tokens": torch.stack(
            [
                vis_dict.get(int(e), torch.full((top_n,), -1, dtype=torch.long))
                for e in batch_entities
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Public loader classes
# ---------------------------------------------------------------------------


class TrainKGLoader:
    """
    Dataloader for model training.
    Yields batches containing triples and 1-hop subgraphs (with target edges masked).
    Automatically augments training data with inverse triples.
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        num_relations: int,
        top_n: int,
        batch_size: int,
        drop_rate: float = 0.0,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        neighbor_index: Optional[NeighborIndex] = None,
    ) -> None:
        _check_dataset(dataset)
        dataset_dir = os.path.join(data_root, dataset)

        # Load multi-modal token dictionaries into memory once
        self.text_dict = load_modal_features(dataset, "textual", top_n=top_n)
        self.vis_dict = load_modal_features(dataset, "visual", top_n=top_n)

        train_triples = load_split(dataset_dir, "train")
        inverse_triples = [(t, r + num_relations, h) for h, r, t in train_triples]
        all_train_triples = train_triples + inverse_triples

        self.neighbor_index = neighbor_index or build_neighbor_index(
            load_splits(dataset_dir, ["train"]), num_relations
        )

        self._loader = DataLoader(
            KGTripleDataset(all_train_triples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=partial(
                _collate_train,
                neighbor_index=self.neighbor_index,
                num_relations=num_relations,
                text_dict=self.text_dict,
                vis_dict=self.vis_dict,
                top_n=top_n,
                drop_rate=drop_rate,
            ),
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


class EntityLoader:
    """
    Dataloader streaming every graph entity exactly once per epoch along with
    its complete 1-hop neighborhood. Ideal for pre-computing entity representations.
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        num_relations: int,
        top_n: int,
        batch_size: int,
        include_valid: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        neighbor_index: Optional[NeighborIndex] = None,
    ) -> None:
        _check_dataset(dataset)
        dataset_dir = os.path.join(data_root, dataset)

        # Load multi-modal token dictionaries into memory once
        self.text_dict = load_modal_features(dataset, "textual", top_n=top_n)
        self.vis_dict = load_modal_features(dataset, "visual", top_n=top_n)

        splits = ["train", "valid"] if include_valid else ["train"]
        self.neighbor_index = neighbor_index or build_neighbor_index(
            load_splits(dataset_dir, splits), num_relations
        )

        self._loader = DataLoader(
            KGEntityDataset(self.neighbor_index),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=partial(
                _collate_entity,
                neighbor_index=self.neighbor_index,
                text_dict=self.text_dict,
                vis_dict=self.vis_dict,
                top_n=top_n,
            ),
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def num_entities(self) -> int:
        return len(self._loader.dataset)


class EvalLoader:
    """
    Dataloader for inference. Yields only evaluation triples without subgraph structures.
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        top_n: int,
        batch_size: int,
        split: str = "valid",
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ) -> None:
        _check_dataset(dataset)
        if split not in ("valid", "test"):
            raise ValueError(
                f"EvalLoader split must be 'valid' or 'test', got '{split}'."
            )
        # Load multi-modal token dictionaries into memory once
        self.text_dict = load_modal_features(dataset, "textual", top_n=top_n)
        self.vis_dict = load_modal_features(dataset, "visual", top_n=top_n)

        dataset_dir = os.path.join(data_root, dataset)
        self._loader = DataLoader(
            KGTripleDataset(load_split(dataset_dir, split)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=partial(
                _collate_eval,
                text_dict=self.text_dict,
                vis_dict=self.vis_dict,
                top_n=top_n,
            ),
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)
