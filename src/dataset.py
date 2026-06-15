import os
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Triple = Tuple[int, int, int]
NeighborIndex = Dict[int, Tuple[Tensor, Tensor]]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = {"DB15K", "MKG-W", "MKG-Y"}


def _check_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {SUPPORTED_DATASETS}.")


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


# ---------------------------------------------------------------------------
# Neighbor index
# ---------------------------------------------------------------------------

def build_neighbor_index(triples: List[Triple], num_relations: int) -> NeighborIndex:
    """
    Build a per-entity neighbor index from triples (undirected: both directions
    are indexed).

    Returns a dict mapping entity_id → (neighbor_ids: LongTensor,
                                         edge_types:   LongTensor).
    """
    adj: Dict[int, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for h, r, t in triples:
        adj[h][0].append(t);  adj[h][1].append(r)
        adj[t][0].append(h);  adj[t][1].append(r + num_relations)

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
) -> Tuple[Tensor, Tensor]:
    """
    Return 1-hop neighbourhood of `entity`, masking only the specific target edge.
    """
    if entity not in neighbor_index:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long)

    neighbors, edge_types = neighbor_index[entity]
    
    # Target Edge Masking: Mask only if BOTH entity and relation match
    if exclude_entity is not None and exclude_relation is not None:
        mask = ~((neighbors == exclude_entity) & (edge_types == exclude_relation))
        neighbors, edge_types = neighbors[mask], edge_types[mask]

    src = torch.full((neighbors.size(0),), entity, dtype=torch.long)
    return torch.stack([src, neighbors], dim=0), edge_types


def _pack_edges(
    edge_indices: List[Tensor],
    edge_attrs: List[Tensor],
) -> Dict[str, Tensor]:
    """
    Concatenate per-sample edge tensors and build a ptr for O(1) slicing.

    Returns:
        edge_index – [2, total_edges]  global entity IDs
        edge_attr  – [total_edges]     relation IDs
        ptr        – [B+1]             ptr[i]:ptr[i+1] selects sample i's edges
    """
    counts = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return {
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr":  torch.cat(edge_attrs,   dim=0),
        "ptr":        ptr,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class KGTripleDataset(Dataset):
    """Wraps a flat list of (head, relation, tail) triples."""

    def __init__(self, triples: List[Triple]) -> None:
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Triple:
        return self.triples[idx]


class KGEntityDataset(Dataset):
    """
    Iterates over every unique entity id present in the neighbor index.
    Used by EntityLoader to stream the full graph entity-by-entity.
    """

    def __init__(self, neighbor_index: NeighborIndex) -> None:
        self.entities = torch.tensor(
            sorted(neighbor_index.keys()), dtype=torch.long
        )  # [N]

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
    num_relations: int,  # Added to compute inverse relation
) -> Dict[str, object]:
    heads, rels, tails = zip(*batch)
    triples_t = torch.tensor(list(zip(heads, rels, tails)), dtype=torch.long)

    h_ei, h_ea, t_ei, t_ea = [], [], [], []
    for h, r, t in zip(heads, rels, tails):
        # Head sub-graph: exclude the forward edge (h -[r]-> t)
        ei, ea = _neighbor_edges(h, neighbor_index, exclude_entity=t, exclude_relation=r)
        h_ei.append(ei); h_ea.append(ea)
        
        # Tail sub-graph: exclude the inverse edge (t -[r_inv]-> h)
        r_inv = r + num_relations
        ei, ea = _neighbor_edges(t, neighbor_index, exclude_entity=h, exclude_relation=r_inv)
        t_ei.append(ei); t_ea.append(ea)

    return {
        "triples":       triples_t,
        "head_neighbor": _pack_edges(h_ei, h_ea),
        "tail_neighbor": _pack_edges(t_ei, t_ea),
    }


def _collate_entity(
    batch: List[int],
    neighbor_index: NeighborIndex,
) -> Dict[str, object]:
    """
    Collate for entity iteration: returns entity ids + their full 1-hop
    neighborhoods (no exclusion).

    Batch dict keys
    ---------------
    entity    LongTensor [B, 1]   entity ids
    neighbor  dict(edge_index [2, E], edge_attr [E], ptr [B+1])
    """
    ei_list, ea_list = [], []
    for e in batch:
        ei, ea = _neighbor_edges(e, neighbor_index)
        ei_list.append(ei); ea_list.append(ea)

    return {
        "entity":   torch.tensor(batch, dtype=torch.long).unsqueeze(1),  # [B, 1]
        "neighbor": _pack_edges(ei_list, ea_list),
    }


def _collate_eval(batch: List[Triple]) -> Dict[str, Tensor]:
    """Collate for evaluation: returns only triples, no graph structure."""
    return {"triples": torch.tensor(list(batch), dtype=torch.long)}


# ---------------------------------------------------------------------------
# Public loader classes
# ---------------------------------------------------------------------------

class TrainKGLoader:
    """
    Dataloader for model training.  Loads ``train.txt`` and returns, for every
    batch, the triples together with the 1-hop neighborhoods of the head and
    tail entities (direct edge excluded).

    Parameters
    ----------
    data_root      : root directory containing per-dataset sub-folders
    dataset        : one of {"DB15K", "MKG-W", "MKG-Y"}
    batch_size     : triples per batch
    shuffle        : whether to shuffle each epoch
    num_workers    : DataLoader worker processes
    neighbor_index : pre-built index (built from all splits if None)

    Batch keys
    ----------
    triples        LongTensor [B, 3]
    head_neighbor  dict(edge_index, edge_attr, ptr)
    tail_neighbor  dict(edge_index, edge_attr, ptr)
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        num_relations: int,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        neighbor_index: Optional[NeighborIndex] = None,
    ) -> None:
        _check_dataset(dataset)
        dataset_dir = os.path.join(data_root, dataset)

        # self.neighbor_index = neighbor_index or build_neighbor_index(
        #     load_splits(dataset_dir, ["train"]), num_relations
        # )

        # self._loader = DataLoader(
        #     KGTripleDataset(load_split(dataset_dir, "train")),
        #     batch_size=batch_size,
        #     shuffle=shuffle,
        #     num_workers=num_workers,
        #     collate_fn=partial(
        #         _collate_train, 
        #         neighbor_index=self.neighbor_index,
        #         num_relations=num_relations
        #     ),
        # )
        # 1. Load original forward triples
        train_triples = load_split(dataset_dir, "train")

        # 2. Augment training data with inverse triples to learn bidirectional embeddings
        # This ensures the model learns valid representations for 'r + num_relations'
        inverse_triples = [
            (t, r + num_relations, h) for h, r, t in train_triples
        ]
        all_train_triples = train_triples + inverse_triples

        self.neighbor_index = neighbor_index or build_neighbor_index(
            load_splits(dataset_dir, ["train"]), num_relations
        )

        self._loader = DataLoader(
            # 3. Pass the augmented dataset to the dataloader
            KGTripleDataset(all_train_triples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(
                _collate_train, 
                neighbor_index=self.neighbor_index,
                num_relations=num_relations
            ),
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


class EntityLoader:
    """
    Dataloader that streams every entity in the graph exactly once per epoch,
    yielding each entity together with its complete 1-hop neighborhood.

    Useful for pre-computing entity representations without leakage concerns.

    Parameters
    ----------
    data_root      : root directory containing per-dataset sub-folders
    dataset        : one of {"DB15K", "MKG-W", "MKG-Y"}
    include_valid  : if True, build the neighbor index from train + valid;
                     otherwise train only
    batch_size     : entities per batch
    shuffle        : whether to shuffle entity order
    num_workers    : DataLoader worker processes
    neighbor_index : pre-built index (built from selected splits if None)

    Batch keys
    ----------
    entity    LongTensor [B, 1]   entity ids
    neighbor  dict(edge_index [2, E], edge_attr [E], ptr [B+1])

    Notes
    -----
    len(loader) == ceil(num_entities / batch_size)
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        num_relations: int,
        batch_size: int,
        include_valid: bool = False,
        shuffle: bool = False,
        num_workers: int = 0,
        neighbor_index: Optional[NeighborIndex] = None,
    ) -> None:
        _check_dataset(dataset)
        dataset_dir = os.path.join(data_root, dataset)

        splits = ["train", "valid"] if include_valid else ["train"]
        self.neighbor_index = neighbor_index or build_neighbor_index(
            load_splits(dataset_dir, splits),
            num_relations
        )

        self._loader = DataLoader(
            KGEntityDataset(self.neighbor_index),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(_collate_entity, neighbor_index=self.neighbor_index),
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
    Dataloader for inference / evaluation.  Loads ``valid.txt`` or ``test.txt``
    and returns only triples — no graph structure is included.

    Parameters
    ----------
    data_root  : root directory containing per-dataset sub-folders
    dataset    : one of {"DB15K", "MKG-W", "MKG-Y"}
    split      : "valid" or "test"
    batch_size : triples per batch
    num_workers: DataLoader worker processes

    Batch keys
    ----------
    triples  LongTensor [B, 3]
    """

    def __init__(
        self,
        data_root: str,
        dataset: str,
        batch_size: int,
        split: str = "valid",
        num_workers: int = 0,
    ) -> None:
        _check_dataset(dataset)
        if split not in ("valid", "test"):
            raise ValueError(f"EvalLoader split must be 'valid' or 'test', got '{split}'.")

        dataset_dir = os.path.join(data_root, dataset)
        self._loader = DataLoader(
            KGTripleDataset(load_split(dataset_dir, split)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_eval,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


# ---------------------------------------------------------------------------
# Smoke-test helpers
# ---------------------------------------------------------------------------

def _graph_stats(neighbor_index: NeighborIndex, num_relations: int) -> None:
    """Print global graph statistics derived from a neighbor index."""
    all_edge_types: List[int] = []
    total_edges = 0
    for _, (_, et) in neighbor_index.items():
        all_edge_types.extend(et.tolist())
        total_edges += et.size(0)

    et_tensor = torch.tensor(all_edge_types, dtype=torch.long)
    num_relation_types = int(et_tensor.unique().size(0))
    num_inverse = int((et_tensor >= num_relations).sum().item())

    degrees = torch.tensor(
        [nb.size(0) for nb, _ in neighbor_index.values()], dtype=torch.float
    )
    print(f"  num_entities        : {len(neighbor_index)}")
    print(f"  num_relation_types  : {num_relation_types} (incl. inverse, base={num_relations})")
    print(f"  num_edge_entries    : {total_edges}")
    print(f"  num_inverse_edges   : {num_inverse}")
    print(f"  avg degree          : {degrees.mean().item():.2f}")
    print(f"  max degree          : {int(degrees.max().item())}")
    print(f"  min degree          : {int(degrees.min().item())}")


def _batch_edge_stats(label: str, ng: Dict[str, Tensor], B: int) -> None:
    """Print per-sample edge counts and relation type diversity for one neighborhood dict."""
    edge_attr = ng["edge_attr"]
    ptr       = ng["ptr"]
    num_types = int(edge_attr.unique().size(0)) if edge_attr.numel() > 0 else 0
    counts    = (ptr[1:] - ptr[:-1]).tolist()
    print(f"  {label}:")
    print(f"    total edges      : {edge_attr.size(0)}")
    print(f"    relation types   : {num_types}")
    print(f"    edges per sample : {[int(c) for c in counts]}")


def _infer_num_relations(dataset_dir: str) -> int:
    """Infer num_relations from the union of relation ids across all splits."""
    rel_ids = set()
    for split in ("train", "valid", "test"):
        fpath = os.path.join(dataset_dir, f"{split}.txt")
        if os.path.exists(fpath):
            for h, r, t in load_triples(fpath):
                rel_ids.add(r)
    return max(rel_ids) + 1

def _infer_dataset_stats(dataset_dir: str) -> Tuple[int, int]:
    """
    Infer (num_entities, num_relations) from the union of entity / relation
    ids across train + valid + test splits.

    num_entities  = |{ all head/tail ids appearing in any split }|
    num_relations = max(relation id) + 1 across all splits
                    (base relations only, before adding inverse edges)
    """
    entity_ids = set()
    rel_ids = set()
    for split in ("train", "valid", "test"):
        fpath = os.path.join(dataset_dir, f"{split}.txt")
        if os.path.exists(fpath):
            for h, r, t in load_triples(fpath):
                entity_ids.add(h)
                entity_ids.add(t)
                rel_ids.add(r)
    num_entities = len(entity_ids)
    num_relations = max(rel_ids) + 1
    return num_entities, num_relations


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ROOT = "./data"
    DS   = "MKG-W"
    B    = 4

    dataset_dir   = os.path.join(ROOT, DS)
    num_entities_total, num_relations = _infer_dataset_stats(dataset_dir)
    print(f"=== Dataset-level stats (train+valid+test union) ===")
    print(f"  num_entities (union)  : {num_entities_total}")
    print(f"  num_relations (base)  : {num_relations}")
    print()

    # ------------------------------------------------------------------ train
    print("=== TrainKGLoader ===")
    train_loader = TrainKGLoader(
        ROOT, DS, num_relations=num_relations, batch_size=B, shuffle=False
    )

    print("-- Global graph stats (neighbor index, train only, fwd+inv edges) --")
    _graph_stats(train_loader.neighbor_index, num_relations)

    batch = next(iter(train_loader))
    triples = batch["triples"]
    hng, tng = batch["head_neighbor"], batch["tail_neighbor"]

    print("-- First batch --")
    print(f"  triples shape    : {triples.shape}")
    _batch_edge_stats("head_neighbor", hng, B)
    _batch_edge_stats("tail_neighbor", tng, B)

    print("-- Exclusion check --")
    for i in range(triples.size(0)):
        h, r, t = triples[i].tolist()
        r_inv = r + num_relations

        # head subgraph: forward edge (h -[r]-> t) must be masked
        s, e = hng["ptr"][i].item(), hng["ptr"][i + 1].item()
        nbrs  = hng["edge_index"][1, s:e].tolist()
        types = hng["edge_attr"][s:e].tolist()
        leaked = any(n == t and rt == r for n, rt in zip(nbrs, types))
        assert not leaked, f"Leakage at sample {i} (head, forward edge)"

        # tail subgraph: inverse edge (t -[r+num_rel]-> h) must be masked
        s, e = tng["ptr"][i].item(), tng["ptr"][i + 1].item()
        nbrs  = tng["edge_index"][1, s:e].tolist()
        types = tng["edge_attr"][s:e].tolist()
        leaked = any(n == h and rt == r_inv for n, rt in zip(nbrs, types))
        assert not leaked, f"Leakage at sample {i} (tail, inverse edge)"
    print("  PASSED\n")

    # --------------------------------------------------------------- entity
    print("=== EntityLoader (train only) ===")
    ent_loader = EntityLoader(
        ROOT, DS, num_relations=num_relations,
        include_valid=False, batch_size=B, shuffle=False,
    )

    print("-- Global graph stats (train only) --")
    _graph_stats(ent_loader.neighbor_index, num_relations)

    batch = next(iter(ent_loader))
    print("-- First batch --")
    print(f"  entity shape     : {batch['entity'].shape}")
    _batch_edge_stats("neighbor", batch["neighbor"], B)
    print(f"  num_entities (loader) : {ent_loader.num_entities}")
    print()

    print("=== EntityLoader (train + valid) ===")
    ent_loader_v = EntityLoader(
        ROOT, DS, num_relations=num_relations,
        include_valid=True, batch_size=B, shuffle=False,
    )
    print("-- Global graph stats (train + valid) --")
    _graph_stats(ent_loader_v.neighbor_index, num_relations)
    print(f"  num_entities (loader) : {ent_loader_v.num_entities}")
    print()

    # ----------------------------------------------------------------- eval
    train_rels = set(r for _, r, _ in load_split(dataset_dir, "train"))

    for split in ("valid", "test"):
        print(f"=== EvalLoader ({split}) ===")
        eval_loader  = EvalLoader(ROOT, DS, split=split, batch_size=B)
        all_triples  = torch.cat([b["triples"] for b in eval_loader], dim=0)
        split_rels   = set(all_triples[:, 1].tolist())

        unseen_rels         = split_rels - train_rels
        unseen_triples_mask = torch.tensor(
            [r in unseen_rels for r in all_triples[:, 1].tolist()]
        )
        num_unseen_triples = int(unseen_triples_mask.sum().item())

        print(f"  total triples      : {all_triples.size(0)}")
        print(f"  num_batches        : {len(eval_loader)}")
        print(f"  relation types     : {len(split_rels)}")
        print(f"  unseen relations   : {len(unseen_rels)} "
              f"(not in train, ids: {sorted(unseen_rels)})")
        print(f"  triples w/ unseen  : {num_unseen_triples} "
              f"({100 * num_unseen_triples / all_triples.size(0):.1f}%)")
        print(f"  first batch shape  : {next(iter(eval_loader))['triples'].shape}")
        print()

    # ----------------------------------------------------------- LUT shift
    # Quick diagnostic: how much does each entity's neighbor SET (not just
    # degree) change between the train-only graph and the train+valid graph?
    # A large fraction of changed entities means the GAT-based ctx_embed for
    # those entities at "val LUT" time differs structurally from "test LUT"
    # time -- a likely source of the val/test metric gap discussed earlier.
    print("=== Train-only vs Train+Valid neighbor-set diff ===")
    common_entities = set(ent_loader.neighbor_index.keys()) & set(ent_loader_v.neighbor_index.keys())
    changed = 0
    for e in common_entities:
        nb_train, _ = ent_loader.neighbor_index[e]
        nb_val,   _ = ent_loader_v.neighbor_index[e]
        if nb_train.size(0) != nb_val.size(0):
            changed += 1
    print(f"  entities present in both graphs : {len(common_entities)}")
    print(f"  entities with degree change     : {changed} "
          f"({100 * changed / max(len(common_entities), 1):.1f}%)")