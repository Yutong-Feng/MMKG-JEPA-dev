import torch
from typing import Dict, List, Set, Tuple


def compute_filtered_ranks(
    scores: torch.Tensor,  # [B, num_entities], lower = better (e.g., L2 dist)
    true_targets: torch.Tensor,  # [B], the ground truth entity IDs
    filter_sets: List[Set[int]],  # List of B sets containing known valid entity IDs
) -> torch.Tensor:  # [B], 1-based ranks
    """
    Computes filtered ranks for any generic target (head or tail).
    Uses fast sparse-coordinate (COO) indexing to apply +inf masks in a single GPU operation.
    """
    B = scores.size(0)
    filtered_scores = scores.clone()
    true_targets_list = true_targets.tolist()

    row_idx = []
    col_idx = []

    # Gather coordinates of all valid answers EXCEPT the ground truth target
    for i, (f_set, true_t) in enumerate(zip(filter_sets, true_targets_list)):
        mask_targets = f_set - {true_t}
        if mask_targets:
            row_idx.extend([i] * len(mask_targets))
            col_idx.extend(list(mask_targets))

    # Apply penalty to known alternatives in ONE vectorized operation
    if row_idx:
        filtered_scores[row_idx, col_idx] = float("inf")

    # Extract the score of the actual ground truth targets
    batch_idx = torch.arange(B, device=scores.device)
    true_target_scores = filtered_scores[batch_idx, true_targets]  # [B]

    # Rank is the number of entities scoring better or equal to the ground truth
    ranks = (
        (filtered_scores <= true_target_scores.unsqueeze(1)).sum(dim=1).long()
    )  # [B]

    return ranks


def compute_all_metrics_filtered(ranks: torch.Tensor) -> Dict[str, float]:
    """
    Compute filtered metrics (MRR, Hit@N) directly from a pre-computed rank tensor.
    The filtering mechanism is assumed to be handled prior to this function
    (e.g., via `compute_filtered_ranks`).
    """
    return _metrics_from_ranks(ranks)


def compute_all_metrics(
    scores: torch.Tensor,
    true_targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute raw (unfiltered) metrics computed over ALL entities.
    """
    batch_idx = torch.arange(scores.size(0), device=scores.device)
    true_target_scores = scores[batch_idx, true_targets]  # [B]

    # Rank is the number of entities scoring better or equal to the ground truth
    ranks = (scores <= true_target_scores.unsqueeze(1)).sum(dim=1).long()  # [B]
    return _metrics_from_ranks(ranks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metrics_from_ranks(ranks: torch.Tensor) -> Dict[str, float]:
    """
    Compute MRR and Hit@1/3/10 from a 1-based rank tensor.
    """
    return {
        "MRR": (1.0 / ranks.float()).mean().item(),
        "Hit@1": (ranks <= 1).float().mean().item(),
        "Hit@3": (ranks <= 3).float().mean().item(),
        "Hit@10": (ranks <= 10).float().mean().item(),
    }
