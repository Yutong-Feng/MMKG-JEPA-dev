import typing
from typing import Any, Dict

from .dataset_old import EntityLoader, EvalLoader, TrainKGLoader


def get_loaders(
    meta_path: str,
    dataset_kwargs: Dict[str, Any],
    exp_kwargs: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """
    Initialize and return dataloaders for training, validation, testing, and entity LUT building.
    """
    # Extract config parameters with safe defaults
    dataset_name = dataset_kwargs["name"]
    train_bs = exp_kwargs["batch_size"]
    eval_bs = exp_kwargs["eval_batch_size"]
    drop_rate = exp_kwargs["drop_rate"]
    num_workers = exp_kwargs["num_workers"]
    prefetch_factor = exp_kwargs["prefetch_factor"]
    top_n = 16

    # 1. Train Loader: For training phase (triples + masked subgraphs)
    train_loader = TrainKGLoader(
        data_root=meta_path,
        dataset=dataset_name,
        num_relations=dataset_kwargs["num_relations"],
        top_n=top_n,
        batch_size=train_bs,
        drop_rate=drop_rate,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # 2. Entity Loader: For building the LUT during inference (entities + full subgraphs)
    # include_valid=True is typical for evaluation to expose valid edges to the GNN context
    val_entity_loader = EntityLoader(
        data_root=meta_path,
        dataset=dataset_name,
        num_relations=dataset_kwargs["num_relations"],
        top_n=top_n,
        include_valid=False,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    test_entity_loader = EntityLoader(
        data_root=meta_path,
        dataset=dataset_name,
        num_relations=dataset_kwargs["num_relations"],
        top_n=top_n,
        include_valid=True,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # 3. Validation Loader: For inference phase (triples only)
    val_loader = EvalLoader(
        data_root=meta_path,
        dataset=dataset_name,
        top_n=top_n,
        split="valid",
        batch_size=eval_bs,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # 4. Test Loader: For inference phase (triples only)
    test_loader = EvalLoader(
        data_root=meta_path,
        dataset=dataset_name,
        top_n=top_n,
        split="test",
        batch_size=eval_bs,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "val_entity_loader": val_entity_loader,
        "test_entity_loader": test_entity_loader,
    }
