import os

import hydra
import hydra.core
import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig
from src import lejepa
from src.engine import MMKGEngine
from src.loader import get_loaders
from src.logger import get_logger
from src.model import KGJEPAModel
from src.loss import lejepa_loss  
from torch import nn


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="default",
)
def main(cfg: DictConfig):
    # mp.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision('high')
    
    dataset_kwargs = cfg["dataset"]
    model_kwargs = cfg["model"]
    exp_kwargs = cfg["exp"]
    seed = cfg["seed"]
    set_seed(seed)

    device_id = exp_kwargs["device_id"]
    DEVICE = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    meta_path = cfg["meta_path"]

    loaders = get_loaders(meta_path, dataset_kwargs, exp_kwargs, DEVICE)

    # Create the multivariate slicing test
    univariate_test = lejepa.univariate.EppsPulley()
    regularization = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test, num_slices=min(1024, model_kwargs["embed_dim"])
    ).to(DEVICE)
    a = exp_kwargs["regularization"]
    loss_func = lambda h, p, t: lejepa_loss(h, p, t, a=a, regularization=regularization)

    model: KGJEPAModel = instantiate(model_kwargs).to(DEVICE)
    # model = torch.compile(model)

    log_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = get_logger(log_folder, log_file="training.log")
    engine = MMKGEngine(
        device=DEVICE,
        model=model,
        logger=logger,
        save_path=log_folder,
        dataset_dir=os.path.join(meta_path, dataset_kwargs["name"]),
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
        val_entity_loader=loaders["val_entity_loader"],
        test_entity_loader=loaders["test_entity_loader"],
        loss_func=loss_func,
        num_relations=dataset_kwargs["num_relations"],
        config=exp_kwargs,
    )
    engine.run()


if __name__ == "__main__":
    main()
