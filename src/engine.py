import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from logging import Logger
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from tqdm import tqdm

from dataset_old import load_splits
from src.metrics import compute_all_metrics_filtered, compute_filtered_ranks


class BaseEngine:
    """Base engine with common functionality."""

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger: logging.Logger,
        save_path: str,
    ):
        self.device = device
        self.model = model.to(device)
        self.logger = logger
        self.save_path = save_path

        # Create save directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Initialize JSON log file
        self.log_file = os.path.join(self.save_path, "training_log.json")

        # Store model parameter count if available
        if hasattr(model, "param_num"):
            self.logger.info(f"Model parameters: {model.param_num():,}")

    def initialize_log_file(self, log_metadata=None):
        """Initialize the JSON log file with metadata."""
        if os.path.exists(self.log_file):
            self.logger.info(
                f"Log file already exists at {self.log_file}, appending to it."
            )
        else:
            if log_metadata is None:
                log_metadata = {}
            with open(self.log_file, "w") as f:
                json.dump(log_metadata, f, indent=2)
            self.logger.info(f"Created new log file at {self.log_file}")

    def _log_epoch_info(self, epoch: Union[int, str], epoch_log: Dict):
        """
        Log epoch information to JSON file.

        Args:
            epoch: Current epoch number
            epoch_log: Information need to be recorded.
        """
        try:
            # Load existing log
            with open(self.log_file, "r") as f:
                log_data = json.load(f)

            # Append new log entry
            log_data["epoch_logs"][epoch] = epoch_log

            # Write back to file
            with open(self.log_file, "w") as f:
                json.dump(log_data, f, indent=2)

            self.logger.info(f"Logged epoch {epoch} info to {self.log_file}")

        except Exception as e:
            self.logger.error(f"Failed to log epoch info: {str(e)}")


class MMKGEngine(BaseEngine):
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger: logging.Logger,
        dataset_dir: str,
        save_path: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        val_entity_loader: torch.utils.data.DataLoader,
        test_entity_loader: torch.utils.data.DataLoader,
        loss_func: Callable,
        num_relations: int,
        config: Dict[str, Any],
    ):
        super().__init__(device, model, logger, save_path)
        self.dataset_dir = dataset_dir

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.val_entity_loader = val_entity_loader
        self.test_entity_loader = test_entity_loader
        self.loss_func = loss_func
        self.num_relations = num_relations

        # Training configuration
        self.current_epoch = 1
        self.max_epochs = config["max_epochs"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.T_0 = config["T_0"]
        self.T_mult = config["T_mult"]
        self.clip_grad_value = config["clip_grad_value"]
        self.accumulation_steps = config["accumulation_steps"]
        self.save_freq = config["save_freq"]

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.lr * 0.01
        )

        # Track best model: for KGC (like MRR or Hit@K), higher is better
        self.best_criteria = -1.0
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[bool, float, float]:
        self.model.train()
        avg_loss = 0.0
        avg_reg = 0.0
        terminate = False

        # Reset gradients at the start of the epoch
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            # Move tensors to device
            batch = {
                k: (
                    v.to(self.device)
                    if isinstance(v, torch.Tensor)
                    else {sk: sv.to(self.device) for sk, sv in v.items()}
                )
                for k, v in batch.items()
            }

            # Forward pass
            pred_embed, tail_ctx_embed = self.model(batch)
            loss, sigreg = self.loss_func(pred_embed, tail_ctx_embed)

            if torch.isnan(loss):
                self.logger.error("Training loss is NaN. Terminating.")
                terminate = True
                break

            # Scale loss to normalize accumulated gradients
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

            # Update metrics using unscaled values for accurate logging
            avg_loss = avg_loss * (
                batch_idx / (batch_idx + 1)
            ) + loss.detach().cpu().item() / (batch_idx + 1)
            avg_reg = avg_reg * (
                batch_idx / (batch_idx + 1)
            ) + sigreg.detach().cpu().item() / (batch_idx + 1)

            pbar.set_description(
                f"Loss: {loss.detach().cpu().item():.4f}, Reg: {sigreg.detach().cpu().item():.4f}, "
                f"Avg Loss: {avg_loss:.4f}, Avg Reg: {avg_reg:.4f}"
            )

            # Gradient accumulation and optimization step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_value
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Handle remaining accumulated gradients at the end of epoch
        if len(self.train_loader) % self.accumulation_steps != 0:
            if self.clip_grad_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_value
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return terminate, avg_loss, avg_reg

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        return self._evaluate(self.val_loader, prefix="Validation")

    def test(self) -> Dict[str, float]:
        """
        Test model on test set.

        Returns:
            Dictionary of test metrics
        """
        return self._evaluate(self.test_loader, prefix="Test")

    def _build_filter_dict(self, dataset_dir: str) -> Dict[tuple, set]:
        # Verify all required split files exist before proceeding
        missing = [
            split
            for split in ["train", "valid", "test"]
            if not os.path.exists(os.path.join(dataset_dir, f"{split}.txt"))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing split files in dataset_dir='{dataset_dir}': "
                f"{[f'{s}.txt' for s in missing]}. "
                f"Make sure you pass the dataset-specific directory "
                f"(e.g. './data/XXX'), not the data root (e.g. './data')."
            )

        filter_dict: Dict[tuple, set] = defaultdict(set)
        all_triples = load_splits(dataset_dir, ["train", "valid", "test"])
        for h, r, t in all_triples:
            # Forward valid answer
            filter_dict[(h, r)].add(t)
            # Inverse valid answer
            filter_dict[(t, r + self.num_relations)].add(h)

        return dict(filter_dict)

    def _evaluate(self, loader, prefix: str, num_samples: int = 16) -> Dict[str, float]:
        """
        Evaluate the model on forward and inverse queries using latent variable sampling.
        """
        self.model.eval()

        total_metrics = {"MRR": 0.0, "Hit@1": 0.0, "Hit@3": 0.0, "Hit@10": 0.0}
        total_samples = 0
        all_ranks = []

        filter_dict = self._build_filter_dict(self.dataset_dir)

        with torch.no_grad():
            self.logger.info(f"[{prefix}] Building entity Look-Up Table (LUT)...")
            if prefix == "Validation":
                self.model.build_lut(self.val_entity_loader, device=self.device)
            elif prefix == "Test":
                self.model.build_lut(self.test_entity_loader, device=self.device)
            else:
                raise ValueError(f"Unknown prefix: {prefix}")

            pbar = tqdm(loader, desc=prefix, file=sys.stdout)
            for batch in pbar:
                triples_cpu = batch["triples"].cpu().numpy()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                triples = batch["triples"]
                batch_size = triples.size(0)

                # ==========================================
                # 1. Forward Prediction: (h, r, ?) -> predict tail
                # ==========================================
                # Pass num_samples to evaluate multiple latent hypotheses
                fw_scores = self.model.retrieve(
                    batch,
                    top_k=None,
                    device=self.device,
                    return_scores=True,
                    num_samples=num_samples,
                )
                fw_targets = triples[:, 2]

                fw_filters = [filter_dict.get((h, r), set()) for h, r, _ in triples_cpu]
                fw_ranks = compute_filtered_ranks(fw_scores, fw_targets, fw_filters)

                # ==========================================
                # 2. Inverse Prediction: (t, r_inv, ?) -> predict head
                # ==========================================
                inv_batch = batch.copy()
                inv_triples = triples.clone()

                inv_triples[:, 0] = triples[:, 2]
                inv_triples[:, 1] = triples[:, 1] + self.num_relations
                inv_triples[:, 2] = triples[:, 0]
                inv_batch["triples"] = inv_triples

                # Pass num_samples to evaluate multiple latent hypotheses for inverse relations
                inv_scores = self.model.retrieve(
                    inv_batch,
                    top_k=None,
                    device=self.device,
                    return_scores=True,
                    num_samples=num_samples,
                )
                inv_targets = inv_triples[:, 2]

                inv_filters = [
                    filter_dict.get((t, r + self.num_relations), set())
                    for _, r, t in triples_cpu
                ]
                inv_ranks = compute_filtered_ranks(inv_scores, inv_targets, inv_filters)

                # ==========================================
                # 3. Update Metrics & Progress Bar
                # ==========================================
                batch_ranks = torch.cat([fw_ranks, inv_ranks], dim=0)
                all_ranks.append(batch_ranks)

                batch_metrics = compute_all_metrics_filtered(batch_ranks)
                current_samples = batch_size * 2
                total_samples += current_samples

                for k in total_metrics:
                    if np.isnan(batch_metrics[k]):
                        self.logger.warning(f"Found NaN in {prefix} metric: {k}.")
                        continue
                    total_metrics[k] += batch_metrics[k] * current_samples

                current_avg = {k: v / total_samples for k, v in total_metrics.items()}
                pbar.set_description(
                    f"{prefix}: "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in current_avg.items())
                )

        # ==========================================
        # 4. Final Aggregation
        # ==========================================
        final_ranks = torch.cat(all_ranks, dim=0)
        final_metrics = compute_all_metrics_filtered(final_ranks)

        self.logger.info(
            f"Final {prefix} Results: "
            + ", ".join(f"{k}: {v:.4f}" for k, v in final_metrics.items())
        )

        return final_metrics

    def _create_state_dict(self, epoch: int) -> dict:
        """Construct the checkpoint state dictionary."""
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_criteria,
            "config": {
                "max_epochs": self.max_epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
        }

    def save_checkpoint(self, epoch: int):
        """Save the latest training checkpoint (overwrites to save disk space)."""
        checkpoint = self._create_state_dict(epoch)
        checkpoint_path = os.path.join(self.save_path, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)

    def save_best_checkpoint(self, epoch: int):
        """Save the best performing model checkpoint."""
        checkpoint = self._create_state_dict(epoch)
        best_path = os.path.join(self.save_path, "best_model.pth")
        torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint: Dict = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "best_val_loss" in checkpoint:
            self.best_criteria = checkpoint["best_val_loss"]
        else:
            self.logger.warning(
                "best_val_loss not found in checkpoint. Setting to infinity."
            )
            self.best_criteria = float("inf")
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
            self.best_epoch = checkpoint["epoch"]
        else:
            self.logger.warning(
                "epoch not found in checkpoint. Setting best_epoch to 0."
            )
            self.best_epoch = 0

        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def run(self):
        """
        Main training loop.
        """
        log_metadata = {
            "training_start_time": datetime.now().isoformat(),
            "max_epochs": self.max_epochs,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "clip_grad_value": self.clip_grad_value,
            "accumulation_steps": self.accumulation_steps,
            "save_freq": self.save_freq,
            "save_dir": self.save_path,
            "epoch_logs": {},
        }
        self.initialize_log_file(log_metadata)

        TERMINATE = False
        self.logger.info("Starting training...")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.logger.info(f"Epoch {epoch}/{self.max_epochs}")

            # Training
            train_start = time()
            TERMINATE, train_loss, train_reg = self.train_epoch()
            train_time = time() - train_start
            train_memory_reserved = torch.cuda.memory_reserved()
            train_memory_allocated = torch.cuda.memory_allocated()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Train loss: {train_loss}, Train reg: {train_reg}")
            self.logger.info(f"Train time: {train_time}s")
            self.logger.info(
                f"Train memory reserved: {train_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Train memory allocated: {train_memory_allocated / (1024 ** 2):.2f} MB"
            )
            self.logger.info(f"Current learning rate: {current_lr}")

            if TERMINATE:
                self.logger.error("Terminating training due to NaN loss.")
                break

            # Validation
            val_start = time()
            val_metrics = self.validate()
            val_time = time() - val_start
            val_memory_reserved = torch.cuda.memory_reserved()
            val_memory_allocated = torch.cuda.memory_allocated()
            self.logger.info(f"Validation metrics: {val_metrics}")
            self.logger.info(f"Val time: {val_time:.2f}s")
            self.logger.info(
                f"Val memory reserved: {val_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Val memory allocated: {val_memory_allocated / (1024 ** 2):.2f} MB"
            )

            # Prepare epoch log entry
            epoch_log = {
                "timestamp": datetime.now().isoformat(),
                "current_learning_rate": current_lr,
                "training": {
                    "loss": float(train_loss),
                    "reg": float(train_reg),
                    "time_seconds": float(train_time),
                    "memory_reserved_mb": float(train_memory_reserved / (1024**2)),
                    "memory_allocated_mb": float(train_memory_allocated / (1024**2)),
                },
                "validation": {
                    "metrics": val_metrics,
                    "time_seconds": float(val_time),
                    "memory_reserved_mb": float(val_memory_reserved / (1024**2)),
                    "memory_allocated_mb": float(val_memory_allocated / (1024**2)),
                },
            }
            self._log_epoch_info(epoch=epoch, epoch_log=epoch_log)

            # Check for best model
            criteria = val_metrics.get("MRR", 0.0)
            is_best = criteria >= self.best_criteria

            if is_best:
                self.best_criteria = criteria
                self.best_epoch = epoch
                self.logger.info(f"New best model! Val MRR: {criteria:.4f}")

            # Save checkpoint
            if (epoch) % self.save_freq == 0:
                self.save_checkpoint(epoch)

            if is_best:
                self.save_best_checkpoint(epoch)

            test_start = time()
            test_metrics = self.test()
            test_time = time() - test_start
            test_memory_reserved = torch.cuda.memory_reserved()
            test_memory_allocated = torch.cuda.memory_allocated()
            self.logger.info(f"Test metrics: {test_metrics}")
            self.logger.info(f"Test time: {test_time:.2f}s")
            self.logger.info(
                f"Test memory reserved: {test_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Test memory allocated: {test_memory_allocated / (1024 ** 2):.2f} MB"
            )

            # Update scheduler
            self.scheduler.step()

        self.logger.info(
            f"Training completed. Best epoch: {self.best_epoch}, Best val loss: {self.best_criteria:.6f}"
        )

        self.load_checkpoint(os.path.join(self.save_path, "best_model.pth"))
        self.logger.info("Evaluating best model on test sets...")
        test_start = time()
        test_metrics = self.test()
        test_time = time() - test_start
        test_memory_reserved = torch.cuda.memory_reserved()
        test_memory_allocated = torch.cuda.memory_allocated()
        self.logger.info(f"Test metrics: {test_metrics}")
        self.logger.info(f"Test time: {test_time:.2f}s")
        self.logger.info(
            f"Test memory reserved: {test_memory_reserved / (1024 ** 2):.2f} MB"
        )
        self.logger.info(
            f"Test memory allocated: {test_memory_allocated / (1024 ** 2):.2f} MB"
        )

        # Prepare epoch log entry
        test_log = {
            "timestamp": datetime.now().isoformat(),
            "current_learning_rate": current_lr,
            "test": {
                "metrics": test_metrics,
                "time_seconds": float(test_time),
                "memory_reserved_mb": float(test_memory_reserved / (1024**2)),
                "memory_allocated_mb": float(test_memory_allocated / (1024**2)),
            },
        }
        self._log_epoch_info(epoch="test", epoch_log=test_log)
