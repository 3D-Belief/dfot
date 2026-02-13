import os
import sys

import hydra
from omegaconf import DictConfig
import torch

from datasets.video import (
    MinecraftAdvancedVideoDataset,
    Kinetics600AdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KMiniAdvancedVideoDataset,
    RealEstate10KOODAdvancedVideoDataset,
    SpocAdvancedVideoDataset,
    HabitatAdvancedVideoDataset,
)
from dfot_utils.hydra_utils import unwrap_shortcuts
from model_wrapper import ModelWrapper

@staticmethod
def _get_shuffle(dataset: torch.utils.data.Dataset, default: bool) -> bool:
    return not isinstance(dataset, torch.utils.data.IterableDataset) and default

@staticmethod
def _get_num_workers(num_workers: int) -> int:
    return min(os.cpu_count(), num_workers)

@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    ckpt_path = cfg.ckpt_path
    model_wrapper = ModelWrapper(cfg, ckpt_path)
    compatible_datasets = dict(
        # video datasets
        minecraft=MinecraftAdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
        realestate10k_ood=RealEstate10KOODAdvancedVideoDataset,
        realestate10k_mini=RealEstate10KMiniAdvancedVideoDataset,
        kinetics_600=Kinetics600AdvancedVideoDataset,
        spoc=SpocAdvancedVideoDataset,
        habitat=HabitatAdvancedVideoDataset,
    )
    dataset = compatible_datasets[cfg.dataset.name](cfg.dataset, split="test")
    split_cfg = cfg.experiment["test"]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=_get_num_workers(split_cfg.data.num_workers),
        shuffle=_get_shuffle(dataset, split_cfg.data.shuffle),
        persistent_workers=False,
        worker_init_fn=lambda worker_id: (
            dataset.worker_init_fn(worker_id)
            if hasattr(dataset, "worker_init_fn")
            else None
        ),
    )
    # get a batch
    batch = next(iter(dataloader))
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
    # run inference
    outputs = model_wrapper.inference(batch)


if __name__ == "__main__":
    sys.argv = unwrap_shortcuts(
        sys.argv, config_path="configurations", config_name="config"
    )
    run()
