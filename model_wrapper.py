import os
import sys
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from algorithms.dfot import DFoTVideo, DFoTVideoPose
import torch

class ModelWrapper:
    """
    A wrapper class for the DFoT model to handle common functionalities
    """

    def __init__(self, cfg: DictConfig, ckpt_path: str):
        self.root_cfg = cfg
        self._build_algo()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.algo.load_state_dict(ckpt["state_dict"], strict=False)
    
    def _build_algo(self):
        """
        Build the lightning module
        """
        self.algo = DFoTVideoPose(self.root_cfg.algorithm)
        self.algo = self.algo.to("cuda")
    
    @torch.no_grad()
    def inference(self, batch) -> dict:
        """
        Inference function
        :param batch: input batch
        :return: generated videos
        """
        return self.algo.inference(batch)