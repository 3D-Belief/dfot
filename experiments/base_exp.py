"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC
from typing import Optional, Union, Dict
import pathlib, shutil

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig

from dfot_utils.print_utils import cyan
from dfot_utils.distributed_utils import rank_zero_print
from dfot_utils.lightning_utils import EMA, load_weights_only
from .data_modules import BaseDataModule

torch.set_float32_matmul_precision("high")

import math
import sys
import torch.distributed as dist  
import os, sys

def master_print(*args, **kwargs):
    is_master = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    if is_master:
        print(*args, **kwargs, flush=True, file=kwargs.pop("file", sys.stdout))

class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger if logger else False
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            rank_zero_print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError
    data_module_cls = BaseDataModule

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = self.data_module_cls(root_cfg, self.compatible_datasets)

    def _build_common_callbacks(self):
        return [EMA(**self.cfg.ema)]

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()

        # ---- choose RESUME vs FINETUNE ----
        resume_ckpt = getattr(self.cfg.training, "resume_from_checkpoint", None)
        finetune_ckpt = getattr(self.cfg.training, "finetune_from", None)

        # If resuming, DO NOT load weights-only (we want loop/optimizer state)
        if resume_ckpt:
            finetune_ckpt = None  # ignore any finetune request

        # If finetuning (weights-only), do it before compile
        if finetune_ckpt:
            if not os.path.exists(str(finetune_ckpt)):
                raise FileNotFoundError(f"finetune_from not found: {finetune_ckpt}")
            load_weights_only(self.algo, finetune_ckpt, strict=False)

        # Optional compile AFTER loading weights (safe for resume or finetune)
        if getattr(self.cfg.training, "compile", False):
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(
                        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
                    )
                    / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )
        callbacks += self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,
            num_sanity_val_steps=(
                int(self.cfg.debug)
                if self.cfg.validation.num_sanity_val_steps is None
                else self.cfg.validation.num_sanity_val_steps
            ),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            reload_dataloaders_every_n_epochs=self.cfg.reload_dataloaders_every_n_epochs,
        )

        self.data_module.trainer = trainer

        def _exists(p): 
            try: return p and os.path.exists(str(p))
            except: return False

        def _safe_get(dc, path, default=None):
            # access like _safe_get(self.root_cfg, "experiment.training.lr")
            cur = dc
            try:
                for k in path.split("."):
                    cur = cur[k] if isinstance(cur, dict) else getattr(cur, k)
                return cur
            except Exception:
                return default

        # world/devices
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count()) * int(_safe_get(self.cfg, "num_nodes", 1))))
            rank = int(os.environ.get("RANK", 0))

        num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
        devices_hint = f"{num_cuda} CUDA (visible) x {_safe_get(self.cfg,'num_nodes',1)} node(s)"

        # dataloaders & datasets
        train_dl = self.data_module.train_dataloader()
        val_dl   = self.data_module.val_dataloader()
        test_dl  = self.data_module.test_dataloader()

        train_ds = getattr(self.data_module, "train_ds", getattr(train_dl, "dataset", None))
        val_ds   = getattr(self.data_module, "val_ds",   getattr(val_dl,   "dataset", None))
        test_ds  = getattr(self.data_module, "test_ds",  getattr(test_dl,  "dataset", None))

        # batch/accum math
        per_gpu_bs = int(_safe_get(self.cfg, "training.batch_size", 1))
        accum      = int(_safe_get(self.cfg, "training.optim.accumulate_grad_batches", 1))
        eff_batch  = per_gpu_bs * max(1, world_size) * max(1, accum)

        # step math (Lightning optimizer steps per epoch ≈ ceil(len(train_dl)/accum))
        len_train_dl = len(train_dl)
        steps_per_epoch = math.ceil(len_train_dl / max(1, accum))
        max_epochs = int(_safe_get(self.cfg, "training.max_epochs", 0))
        max_steps  = int(_safe_get(self.cfg, "training.max_steps", 0)) or steps_per_epoch * max_epochs

        # dataset-specific extras
        n_frames   = getattr(train_ds, "n_frames", None)
        frame_skip = _safe_get(self.root_cfg, "frame_skip", _safe_get(self.cfg, "frame_skip", None))
        max_frames = _safe_get(self.root_cfg, "max_frames", _safe_get(self.cfg, "max_frames", None))
        epoch_clips = (len(train_ds) if train_ds is not None else None)
        clips_available = None
        try:
            if train_ds is not None and hasattr(train_ds, "metadata"):
                clips_available = sum(max(train_ds.video_length(m) - int(n_frames) + 1, 0) for m in train_ds.metadata) if n_frames else None
        except Exception:
            pass

        # data stats & aug
        resolution = int(_safe_get(self.root_cfg, "resolution", _safe_get(self.cfg, "resolution", 0)))
        data_mean  = _safe_get(self.root_cfg, "dataset.data_mean", None)
        data_std   = _safe_get(self.root_cfg, "dataset.data_std",  None)
        aug = _safe_get(self.root_cfg, "dataset.augmentation", {})

        # LR & schedule
        lr = float(_safe_get(self.cfg, "training.lr", 0.0))
        sched = _safe_get(self.root_cfg, "algorithm.lr_scheduler.name", "unknown")
        warmup = _safe_get(self.root_cfg, "algorithm.lr_scheduler.num_warmup_steps", None)
        sched_total = _safe_get(self.root_cfg, "algorithm.lr_scheduler.num_training_steps", None)

        # checkpoints
        ckpt_cfg   = _safe_get(self.cfg, "training.checkpointing", {}) or {}
        monitor    = ckpt_cfg.get("monitor", None)
        mode       = ckpt_cfg.get("mode", None)
        save_top_k = ckpt_cfg.get("save_top_k", None)
        save_last  = ckpt_cfg.get("save_last", None)
        every_steps = ckpt_cfg.get("every_n_train_steps", None)
        every_epochs = ckpt_cfg.get("every_n_epochs", None)
        finetune_from = _safe_get(self.cfg, "training.finetune_from", None)

        # model params
        total_params = sum(p.numel() for p in self._build_algo().parameters())
        trainable_params = sum(p.numel() for p in self._build_algo().parameters() if p.requires_grad)

        # dataloader workers
        tw = getattr(train_dl, "num_workers", None)
        vw = getattr(val_dl,   "num_workers", None)
        pw = getattr(test_dl,  "num_workers", None)

        master_print("────────────────────────────────────────────────────────────────────────")
        master_print("Run summary")
        master_print(f"• World: rank={rank} | world_size={world_size} | devices (visible)={devices_hint}")
        master_print(f"• Precision={_safe_get(self.cfg,'training.precision', '32')} | Compile={bool(_safe_get(self.cfg,'training.compile', False))}")
        master_print(f"• Per-GPU batch={per_gpu_bs} | Accum={accum} ⇒ Effective batch={eff_batch}")
        master_print(f"• len(train_dl)={len_train_dl} | steps/epoch≈ceil({len_train_dl}/{accum})={steps_per_epoch}")
        master_print(f"• Max epochs={max_epochs} | Max steps={max_steps} (scheduler total={sched_total})")
        if epoch_clips is not None:
            master_print(f"• Train epoch clips=len(train_ds)={epoch_clips}")
        if clips_available is not None:
            master_print(f"• Total clips available in metadata={clips_available}")
        if n_frames is not None:
            master_print(f"• Clip geometry: n_frames={n_frames} (=1 + (max_frames-1)*frame_skip) | frame_skip={frame_skip} | max_frames={max_frames}")
        master_print(f"• Resolution={resolution} | data_mean={data_mean} | data_std={data_std}")
        master_print(f"• Augmentation: {aug}")
        master_print(f"• Dataloader workers: train={tw} val={vw} test={pw}")
        master_print(f"• Optim: lr={lr:g} | Scheduler={sched} | warmup_steps={warmup}")
        master_print(f"• Checkpointing: monitor={monitor} mode={mode} save_top_k={save_top_k} save_last={save_last} "
                    f"| every_n_steps={every_steps} every_n_epochs={every_epochs}")
        master_print(f"• Finetune from: {finetune_from} | exists={_exists(finetune_from)}")
        master_print(f"• Params: total={total_params/1e6:.2f}M | trainable={trainable_params/1e6:.2f}M")
        # optional: show free disk where checkpoints go
        try:
            outdir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
            free_gb = shutil.disk_usage(outdir).free / (1024**3)
            master_print(f"• Output dir: {outdir} | free space ≈ {free_gb:.1f} GB")
        except Exception:
            pass
        master_print("────────────────────────────────────────────────────────────────────────")

        ckpt_path = str(resume_ckpt) if resume_ckpt else None
        trainer.fit(self.algo, datamodule=self.data_module, ckpt_path=ckpt_path)

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.test.inference_mode,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )
