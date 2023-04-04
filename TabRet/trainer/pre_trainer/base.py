import logging
import os

import numpy as np
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as ext
import torch
import torch.distributed as dist
import yaml
from pytorch_pfn_extras.training import ExtensionsManager
from pytorch_pfn_extras.training.triggers import EarlyStoppingTrigger
from torch import Tensor
from torch.cuda.amp import autocast

from ..base import BaseTrainer

logger = logging.getLogger(__name__)


class BasePreTrainer(BaseTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.save_best_model = True
        self.load_best_model = False
        self.frec_snapshot = 1

        if self.config.multi_node:
            self.config.batch_size = int(self.config.batch_size / self.config.world_size)
            self.config.eval_batch_size = int(self.config.eval_batch_size / self.config.world_size)
            if self.is_root:
                logger.info(f"batch_size {self.config.batch_size * self.config.world_size} -> {self.config.batch_size}")
                logger.info(
                    f"eval_batch_size {self.config.eval_batch_size * self.config.world_size} -> {self.config.eval_batch_size}"
                )

    def get_pre_manager(self, iters_per_epoch):
        if self.is_root:
            min_value = ext.MinValue("pre-val/loss")
            if os.path.exists("checkpoints_pre/best.yaml"):
                with open("checkpoints_pre/best.yaml", "r") as f:
                    best = yaml.safe_load(f)
                min_value.load_state_dict(best)
        else:
            min_value = None

        my_extensions = self.get_default_extensions(
            "pval",
            self.evaluate,
            min_value,
            ["pre-train/loss", "pre-val/loss", "lr/pre"],
        )

        ppe_writer = ppe.writing.SimpleWriter(out_dir=f"{self.save_dir}/checkpoints_pre")
        manager = ExtensionsManager(
            self.model,
            self.optimizer,
            self.config.epochs,
            extensions=my_extensions,
            writer=ppe_writer,
            iters_per_epoch=iters_per_epoch,
            stop_trigger=EarlyStoppingTrigger(
                monitor="pre-val/loss",
                patience=self.config.patience,
                max_trigger=(self.config.epochs, "epoch"),
            ),
        )
        saver_rank = 0 if self.config.multi_node else None
        manager.extend(
            ext.snapshot_object(
                self.model_wo_ddp,
                "model_epoch_{.updater.epoch:03d}",
                saver_rank=saver_rank,
            ),
            trigger=(self.frec_snapshot, "epoch"),
        )
        manager.extend(
            ext.snapshot(
                filename="snapshot_epoch_{.updater.epoch:03d}",
                saver_rank=saver_rank,
                n_retains=self.config.n_retains,
                autoload=True,
            ),
            trigger=(self.frec_snapshot, "epoch"),
        )
        return manager

    @torch.no_grad()
    def evaluate(self, **batch):
        self.model.eval()
        with autocast(enabled=self.scaler is not None):
            loss = self.forward_pre(batch)
        ppe.reporting.report({"pre-val/loss": loss.item()})

    def forward_pre(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

    def training(self):
        dataloader = self.datamodule.dataloader(
            "pre",
            self.config.batch_size,
            drop_last=self.config.drop_last,
        )
        scheduler = self.get_scheduler()

        manager = self.get_pre_manager(iters_per_epoch=len(dataloader))

        best_score = np.Inf
        while not manager.stop_trigger:
            self.model.train()
            if scheduler is not None:
                scheduler.step(manager.epoch)
            for batch in dataloader:
                with manager.run_iteration():
                    self.optimizer.zero_grad()
                    with autocast(enabled=self.scaler is not None):
                        loss = self.forward_pre(batch)

                    assert not torch.isnan(loss).any()

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    if self.config.multi_node:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()
                    ppe.reporting.report({"pre-train/loss": loss.item()})
                    ppe.reporting.report({"lr/pre": self.optimizer.param_groups[0]["lr"]})

            if self.is_root:
                best_conf = manager.get_extension("min_value").state_dict()
                if best_score > best_conf["_best_trigger"]["_best_value"]:
                    best_score = best_conf["_best_trigger"]["_best_value"]
                    if self.save_best_model:
                        torch.save(self.model_wo_ddp.state_dict(), "checkpoints_pre/best_model")
                    with open("checkpoints_pre/best.yaml", "w") as f:
                        yaml.dump(best_conf, f)
                if (
                    os.path.exists(f"checkpoints_pre/model_epoch_{manager.epoch:03d}")
                    and manager.epoch % self.config.store_interval != 0
                ):
                    os.remove(f"checkpoints_pre/model_epoch_{manager.epoch:03d}")

        if self.is_root:
            best_epoch = manager.get_extension("min_value").state_dict()["_best_epoch"]
            if self.load_best_model:
                logger.info(f"Load best model: {best_epoch} epoch model.")
                self.model_wo_ddp.load_state_dict(torch.load("checkpoints_pre/best_model"))
            else:
                logger.info(f"Best model is {best_epoch} epoch model.")
