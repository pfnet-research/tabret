import logging
import os
from typing import Dict

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

from ...base import BaseTrainer
from ..ood.utils import get_diff_columns

logger = logging.getLogger(__name__)


class Retokenizing(BaseTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.ret_config = config.fine_conf.ret
        self.save_best_model = True
        self.load_best_model = False
        self.frec_snapshot = 1

    def retokenizing_init(self):
        diff_columns, continuous_columns, cat_cardinality_dict = get_diff_columns(self.datamodule)

        self.model.add_attribute(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
        )
        if self.train_config.para_freeze:
            self.model.freeze_parameters_wo_specific_columns(diff_columns)

        if not self.train_config.mask_token_freeze:
            self.model.unfreeze_mask_token()

        if self.train_config.except_decoder:
            self.model.unfreeze_decoder()

        self.model_init()
        self.optimizer_init(self.ret_config.lr)

    def get_ret_manager(self):
        if self.is_root:
            min_value = ext.MinValue("ret/val/loss")
            if os.path.exists("checkpoints_ret/best.yaml"):
                with open("checkpoints_ret/best.yaml", "r") as f:
                    best = yaml.safe_load(f)
                min_value.load_state_dict(best)
        else:
            min_value = None

        my_extensions = self.get_default_extensions(
            "fval",
            self.evaluate_ret,
            min_value,
            ["ret/train/loss", "ret/val/loss", "lr/ret"],
        )

        ppe_writer = ppe.writing.SimpleWriter(out_dir=f"{self.save_dir}/checkpoints_ret")
        manager = ExtensionsManager(
            self.model,
            self.optimizer,
            self.ret_config.epochs,
            extensions=my_extensions,
            writer=ppe_writer,
            iters_per_epoch=len(self.datamodule.dataloader("fine", self.ret_config.batch_size)),
            stop_trigger=EarlyStoppingTrigger(
                monitor="ret/val/loss",
                patience=self.ret_config.patience,
                max_trigger=(self.ret_config.epochs, "epoch"),
            ),
        )
        manager.extend(
            ext.snapshot_object(
                self.model_wo_ddp,
                "model_epoch_{.updater.epoch:03d}",
            ),
            trigger=(self.frec_snapshot, "epoch"),
        )
        manager.extend(
            ext.snapshot(
                filename="snapshot_epoch_{.updater.epoch:03d}",
                n_retains=self.config.n_retains,
                autoload=True,
            ),
            trigger=(self.frec_snapshot, "epoch"),
        )
        return manager

    @torch.no_grad()
    def evaluate_ret(self, **batch):
        self.model.eval()
        with autocast(enabled=self.scaler is not None):
            loss = self.forward_ret(batch)
        ppe.reporting.report({"ret/val/loss": loss.item()})

    def retokenizing(self):
        manager = self.get_ret_manager()
        dataloader = self.datamodule.dataloader("fine", self.ret_config.batch_size)
        scheduler = self.get_scheduler()

        best_score = np.Inf
        while not manager.stop_trigger:
            self.model.train()
            if scheduler is not None:
                scheduler.step(manager.epoch)
            for batch in dataloader:
                with manager.run_iteration():
                    self.optimizer.zero_grad()
                    with autocast(enabled=self.scaler is not None):
                        loss = self.forward_ret(batch)

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
                    ppe.reporting.report({"ret/train/loss": loss.item()})
                    ppe.reporting.report({"lr/ret": self.optimizer.param_groups[0]["lr"]})

            best_conf = manager.get_extension("min_value").state_dict()
            if best_score > best_conf["_best_trigger"]["_best_value"]:
                best_score = best_conf["_best_trigger"]["_best_value"]
                torch.save(self.model_wo_ddp.state_dict(), "checkpoints_ret/best_model")
                with open("checkpoints_ret/best.yaml", "w") as f:
                    yaml.dump(best_conf, f)
            if os.path.exists(f"checkpoints_ret/model_epoch_{manager.epoch:03d}"):
                os.remove(f"checkpoints_ret/model_epoch_{manager.epoch:03d}")

        best_epoch = manager.get_extension("min_value").state_dict()["_best_epoch"]
        logger.info(f"Load best model: {best_epoch} epoch model.")
        self.model_wo_ddp.load_state_dict(torch.load("checkpoints_ret/best_model"))

    def forward_ret(self, data: Dict[str, Tensor]) -> Tensor:
        cont = data["continuous"]
        cate = data["categorical"]
        cont, cate = self.apply_device(cont, cate)
        loss, _, _ = self.model(
            cont,
            cate,
            self.ret_config.mask_ratio,
        )
        return loss
