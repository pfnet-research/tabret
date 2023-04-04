import logging
import os
from glob import glob

import numpy as np
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as ext
import sklearn
import torch
import torch.nn.functional as F
import yaml
from pytorch_pfn_extras.training import ExtensionsManager
from pytorch_pfn_extras.training.triggers import EarlyStoppingTrigger
from torch import Tensor
from torch.cuda.amp import autocast

from ..base import BaseTrainer
from ..utils import save_json
from .utils import load_except_decoder, load_pre_model, load_wo_feature_embed

logger = logging.getLogger(__name__)


class BaseFineTrainer(BaseTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.train_config = config.fine_conf
        task = self.datamodule.task
        self.monitor_metric = "rmse" if task == "regression" else "auc" if task == "binary" else "acc"
        self.loss_fn = F.cross_entropy if self.datamodule.task != "regression" else F.mse_loss

    def load_pretraind_model(self):
        if self.train_config.pre_path is not None:
            if hasattr(self.train_config, "except_decoder") and self.train_config.except_decoder:
                load_except_decoder(self.model, self.train_config.pre_path)
            else:
                load_pre_model(self.model, self.train_config.pre_path)
        elif hasattr(self.train_config, "trans_path") and self.train_config.trans_path is not None:
            load_wo_feature_embed(self.model, self.train_config.trans_path)
        else:
            logger.info("Not pre-trained model.")

    def update_best_value(self, best_value, new_value):
        if self.datamodule.task == "regression":
            return best_value > new_value
        else:
            return best_value < new_value

    def get_fine_manager(self, iters_per_epoch):
        if self.is_root:
            best_value = ext.BestValue(
                f"fine-val/{self.monitor_metric}",
                self.update_best_value,
            )
            if os.path.exists(f"{self.save_dir}/checkpoints_fine/best.yaml"):
                with open(f"{self.save_dir}/checkpoints_fine/best.yaml", "r") as f:
                    best = yaml.safe_load(f)
                best_value.load_state_dict(best)
        else:
            best_value = None

        if self.datamodule.task == "binary":
            metrics = ["acc", "auc"]
        elif self.datamodule.task == "multiclass":
            metrics = ["acc"]
        else:
            metrics = ["rmse"]

        my_extensions = self.get_default_extensions(
            "fval",
            self.evaluate,
            best_value,
            ["fine-train/loss", "fine-val/loss"] + [f"fine-val/{k}" for k in metrics] + ["lr/fine"],
        )

        ppe_writer = ppe.writing.SimpleWriter(out_dir=f"{self.save_dir}/checkpoints_fine")
        manager = ExtensionsManager(
            self.model,
            self.optimizer,
            self.config.epochs,
            extensions=my_extensions,
            writer=ppe_writer,
            iters_per_epoch=iters_per_epoch,
            stop_trigger=EarlyStoppingTrigger(
                monitor=f"fine-val/{self.monitor_metric}",
                patience=self.config.patience,
                mode="max" if self.monitor_metric != "rmse" else "min",
                max_trigger=(self.config.epochs, "epoch"),
            ),
        )
        manager.extend(
            ext.snapshot_object(
                self.model_wo_ddp,
                "model_epoch_{.updater.epoch:03d}",
            ),
            trigger=(100, "epoch"),
        )
        manager.extend(
            ext.snapshot(
                target=self.model_wo_ddp,
                filename="snapshot_epoch_{.updater.epoch:03d}",
                n_retains=self.config.n_retains,
                autoload=True,
            ),
            trigger=(100, "epoch"),
        )
        return manager

    @torch.no_grad()
    def evaluate(self, **batch):
        self.model.eval()
        with autocast(enabled=self.scaler is not None):
            loss, prediction = self.forward_fine(batch)
        # prediction = prediction["target"]
        target = batch["target"].numpy()

        report = {"fine-val/loss": loss.item()}
        if self.datamodule.task == "binary":
            label = prediction.cpu().numpy().argmax(1)
            prediction = torch.softmax(prediction, 1)[:, 1].cpu().numpy()
            score = {
                "acc": sklearn.metrics.accuracy_score(target, label),
                "auc": sklearn.metrics.roc_auc_score(target, prediction),
            }
        elif self.datamodule.task == "multiclass":
            prediction = prediction.cpu().numpy().argmax(1)
            score = {"acc": sklearn.metrics.accuracy_score(target, prediction.cpu().numpy())}
        else:
            assert self.datamodule.task == "regression"
            score = {
                "rmse": sklearn.metrics.mean_squared_error(target, prediction.cpu().numpy()) ** 0.5
                * self.datamodule.y_std
            }
        report.update({f"fine-val/{k}": s for k, s in score.items()})
        ppe.reporting.report(report)

    def cal_loss(self, pred, target):
        if self.datamodule.task != "regression":
            target = target.squeeze(-1).long()
        else:
            target = target.unsqueeze(1).float()
        return self.loss_fn(pred, target)

    def forward_fine(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError()

    def training(self):
        dataloader = self.datamodule.dataloader("fine", self.config.batch_size)
        manager = self.get_fine_manager(iters_per_epoch=len(dataloader))

        scheduler = self.get_scheduler()

        best_score = -np.inf if self.datamodule.task != "regression" else np.inf
        while not manager.stop_trigger:
            self.model.train()
            if scheduler is not None:
                scheduler.step(manager.epoch)
            for batch in dataloader:
                with manager.run_iteration():
                    self.optimizer.zero_grad()
                    with autocast(enabled=self.scaler is not None):
                        loss, _ = self.forward_fine(batch)

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    ppe.reporting.report({"fine-train/loss": loss.item()})
                    ppe.reporting.report({"lr/fine": self.optimizer.param_groups[0]["lr"]})

            best_conf = manager.get_extension("best_value").state_dict()
            if self.update_best_value(best_score, best_conf["_best_trigger"]["_best_value"]):
                best_score = best_conf["_best_trigger"]["_best_value"]
                torch.save(self.model_wo_ddp.state_dict(), f"{self.save_dir}/checkpoints_fine/best_model")
                with open(f"{self.save_dir}/checkpoints_fine/best.yaml", "w") as f:
                    yaml.dump(manager.get_extension("best_value").state_dict(), f)
            if os.path.exists(f"{self.save_dir}/checkpoints_fine/model_epoch_{manager.epoch:03d}"):
                os.remove(f"{self.save_dir}/checkpoints_fine/model_epoch_{manager.epoch:03d}")

        best_epoch = manager.get_extension("best_value").state_dict()["_best_epoch"]
        # logger.info(f"Best model is {best_epoch} epoch model.")
        logger.info(f"Load {best_epoch} epoch model.")
        self.model_wo_ddp.load_state_dict(torch.load(f"{self.save_dir}/checkpoints_fine/best_model"))

    @torch.no_grad()
    def get_pred_target_losses(self, mode: str):
        self.model.eval()
        prediction = []
        target = []
        losses = []
        for batch in self.datamodule.dataloader(mode, self.config.eval_batch_size):
            b_target = batch["target"].to(self.device)
            with autocast(enabled=self.scaler is not None):
                loss, b_pred = self.forward_fine(batch)
            prediction.append(b_pred)
            target.append(b_target)
            losses.append(loss.item())
        prediction = torch.cat(prediction).squeeze(1)
        target = torch.cat(target)
        losses = np.array(losses).mean()

        return prediction, target, losses

    def get_evaluate(self, mode="test"):
        prediction, target, losses = self.get_pred_target_losses(mode)
        target = target.cpu().numpy()

        if self.datamodule.task == "binary":
            label = prediction.cpu().numpy().argmax(1)
            prediction = torch.softmax(prediction, 1)[:, 1].cpu().numpy()
            score = {
                "ACC": sklearn.metrics.accuracy_score(target, label),
                "AUC": sklearn.metrics.roc_auc_score(target, prediction),
            }
        elif self.datamodule.task == "multiclass":
            prediction = prediction.cpu().numpy().argmax(1)
            score = {"ACC": sklearn.metrics.accuracy_score(target, prediction.cpu().numpy())}
        else:
            score = {
                "RMSE": sklearn.metrics.mean_squared_error(target, prediction.cpu().numpy()) ** 0.5
                * self.datamodule.y_std
            }

        score.update({"Loss": losses})
        return score

    def print_evaluate(self, mode="test"):
        score = self.get_evaluate(mode)
        save_json(score, "./")
        result = ""
        for k, v in score.items():
            self.writer.add_scalar(f"test/{k}", v, 0)
            result += f"{k}: {v:.4f}, "
        logger.info(f"Test ==> {result}")
        if not self.config.store:
            paths = [f"{self.save_dir}/checkpoints_fine/best_model"] + glob(
                f"{self.save_dir}/checkpoints_fine/snapshot_epoch_*"
            )
            for path in paths:
                if os.path.isfile(path):
                    os.remove(path)
