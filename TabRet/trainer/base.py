import logging
from typing import List, Optional

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as ext
import torch
import zero
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from data import get_datamodule

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, datamodule=None):
        self.config = config
        self.is_root = config.world_rank == 0
        self.device = torch.device(f"cuda:{config.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if self.device != torch.device("cpu"):
            torch.cuda.set_device(self.device)
        zero.improve_reproducibility(config.seed)
        torch.backends.cudnn.benchmark = True

        if datamodule is None:
            self.datamodule = get_datamodule(config)
        else:
            self.datamodule = datamodule

        self.save_dir = "."
        if self.is_root:
            self.writer = SummaryWriter(log_dir="./")

        if self.config.mixed_fp16:
            if self.is_root:
                logger.info("Initialize automatic mixed fp16 training")
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def model_init(self):
        self.model.to(self.device)

        self.model_wo_ddp = self.model
        if self.config.multi_node:
            if self.is_root:
                logger.info("Setting up distributed data parallel model")
            self.model = DistributedDataParallel(self.model, device_ids=[self.config.local_rank])
            self.model_wo_ddp = self.model.module

    def optimizer_init(self, base_lr=None):
        if base_lr is None:
            base_lr = self.config.lr
        lr = base_lr * self.config.batch_size / 256
        if self.config.multi_node:
            lr *= self.config.world_size

        if self.is_root:
            logger.info(f"Set learning rate: {lr}")

        self.optimizer = AdamW(self.model_wo_ddp.optimization_param_groups(), lr=lr, weight_decay=1e-5)

    def apply_device(self, cont, cate):
        if len(cont) != 0:
            for key in cont.keys():
                cont[key] = cont[key].to(self.device)
        else:
            cont = None
        if len(cate) != 0:
            for key in cate.keys():
                cate[key] = cate[key].to(self.device)
        else:
            cate = None
        return cont, cate

    def apply_device_concat(self, cont, cate):
        if len(cont) != 0:
            cont_values = []
            for c in cont.values():
                cont_values.append(c.unsqueeze(1))
            cont = torch.cat(cont_values, dim=1).to(self.device)
        else:
            cont = None
        if len(cate) != 0:
            cate_values = []
            for c in cate.values():
                cate_values.append(c.unsqueeze(1))
            cate = torch.cat(cate_values, dim=1).to(self.device)
        else:
            cate = None
        return cont, cate

    def get_default_extensions(
        self,
        evaluator: str,
        eval_func,
        best_value: Optional[ext.BestValue],
        logs: List[str],
    ):
        my_extensions = [
            ext.Evaluator(
                self.datamodule.dataloader(
                    evaluator,
                    self.config.eval_batch_size,
                    drop_last=self.config.drop_last,
                ),
                self.model,
                eval_func=lambda **batch: eval_func(**batch),
                progress_bar=self.is_root,
            ),
        ]
        if self.is_root:

            @ppe.training.make_extension(trigger=(1, "epoch"))
            def tensorboard_writer(manager):
                m = manager.observation
                for log in logs:
                    self.writer.add_scalar(log, m[log], manager.epoch)

            my_extensions += [
                ext.LogReport(),
                ext.ProgressBar(),
                best_value,
                ext.observe_lr(optimizer=self.optimizer),
                ext.PrintReport(["epoch", "iteration"] + logs),
                tensorboard_writer,
            ]
        return my_extensions

    def get_scheduler(self):
        if self.config.scheduler.name is None:
            logger.info("No scheduler.")
            return None

        logger.info(f"Scheduler: {self.config.scheduler.name}")
        if self.config.scheduler.name == "Cosine":
            scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=self.config.epochs,
                **self.config.scheduler.param,
            )
        return scheduler
