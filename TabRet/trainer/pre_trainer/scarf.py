from copy import copy
from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor

from model import SCARF, NT_Xent

from .base import BasePreTrainer


def scarf_augument(x: Dict[str, Tensor], mask_ratio, is_cate: bool = False):
    b = list(x.values())[0].shape[0]
    new_x = {}
    for key in x.keys():
        mask = torch.bernoulli(torch.ones_like(x[key]) * mask_ratio)
        idx = torch.randint(0, b, size=(b,))
        x_bar = x[key][idx].detach().clone()
        x_tilde = x[key] * (1 - mask) + x_bar * mask
        if is_cate:
            x_tilde = x_tilde.to(torch.long)
        new_x[key] = x_tilde
    return new_x


class SCARFPreTrainer(BasePreTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.train_config = config.pre_conf
        self.model_config = config.model

        continuous_columns = copy(self.datamodule.continuous_columns)
        cat_cardinality_dict = copy(self.datamodule.cat_cardinality_dict)
        self.model = SCARF.make(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            enc_transformer_config=self.model_config.encoder,
            projection_dim=self.model_config.projection_dim,
        )
        self.model_init()
        self.optimizer_init()
        world_size = dist.get_world_size() if self.config.multi_node else 1
        self.criterion = NT_Xent(
            batch_size=self.config.batch_size,
            temperature=self.train_config.temperature,
            world_size=world_size if self.train_config.all_gather else 1,
        )

        self.val_criterion = NT_Xent(
            batch_size=self.config.eval_batch_size,
            temperature=self.train_config.temperature,
            world_size=world_size if self.train_config.all_gather else 1,
        )

    def forward_pre(self, data: Dict[str, Tensor]):
        cont = data["continuous"]
        cate = data["categorical"]
        cont, cate = self.apply_device(cont, cate)
        if cont is not None:
            positive_cont = scarf_augument(cont, self.train_config.mask_ratio)
        else:
            positive_cont = None
        if cate is not None:
            positive_cate = scarf_augument(cate, self.train_config.mask_ratio, is_cate=True)
        else:
            positive_cate = None
        z0, z1 = self.model(cont, cate, positive_cont, positive_cate, dict(self.train_config.column_shuffle))

        if self.model.training:
            loss = self.criterion(z0, z1)
        else:
            loss = self.val_criterion(z0, z1)
        return loss
