from copy import copy
from typing import Dict

from torch import Tensor

from model import FTTransTabForCL

from .base import BasePreTrainer


class TrasTabPreTrainer(BasePreTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.train_config = config.pre_conf
        self.model_config = config.model

        continuous_columns = copy(self.datamodule.continuous_columns)
        categorical_columns = copy(self.datamodule.categorical_columns)
        cat_cardinality_dict = copy(self.datamodule.cat_cardinality_dict)
        self.model = FTTransTabForCL(
            cat_cardinality_dict=cat_cardinality_dict,
            categorical_columns=categorical_columns,
            numerical_columns=continuous_columns,
            overlap_ratio=self.train_config.overlap_ratio,
            num_partition=self.train_config.num_partition,
            supervised=False,
            device=self.device,
            multi_node=self.config.multi_node and self.train_config.all_gather,
            **self.model_config.encoder,
        )

        self.model_init()
        self.optimizer_init()

    def forward_pre(self, data: Dict[str, Tensor]) -> Tensor:
        cont = data["continuous"]
        cate = data["categorical"]
        cont, cate = self.apply_device(cont, cate)
        if self.model.training:
            _, loss = self.model(cont, cate, dict(self.train_config.column_shuffle))
        else:
            _, loss = self.model(cont, cate)
        return loss
