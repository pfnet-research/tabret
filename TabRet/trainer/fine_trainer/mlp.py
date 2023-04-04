from copy import copy
from typing import Dict

from torch import Tensor

from model import MLP

from ..utils import column_shuffle
from .base import BaseFineTrainer


class MLPTrainer(BaseFineTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model_config = config.model

        continuous_columns = len(copy(self.datamodule.continuous_columns))
        cat_cardinality = list(copy(self.datamodule.cat_cardinality_dict).values())
        d_out = self.datamodule.d_out
        if self.datamodule.task == "binary":
            d_out = 2

        self.model = MLP(
            d_in=continuous_columns,
            categories=cat_cardinality if cat_cardinality != [] else None,
            d_out=d_out,
            **self.model_config,
        )
        self.model_init()
        self.optimizer_init()

    def forward_fine(self, data: Dict[str, Tensor]):
        cont = data["continuous"]
        cate = data["categorical"]
        target = data["target"].to(self.device)

        # if self.config.column_shuffle.ratio > 0:
        #     cont, cate, _ = column_shuffle(cont, cate, self.config.column_shuffle.ratio)

        cont, cate = self.apply_device_concat(cont, cate)
        logits = self.model(cont, cate)
        loss = self.cal_loss(logits, target)
        return loss, logits
