from copy import copy
from typing import Dict

from rtdl import FTTransformer
from torch import Tensor

from ..utils import column_shuffle
from .base import BaseFineTrainer


class FTTransTrainer(BaseFineTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model_config = config.model

        continuous_columns = len(copy(self.datamodule.continuous_columns))
        cat_cardinality = list(copy(self.datamodule.cat_cardinality_dict).values())
        d_out = self.datamodule.d_out
        if self.datamodule.task == "binary":
            d_out = 2

        self.model = FTTransformer.make_default(
            n_blocks=self.model_config.n_blocks,
            n_num_features=continuous_columns,
            cat_cardinalities=cat_cardinality,
            last_layer_query_idx=[-1],
            d_out=d_out,
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
