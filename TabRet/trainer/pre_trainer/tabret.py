from copy import copy
from typing import Dict

from torch import Tensor

from model import TabRet

from .base import BasePreTrainer


class TabRetPreTrainer(BasePreTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.train_config = config.pre_conf
        self.model_config = config.model

        continuous_columns = copy(self.datamodule.continuous_columns)
        cat_cardinality_dict = copy(self.datamodule.cat_cardinality_dict)
        self.model = TabRet.make(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            enc_transformer_config=self.model_config.encoder,
            dec_transformer_config=self.model_config.decoder,
        )
        self.model_init()
        self.optimizer_init()

    def forward_pre(self, data: Dict[str, Tensor]) -> Tensor:
        cont = data["continuous"]
        cate = data["categorical"]
        cont, cate = self.apply_device(cont, cate)
        if self.model.training:
            loss, _, _ = self.model(
                cont,
                cate,
                self.train_config.mask_ratio,
                dict(self.train_config.column_shuffle),
            )
        else:
            loss, _, _ = self.model(
                cont,
                cate,
                self.train_config.mask_ratio,
            )
        return loss
