import logging
from copy import copy
from typing import Dict

import torch
from hydra.utils import to_absolute_path
from torch import Tensor

from model import FTTransTabClassifier

from ..base import BaseFineTrainer

logger = logging.getLogger(__name__)


class TransTabOODTrainer(BaseFineTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.model_config = config.model

        self.continuous_columns = copy(self.datamodule.continuous_columns)
        self.categorical_columns = copy(self.datamodule.categorical_columns)
        self.cat_cardinality_dict = copy(self.datamodule.cat_cardinality_dict)

        self.model = FTTransTabClassifier(
            cat_cardinality_dict=self.cat_cardinality_dict,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.continuous_columns,
            **self.model_config.encoder,
        )

        self.transfer_pre_model(self.train_config.pre_path)
        self.model_init()
        self.optimizer_init()

    def transfer_pre_model(self, path):
        state_dict = torch.load(
            to_absolute_path(path),
            map_location=torch.device("cpu"),
        )
        missing_keys, _ = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {missing_keys}")

    def forward_fine(self, data: Dict[str, Tensor]):
        cont = data["continuous"]
        cate = data["categorical"]
        target = data["target"].to(self.device)

        cont, cate = self.apply_device(cont, cate)
        logits = self.model(cont, cate)
        loss = self.cal_loss(logits, target)
        return loss, logits
