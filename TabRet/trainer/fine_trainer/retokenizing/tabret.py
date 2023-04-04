import logging
from copy import copy
from typing import Dict

from torch import Tensor

from model import TabRet, TabRetClassifier

from ..base import BaseFineTrainer
from .retokenizing import Retokenizing

logger = logging.getLogger(__name__)


class TabRetokenOODTrainer(BaseFineTrainer, Retokenizing):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.model_config = config.model

        self.has_pre = hasattr(self.datamodule, "pre_continuous_columns")
        if self.has_pre:
            continuous_columns = copy(self.datamodule.pre_continuous_columns)
            cat_cardinality_dict = copy(self.datamodule.pre_cat_cardinality_dict)
        else:
            continuous_columns = copy(self.datamodule.continuous_columns)
            cat_cardinality_dict = copy(self.datamodule.cat_cardinality_dict)
            assert self.train_config.pre_path is None, "All params in pre-trained model cannot be loaded."
            assert self.train_config.trans_path is not None, "In this case, this option must be specified."

        self.model = TabRet.make(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            enc_transformer_config=self.model_config.encoder,
            dec_transformer_config=self.model_config.decoder,
        )

        self.load_pretraind_model()

    def training(self):
        self.retokenizing_init()
        self.retokenizing()

        self.fine_tune_init()
        super().training()

    def fine_tune_init(self):
        self.model.freeze_parameters()
        output_dim = self.datamodule.d_out
        if self.datamodule.task == "binary":
            output_dim += 1
        self.model = TabRetClassifier(self.model, output_dim)
        self.model.show_trainable_parameter()

        self.model_init()
        self.optimizer_init()

    def forward_fine(self, data: Dict[str, Tensor]):
        cont = data["continuous"]
        cate = data["categorical"]
        target = data["target"].to(self.device)

        cont, cate = self.apply_device(cont, cate)
        logits = self.model(cont, cate)
        loss = self.cal_loss(logits, target)
        return loss, logits
