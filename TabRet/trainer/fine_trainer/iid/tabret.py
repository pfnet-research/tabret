import logging
from copy import copy
from typing import Dict

from torch import Tensor

from model import TabRet, TabRetClassifier

from ..base import BaseFineTrainer

logger = logging.getLogger(__name__)


class TabRetIIDTrainer(BaseFineTrainer):
    def __init__(self, config, datamodule=None):
        super().__init__(config, datamodule)
        self.model_config = config.model

        continuous_columns = copy(self.datamodule.continuous_columns)
        cat_cardinality_dict = copy(self.datamodule.cat_cardinality_dict)
        self.model = TabRet.make(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            enc_transformer_config=self.model_config.encoder,
            dec_transformer_config=self.model_config.decoder,
        )

        self.load_pretraind_model()
        self.fine_tune_init()
        self.model_init()
        self.optimizer_init()

    def fine_tune_init(self):
        if self.train_config.para_freeze:
            self.model.freeze_parameters()
        elif self.train_config.trans_freeze:
            self.model.freeze_transfomer()

        if not self.train_config.mask_token_freeze:
            self.model.unfreeze_mask_token()

        if self.train_config.except_decoder:
            self.model.unfreeze_decoder()

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
