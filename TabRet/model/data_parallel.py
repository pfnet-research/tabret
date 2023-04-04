from typing import Dict, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from .tabret.tabret import TabRet


class DataParallel(nn.Module):
    def __init__(self, model: TabRet):
        super().__init__()
        self.net = nn.DataParallel(model)

    def freeze_parameters_wo_specific_columns(self, columns: List[str]):
        self.net.module.freeze_parameters_wo_specific_columns(columns)

    def freeze_parameters(self):
        self.net.module.freeze_parameters()

    def unfreeze_parameters(self):
        self.net.module.unfreeze_parameters()

    def add_attribute(
        self,
        continuous_columns: Optional[List[str]] = None,
        cat_cardinality_dict: Optional[Dict[str, int]] = None,
    ):
        self.net.module.add_attribute(continuous_columns, cat_cardinality_dict)

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        mask_ratio: Union[float, List[int]],
    ):
        loss, preds, mask = self.net(x_num, x_cat, mask_ratio)
        return loss.mean(), preds, mask
