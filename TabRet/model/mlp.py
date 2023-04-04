import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# The following code is copied and modified from https://github.com/Yura52/tabular-dl-revisiting-models (MIT License)
# Original code: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/mlp.py
# Modified by: somaonishi
class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        categories: Optional[List[int]] = None,
        d_layers: List[int] = [256, 256],
        dropout: float = 0.1,
        d_embedding: int = 64,
    ) -> None:
        super().__init__()

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # print(f'{self.category_embeddings.weight.shape=}')

        self.layers = nn.ModuleList([nn.Linear(d_layers[i - 1] if i else d_in, x) for i, x in enumerate(d_layers)])
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def optimization_param_groups(self) -> List[Dict[str, Any]]:
        return [
            {"params": [v for k, v in self.named_parameters() if ".bias" not in k]},
            {
                "params": [v for k, v in self.named_parameters() if ".bias" in k],
                "weight_decay": 0.0,
            },
        ]

    def forward(self, x_num, x_cat):
        if x_cat is not None:
            x_cat = self.category_embeddings(x_cat + self.category_offsets[None])  # type: ignore
            x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)
        else:
            x = x_num

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        return x
