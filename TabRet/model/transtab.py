import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from transtab import TransTabClassifier, TransTabForCL, TransTabModel
from transtab.trainer_utils import get_parameter_names

from .tabret.ft_transformer import FeatureTokenizer

logger = logging.getLogger(__name__)


class _TransTabModel(TransTabModel):
    def optimization_param_groups(self):
        non_decay_parameters = get_parameter_names(self, [nn.LayerNorm, FeatureTokenizer])
        decay_parameters = [name for name in non_decay_parameters if ".bias" not in name]
        return [
            {"params": [p for n, p in self.named_parameters() if n in decay_parameters]},
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]


class TransTabLinearClassifier(nn.Module):
    def __init__(self,
        num_class,
        hidden_dim=128) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:,0,:] # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits

class FTTransTabClassifier(TransTabClassifier, _TransTabModel):
    def __init__(
        self,
        cat_cardinality_dict=None,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation="relu",
        device="cuda:0",
        **kwargs
    ) -> None:
        super().__init__(
            categorical_columns,
            numerical_columns,
            binary_columns,
            feature_extractor,
            num_class,
            hidden_dim,
            num_layer,
            num_attention_head,
            hidden_dropout_prob,
            ffn_dim,
            activation,
            device,
            **kwargs
        )

        self.input_encoder = FeatureTokenizer(
            numerical_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            d_token=hidden_dim,
        )
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)

    def freeze_parameters_wo_specific_columns(self, columns: List[str]):
        logger.info("Parameters w/o specific columns were frozen.")
        for name, p in self.named_parameters():
            name_split = name.split(".")
            if len(name_split) > 1 and (name_split[-1] in columns or name_split[-2] in columns):
                p.requires_grad = True
                continue
            p.requires_grad = False

    def add_attribute(
        self,
        continuous_columns: Optional[List[str]] = None,
        cat_cardinality_dict: Optional[Dict[str, int]] = None,
    ):
        assert (
            continuous_columns is not None or cat_cardinality_dict is not None
        ), "At least one of n_num and cardinalities must be presented"
        self.input_encoder.add_attribute(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
        )

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
    ):
        x = self.input_encoder(x_num, x_cat)
        b, n, _ = x.shape
        x = {"embedding": x, "attention_mask": torch.ones(b, n, device=x.device)}
        x = self.cls_token(**x)

        x = self.encoder(**x)

        logits = self.clf(x)
        return logits


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class FTTransTabForCL(TransTabForCL, _TransTabModel):
    def __init__(
        self,
        cat_cardinality_dict=None,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.1,
        num_partition=2,
        supervised=False,
        temperature=10,
        base_temperature=10,
        activation="relu",
        device="cuda:0",
        multi_node=False,
        **kwargs
    ) -> None:
        super().__init__(
            categorical_columns,
            numerical_columns,
            binary_columns,
            feature_extractor,
            hidden_dim,
            num_layer,
            num_attention_head,
            hidden_dropout_prob,
            ffn_dim,
            projection_dim,
            overlap_ratio,
            num_partition,
            supervised,
            temperature,
            base_temperature,
            activation,
            device,
            **kwargs
        )

        self.input_encoder = FeatureTokenizer(
            numerical_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            d_token=hidden_dim,
        )

        self.multi_node = multi_node

    def column_shuffle(self, x, column_shuffle_ratio):
        N, L, _ = x.shape
        num_noise = int(L * column_shuffle_ratio)

        noise = torch.rand(L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove

        shuffle_idx_shift = torch.cat([ids_shuffle[:, None], torch.randint(N, (L, 1), device=x.device)], dim=1)[
            :num_noise
        ]

        for idx, shift in shuffle_idx_shift:
            x[:, idx] = x[:, idx].roll(shift.item(), 0)
        return x

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        keys = []
        if x_num is not None:
            keys += list(x_num.keys())
        if x_cat is not None:
            keys += list(x_cat.keys())

        sub_x_list = self._build_positive_pairs(keys, x_num, x_cat, self.num_partition)

        # do positive sampling
        feat_x_list = []
        for sub_x_num, sub_x_cat in sub_x_list:
            x = self.input_encoder(sub_x_num, sub_x_cat)
            b, n, _ = x.shape

            if col_shuffle is not None and col_shuffle["ratio"] > 0:
                x = self.column_shuffle(x, col_shuffle["ratio"])

            x = {"embedding": x, "attention_mask": torch.ones(b, n, device=x.device)}
            x = self.cls_token(**x)

            x = self.encoder(**x)

            feat_x_proj = x[:, 0, :]  # take cls embedding
            feat_x_proj = self.projection_head(feat_x_proj)
            if self.multi_node:
                feat_x_proj = torch.cat(GatherLayer.apply(feat_x_proj), dim=0)
            feat_x_list.append(feat_x_proj)
        feat_x_multiview = torch.stack(feat_x_list, axis=1)  # bs, n_view, emb_dim
        # compute cl loss (multi-view InfoNCE loss)
        loss = self.self_supervised_contrastive_loss(feat_x_multiview)
        return None, loss

    def self_supervised_contrastive_loss(self, features):
        """Compute the self-supervised VPCL loss.

        Parameters
        ----------
        features: torch.Tensor
            the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.

        Returns
        -------
        loss: torch.Tensor
            the computed self-supervised VPCL loss.
        """
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=features.device).view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        # print(F.one_hot(labels.squeeze(), batch_size))

        contrast_count = features.shape[1]
        # [[0,1],[2,3]] -> [0,2,1,3]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def _build_positive_pairs(
        self,
        x_cols: List[str],
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        num_partition,
    ):
        sub_col_list = np.array_split(np.array(x_cols), num_partition)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_cols in enumerate(sub_col_list):
            if overlap > 0 and i < num_partition - 1:
                sub_cols = np.concatenate([sub_cols, sub_col_list[i + 1][:overlap]])
            elif overlap > 0 and i == num_partition - 1:
                sub_cols = np.concatenate([sub_cols, sub_col_list[i - 1][-overlap:]])
            sub_x_num_dict = {}
            sub_x_cat_dict = {}
            for col in sub_cols:
                if col in x_num.keys():
                    sub_x_num_dict[col] = x_num[col].detach().clone()
                else:
                    sub_x_cat_dict[col] = x_cat[col].detach().clone()
            sub_x_num_dict = sub_x_num_dict if sub_x_num_dict != {} else None
            sub_x_cat_dict = sub_x_cat_dict if sub_x_cat_dict != {} else None
            sub_x_list.append((sub_x_num_dict, sub_x_cat_dict))
        return sub_x_list
