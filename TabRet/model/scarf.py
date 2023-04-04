import logging
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from rtdl.modules import _TokenInitialization
from torch import Tensor
from transtab.trainer_utils import get_parameter_names

from .tabret.ft_transformer import FeatureTokenizer, Transformer

logger = logging.getLogger(__name__)


# The following code is copied and modified from https://github.com/Spijkervet/SimCLR/ (MIT License)
# Original code: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
# Modified by: somaonishi
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


# The following code is copied and modified from https://github.com/Spijkervet/SimCLR/ (MIT License)
# Original code: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
# Modified by: somaonishi
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class SCARF(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        feature_tokenizer: FeatureTokenizer,
        encoder: Transformer,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        projection_dim: int = 128,
        initialization: str = "uniform",
    ):
        super().__init__()
        self.keys = continuous_columns + list(cat_cardinality_dict.keys())

        self.feature_tokenizer = feature_tokenizer

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder = encoder
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(encoder_embed_dim, encoder_embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(encoder_embed_dim, projection_dim, bias=False),
        )

        # initialization
        self.initialization_ = _TokenInitialization(initialization)
        self.initialization_.apply(self.cls_token, encoder_embed_dim)

    @classmethod
    def make(
        cls,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        enc_transformer_config,
        projection_dim=128,
    ):
        feature_tokenizer = FeatureTokenizer(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            d_token=enc_transformer_config["d_token"],
        )

        encoder = Transformer(**enc_transformer_config)

        return SCARF(
            encoder_embed_dim=enc_transformer_config["d_token"],
            feature_tokenizer=feature_tokenizer,
            encoder=encoder,
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            projection_dim=projection_dim,
        )

    def optimization_param_groups(self):
        no_wd_names = ["feature_tokenizer", "normalization", ".bias"]
        assert isinstance(getattr(self, no_wd_names[0], None), FeatureTokenizer)

        non_decay_parameters = get_parameter_names(self, [nn.LayerNorm, FeatureTokenizer])
        decay_parameters = [name for name in non_decay_parameters if ".bias" not in name]
        return [
            {"params": [p for n, p in self.named_parameters() if n in decay_parameters]},
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

    def freeze_parameters_wo_specific_columns(self, columns: List[str]):
        logger.info("Parameters w/o specific columns were frozen.")
        for name, p in self.named_parameters():
            name_split = name.split(".")
            if len(name_split) > 1 and (name_split[-1] in columns or name_split[-2] in columns):
                p.requires_grad = True
                continue
            p.requires_grad = False

    def freeze_transfomer(self):
        logger.info("Parameters in Transfomer were frozen.")
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_parameters(self):
        logger.info("All parameters were frozen.")
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):
        logger.info("All parameters were unfrozen.")
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_cls_token(self):
        self.cls.requires_grad = True

    def show_trainable_parameter(self):
        trainable_list = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_list.append(name)
        trainable = ", ".join(trainable_list)
        logger.info(f"Trainable parameters: {trainable}")

    def show_frozen_parameter(self):
        frozen_list = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                frozen_list.append(name)
        frozen = ", ".join(frozen_list)
        logger.info(f"Frozen parameters: {frozen}")

    def add_attribute(
        self,
        continuous_columns: Optional[List[str]] = None,
        cat_cardinality_dict: Optional[Dict[str, int]] = None,
    ):
        assert (
            continuous_columns is not None or cat_cardinality_dict is not None
        ), "At least one of n_num and cardinalities must be presented"
        self.feature_tokenizer.add_attribute(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
        )

    def column_shuffle(self, x, column_shuffle_ratio, mask: List[int] = None):
        N, L, _ = x.shape
        num_noise = int(L * column_shuffle_ratio)

        noise = torch.rand(L, device=x.device)  # noise in [0, 1]
        if mask is not None:
            assert num_noise + len(mask) < L
            noise[mask] = 1
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove

        shuffle_idx_shift = torch.cat([ids_shuffle[:, None], torch.randint(N, (L, 1), device=x.device)], dim=1)[
            :num_noise
        ]

        for idx, shift in shuffle_idx_shift:
            x[:, idx] = x[:, idx].roll(shift.item(), 0)
        return x, shuffle_idx_shift

    def forward_encoder(
        self,
        x_num,
        x_cat,
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        x = self.feature_tokenizer(x_num, x_cat)

        if col_shuffle is not None and col_shuffle["ratio"] > 0:
            x, _ = self.column_shuffle(x, col_shuffle["ratio"])

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward_decoder(self, x):
        x = x[:, 0]  # cls token

        z = self.projector(x)
        return z

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        positive_num: Optional[Dict[str, Tensor]],
        positive_cat: Optional[Dict[str, Tensor]],
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        x0 = self.forward_encoder(x_num, x_cat)
        z0 = self.forward_decoder(x0)

        x1 = self.forward_encoder(positive_num, positive_cat, col_shuffle)
        z1 = self.forward_decoder(x1)
        return z0, z1


class SCARFClassifier(nn.Module):
    class Classifier(nn.Module):
        def __init__(self, input_dim, output_dim) -> None:
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x) -> Tensor:
            x = x[:, 0]
            logits = self.fc(x)
            return logits

    def __init__(self, encoder: SCARF, output_dim: int) -> None:
        super().__init__()

        self.encoder = encoder

        self.classifier = SCARFClassifier.Classifier(self.encoder.cls_token.shape[-1], output_dim)
        if output_dim > 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def optimization_param_groups(self):
        no_wd_names = ["feature_tokenizer", "normalization", ".bias"]
        assert isinstance(getattr(self.encoder, no_wd_names[0], None), FeatureTokenizer)

        def needs_wd(name):
            return all(x not in name for x in no_wd_names)

        return [
            {"params": [v for k, v in self.named_parameters() if needs_wd(k)]},
            {
                "params": [v for k, v in self.named_parameters() if not needs_wd(k)],
                "weight_decay": 0.0,
            },
        ]

    def show_trainable_parameter(self):
        trainable_list = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_list.append(name)
        trainable = ", ".join(trainable_list)
        logger.info(f"Trainable parameters: {trainable}")

    def forward(self, x_num, x_cat):
        x = self.encoder.forward_encoder(x_num, x_cat)
        logit = self.classifier(x)
        return logit
