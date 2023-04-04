import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl.modules import _TokenInitialization
from torch import Tensor
from transtab.trainer_utils import get_parameter_names

from .ft_transformer import FeatureTokenizer, Transformer

logger = logging.getLogger(__name__)


class TabRet(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        feature_tokenizer: FeatureTokenizer,
        encoder: Transformer,
        decoder: Transformer,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        initialization: str = "uniform",
    ):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.keys = continuous_columns + list(cat_cardinality_dict.keys())

        self.feature_tokenizer = feature_tokenizer
        self.ft_norm = nn.LayerNorm(encoder_embed_dim)
        self.alignment_layer = nn.Linear(encoder_embed_dim, encoder_embed_dim, bias=True)
        self.encoder = encoder
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(Tensor(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.ParameterDict()
        if continuous_columns is not None:
            self.decoder_pos_embed.update(
                {key: nn.Parameter(Tensor(1, 1, decoder_embed_dim)) for key in continuous_columns}
            )
        if cat_cardinality_dict is not None:
            self.decoder_pos_embed.update(
                {key: nn.Parameter(Tensor(1, 1, decoder_embed_dim)) for key in cat_cardinality_dict.keys()}
            )
        self.decoder = decoder

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.projectors = nn.ModuleDict()
        if continuous_columns is not None:
            self.projectors.update({key: nn.Linear(decoder_embed_dim, 1) for key in continuous_columns})
        if cat_cardinality_dict is not None:
            self.projectors.update(
                nn.ModuleDict(
                    {key: nn.Linear(decoder_embed_dim, out_dim) for key, out_dim in cat_cardinality_dict.items()}
                )
            )

        # initialization
        self.initialization_ = _TokenInitialization(initialization)
        self.initialization_.apply(self.mask_token, decoder_embed_dim)
        for parameter in self.decoder_pos_embed.values():
            self.initialization_.apply(parameter, decoder_embed_dim)

    @classmethod
    def make(
        cls,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        enc_transformer_config,
        dec_transformer_config,
    ):
        feature_tokenizer = FeatureTokenizer(
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
            d_token=enc_transformer_config["d_token"],
        )

        encoder = Transformer(**enc_transformer_config)
        decoder = Transformer(**dec_transformer_config)

        return TabRet(
            encoder_embed_dim=enc_transformer_config["d_token"],
            decoder_embed_dim=dec_transformer_config["d_token"],
            feature_tokenizer=feature_tokenizer,
            encoder=encoder,
            decoder=decoder,
            continuous_columns=continuous_columns,
            cat_cardinality_dict=cat_cardinality_dict,
        )

    def optimization_param_groups(self):
        no_wd_names = ["feature_tokenizer", "normalization", "_norm", ".bias"]
        assert isinstance(getattr(self, no_wd_names[0], None), FeatureTokenizer)
        assert sum(1 for name, _ in self.named_modules() if no_wd_names[1] in name) == (
            len(self.encoder.blocks) + len(self.decoder.blocks)
        ) * 2 - int(
            "attention_normalization" not in self.encoder.blocks[0]
        ) - int(  # type: ignore
            "attention_normalization" not in self.decoder.blocks[0]
        )

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
        for p in self.alignment_layer.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder_embed.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_decoder(self):
        logger.info("Parameters in Decoder were unfrozen.")
        for p in self.decoder.parameters():
            p.requires_grad = True

    def freeze_parameters(self):
        logger.info("All parameters were frozen.")
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):
        logger.info("All parameters were unfrozen.")
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_mask_token(self):
        self.mask_token.requires_grad = True

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

        # add decoder pos embedding
        if continuous_columns is not None:
            for key in continuous_columns:
                if key in self.decoder_pos_embed.keys():
                    continue
                pos_embed_add = nn.Parameter(Tensor(1, 1, self.decoder_embed_dim))
                self.initialization_.apply(pos_embed_add, self.decoder_embed_dim)
                self.decoder_pos_embed.update({key: pos_embed_add})

                self.projectors.update({key: nn.Linear(self.decoder_embed_dim, 1)})

        if cat_cardinality_dict is not None:
            for key, cardinality in cat_cardinality_dict.items():
                if key in self.decoder_pos_embed.keys():
                    continue
                pos_embed_add = nn.Parameter(Tensor(1, 1, self.decoder_embed_dim))
                self.initialization_.apply(pos_embed_add, self.decoder_embed_dim)
                self.decoder_pos_embed.update({key: pos_embed_add})

                self.projectors.update({key: nn.Linear(self.decoder_embed_dim, cardinality)})

    def save_attention_map(self, x, keys=None):
        if keys is None:
            keys = self.keys

        import os

        os.makedirs("attention/")
        layer = list(range(len(self.encoder.blocks))) + ["all"]
        for i in layer:
            attention = self.encoder.get_attention(x, layer_idx=i)
            print(attention.sum(-1).mean())
            import pandas as pd

            df = pd.DataFrame(attention.mean(0).cpu().numpy(), index=keys, columns=keys)
            df.to_csv(f"./attention/attention_{i}.csv")
        exit()

    def random_masking(
        self,
        x,
        mask_ratio: Union[int, float],
        shuffle_idx_shift: Tensor = None,
    ):
        N, L, D = x.shape

        if mask_ratio >= 1:
            len_keep = L - mask_ratio
        else:
            len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if shuffle_idx_shift is not None:
            assert len_keep - len(shuffle_idx_shift) > 0
            noise[:, shuffle_idx_shift[:, 0]] = 0

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def custom_masking(self, x, mask_idx):
        N, L, D = x.shape

        mask = torch.zeros(N, L, device=x.device)
        mask[:, mask_idx] = 1

        x_masked = x[:, (1 - mask[0]).bool()]

        ids_restore = torch.ones(N, L, device=x.device) * torch.arange(L, device=x.device)
        return x_masked, mask, ids_restore.to(torch.int64)

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
        mask_ratio: Union[float, int, List[int]],
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.ft_norm(x)
        x = self.alignment_layer(x)
        # self.save_attention_map(x, list(x_num.keys()) + list(x_cat.keys()))

        if col_shuffle is not None and col_shuffle["ratio"] > 0:
            x, shuffle_idx_shift = self.column_shuffle(
                x,
                col_shuffle["ratio"],
                mask=mask_ratio if type(mask_ratio) == list else None,
            )
        else:
            shuffle_idx_shift = None

        # masking: length -> length * mask_ratio
        if type(mask_ratio) != list:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, shuffle_idx_shift)
        else:
            x, mask, ids_restore = self.custom_masking(x, mask_ratio)

        x = self.encoder(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore, shuffle_idx_shift

    def forward_decoder(
        self,
        x,
        ids_restore,
        keys,
        shuffle_idx_shift: Optional[Tensor] = None,
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        # embed tokens
        x = self.decoder_embed(x)
        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_token], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if shuffle_idx_shift is not None:
            _, _, d = x.shape
            for idx, shift in shuffle_idx_shift:
                if col_shuffle["mode"] == "concat":
                    x[:, idx, : d // 2] = x[:, idx, : d // 2].roll(-shift.item(), 0)
                elif col_shuffle["mode"] == "unshuffle":
                    x[:, idx] = x[:, idx].roll(-shift.item(), 0)
                elif col_shuffle["mode"] == "shuffle":
                    break

        for i, key in enumerate(keys):
            x[:, i] = x[:, i] + self.decoder_pos_embed[key]

        x = self.decoder(x)
        x = self.decoder_norm(x)

        x_col = {}
        for i, key in enumerate(keys):
            x_col[key] = self.projectors[key](x[:, i])
        return x_col

    def forward_loss(self, x_num, x_cat, pred, mask):
        """
        x_num: Dict[L_num, N]
        x_cat: Dict[L_cat, N]
        preds Prediction list for each attribute
        mask: [N, L], 0 is keep, 1 is remove
        """
        if x_num is not None:
            n_num = len(x_num)
        else:
            n_num = 0
            x_num = {}
        if x_cat is None:
            x_cat = {}

        all_loss = 0
        for i, (key, x) in enumerate(x_num.items()):
            loss = F.mse_loss(pred[key].squeeze(), x, reduction="none")
            all_loss += (loss * mask[:, i]).sum()

        for i, (key, x) in enumerate(x_cat.items()):
            loss = F.cross_entropy(pred[key], x, reduction="none")
            all_loss += (loss * mask[:, i + n_num]).sum()

        all_loss /= mask.sum()
        return all_loss

    def forward(
        self,
        x_num: Optional[Dict[str, Tensor]],
        x_cat: Optional[Dict[str, Tensor]],
        mask_ratio: Union[float, List[int]],
        col_shuffle: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        keys = []
        if x_num is not None:
            keys += list(x_num.keys())
        if x_cat is not None:
            keys += list(x_cat.keys())
        x, mask, ids_restore, shuffle_idx_shift = self.forward_encoder(x_num, x_cat, mask_ratio, col_shuffle)
        preds = self.forward_decoder(x, ids_restore, keys, shuffle_idx_shift, col_shuffle)
        loss = self.forward_loss(x_num, x_cat, preds, mask)
        return loss, preds, mask


if __name__ == "__main__":
    import rtdl

    x_num = {f"num_{i}": torch.randn(4) for i in range(3)}
    x_cat = {f"cate_{i}": torch.tensor([1 for _ in range(4)]) for i in range(1)}
    # x_cat = None
    # x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
    e_conf = rtdl.FTTransformer.get_default_transformer_config(n_blocks=3)
    d_conf = rtdl.FTTransformer.get_default_transformer_config(n_blocks=1)
    mae = TabRet.make(
        [f"num_{i}" for i in range(3)],
        {f"cate_{i}": 3 for i in range(2)},
        e_conf,
        d_conf,
    )
    print(len(mae.optimization_param_groups()[1]["params"]))
    # print(mae)
    for pred in mae(x_num, x_cat, 0.001):
        print(pred)
