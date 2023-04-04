import logging
from copy import deepcopy

import torch
from hydra.utils import to_absolute_path
from model import TabRet

logger = logging.getLogger(__name__)


def load_pre_model(model, path):
    logger.info(f"Loading the model in path {path}.")
    model.cpu()
    state_dict = torch.load(
        to_absolute_path(path),
        map_location=torch.device("cpu"),
    )

    if "encoder_embed.weight" in state_dict.keys() and isinstance(model, TabRet):
        state_dict["alignment_layer.weight"] = deepcopy(state_dict["encoder_embed.weight"])
        state_dict["alignment_layer.bias"] = deepcopy(state_dict["encoder_embed.bias"])
        del state_dict["encoder_embed.weight"], state_dict["encoder_embed.bias"]

    model.load_state_dict(state_dict)
    logger.info("Pre-trained model is successfully loaded.")


def load_wo_feature_embed(model, path):
    logger.info(f"Loading the model in path {model}.")
    model.cpu()
    state_dict = torch.load(
        to_absolute_path(path),
        map_location=torch.device("cpu"),
    )
    state_dict_ft_norm = {}
    state_dict_enc_emb = {}
    state_dict_al_norm = {}
    state_dict_enc = {}
    state_dict_enc_norm = {}
    state_dict_dec = {}
    state_dict_dec_emb = {}
    state_dict_dec_norm = {}
    for name, para in state_dict.items():
        if name.split(".")[0] == "ft_norm":
            state_dict_ft_norm[name.replace("ft_norm.", "")] = para
        if name.split(".")[0] == "encoder_embed":
            state_dict_enc_emb[name.replace("encoder_embed.", "")] = para
        if name.split(".")[0] == "al_norm":
            state_dict_al_norm[name.replace("al_norm.", "")] = para
        if name.split(".")[0] == "encoder":
            if name.split(".")[1] == "transformer":
                state_dict_enc[name.replace("encoder.transformer.", "")] = para
            elif name.split(".")[1] != "feature_tokenizer":
                state_dict_enc[name.replace("encoder.", "")] = para
        if name.split(".")[0] == "decoder":
            state_dict_dec[name.replace("decoder.", "")] = para
        if name.split(".")[0] == "encoder_norm":
            state_dict_enc_norm[name.replace("encoder_norm.", "")] = para
        if name.split(".")[0] == "decoder_embed":
            state_dict_dec_emb[name.replace("decoder_embed.", "")] = para
        if name.split(".")[0] == "decoder_norm":
            state_dict_dec_norm[name.replace("decoder_norm.", "")] = para

    model.ft_norm.load_state_dict(state_dict_ft_norm)
    model.alignment_layer.load_state_dict(state_dict_enc_emb)
    model.model_wo_ddp.al_norm.load_state_dict(state_dict_al_norm)
    model.encoder.load_state_dict(state_dict_enc)
    model.encoder_norm.load_state_dict(state_dict_enc_norm)
    model.decoder.load_state_dict(state_dict_dec)
    model.decoder_embed.load_state_dict(state_dict_dec_emb)
    model.decoder_norm.load_state_dict(state_dict_dec_norm)
    with torch.no_grad():
        model.mask_token.copy_(state_dict["mask_token"])


def load_except_decoder(model, path):
    logger.info(f"Loading the model in path {path}.")
    model.cpu()
    state_dict = torch.load(
        to_absolute_path(path),
        map_location=torch.device("cpu"),
    )

    if "encoder_embed.weight" in state_dict.keys() and isinstance(model, TabRet):
        state_dict["alignment_layer.weight"] = deepcopy(state_dict["encoder_embed.weight"])
        state_dict["alignment_layer.bias"] = deepcopy(state_dict["encoder_embed.bias"])
        del state_dict["encoder_embed.weight"], state_dict["encoder_embed.bias"]

    state_dict_dec = model.decoder.state_dict()
    model.load_state_dict(state_dict)
    model.decoder.load_state_dict(state_dict_dec)
