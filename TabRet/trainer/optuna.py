import logging
import os
import shutil
from copy import deepcopy
from glob import glob
from typing import List, Optional

import numpy as np
import optuna
import trainer as t
from trainer.fine_trainer import BaseFineTrainer

logger = logging.getLogger(__name__)


def get_mask_ratio(path_list: List[str]):
    mask_ratio_list = []
    for p in path_list:
        p_split = p.split("/")
        mask_ratio = float(p_split[-3])
        if mask_ratio not in mask_ratio_list:
            mask_ratio_list.append(mask_ratio)
    return mask_ratio_list


def get_opt_pre_path(trial: optuna.Trial, pre_path_list):
    if pre_path_list is None:
        return None
    p_split = pre_path_list[0].split("/")
    mask_ratio_list = get_mask_ratio(pre_path_list)
    mask_ratio = trial.suggest_categorical("pre_mask_ratio", mask_ratio_list)
    p_split[-3] = str(mask_ratio)
    pre_path = "/".join(p_split)
    return pre_path


def get_partitions(path_list: List[str]):
    partition__list = []
    for p in path_list:
        p_split = p.split("/")
        partition = int(p_split[-3])
        if partition not in partition__list:
            partition__list.append(partition)
    return partition__list


def get_transtab_pre_path(trial: optuna.Trial, pre_path_list: Optional[List] = None):
    if pre_path_list is None:
        return None
    p_split = pre_path_list[0].split("/")
    partition_list = get_partitions(pre_path_list)
    partition = trial.suggest_categorical("partition", partition_list)
    p_split[-3] = str(partition)
    pre_path = "/".join(p_split)
    return pre_path


def set_iid_ood_params(trial: optuna.Trial, pre_path_list, config):
    pre_path = get_opt_pre_path(trial, pre_path_list)
    config.fine_conf.pre_path = pre_path
    config.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)


def set_retokenizing_params(trial: optuna.Trial, pre_path_list, config):
    pre_path = get_opt_pre_path(trial, pre_path_list)
    config.fine_conf.pre_path = pre_path
    config.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    config.fine_conf.ret.lr = trial.suggest_float("lr_ret", 1e-4, 1e-1, log=True)
    # config.fine_conf.ret.mask_ratio = trial.suggest_float("mask_ratio", 0.01, 0.8)


def set_mlp_params(trial: optuna.Trial, config):
    config.model.d_layers = [trial.suggest_int("layer_size", 1, 512) for _ in range(trial.suggest_int("layers", 1, 8))]
    config.model.dropout = trial.suggest_float("dropout", 0, 0.5)
    config.model.d_embedding = trial.suggest_int("d_embedding", 64, 512)
    config.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)


def set_fttrans_params(trial: optuna.Trial, config):
    config.model.n_blocks = trial.suggest_int("n_blocks", 1, 6)
    config.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)


def set_scarf_params(trial: optuna.Trial, config):
    config.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)


def set_transtab_params(trial: optuna.Trial, pre_path_list, config):
    pre_path = get_transtab_pre_path(trial, pre_path_list)
    config.fine_conf.pre_path = pre_path
    config.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)


def set_opt_params(Trainer, trial: optuna.Trial, pre_path_list, config):
    if Trainer == t.TabRetIIDTrainer or Trainer == t.TabRetOODTrainer:
        set_iid_ood_params(trial, pre_path_list, config)
    elif Trainer == t.TabRetokenOODTrainer:
        set_retokenizing_params(trial, pre_path_list, config)
    elif Trainer == t.MLPTrainer:
        set_mlp_params(trial, config)
    elif Trainer == t.FTTransTrainer:
        set_fttrans_params(trial, config)
    elif Trainer == t.SCARFIIDTrainer or Trainer == t.SCARFOODTrainer:
        set_scarf_params(trial, config)
    elif Trainer == t.TransTabIIDTrainer or Trainer == t.TransTabOODTrainer:
        set_transtab_params(trial, pre_path_list, config)
    else:
        raise ValueError(f"Unexpected value: {Trainer}")


def delete_dir_and_file():
    tensorboard_files = glob("events.out.tfevents.*")
    checkpoint_dir = glob("checkpoints_*")
    for f in tensorboard_files:
        os.remove(f)
    for d in checkpoint_dir:
        shutil.rmtree(d)


def get_path_list(paths: str):
    return glob(paths)


class OptunaSearch:
    def __init__(self, Trainer, config) -> None:
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler("./optuna.log"))
        self.Trainer = Trainer
        self.config = config
        self.pre_paths = config.fine_conf.pre_path if "pre_path" in config.fine_conf else None
        self.is_first_trial = True
        self.best_score = -np.inf

    def first_trial(self, trainer: BaseFineTrainer):
        self.datamodule = deepcopy(trainer.datamodule)
        self.is_first_trial = False

    def objective(self, trial: optuna.Trial):
        delete_dir_and_file()

        config_cpy = deepcopy(self.config)
        set_opt_params(
            self.Trainer,
            trial,
            get_path_list(self.pre_paths) if self.pre_paths is not None else None,
            config_cpy,
        )

        if self.is_first_trial:
            t_obj: BaseFineTrainer = self.Trainer(config_cpy)
            self.first_trial(t_obj)
        else:
            t_obj: BaseFineTrainer = self.Trainer(config_cpy, datamodule=deepcopy(self.datamodule))

        t_obj.training()

        score = t_obj.get_evaluate("fval")

        if self.best_score < score["AUC"]:
            self.best_score = score["AUC"]
            t_obj.print_evaluate("test")

        del t_obj.datamodule, t_obj, config_cpy
        return score["AUC"]

    def run(self, use_storage=False):
        if use_storage:
            storage = optuna.storages.RDBStorage(
                url=os.environ["OPTUNA_STORAGE"],
                heartbeat_interval=60,
                grace_period=120,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
            )
        else:
            storage = None
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            load_if_exists=True,
        )
        study.optimize(self.objective, callbacks=[optuna.study.MaxTrialsCallback(100)])
        best_params = study.best_params
        logger.info(f"Best params: {best_params}")
        delete_dir_and_file()
