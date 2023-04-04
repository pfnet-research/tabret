import logging
from typing import Dict, Optional, Sequence, Union

import data.datasets as datasets
import numpy as np
import pandas as pd
import torch
from data.datasets.tabular_dataframe import TabularDataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# The following code is copied and modified from https://github.com/pfnet-research/deep-table (MIT License)
# Original code: https://github.com/pfnet-research/deep-table/blob/master/deep_table/data/data_module.py
# Modified by: somaonishi
class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str = "binary",
        continuous_columns: Optional[Sequence[str]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        target: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """
        Args:
            data (pandas.DataFrame): DataFrame.
            task (str): One of "binary", "multiclass", "regression".
                Defaults to "binary".
            continuous_cols (sequence of str, optional): Sequence of names of
                continuous features (columns). Defaults to None.
            categorical_cols (sequence of str, optional): Sequence of names of
                categorical features (columns). Defaults to None.
            target (str, optional): If None, `np.zeros` is set as target.
                Defaults to None.
        """
        super().__init__()
        self.data = data
        self.task = task
        self.num = data.shape[0]
        self.continuous_columns = continuous_columns if continuous_columns else []
        self.categorical_columns = categorical_columns if categorical_columns else []

        if target:
            self.target = data[target].values
            if isinstance(target, str):
                self.target = self.target.reshape(-1, 1)
        else:
            self.target = np.zeros((self.num, 1))

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            dict[str, Tensor]:
                The returned dict has the keys {"target", "continuous", "categorical"}
                and its values. If no continuous/categorical features, the returned value is `[]`.
        """
        x = {
            "continuous": {key: torch.tensor(self.data[key].values[idx]).float() for key in self.continuous_columns}
            if self.continuous_columns
            else {},
            "categorical": {key: torch.tensor(self.data[key].values[idx]).long() for key in self.categorical_columns}
            if self.categorical_columns
            else {},
        }
        if self.task == "multiclass":
            x["target"] = torch.LongTensor(self.target[idx])
        elif self.task in ["binary", "regression"]:
            x["target"] = torch.tensor(self.target[idx])
        else:
            raise ValueError(f"task: {self.task} must be 'multiclass' or 'binary' or 'regression'")
        return x


# The following code is copied and modified from https://github.com/pfnet-research/deep-table (MIT License)
# Original code: https://github.com/pfnet-research/deep-table/blob/master/deep_table/data/data_module.py
# Modified by: somaonishi
class TabularDatamodule:
    def __init__(
        self,
        dataset: TabularDataFrame,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: int = 3,
        multi_node: bool = False,
        seed: int = 42,
    ) -> None:
        # self.dataset = dataset
        dataframes = dataset.processed_dataframes(test_size=0.1, random_state=seed)
        logger.info("Conversion from raw data to pandas has been successfully completed.")
        self.pre = dataframes["pre"]
        self.pval = dataframes["pval"]
        self.fine = dataframes["fine"]
        self.fval = dataframes["fval"]
        self.test = dataframes["test"]

        self.__num_categories = dataset.num_categories()
        self.continuous_columns = dataset.continuous_columns
        self.categorical_columns = dataset.categorical_columns
        self.cat_cardinality_dict = dataset.cat_cardinality_dict(True)
        print(self.cat_cardinality_dict)
        self.target = dataset.target_columns
        self.d_out = dataset.dim_out

        if hasattr(dataset, "pre_all_columns"):
            self.pre_continuous_columns = dataset.pre_continuous_columns
            self.pre_categorical_columns = dataset.pre_categorical_columns
            self.pre_cat_cardinality_dict = dataset.pre_cat_cardinality_dict(True)

        if len(self.fval) > 10000:
            self.fval, _ = train_test_split(
                self.fval, train_size=10000, random_state=seed, stratify=self.fval[self.target]
            )
            dataframes["fval"] = self.fval

        msg = ""
        for k, v in dataframes.items():
            msg += f"{k}: {len(v)}, "
        logger.info(msg)

        self.task = dataset.task
        self.train_sampler = train_sampler
        self.num_workers = num_workers
        self.multi_node = multi_node

        if self.task == "regression":
            self.y_std = dataset.y_std

    @property
    def num_categories(self) -> int:
        return self.__num_categories

    @property
    def num_continuous_features(self) -> int:
        return len(self.continuous_columns)

    @property
    def num_categorical_features(self) -> int:
        return len(self.categorical_columns)

    def dataloader(self, mode: str, batch_size: int, drop_last=False):
        assert mode in {"pre", "pval", "fine", "fval", "test"}
        if not hasattr(self, mode):
            return None
        data = getattr(self, mode)

        if hasattr(self, "pre_continuous_columns") and "p" in mode:
            dataset = TabularDataset(
                data=data,
                task=self.task,
                categorical_columns=self.pre_categorical_columns,
                continuous_columns=self.pre_continuous_columns,
            )
        else:
            dataset = TabularDataset(
                data=data,
                task=self.task,
                categorical_columns=self.categorical_columns,
                continuous_columns=self.continuous_columns,
                target=self.target,
            )

        if mode == "pre" or mode == "fine":
            if self.multi_node:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)
        else:
            if self.multi_node:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            else:
                sampler = None

        return DataLoader(
            dataset,
            batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            multiprocessing_context="fork",
            drop_last=drop_last,
        )


def get_datamodule(config) -> TabularDatamodule:
    logger.info(f"Start {config.data} data loading...")
    dataset = getattr(datasets, config.data)(config=config)
    logger.info("Data loading has been successfully completed.")

    return TabularDatamodule(
        dataset,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 3,
        multi_node=config.multi_node,
        seed=config.seed,
    )
