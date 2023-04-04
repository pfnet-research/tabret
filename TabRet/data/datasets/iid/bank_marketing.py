import numpy as np
import pandas as pd

from .iid_base import IIDBaseDataFrame


class Bank(IIDBaseDataFrame):
    resources = [
        ("bank-full.csv", "5d7c39d7b8804f071cdd1f2a7c460872"),
    ]

    dim_out = 1

    all_columns = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]

    continuous_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "pdays",
        "previous",
    ]

    categorical_columns = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "campaign",
        "poutcome",
    ]

    target_columns = ["y"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=42).permutation(45211)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "bank/bank-full.csv", sep=";")
        df["y"] = df["y"].replace({"no": 0, "yes": 1})
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
