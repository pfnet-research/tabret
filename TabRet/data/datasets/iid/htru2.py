import numpy as np
import pandas as pd

from .iid_base import IIDBaseDataFrame


class HTRU2(IIDBaseDataFrame):
    resources = [
        ("HTRU_2.csv", "5d7c39d7b8804f071cdd1f2a7c460872"),
    ]

    dim_out = 1

    all_columns = [f"x{i}" for i in range(8)] + ["y"]

    continuous_columns = [f"x{i}" for i in range(8)]

    categorical_columns = []

    target_columns = ["y"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=42).permutation(17898)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "htru/HTRU_2.csv", sep=",", header=None)
        df.columns = self.all_columns
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
