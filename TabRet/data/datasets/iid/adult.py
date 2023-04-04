import numpy as np
import pandas as pd

from .iid_base import IIDBaseDataFrame


class Adult(IIDBaseDataFrame):
    mirrors = [
        "https://www.kaggle.com/datasets/lodetomasi1995/income-classification",
    ]

    dim_out = 1

    all_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    continuous_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    target_columns = ["income"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=42).permutation(32561)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "adult/income_evaluation.csv", sep=",")
        df.columns = self.all_columns
        df["income"] = df["income"].replace({" <=50K": 0, " >50K": 1})

        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
