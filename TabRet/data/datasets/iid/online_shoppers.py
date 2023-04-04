import numpy as np
import pandas as pd

from .iid_base import IIDBaseDataFrame


class OnlineShoppers(IIDBaseDataFrame):
    resources = [
        ("online_shoppers_intention.csv", "5d7c39d7b8804f071cdd1f2a7c460872"),
    ]

    dim_out = 1

    all_columns = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        "Revenue",
    ]

    continuous_columns = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]

    categorical_columns = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]

    target_columns = ["Revenue"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=42).permutation(12330)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "online_shoppers/online_shoppers_intention.csv", sep=",")
        df["Revenue"] = df["Revenue"].replace({False: 0, True: 1})
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
