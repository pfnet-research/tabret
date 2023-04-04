import numpy as np
import pandas as pd

from ..ood_base import OODBaseDataFrame


class DiabetesLBaseDataFrame(OODBaseDataFrame):
    pre_target_columns = ["Diabetes"]

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)

        idx = np.random.RandomState(seed=42).permutation(2038772)
        self.pre_train_idx = idx[: int(len(idx) * 0.8)]

    def raw_pre_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / "diabetesl/all.csv")
        self.pre_all_columns = list(df.columns)
        self.pre_categorical_columns = list(
            df.loc[:, (df.dtypes == "object") | (df.dtypes == "int64")].drop("Diabetes", axis=1).columns
        )
        self.pre_continuous_columns = list(df.loc[:, df.dtypes == "float64"].columns)
        return df.iloc[self.pre_train_idx]
