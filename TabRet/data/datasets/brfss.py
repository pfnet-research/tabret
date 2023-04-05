from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from .tabular_dataframe import TabularDataFrame


class BRFSS(TabularDataFrame):
    dim_out = 1

    target_columns = ["Diabetes"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)

        idx = np.random.RandomState(seed=42).permutation(2038772)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def download(self):
        pass

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "brfss/all.csv")
        self.all_columns = list(df.columns)
        self.categorical_columns = list(
            df.loc[:, (df.dtypes == "object") | (df.dtypes == "int64")].drop("Diabetes", axis=1).columns
        )
        self.continuous_columns = list(df.loc[:, df.dtypes == "float64"].columns)
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        df_train = self.raw_dataframe(train=True)
        df_test = self.raw_dataframe(train=False)

        df_train, df_val = train_test_split(df_train, stratify=df_train["Diabetes"], test_size=0.1, random_state=42)

        df_pre, df_fine = train_test_split(
            df_train,
            stratify=df_train["Diabetes"],
            test_size=0.01,
            random_state=42,
        )

        assert self.config.fine_num <= len(df_fine), f"f_num must be less than or equal to {len(df_fine)}."

        _, df_fine = train_test_split(
            df_fine,
            stratify=df_fine["Diabetes"],
            test_size=self.config.fine_num,
            random_state=self.config.seed,
        )
        dfs = {
            "pre": df_pre,
            "pval": df_val,
            "fine": df_fine,
            "test": df_test,
        }

        cont_enc = QuantileTransformer(output_distribution="normal").fit(df_pre[self.continuous_columns])
        if self.is_xgb:
            cate_enc = OneHotEncoder(handle_unknown="ignore").fit(df_pre[self.categorical_columns])
        else:
            cate_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                df_pre[self.categorical_columns]
            )

        for key, df in dfs.items():
            df = df.reset_index(drop=True)
            if not self.is_tree:
                dfs[key][self.continuous_columns] = cont_enc.transform(df[self.continuous_columns])
            if self.is_xgb:
                new_df = pd.DataFrame(cate_enc.transform(df[self.categorical_columns]).toarray().astype("int64"))
                df = pd.concat([df, new_df], axis=1)
                dfs[key] = df.drop(self.categorical_columns, axis=1)
            else:
                dfs[key][self.categorical_columns] = cate_enc.transform(df[self.categorical_columns]) + 1

        if self.is_xgb:
            self.continuous_columns += list(new_df.columns)
            self.categorical_columns = []

        dfs["fval"] = dfs["pval"]

        _, dfs["fval"] = train_test_split(
            dfs["fval"],
            stratify=dfs["fval"]["Diabetes"],
            test_size=10000,
            random_state=self.config.seed,
        )

        for k, v in dfs.items():
            print(f"{k}: {len(v)}", end=", ")
        print()
        return dfs
