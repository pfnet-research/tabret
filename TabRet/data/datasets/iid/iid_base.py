from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from ..tabular_dataframe import TabularDataFrame


def get_colmns(columns, categories):
    columns_after = []
    for col, cates in zip(columns, categories):
        for cate in cates:
            columns_after.append(f"{col}-{cate}")
    return columns_after


class IIDBaseDataFrame(TabularDataFrame):
    def __init__(self, config, download: bool = True) -> None:
        super().__init__(config=config, download=download)

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        df_train = self.raw_dataframe(train=True)
        df_test = self.raw_dataframe(train=False)

        df_train, df_val = train_test_split(df_train, stratify=df_train[self.target_columns], test_size=1 / 8)

        df_pre, df_fine = train_test_split(
            df_train,
            stratify=df_train[self.target_columns],
            test_size=1 / 7,
        )

        assert self.config.fine_num <= len(df_fine), f"f_num must be less than or equal to {len(df_fine)}."

        _, df_fine = train_test_split(
            df_fine,
            stratify=df_fine[self.target_columns],
            test_size=self.config.fine_num,
            random_state=self.config.seed,
        )

        dfs = {
            "pre": df_pre,
            "pval": df_val,
            "fine": df_fine,
            "test": df_test,
        }
        # preprocessing
        if hasattr(self, "categorical_columns"):
            if self.is_xgb:
                cate_enc = OneHotEncoder(handle_unknown="ignore").fit(df_fine[self.categorical_columns])
            else:
                cate_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                    df_pre[self.categorical_columns]
                )
        if hasattr(self, "continuous_columns"):
            cont_enc = QuantileTransformer(output_distribution="normal").fit(df_pre[self.continuous_columns])
        for key in dfs.keys():
            dfs[key] = dfs[key].reset_index(drop=True)
            if hasattr(self, "categorical_columns"):
                if self.is_xgb:
                    # to one-hot
                    new_df = pd.DataFrame(
                        cate_enc.transform(dfs[key][self.categorical_columns]).toarray().astype("int64"),
                        columns=get_colmns(self.categorical_columns, cate_enc.categories_),
                    )
                    df = pd.concat([dfs[key], new_df], axis=1)
                    dfs[key] = df.drop(self.categorical_columns, axis=1)
                else:
                    dfs[key][self.categorical_columns] = cate_enc.transform(dfs[key][self.categorical_columns]) + 1
            if not self.is_tree:
                if hasattr(self, "continuous_columns"):
                    dfs[key][self.continuous_columns] = cont_enc.transform(dfs[key][self.continuous_columns])

        if self.is_xgb:
            self.continuous_columns = list(dfs["fine"].drop(self.target_columns, axis=1).columns)
            self.categorical_columns = []

        dfs["fval"] = dfs["pval"]
        return dfs
