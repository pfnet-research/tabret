from copy import copy
from typing import Dict, Optional

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


class OODBaseDataFrame(TabularDataFrame):
    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)

    def cat_cardinality_dict(self, use_unk: bool = True) -> Optional[Dict[str, int]]:
        cardinality_dict = copy(self._cat_cardinality_dict)
        for k in self._cat_cardinality_dict.keys():
            if use_unk:
                cardinality_dict[k] += 1
        return cardinality_dict

    def processed_dataframes(self, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        df_pre = self.raw_pre_dataframe()
        df_train = self.raw_dataframe(train=True)
        df_test = self.raw_dataframe(train=False)

        if hasattr(self, "pre_target_columns"):
            df_pre, df_val = train_test_split(
                df_pre, stratify=df_pre[self.pre_target_columns], test_size=0.1, random_state=42
            )
            df_pre, _ = train_test_split(
                df_pre,
                stratify=df_pre[self.pre_target_columns],
                test_size=0.01,
                random_state=42,
            )
        else:
            df_pre, df_val = train_test_split(df_pre, test_size=0.1, random_state=42)

        if self.task != "regression":
            df_fine, df_fval = train_test_split(
                df_train,
                stratify=df_train[self.target_columns],
                train_size=self.config.fine_num,
                random_state=self.config.seed,
            )
        else:
            df_fine, df_fval = train_test_split(
                df_train,
                train_size=self.config.fine_num,
                random_state=self.config.seed,
            )

        dfs = {
            "pre": df_pre,
            "pval": df_val,
            "fine": df_fine,
            "fval": df_fval,
            "test": df_test,
        }
        fine_keys = ["fine", "fval", "test"]

        # common columns encoding
        if hasattr(self, "common_cont_columns"):
            cont_enc = QuantileTransformer(output_distribution="normal").fit(df_pre[self.common_cont_columns])
        if hasattr(self, "common_cate_columns"):
            if self.is_xgb:
                common_cate_enc = OneHotEncoder(handle_unknown="ignore").fit(df_pre[self.common_cate_columns])
            else:
                common_cate_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                    df_pre[self.common_cate_columns]
                )
        # diff columns encoding
        if hasattr(self, "diff_cont_columns"):
            diff_cont_enc = QuantileTransformer(output_distribution="normal").fit(df_fine[self.diff_cont_columns])
        if hasattr(self, "diff_cate_columns"):
            if self.is_xgb:
                diff_cate_enc = OneHotEncoder(handle_unknown="ignore").fit(df_fine[self.diff_cate_columns])
            else:
                diff_cate_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                    df_fine[self.diff_cate_columns]
                )

        for key in fine_keys:
            dfs[key] = dfs[key].reset_index(drop=True)
            if not self.is_tree:
                if hasattr(self, "common_cont_columns"):
                    dfs[key][self.common_cont_columns] = cont_enc.transform(dfs[key][self.common_cont_columns])
                if hasattr(self, "diff_cont_columns"):
                    dfs[key][self.diff_cont_columns] = diff_cont_enc.transform(dfs[key][self.diff_cont_columns])
            if hasattr(self, "common_cate_columns"):
                if self.is_xgb:
                    # to one-hot
                    new_df = pd.DataFrame(
                        common_cate_enc.transform(dfs[key][self.common_cate_columns]).toarray().astype("int64"),
                        columns=get_colmns(self.common_cate_columns, common_cate_enc.categories_),
                    )
                    df = pd.concat([dfs[key], new_df], axis=1)
                    dfs[key] = df.drop(self.common_cate_columns, axis=1)
                else:
                    dfs[key][self.common_cate_columns] = (
                        common_cate_enc.transform(dfs[key][self.common_cate_columns]) + 1
                    )
            if hasattr(self, "diff_cate_columns"):
                if self.is_xgb:
                    # to one-hot
                    new_df = pd.DataFrame(
                        diff_cate_enc.transform(dfs[key][self.diff_cate_columns]).toarray().astype("int64"),
                        columns=get_colmns(self.diff_cate_columns, diff_cate_enc.categories_),
                    )
                    df = pd.concat([dfs[key], new_df], axis=1)
                    dfs[key] = df.drop(self.diff_cate_columns, axis=1)
                else:
                    dfs[key][self.diff_cate_columns] = diff_cate_enc.transform(dfs[key][self.diff_cate_columns]) + 1

        if self.is_xgb:
            self.continuous_columns = list(dfs["fine"].drop(self.target_columns, axis=1).columns)
            print(self.continuous_columns)
            self.categorical_columns = []

        self._cat_cardinality_dict = {}
        if hasattr(self, "common_cate_columns"):
            self._cat_cardinality_dict.update(self.get_cardinality_dict(common_cate_enc))
        if hasattr(self, "diff_cate_columns"):
            self._cat_cardinality_dict.update(self.get_cardinality_dict(diff_cate_enc))

        # alignment
        self._cat_cardinality_dict = {key: self._cat_cardinality_dict[key] for key in self.categorical_columns}

        if self.task == "regression":
            self.y_mean = dfs["fine"][self.target_columns].to_numpy().mean()
            self.y_std = dfs["fine"][self.target_columns].to_numpy().std()
            for key in fine_keys:
                dfs[key] = self._regression_encoder(dfs[key])

        for k, v in dfs.items():
            print(f"{k}: {len(v)}", end=", ")
        print()
        return dfs

    def _regression_encoder(self, df):
        df[self.target_columns] = (df[self.target_columns] - self.y_mean) / self.y_std
        return df
