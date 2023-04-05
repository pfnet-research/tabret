import numpy as np
import pandas as pd

from .brfss_base import BRFSSBaseDataFrame


class Stroke(BRFSSBaseDataFrame):
    dim_out = 1

    all_columns = [
        "hypertension",
        "heart_disease",
        "work_type",
        "Residence_type",
        "SEX",
        "_AGEG5YR",
        "MARITAL",
        "_BMI5",
        "SMOKE100",
        "stroke",
    ]

    continuous_columns = [
        "avg_glucose_level",
        "_BMI5",
    ]

    categorical_columns = [
        "hypertension",
        "heart_disease",
        "work_type",
        "Residence_type",
        "SEX",
        "_AGEG5YR",
        "MARITAL",
        "SMOKE100",
    ]

    common_cont_columns = ["_BMI5"]

    common_cate_columns = [
        "SEX",
        "_AGEG5YR",
        "MARITAL",
        "SMOKE100",
    ]

    diff_cont_columns = [
        "avg_glucose_level",
    ]

    diff_cate_columns = [
        "hypertension",
        "heart_disease",
        "work_type",
        "Residence_type",
    ]

    target_columns = ["stroke"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=config.seed).permutation(4909)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "stroke/healthcare-dataset-stroke-data.csv")
        # gender
        df["SEX"] = df["gender"].replace({"Male": 1, "Female": 2})
        # Age
        df = df.dropna()
        low = np.arange(25, 76, 5)
        high = np.arange(29, 80, 5)
        df.loc[df["age"] < 18, ["_AGEG5YR"]] = 14
        df.loc[(18 <= df["age"]) & (df["age"] <= 24), ["_AGEG5YR"]] = 1
        df.loc[df["age"] >= 80, ["_AGEG5YR"]] = 13
        for k, i, j in zip(range(2, 13), low, high):
            df.loc[(i <= df["age"]) & (df["age"] <= j), ["_AGEG5YR"]] = k
        # ever_married
        df["MARITAL"] = df["ever_married"].replace({"No": 5, "Yes": 1})
        # BMI
        df["_BMI5"] = df["bmi"] * 100
        # smoke
        df["SMOKE100"] = df["smoking_status"].replace(
            {"smokes": 1, "formerly smoked": 1, "never smoked": 2, "Unknown": 7}
        )
        # drop
        df = df.drop(["id", "gender", "age", "ever_married", "bmi", "smoking_status"], axis=1)
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
