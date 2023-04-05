import numpy as np
import pandas as pd

from .brfss_base import BRFSSBaseDataFrame


class PKIHD(BRFSSBaseDataFrame):
    dim_out = 1

    all_columns = [
        "HeartDisease",
        "AlcoholDrinking",
        "PhysicalHealth",
        "MentalHealth",
        "DiffWalking",
        "Race",
        "Diabetic",
        "PhysicalActivity",
        "SleepTime",
        "_BMI5",
        "SMOKE100",
        "CVDSTRK3",
        "SEX",
        "_AGEG5YR",
        "GENHLTH",
        "ASTHMA3",
        "CHCKIDNY",
        "CHCSCNCR",
    ]

    continuous_columns = [
        "SleepTime",
        "_BMI5",
    ]

    categorical_columns = [
        "AlcoholDrinking",
        "PhysicalHealth",
        "MentalHealth",
        "DiffWalking",
        "Race",
        "Diabetic",
        "PhysicalActivity",
        "SMOKE100",
        "CVDSTRK3",
        "SEX",
        "_AGEG5YR",
        "GENHLTH",
        "ASTHMA3",
        "CHCKIDNY",
        "CHCSCNCR",
    ]

    common_cont_columns = ["_BMI5"]

    common_cate_columns = [
        "SMOKE100",
        "CVDSTRK3",
        "SEX",
        "_AGEG5YR",
        "GENHLTH",
        "ASTHMA3",
        "CHCKIDNY",
        "CHCSCNCR",
    ]

    diff_cont_columns = ["SleepTime"]

    diff_cate_columns = [
        "AlcoholDrinking",
        "PhysicalHealth",
        "MentalHealth",
        "DiffWalking",
        "Race",
        "Diabetic",
        "PhysicalActivity",
    ]

    target_columns = ["HeartDisease"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=config.seed).permutation(319795)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "pkihd/heart_2020_cleaned.csv")
        df["_BMI5"] = df["BMI"] * 100
        df["SMOKE100"] = df["Smoking"].replace({"Yes": 1, "No": 2})
        df["CVDSTRK3"] = df["Stroke"].replace({"Yes": 1, "No": 2})
        df["SEX"] = df["Sex"].replace({"Male": 1, "Female": 2})
        df["_AGEG5YR"] = df["AgeCategory"].replace(
            {
                "18-24": 1,
                "25-29": 2,
                "30-34": 3,
                "35-39": 4,
                "40-44": 5,
                "45-49": 6,
                "50-54": 7,
                "55-59": 8,
                "60-64": 9,
                "65-69": 10,
                "70-74": 11,
                "75-79": 12,
                "80 or older": 13,
            }
        )
        df["GENHLTH"] = df["GenHealth"].replace({"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5})
        df["ASTHMA3"] = df["Asthma"].replace({"Yes": 1, "No": 2})
        df["CHCKIDNY"] = df["KidneyDisease"].replace({"Yes": 1, "No": 2})
        df["CHCSCNCR"] = df["SkinCancer"].replace({"Yes": 1, "No": 2})
        df = df.drop(
            ["BMI", "Smoking", "Stroke", "Sex", "AgeCategory", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"],
            axis=1,
        )
        df["HeartDisease"] = df["HeartDisease"].replace({"No": 0, "Yes": 1})
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
