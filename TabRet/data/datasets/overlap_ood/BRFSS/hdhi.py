import numpy as np
import pandas as pd

from .brfss_base import BRFSSBaseDataFrame


class HDHI(BRFSSBaseDataFrame):
    dim_out = 1

    all_columns = [
        "HeartDiseaseorAttack",
        "HighBP",
        "HighChol",
        "CholCheck",
        "Diabetes",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "_BMI5",
        "SMOKE100",
        "CVDSTRK3",
        "_TOTINDA",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "SEX",
        "_AGEG5YR",
        "EDUCA",
        "INCOME2",
    ]

    continuous_columns = [
        "_BMI5",
    ]

    categorical_columns = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Diabetes",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "SMOKE100",
        "CVDSTRK3",
        "_TOTINDA",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "SEX",
        "_AGEG5YR",
        "EDUCA",
        "INCOME2",
    ]

    common_cont_columns = ["_BMI5"]

    common_cate_columns = [
        "SMOKE100",
        "CVDSTRK3",
        "_TOTINDA",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "SEX",
        "_AGEG5YR",
        "EDUCA",
        "INCOME2",
    ]

    diff_cate_columns = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Diabetes",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
    ]

    target_columns = ["HeartDiseaseorAttack"]

    task = "binary"

    def __init__(self, config, download: bool = False) -> None:
        super().__init__(config=config, download=download)
        idx = np.random.RandomState(seed=config.seed).permutation(253680)
        self.train_idx = idx[: int(len(idx) * 0.8)]
        self.test_idx = idx[int(len(idx) * 0.8) :]

    def raw_dataframe(self, train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(self.root / "hdhi/heart_disease_health_indicators_BRFSS2015.csv")
        # be divided by 100 and then be rounded by 0
        df["BMI"] = df["BMI"] * 100
        # be changed 2 -> 0 and be deleted 7, 9
        df["Smoker"] = df["Smoker"].replace({0: 2})
        # be changed 2 -> 0 and be deleted 7, 9
        df["Stroke"] = df["Stroke"].replace({0: 2})
        # be changed 2 -> 0 and be deleted 9
        df["PhysActivity"] = df["PhysActivity"].replace({0: 2})
        # be changed 2 -> 0 and be deleted 7, 9
        df["AnyHealthcare"] = df["AnyHealthcare"].replace({0: 2})
        # be changed 2 -> 0 and be deleted 7, 9
        df["NoDocbcCost"] = df["NoDocbcCost"].replace({0: 2})
        # be deleted 7, 9
        df["GenHlth"] = df["GenHlth"]
        # be changd 2 -> 0
        df["Sex"] = df["Sex"].replace({0: 2})
        # be deleted 14
        df["Age"] = df["Age"]
        # be deleted 9
        df["Education"] = df["Education"]
        # be deleted 77, 79
        df["Income"] = df["Income"]
        df = df.rename(
            columns={
                "BMI": "_BMI5",
                "Smoker": "SMOKE100",
                "Stroke": "CVDSTRK3",
                "PhysActivity": "_TOTINDA",
                "AnyHealthcare": "HLTHPLN1",
                "NoDocbcCost": "MEDCOST",
                "GenHlth": "GENHLTH",
                "Sex": "SEX",
                "Age": "_AGEG5YR",
                "Education": "EDUCA",
                "Income": "INCOME2",
            }
        )
        if train:
            df = df.iloc[self.train_idx]
        else:
            df = df.iloc[self.test_idx]
        return df
