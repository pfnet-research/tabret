import logging

import numpy as np
import optuna
import sklearn
import zero
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from data import get_datamodule
from trainer.utils import save_json

logger = logging.getLogger(__name__)


class LRTrainer:
    def __init__(self, config) -> None:
        self.config = config
        zero.improve_reproducibility(config.seed)
        self.datamodule = get_datamodule(config)
        self.feature_cols = self.datamodule.continuous_columns + self.datamodule.categorical_columns
        self.target_col = self.datamodule.target[0]

    def training(self):
        self.model = LogisticRegression(max_iter=1000, random_state=self.config.seed)
        self.model.fit(
            self.datamodule.fine[self.feature_cols],
            self.datamodule.fine[self.target_col],
        )

    def print_evaluate(self):
        test = self.datamodule.test
        target = test[self.target_col]

        prediction = self.model.predict_proba(test[self.feature_cols])[:, 1]
        if self.datamodule.task == "binary":
            label = (prediction > 0.5).astype(np.int64)
            score = {
                "ACC": sklearn.metrics.accuracy_score(target, label),
                "AUC": sklearn.metrics.roc_auc_score(target, prediction),
            }
        elif self.datamodule.task == "multiclass":
            prediction = prediction.argmax(1)
            score = {"Acc": sklearn.metrics.accuracy_score(target, prediction)}
        else:
            score = {"Acc": sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * self.datamodule.y_std}

        save_json(score, "./")
        result = ""
        for k, v in score.items():
            result += f"{k}: {v:.3f}, "
        logger.info(f"Test Score ==> {result}")
