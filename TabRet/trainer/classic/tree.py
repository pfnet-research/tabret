import logging
import os

import numpy as np
import optuna
import sklearn
import xgboost as xgb
import zero
from catboost import CatBoostClassifier, Pool
from torch.utils.tensorboard import SummaryWriter

from data import get_datamodule
from trainer.utils import save_json

logger = logging.getLogger(__name__)


class TreeTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.is_root = config.world_rank == 0
        zero.improve_reproducibility(config.seed)
        self.datamodule = get_datamodule(config)
        self.feature_cols = self.datamodule.continuous_columns + self.datamodule.categorical_columns
        self.target_col = self.datamodule.target[0]
        if self.is_root:
            self.writer = SummaryWriter(log_dir="./")

    def objective(self):
        raise NotImplementedError()

    def search_hyper_params(self):
        if self.config.use_storage:
            storage = optuna.storages.RDBStorage(
                url=os.environ["OPTUNA_STORAGE"],
                heartbeat_interval=60,
                grace_period=120,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
            )
        else:
            storage = None
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            load_if_exists=True,
        )
        study.optimize(self.objective, callbacks=[optuna.study.MaxTrialsCallback(self.config.n_trials)])
        best_params = study.best_params
        logger.info(f"Best params: {best_params}")
        return best_params

    def print_evaluate(self):
        test = self.datamodule.test
        target = test[self.target_col]

        prediction = self.get_prediction(test)
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

        if self.is_root:
            save_json(score, "./")
            result = ""
            for k, v in score.items():
                result += f"{k}: {v:.3f}, "
                self.writer.add_scalar(f"test/{k}", v, 0)
            logger.info(f"Test Score ==> {result}")


class XGBoostTrainer(TreeTrainer):
    def objective(self, trial: optuna.Trial):
        zero.improve_reproducibility(self.config.seed)
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 11),
            "n_estimators": trial.suggest_int("n_estimators", 100, 5900, 200),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 1e2),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.7, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 7, log=True),
            "lambda": trial.suggest_float("lambda", 1, 4, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True),
        }
        if self.datamodule.task == "binary":
            params.update({"objective": "binary:logistic", "eval_metric": "auc"})
        elif self.datamodule.task == "multiclass":
            params.update(
                {"objective": "multi:softprob", "num_class": self.datamodule.d_out, "eval_metric": "mlogloss"}
            )
        else:
            params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

        fine, val = self.datamodule.fine, self.datamodule.fval
        model = xgb.XGBClassifier(
            **params,
            random_state=self.config.seed,
            use_label_encoder=False,
            early_stopping_rounds=self.config.patience,
        )
        model.fit(
            fine[self.feature_cols],
            fine[self.target_col],
            eval_set=[(val[self.feature_cols], val[self.target_col])],
            verbose=False,
        )
        pred = model.predict_proba(val[self.feature_cols])[:, 1]
        auc = sklearn.metrics.roc_auc_score(val[self.target_col], pred)
        return auc

    def training(self):
        best_params = self.search_hyper_params()

        if self.datamodule.task == "binary":
            best_params.update({"objective": "binary:logistic", "eval_metric": "auc"})
        elif self.datamodule.task == "multiclass":
            best_params.update(
                {"objective": "multi:softprob", "num_class": self.datamodule.d_out, "eval_metric": "mlogloss"}
            )
        else:
            best_params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

        fine, val = self.datamodule.fine, self.datamodule.fval

        self.model = xgb.XGBClassifier(
            **best_params,
            random_state=self.config.seed,
            use_label_encoder=False,
            early_stopping_rounds=self.config.patience,
        )
        self.model.fit(
            fine[self.feature_cols],
            fine[self.target_col],
            eval_set=[(val[self.feature_cols], val[self.target_col])],
            verbose=False,
        )

    def get_prediction(self, test):
        return self.model.predict_proba(test[self.feature_cols])[:, 1]


class CatBoostTrainer(TreeTrainer):
    def objective(self, trial: optuna.Trial):
        zero.improve_reproducibility(self.config.seed)
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        }

        if self.datamodule.task == "binary":
            params.update({"eval_metric": "AUC"})
        elif self.datamodule.task == "multiclass":
            params.update({"eval_metric": "ACC"})
        else:
            params.update({"eval_metric": "RMSE"})

        train, val = self.datamodule.fine, self.datamodule.fval
        cate_cols = self.datamodule.categorical_columns
        train[cate_cols] = train[cate_cols].astype(int)
        val[cate_cols] = val[cate_cols].astype(int)

        train_pool = Pool(train[self.feature_cols], train[self.target_col], cat_features=cate_cols)
        val_pool = Pool(val[self.feature_cols], val[self.target_col], cat_features=cate_cols)
        model = CatBoostClassifier(
            **params,
            random_seed=self.config.seed,
            verbose=0,
            od_pval=0.001,
            iterations=2000,
            early_stopping_rounds=self.config.patience,
        )
        model.fit(
            train_pool,
            eval_set=(val_pool),
        )

        pred = model.predict_proba(val_pool)[:, 1]
        auc = sklearn.metrics.roc_auc_score(val[self.target_col], pred)
        return auc

    def training(self):
        best_params = self.search_hyper_params()

        if self.datamodule.task == "binary":
            best_params.update({"eval_metric": "AUC"})
        elif self.datamodule.task == "multiclass":
            best_params.update({"eval_metric": "ACC"})
        else:
            best_params.update({"eval_metric": "RMSE"})

        train, val = self.datamodule.fine, self.datamodule.fval
        cate_cols = self.datamodule.categorical_columns
        train[cate_cols] = train[cate_cols].astype(int)
        val[cate_cols] = val[cate_cols].astype(int)

        train_pool = Pool(train[self.feature_cols], train[self.target_col], cat_features=cate_cols)
        val_pool = Pool(val[self.feature_cols], val[self.target_col], cat_features=cate_cols)
        self.model = CatBoostClassifier(
            **best_params,
            random_seed=self.config.seed,
            verbose=0,
            od_pval=0.001,
            iterations=2000,
            early_stopping_rounds=self.config.patience,
        )
        self.model.fit(
            train_pool,
            eval_set=(val_pool),
        )

    def get_prediction(self, test):
        test = self.datamodule.test
        cate_cols = self.datamodule.categorical_columns
        test[cate_cols] = test[cate_cols].astype(int)
        test_pool = Pool(test[self.feature_cols], test[self.target_col], cat_features=cate_cols)
        return self.model.predict_proba(test_pool)[:, 1]
