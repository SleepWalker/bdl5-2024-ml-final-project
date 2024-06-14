import pandas as pd
import optuna
from .model_lgb import train_multiclass


def multiclass_objective(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metric_fn: callable,
    num_class: int,
    seed: int = None,
):
    params = {
        # # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        # "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 50, step=2),
        # "max_depth": trial.suggest_int("max_depth", 1, 5),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100, step=5),
        # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "bagging_fraction": trial.suggest_float(
        #     "bagging_fraction", 0.2, 0.90, step=0.1
        # ),
        # "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        # "feature_fraction": trial.suggest_float(
        #     "feature_fraction", 0.2, 0.90, step=0.1
        # ),
        #     'objective': 'multiclass',
        # 'metric': 'multi_logloss',
        # 'num_class': 3,
        # 'boosting_type': 'gbdt',
        # 'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        # 'random_state': 42,
        # 'verbosity': -1
        "verbosity": -1,
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 0, 50, step=10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # this param is ignored if num_leaves specified
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    if trial.should_prune():
        raise optuna.TrialPruned()

    predict, model = train_multiclass(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_class=num_class,
        seed=seed,
        params=params,
    )
    y_pred = predict(X_test)

    return metric_fn(y_test, y_pred)
