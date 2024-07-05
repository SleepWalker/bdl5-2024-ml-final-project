import pandas as pd
import optuna
from .model_lgb import train_multiclass


def multiclass_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metric_fn: callable,
    num_class: int,
    seed: int = None,
    cv: bool = False,
    optimize_overfitting: bool = False,
):
    # see: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    params = {
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_iterations": trial.suggest_int("num_iterations", 70, 1000, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 1000, step=5),
        "max_depth": trial.suggest_int("max_depth", -1, 41, step=2),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
        # which fraction of dataset to create random sample
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1, step=0.1),
        # how often (each n iteration) we will change the bagging sample (the subset of data)
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        # regularization and overfitting reduction
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 1),
        "path_smooth": trial.suggest_float("path_smooth", 0, 1),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
    }

    # Trial.should_prune is not supported for multi-objective optimization.
    # therefore we can not use it with optimize_overfitting
    if not optimize_overfitting and trial.should_prune():
        raise optuna.TrialPruned()

    if cv:
        metric_value = train_multiclass(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_class=num_class,
            seed=seed,
            params=params,
            cv={"metric_fn": metric_fn},
            optimize_overfitting=optimize_overfitting,
        )

        return metric_value
    else:
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
