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
    # overrides to narrow down the search
    # {"param": [from, to], "param2": [[cat1, cat2, cat3]]}
    param_overrides: dict = {},
):
    # see: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    config = {
        "boosting_type": ["categorical", [["gbdt", "dart"]]],
        "num_iterations": ["int", [70, 1000], {"step": 10}],
        "learning_rate": ["float", [1e-5, 1], {"log": True}],
        "num_leaves": ["int", [2, 100]],
        "min_data_in_leaf": ["int", [5, 1000], {"step": 5}],
        "max_depth": ["int", [-1, 41], {"step": 2}],
        "feature_fraction": ["float", [0.2, 1], {"step": 0.1}],
        # which fraction of dataset to create random sample
        "bagging_fraction": ["float", [0.2, 1], {"step": 0.1}],
        # how often (each n iteration) we will change the bagging sample (the subset of data)
        "bagging_freq": ["int", [1, 10]],
        # regularization and overfitting reduction
        "lambda_l1": ["float", [1e-8, 20], {"log": True}],
        "lambda_l2": ["float", [1e-8, 20], {"log": True}],
        "min_gain_to_split": ["float", [0, 1], {"step": 0.01}],
        "path_smooth": ["float", [0, 1], {"step": 0.01}],
        "extra_trees": ["categorical", [[True, False]]],
    }
    params = {
        "verbosity": -1,
    }

    for key, value in config.items():
        type = value[0]
        args = param_overrides[key] if key in param_overrides else value[1]
        kwargs = value[2] if len(value) == 3 else {}

        params[key] = getattr(trial, f"suggest_{type}")(key, *args, **kwargs)


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
