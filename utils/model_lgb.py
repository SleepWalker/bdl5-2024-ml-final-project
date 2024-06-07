import lightgbm as lgb
import pandas as pd
import numpy as np


def train_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: dict,
    num_class: int,
    seed: int = None,
):
    dtrain = lgb.Dataset(X_train, y_train)
    dtest = lgb.Dataset(X_test, y_test)

    orig_params = params
    params = {
        "num_class": num_class,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "seed": seed,
    }
    params.update(orig_params)

    model_gb = lgb.train(
        params,
        dtrain,
        valid_sets=[dtest],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=0),
        ],
    )
    model_gb_predict = lambda X: np.argmax(
        model_gb.predict(X, num_iteration=model_gb.best_iteration), axis=1
    )

    return model_gb_predict, model_gb
