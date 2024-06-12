import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path


def train_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: dict,
    num_class: int,
    seed: int = None,
    # if specified, we will save the model
    name: str = None,
):
    model_dir = "./models"
    model_path = f"{model_dir}/{name}.lgb.txt" if name else None
    should_train = not model_path or not Path(model_path).is_file()

    if should_train:
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

        model_lgb = lgb.train(
            params,
            dtrain,
            valid_sets=[dtest],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=0),
            ],
        )

        if model_path:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            # will save best iteration
            model_lgb.save_model(model_path)

    if model_path:
        # load from file even if we've just trained this model
        # just to test that io works
        model_lgb = lgb.Booster(model_file=model_path)

    model_lgb_predict = lambda X: np.argmax(
        model_lgb.predict(X, num_iteration=model_lgb.best_iteration), axis=1
    )

    return model_lgb_predict, model_lgb
