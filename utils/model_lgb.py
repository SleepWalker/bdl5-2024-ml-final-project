import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path

model_dir = "./models"


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
    cv: dict = None,
):
    model_path = get_model_path(name) if name else None
    should_train = not model_path or not Path(model_path).is_file()

    if should_train:
        dtrain = lgb.Dataset(X_train, y_train)
        dtest = lgb.Dataset(X_test, y_test)

        orig_params = params
        params = {
            "num_class": num_class,
            "objective": "multiclass",
            "metric": "multi_logloss",
            # for CV it's recommended to use random initialization for estimators
            # https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
            "seed": None if cv else seed,
        }
        params.update(orig_params)

        if cv:
            # TODO: we should split the function because for
            # cv we have different return value
            metric_fn = cv["metric_fn"]
            result = lgb.cv(
                params,
                dtrain,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=0),
                ],
                nfold=5,
                stratified=True,
                shuffle=True,
                seed=seed,
                feval=lambda preds, train_data: (
                    "accuracy",
                    metric_fn(
                        train_data.get_label(),
                        np.argmax(preds, axis=1),
                    ),
                    True,
                ),
            )

            # the last item is the best iteration
            return result["valid accuracy-mean"][-1]
        else:
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
        return load(name)

    return get_model_with_predict(model_lgb)


def load(name: str) -> tuple[callable, lgb.Booster]:
    model_lgb = lgb.Booster(model_file=get_model_path(name))

    return get_model_with_predict(model_lgb)


def get_model_with_predict(model_lgb: lgb.Booster):
    def predict(X, proba=False):
        preds = model_lgb.predict(X, num_iteration=model_lgb.best_iteration)

        if not proba and model_lgb._Booster__num_class > 1:
            preds = np.argmax(preds, axis=1)

        return preds

    return predict, model_lgb


def get_model_path(name: str) -> str:
    return f"{model_dir}/{name}.lgb.txt"
