from BorutaShap import BorutaShap
from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from . import io


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
    optimize_overfitting: bool = False,
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
                eval_train_metric=optimize_overfitting,
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
            return (
                (
                    result["train accuracy-mean"][-1]
                    - result["valid accuracy-mean"][-1],
                    result["valid accuracy-mean"][-1],
                )
                if optimize_overfitting
                else result["valid accuracy-mean"][-1]
            )
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


def boruta_select_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: dict,
    # path without extension, e.g. ./data/borutashap_feature_importance
    output_basename: str,
    n_trials: int = 50,
    seed: int = None,
    classification: bool = True,
):
    def calc_boruta():
        Feature_Selector = BorutaShap(
            importance_measure="shap",
            classification=classification,
            model=LGBMClassifier(**params),
        )

        Feature_Selector.fit(
            X=X.fillna(-1),
            y=y,
            # unfortunately the 'test' value does not makes the lib to measure importance
            # on test set https://github.com/Ekeany/Boruta-Shap/issues/126
            # moreover this lib makes train/test split under the hood so we can not feed
            # it with our test set
            train_or_test="train",
            random_state=seed,
            n_trials=n_trials,
        )

        Feature_Selector.results_to_csv(output_basename)

        return pd.read_csv(f"{output_basename}.csv")

    df_boruta = io.run_cached(f"{output_basename}.parquet", calc_boruta)

    attr_important = df_boruta[df_boruta["Decision"] == "Accepted"][
        "Features"
    ].values.tolist()
    attr_tentative = df_boruta[df_boruta["Decision"] == "Tentative"][
        "Features"
    ].values.tolist()
    attr_rejected = df_boruta[df_boruta["Decision"] == "Rejected"][
        "Features"
    ].values.tolist()

    print(f"{len(attr_important)} attributes confirmed important: {attr_important}")
    print(f"\n\n{len(attr_tentative)} attributes confirmed tentative: {attr_tentative}")
    print(f"\n\n{len(attr_rejected)} attributes confirmed unimportant: {attr_rejected}")

    return {
        "df_boruta": df_boruta,
        "important": attr_important,
        "tentative": attr_tentative,
        "rejected": attr_rejected,
    }
