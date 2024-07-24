import pandas as pd
import lightgbm as lgb
from utils import io
from utils import compress_df
import utils.model_lgb as model_lgb
from project import bnum
from project import dpi
import time

"""
This package contains all functions required to train and execute production model
"""

RANDOM_SEED = 42
TARGET_KEY = "target"
CLASS_NUM = 5

best_params = {
    "boosting_type": "gbdt",
    "num_iterations": 350,
    "learning_rate": 0.27831449040881007,
    "num_leaves": 14,
    "min_data_in_leaf": 1360,
    "max_depth": 1,
    "feature_fraction": 0.9000000000000001,
    "bagging_fraction": 0.4,
    "bagging_freq": 2,
    "lambda_l1": 0.0011830914968329104,
    "lambda_l2": 0.012417262351779968,
    "min_gain_to_split": 0.6889417143854851,
    "path_smooth": 0.03645227176885784,
    "extra_trees": True,
}


def preprocess(
    fe_train_path: str,
    fe_test_path: str,
    bnum_train_path: str,
    bnum_test_path: str,
    dpi_train_path: str,
    dpi_test_path: str,
    cache_key: str = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    separator = "=" * 10
    print(f"\n{separator}\nPreparing Train dataset\n{separator}")

    X_train, y_train = preprocess_dataset(
        fe_path=fe_train_path,
        bnum_path=bnum_train_path,
        dpi_path=dpi_train_path,
        cache_key=f"{cache_key}_train" if cache_key else None,
    )

    print(f"\n{separator}\nPreparing Test dataset\n{separator}")

    X_test, y_test = preprocess_dataset(
        fe_path=fe_test_path,
        bnum_path=bnum_test_path,
        dpi_path=dpi_test_path,
        cache_key=f"{cache_key}_test" if cache_key else None,
    )

    return X_train, y_train, X_test, y_test


def preprocess_dataset(
    fe_path: str,
    bnum_path: str,
    dpi_path: str,
    cache_key: str = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df_fe = pd.read_parquet(fe_path)
    df_bnum = pd.read_parquet(bnum_path)
    df_dpi = pd.read_parquet(dpi_path)

    X = compile_features(
        df_fe=df_fe,
        df_bnum=df_bnum,
        df_dpi=df_dpi,
        cache_key=cache_key,
    )

    # lgb requires class to be zero-based
    y = df_fe[TARGET_KEY] - 1

    return X, y


def compile_features(
    df_fe: pd.DataFrame,
    df_bnum: pd.DataFrame,
    df_dpi: pd.DataFrame,
    cache_key: str = None,
) -> pd.DataFrame:
    preprocess_start = time.time()
    features = io.read_json("./data/all_in_one_cv_features.json")
    intersect = lambda lst: list(set(features).intersection(lst))

    fe_columns = intersect(df_fe.columns)

    X = df_fe[fe_columns]

    print(f"\nExtracted {len(fe_columns)} columns from fe dataset:\n {fe_columns}")

    bnum_extra = bnum.collect_extra_fe(df_bnum)
    bnum_extra_columns = intersect(bnum_extra.columns)

    if len(bnum_extra_columns):
        print(
            f"\nExtracted {len(bnum_extra_columns)} columns from bnum extra features:\n {bnum_extra_columns}"
        )

        X = X.merge(
            bnum_extra[bnum_extra_columns],
            how="left",
            left_index=True,
            right_index=True,
        )

    bnum_features = run_with_cache(
        fn=lambda: bnum.flatten(
            df_bnum=df_bnum,
            features=features,
        ),
        step="bnum",
        cache_key=cache_key,
    )

    if len(bnum_features):
        print(
            f"\nExtracted {len(bnum_features.columns)} columns from bnum dataset:\n {bnum_features.columns.to_list()}"
        )

        X = X.merge(
            bnum_features,
            how="left",
            left_index=True,
            right_index=True,
        )

    dpi_extra = dpi.collect_extra_fe(df_dpi)
    dpi_extra_columns = intersect(dpi_extra.columns)

    if len(dpi_extra_columns):
        print(
            f"\nExtracted {len(dpi_extra_columns)} columns from dpi extra features:\n {dpi_extra_columns}"
        )

        X = X.merge(
            dpi_extra[dpi_extra_columns],
            how="left",
            left_index=True,
            right_index=True,
        )

    dpi_features = run_with_cache(
        fn=lambda: dpi.flatten(
            df_dpi=df_dpi,
            features=features,
        ),
        step="dpi",
        cache_key=cache_key,
    )

    if len(dpi_features):
        interaction_features = []
        normal_features = []

        for key in dpi_features.columns:
            if "daily" in key:
                interaction_features.append(key)
            else:
                normal_features.append(key)

        if len(normal_features):
            print(
                f"\nExtracted {len(normal_features)} columns from dpi dataset:\n {normal_features}"
            )

        if len(interaction_features):
            print(
                f"\nExtracted {len(interaction_features)} interaction columns from dpi dataset:\n {interaction_features}"
            )

        X = X.merge(
            dpi_features,
            how="left",
            left_index=True,
            right_index=True,
        ).reindex(columns=features)

    print(f"\nTotal features: {X.shape}")

    result = compress_df(X)
    print(f"Preprocessing time: {time.time() - preprocess_start:.2f} seconds")

    return result


def run_with_cache(
    fn: callable,
    step: str,
    cache_key: str = None,
):
    if not cache_key:
        return fn()

    return io.run_cached(f"./data/pipeline_cache_{cache_key}_step_{step}.parquet", fn)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = None,
) -> tuple[callable, lgb.Booster]:
    predict, model = model_lgb.train_multiclass(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=best_params,
        num_class=CLASS_NUM,
        seed=RANDOM_SEED,
        name=name,
    )

    return predict, model


def predict(
    model_name: str,
    df_fe: pd.DataFrame,
    df_bnum: pd.DataFrame,
    df_dpi: pd.DataFrame,
):
    predict, model = model_lgb.load(model_name)

    X = compile_features(
        df_fe=df_fe,
        df_bnum=df_bnum,
        df_dpi=df_dpi,
    )

    prediction_start = time.time()
    y_pred = predict(X)
    print(f"Prediction time: {time.time() - prediction_start:.2f} seconds")

    labels = ["<20", "20-30", "30-40", "40-50", ">50"]

    return pd.DataFrame(
        {
            "abon_id": df_fe.index.to_list(),
            "cat_index": y_pred,
            "category": map(lambda index: labels[index], y_pred),
        }
    ).set_index("abon_id")
