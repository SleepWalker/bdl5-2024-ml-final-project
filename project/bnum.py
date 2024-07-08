import pandas as pd
import numpy as np
from utils import compress_df
from utils import io
from utils import eda

TARGET_KEY = "target"


def collect_extra_fe(df: pd.DataFrame):
    col_to_agg = [
        "cnt_sms_out",
        "cnt_sms_in",
        "call_cnt_in",
        "call_cnt_out",
        "call_dur_in",
        "call_dur_out",
    ]

    df_abon_extra_fe = (
        df.groupby("abon_id")[col_to_agg]
        .sum()
        .rename(
            columns={key: f"total_{key}" for key in col_to_agg},
        )
    )
    df_abon_extra_fe["total_call_cnt"] = (
        df_abon_extra_fe["total_call_cnt_in"] + df_abon_extra_fe["total_call_cnt_out"]
    )
    df_abon_extra_fe["total_call_dur"] = (
        df_abon_extra_fe["total_call_dur_in"] + df_abon_extra_fe["total_call_dur_out"]
    )
    df_abon_extra_fe["total_cnt_sms"] = (
        df_abon_extra_fe["total_cnt_sms_in"] + df_abon_extra_fe["total_cnt_sms_out"]
    )
    df_abon_extra_fe["avg_cnt_sms"] = (
        df_abon_extra_fe["total_cnt_sms"] / df.groupby("abon_id")["bnum"].count()
    )
    df_abon_extra_fe["avg_call_dur_out"] = (
        (
            df_abon_extra_fe["total_call_dur_out"]
            / df_abon_extra_fe["total_call_cnt_out"]
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df_abon_extra_fe["avg_call_dur_in"] = (
        (df_abon_extra_fe["total_call_dur_in"] / df_abon_extra_fe["total_call_cnt_in"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df_abon_extra_fe["avg_call_dur"] = (
        (df_abon_extra_fe["total_call_dur"] / df_abon_extra_fe["total_call_cnt"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df_abon_extra_fe["sms_per_call_cnt"] = (
        (df_abon_extra_fe["total_cnt_sms"] / df_abon_extra_fe["total_call_cnt"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df_abon_extra_fe["avg_sms_per_call_dur"] = (
        (df_abon_extra_fe["avg_cnt_sms"] / df_abon_extra_fe["avg_call_dur"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df_abon_extra_fe["total_bnum"] = df.groupby("abon_id")["bnum"].count()

    eda.print_missing(df_abon_extra_fe)

    return compress_df(df_abon_extra_fe.reset_index().set_index("abon_id"))


def preprocess(
    name: str,
    bnum_path: str,
    fe_path: str,
    bnum_selection_path: str,
    feature_selection_path: str = None,
):
    destination_path = f"./data/{name}.parquet"

    def do_process():
        df_bnum = pd.read_parquet(bnum_path)
        all_bnums = io.read_json(bnum_selection_path)

        df_abon_extra_fe = collect_extra_fe(df_bnum)

        X = flatten(
            df_bnum=df_bnum,
            bnum_list=all_bnums,
            features=(
                io.read_json(feature_selection_path) if feature_selection_path else []
            ),
        )

        return X.merge(
            df_abon_extra_fe,
            how="left",
            left_index=True,
            right_index=True,
        )

    X = io.run_cached(destination_path, do_process)

    df_targets = pd.read_parquet(
        fe_path,
        columns=[TARGET_KEY],
    )

    # lgb requires class to be zero-based
    y = df_targets.loc[X.index.to_list()] - 1

    if feature_selection_path:
        features = io.read_json(feature_selection_path)
        X = X[features]

    return (X, y)


def flatten(
    df_bnum: pd.DataFrame,
    # list of bnum to include in resulting dataset
    bnum_list: list = [],
    # list of aggregated feature names (the names from resulting dataset) to include in final dataset
    features: list = [],
) -> pd.DataFrame:
    # expose abon_id index as column
    df_bnum = df_bnum.reset_index()

    if len(bnum_list):
        # drop bnums we do not support
        df_bnum = df_bnum[df_bnum["bnum"].isin(bnum_list)]

    # if we will store all the columns we will get a huge dataset because
    # it will be sparse matrix with thousands of columns
    # so we will merge in/out together
    col_to_agg = [
        "cnt_sms_out",
        "cnt_sms_in",
        "call_cnt_in",
        "call_cnt_out",
        "call_dur_in",
        "call_dur_out",
    ]
    df_bnum["cnt_sms"] = df_bnum["cnt_sms_out"] + df_bnum["cnt_sms_in"]
    df_bnum["call_cnt"] = df_bnum["call_cnt_out"] + df_bnum["call_cnt_in"]
    df_bnum["call_dur"] = df_bnum["call_dur_out"] + df_bnum["call_dur_in"]
    df_bnum = df_bnum.drop(columns=col_to_agg)

    # group dataset by subscriber and flatten all related to subscriber data in one row
    col_to_flatten = [
        "cnt_sms",
        "call_cnt",
        "call_dur",
    ]

    index_map = {}
    columns = ["abon_id"]
    bnum_to_pick = set()

    # create columns that we will require to populate
    for bnum in df_bnum["bnum"].unique():
        for source_key in col_to_flatten:
            key = f"{source_key}_{bnum}".replace(" ", "_")

            if len(features) and not (key in features):
                continue

            index_map[key] = len(columns)
            columns.append(key)
            bnum_to_pick.add(bnum)

    # -1 because we will pre-populate abon_id
    empty_row = [0] * (len(columns) - 1)
    abon_bnums = df_bnum[df_bnum["bnum"].isin(list(bnum_to_pick))].groupby("abon_id")

    def row_generator():
        for abon_id, df_abon in abon_bnums:
            abon_row = [abon_id] + empty_row

            for _, bnum_row in df_abon.iterrows():
                for source_key in col_to_flatten:
                    key = f"{source_key}_{bnum_row['bnum']}".replace(" ", "_")

                    if key in index_map:
                        abon_row[index_map[key]] = bnum_row[source_key]

            yield abon_row

    # import csv
    # from pathlib import Path
    # from utils import compress_df

    # if not Path("./flat-bnums.csv").is_file():
    #     with open("./flat-bnums.csv", "w") as file:
    #         writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    #         writer.writerow(columns)
    #         writer.writerows(row_generator())

    # X = compress_df(pd.read_csv('./flat-bnums.csv'))

    return compress_df(
        pd.DataFrame(
            row_generator(),
            columns=columns,
        )
    ).set_index("abon_id")
