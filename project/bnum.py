from pathlib import Path
import pandas as pd
from utils import compress_df
from utils import io

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
        df_abon_extra_fe["total_call_dur_out"] / df_abon_extra_fe["total_call_cnt_out"]
    ).fillna(0)
    df_abon_extra_fe["avg_call_dur_in"] = (
        df_abon_extra_fe["total_call_dur_in"] / df_abon_extra_fe["total_call_cnt_in"]
    ).fillna(0)
    df_abon_extra_fe["avg_call_dur"] = (
        df_abon_extra_fe["total_call_dur"] / df_abon_extra_fe["total_call_cnt"]
    ).fillna(0)
    df_abon_extra_fe["sms_per_call_cnt"] = (
        df_abon_extra_fe["total_cnt_sms"] / df_abon_extra_fe["total_call_cnt"]
    ).fillna(0)
    df_abon_extra_fe["avg_sms_per_call_dur"] = (
        df_abon_extra_fe["avg_cnt_sms"] / df_abon_extra_fe["avg_call_dur"]
    ).fillna(0)
    df_abon_extra_fe["total_bnum"] = df.groupby("abon_id")["bnum"].count()

    return compress_df(df_abon_extra_fe.reset_index().set_index("abon_id"))


def preprocess(
    name: str,
    bnum_path: str,
    fe_path: str,
    bnum_selection_path: str,
    feature_selection_path: str = None,
):
    destination_path = f"./data/{name}.parquet"

    if Path(destination_path).is_file():
        X = pd.read_parquet(destination_path).rename(
            # migrate already created datasets
            # TODO: can remove if we re-generate datasets
            columns=lambda name: name.replace(" ", "_"),
        )
    else:
        df_num = pd.read_parquet(bnum_path).rename(
            columns=lambda name: name.replace(" ", "_"),
        )

        # expose index as column
        df_num["abon_id"] = df_num.index
        df_num = df_num.reset_index(drop=True)

        df_abon_extra_fe = collect_extra_fe(df_num)

        all_bnums = io.read_json(bnum_selection_path)

        # drop bnums we do not support
        df_num = df_num[df_num["bnum"].isin(all_bnums)]

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
        df_num["cnt_sms"] = df_num["cnt_sms_out"] + df_num["cnt_sms_in"]
        df_num["call_cnt"] = df_num["call_cnt_out"] + df_num["call_cnt_in"]
        df_num["call_dur"] = df_num["call_dur_out"] + df_num["call_dur_in"]
        df_num = df_num.drop(columns=col_to_agg)

        # group dataset by subscriber and flatten all related to subscriber data in one row
        abon_nums = df_num.groupby("abon_id")
        col_to_flatten = [
            "cnt_sms",
            "call_cnt",
            "call_dur",
        ]

        index_map = {}
        columns = ["abon_id"]

        # create columns that we will require to populate
        for bnum in all_bnums:
            for source_key in col_to_flatten:
                key = f"{source_key}_{bnum}"
                index_map[key] = len(columns)
                columns.append(key)

        empty_row = [0] * (len(columns) - 1)

        def row_generator():
            for abon_id, df_abon in abon_nums:
                abon_row = [abon_id] + empty_row

                for _, bnum_row in df_abon.iterrows():
                    for source_key in col_to_flatten:
                        key = f"{source_key}_{bnum_row['bnum']}"
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

        X = compress_df(pd.DataFrame(row_generator(), columns=columns)).set_index(
            "abon_id"
        )
        X = X.merge(
            df_abon_extra_fe,
            how="left",
            left_index=True,
            right_index=True,
        )
        X.to_parquet(destination_path)

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
