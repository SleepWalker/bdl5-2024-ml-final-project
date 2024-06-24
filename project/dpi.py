from pathlib import Path
import pandas as pd
from utils import compress_df
from utils import io
from utils import eda

TARGET_KEY = "target"


def collect_extra_fe(df: pd.DataFrame):
    col_to_agg = [
        "MAX_of_day_cnt",
        "SUM_of_Count_events",
        "SUM_of_Duration_sec",
        "SUM_of_Volume_kb",
    ]

    df_dpi_extra_fe = (
        df.groupby("abon_id")[col_to_agg]
        .sum()
        .rename(
            columns={key: f"total_{key}" for key in col_to_agg},
        )
    )
    df_dpi_extra_fe["total_apps"] = df.groupby("abon_id")["Application"].count()

    df_dpi_extra_fe["avg_daily_session_sec_sum"] = (
        df_dpi_extra_fe["total_SUM_of_Duration_sec"]
        / df_dpi_extra_fe["total_MAX_of_day_cnt"]
    )
    df_dpi_extra_fe["avg_daily_session_cnt"] = (
        df_dpi_extra_fe["total_SUM_of_Count_events"]
        / df_dpi_extra_fe["total_MAX_of_day_cnt"]
    )
    df_dpi_extra_fe["avg_daily_traffic_kb"] = (
        df_dpi_extra_fe["total_SUM_of_Volume_kb"]
        / df_dpi_extra_fe["total_MAX_of_day_cnt"]
    )

    df_dpi_extra_fe["avg_session_sec"] = (
        df_dpi_extra_fe["total_SUM_of_Duration_sec"]
        / df_dpi_extra_fe["total_SUM_of_Count_events"]
    )
    df_dpi_extra_fe["avg_session_traffic_kb"] = (
        df_dpi_extra_fe["total_SUM_of_Volume_kb"]
        / df_dpi_extra_fe["total_SUM_of_Count_events"]
    )

    df_dpi_extra_fe["avg_session_bandwidth_kbs"] = (
        df_dpi_extra_fe["total_SUM_of_Volume_kb"]
        / df_dpi_extra_fe["total_SUM_of_Duration_sec"]
    ).fillna(0)

    df_dpi_extra_fe["avg_days_per_app"] = (
        df_dpi_extra_fe["total_MAX_of_day_cnt"] / df_dpi_extra_fe["total_apps"]
    )
    df_dpi_extra_fe["avg_session_sec_sum_per_app"] = (
        df_dpi_extra_fe["total_SUM_of_Duration_sec"] / df_dpi_extra_fe["total_apps"]
    )
    df_dpi_extra_fe["avg_session_cnt_per_app"] = (
        df_dpi_extra_fe["total_SUM_of_Count_events"] / df_dpi_extra_fe["total_apps"]
    )
    df_dpi_extra_fe["avg_traffic_kb_per_app"] = (
        df_dpi_extra_fe["total_SUM_of_Volume_kb"] / df_dpi_extra_fe["total_apps"]
    )
    df_dpi_extra_fe["avg_daily_session_sec_per_app"] = (
        df_dpi_extra_fe["avg_daily_session_sec_sum"] / df_dpi_extra_fe["total_apps"]
    )
    df_dpi_extra_fe["avg_daily_session_cnt_per_app"] = (
        df_dpi_extra_fe["avg_daily_session_cnt"] / df_dpi_extra_fe["total_apps"]
    )
    df_dpi_extra_fe["avg_daily_traffic_kb_per_app"] = (
        df_dpi_extra_fe["avg_daily_traffic_kb"] / df_dpi_extra_fe["total_apps"]
    )

    eda.print_missing(df_dpi_extra_fe)

    return compress_df(df_dpi_extra_fe.reset_index().set_index("abon_id"))


def preprocess(
    name: str,
    dpi_path: str,
    fe_path: str,
    dpi_selection_path: str,
    feature_selection_path: str = None,
):
    destination_path = f"./data/{name}.parquet"

    def do_process():
        df_dpi = pd.read_parquet(dpi_path)

        # expose index as column
        df_dpi["abon_id"] = df_dpi.index
        df_dpi = df_dpi.reset_index(drop=True)

        df_abon_extra_fe = collect_extra_fe(df_dpi)

        all_dpi = io.read_json(dpi_selection_path)

        # drop apps we do not support
        df_dpi = df_dpi[df_dpi["Application"].isin(all_dpi)]

        # group dataset by subscriber and flatten all related to subscriber data in one row
        abon_apps = df_dpi.groupby("abon_id")
        col_to_flatten = [
            "MAX_of_day_cnt",
            "SUM_of_Duration_sec",
            "SUM_of_Count_events",
            "SUM_of_Volume_kb",
            # the rate of particular app daily activity among all subscriber apps
            "daily_session_dur_rate",
            "daily_session_cnt_rate",
            "daily_traffic_rate",
        ]

        index_map = {}
        columns = ["abon_id"]

        # create columns that we will require to populate
        for app in all_dpi:
            for source_key in col_to_flatten:
                key = f"{source_key}_{app}"
                index_map[key] = len(columns)
                columns.append(key)

        # -1 because we will pre-populate abon_id
        empty_row = [0] * (len(columns) - 1)

        def row_generator():
            for abon_id, df_abon in abon_apps:
                abon_row = [abon_id] + empty_row

                # add extra features to rank app importance among other subscriber's apps
                df_abon["daily_session_dur"] = (
                    df_abon["SUM_of_Duration_sec"] / df_abon["MAX_of_day_cnt"]
                )
                df_abon["daily_session_dur_rate"] = (
                    df_abon["daily_session_dur"] / df_abon["daily_session_dur"].sum()
                )

                df_abon["daily_session_cnt"] = (
                    df_abon["SUM_of_Count_events"] / df_abon["MAX_of_day_cnt"]
                )
                df_abon["daily_session_cnt_rate"] = (
                    df_abon["daily_session_cnt"] / df_abon["daily_session_cnt"].sum()
                )

                df_abon["daily_traffic"] = (
                    df_abon["SUM_of_Volume_kb"] / df_abon["MAX_of_day_cnt"]
                )
                df_abon["daily_traffic_rate"] = (
                    df_abon["daily_traffic"] / df_abon["daily_traffic"].sum()
                )

                for _, app_row in df_abon.iterrows():
                    for source_key in col_to_flatten:
                        # pandas messes app dtypes after iterrows(). Ensure app is int
                        key = f"{source_key}_{int(app_row['Application'])}"
                        abon_row[index_map[key]] = app_row[source_key]

                yield abon_row

        X = compress_df(pd.DataFrame(row_generator(), columns=columns)).set_index(
            "abon_id"
        )
        X = X.merge(
            df_abon_extra_fe,
            how="left",
            left_index=True,
            right_index=True,
        )

        return X

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
