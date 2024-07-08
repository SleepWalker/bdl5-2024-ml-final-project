import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_missing(df: pd.DataFrame):
    df_inf = df.select_dtypes(include=["int", "float"])
    df_inf = df_inf[df_inf.isin([np.inf, -np.inf])][
        df_inf.columns.to_series()[np.isinf(df_inf).any()]
    ]

    if len(df_inf.columns):
        print(f"Found {len(df_inf.columns)} columns with inf values")

        with pd.option_context("display.max_rows", None):
            print(df_inf[df_inf.isin([np.inf, -np.inf]).any(axis=1)])
    else:
        print("No columns with inf found!")

    na_vector = df.isna().sum()
    missing_values_totals = na_vector[na_vector > 0].sort_values(ascending=False)
    missing_values_percent = (missing_values_totals / len(df) * 100).round(2)
    missing_values_stats = pd.concat(
        [
            missing_values_totals,
            missing_values_percent,
        ],
        axis=1,
        keys=["Total", "Rate"],
    )

    missing_values_feature_count = len(missing_values_stats)

    print("Missing values report:")

    if missing_values_feature_count == 0:
        print("No missing values found!")

        return missing_values_stats

    print(f"Features count without missing values: {len(na_vector[na_vector == 0])}\n")

    print(
        f"Percent of missing values by feature (features count {missing_values_feature_count}):"
    )

    with pd.option_context("display.max_rows", None):
        print(missing_values_stats)

    return missing_values_stats


def print_uniq(df: pd.DataFrame):
    # TODO: df.describe(include="object") may provide good info too
    uniq_counts = df.nunique().sort_values()

    print("Unique values report:")

    with pd.option_context("display.max_rows", None):
        print(
            pd.DataFrame(
                {
                    "uniq": uniq_counts,
                    "rate": uniq_counts / len(df),
                }
            )
        )

    print(f"Total rows: {len(df)}")


def plot_na(df: pd.DataFrame):
    plt.figure(figsize=(15, 10))

    sns.heatmap(df.isna().transpose(), cbar_kws={"label": "Missing Data"})
