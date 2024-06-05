import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def print_missing(df: pd.DataFrame):
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
    print(missing_values_stats)

    return missing_values_stats


def print_uniq(df: pd.DataFrame):
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
