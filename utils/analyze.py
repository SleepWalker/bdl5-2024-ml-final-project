import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    classification_report as sklearn_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    RocCurveDisplay,
)


def classification_report(
    y_true_tr,
    y_pred_tr,
    y_true_val=None,
    y_pred_val=None,
    report=False,
):
    def round(input: float) -> float:
        return np.round(input, 4)

    def collect_metrics(y_true, y_pred, dest):
        dest["roc_auc"] = round(roc_auc_score(y_true, y_pred))
        dest["accuracy"] = round(accuracy_score(y_true, y_pred))
        dest["precision"] = round(precision_score(y_true, y_pred))
        dest["recall"] = round(recall_score(y_true, y_pred))
        dest["f1_score"] = round(f1_score(y_true, y_pred))

    print("{:<15} {:<10} {:<10} {:<10}".format("Metrics", "Train", "Test", "\u0394"))

    metrics_dict = {}
    collect_metrics(y_true_tr, y_pred_tr, metrics_dict)

    if y_true_val is not None:
        metrics_dict_test = {}
        collect_metrics(y_true_val, y_pred_val, metrics_dict_test)

        for metrics, value in metrics_dict.items():
            value_test = metrics_dict_test[metrics]
            diff = round(metrics_dict_test[metrics] - value)
            print(
                "{:<15} {:<10} {:<10} {:<10}".format(metrics, value, value_test, diff)
            )
    else:
        for metrics, value in metrics_dict.items():
            print("{:<15} {:<10}".format(metrics, value))

    if report:
        print("\n")
        print("Train:")
        print(sklearn_report(y_true_tr, y_pred_tr))
        if not y_true_val.empty:
            print("Test:")
            print(sklearn_report(y_true_val, y_pred_val))


def multiclass_report(
    y_true_tr: np.ndarray,
    y_proba_tr: np.ndarray,
    labels: list,
    y_true_val: np.ndarray = None,
    y_proba_val: np.ndarray = None,
    report=False,
    visualization=False,
):
    def round(input: float) -> float:
        return np.round(input, 4)

    def collect_metrics(y_true, y_proba, dest: dict):
        dest["roc_auc"] = round(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
            )
        )

        y_pred = np.argmax(y_proba, axis=1)

        dest["accuracy"] = round(accuracy_score(y_true, y_pred))
        dest["precision"] = round(precision_score(y_true, y_pred, average="micro"))
        dest["recall"] = round(recall_score(y_true, y_pred, average="micro"))
        dest["f1_score"] = round(f1_score(y_true, y_pred, average="micro"))

        # collect for each class
        # dest.update(
        #     {
        #         f"precision_c_{i}": round(x)
        #         for i, x in enumerate(precision_score(y_true, y_pred, average=None))
        #     }
        # )
        # dest.update(
        #     {
        #         f"recall_c_{i}": round(x)
        #         for i, x in enumerate(recall_score(y_true, y_pred, average=None))
        #     }
        # )
        # dest.update(
        #     {
        #         f"f1_c_{i}": round(x)
        #         for i, x in enumerate(f1_score(y_true, y_pred, average=None))
        #     }
        # )

    print("{:<15} {:<10} {:<10} {:<10}".format("Metrics", "Train", "Test", "\u0394"))

    metrics_dict = {}
    collect_metrics(y_true_tr, y_proba_tr, metrics_dict)

    if y_true_val is not None:
        metrics_dict_test = {}
        collect_metrics(y_true_val, y_proba_val, metrics_dict_test)

        for metrics, value in metrics_dict.items():
            value_test = metrics_dict_test[metrics]
            diff = round(metrics_dict_test[metrics] - value)
            print(
                "{:<15} {:<10} {:<10} {:<10}".format(metrics, value, value_test, diff)
            )
    else:
        for metrics, value in metrics_dict.items():
            print("{:<15} {:<10}".format(metrics, value))

    if report:
        y_tr = np.argmax(y_proba_tr, axis=1)

        print("\n")
        print("Train:")
        print(sklearn_report(y_true_tr, y_tr))

        if visualization:
            plot_confusion_matrix(
                y_true_tr,
                y_tr,
                labels=labels,
                subtitle="Recall. Train dataset",
                normalize="true",
            )
            plot_confusion_matrix(
                y_true_tr,
                y_tr,
                labels=labels,
                subtitle="Precision. Train dataset",
                normalize="pred",
            )
            plot_multiclass_roc_curve(
                y_true_tr,
                y_proba_tr,
                labels=labels,
                subtitle="Train dataset",
            )
            plot_scoring_distribution(
                y_true_tr,
                y_proba_tr,
                labels=labels,
                subtitle="Train dataset",
            )

        if not y_true_val.empty:
            y_val = np.argmax(y_proba_val, axis=1)

            print("Test:")
            print(sklearn_report(y_true_val, y_val))

            if visualization:
                plot_confusion_matrix(
                    y_true_val,
                    y_val,
                    labels=labels,
                    subtitle="Recall. Test dataset",
                    normalize="true",
                )
                plot_confusion_matrix(
                    y_true_val,
                    y_val,
                    labels=labels,
                    subtitle="Precision. Test dataset",
                    normalize="pred",
                )
                plot_multiclass_roc_curve(
                    y_true_val,
                    y_proba_val,
                    labels=labels,
                    subtitle="Test dataset",
                )
                plot_multiclass_roc_curve_train_vs_test(
                    y_true_tr=y_true_tr,
                    y_proba_tr=y_proba_tr,
                    y_true_val=y_true_val,
                    y_proba_val=y_proba_val,
                    labels=labels,
                )
                plot_scoring_distribution(
                    y_true_val,
                    y_proba_val,
                    labels=labels,
                    subtitle="Test dataset",
                )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    subtitle: str = None,
    normalize=None,
):
    data = confusion_matrix(y_true, y_pred, normalize=normalize)

    title = "Confusion matrix"

    if subtitle:
        title = f"{title} ({subtitle})"

    px.imshow(
        pd.DataFrame(data, index=labels, columns=labels),
        text_auto=".0%",
        title=title,
        height=800,
        width=800,
    ).update_xaxes(
        showgrid=False,
        title_text="predicted label",
    ).update_yaxes(
        showgrid=False,
        title_text="actual label",
    ).update_layout(
        title_font_size=30,
        font_size=15,
    ).show()


def plot_multiclass_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: list,
    subtitle: str = None,
):
    num_classes = y_pred_proba.shape[1]

    assert num_classes == len(
        labels
    ), f"y_pred_proba size {num_classes} does not match labels count {len(labels)}"

    label_binarizer = LabelBinarizer().fit(np.arange(num_classes))
    y_true_ohe = label_binarizer.transform(y_true)

    fig, ax = plt.subplots(figsize=(6, 6))

    RocCurveDisplay.from_predictions(
        y_true_ohe.ravel(),
        y_pred_proba.ravel(),
        name="micro-average ROC curve",
        color="deeppink",
        linestyle=":",
        linewidth=4,
        ax=ax,
    )

    for class_id, label in enumerate(labels):
        RocCurveDisplay.from_predictions(
            y_true_ohe[:, class_id],
            y_pred_proba[:, class_id],
            name=f"ROC for {label}",
            ax=ax,
            # plot only for last class because it's the same for all
            plot_chance_level=(class_id == len(labels) - 1),
        )

    title = "One-vs-Rest Receiver Operating Characteristic (OVR ROC)"

    if subtitle:
        title = f"{title}\n{subtitle}"

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )


def plot_multiclass_roc_curve_train_vs_test(
    y_true_tr: np.ndarray,
    y_proba_tr: np.ndarray,
    y_true_val: np.ndarray,
    y_proba_val: np.ndarray,
    labels: list,
):
    num_classes = y_proba_tr.shape[1]

    assert num_classes == len(
        labels
    ), f"y_pred_proba size {num_classes} does not match labels count {len(labels)}"

    label_binarizer = LabelBinarizer().fit(np.arange(num_classes))
    y_true_tr_ohe = label_binarizer.transform(y_true_tr)
    y_true_val_ohe = label_binarizer.transform(y_true_val)

    fig, ax = plt.subplots(figsize=(6, 6))

    RocCurveDisplay.from_predictions(
        y_true_tr_ohe.ravel(),
        y_proba_tr.ravel(),
        name="ROC for Train",
        ax=ax,
    )

    RocCurveDisplay.from_predictions(
        y_true_val_ohe.ravel(),
        y_proba_val.ravel(),
        name="ROC for Test",
        ax=ax,
        plot_chance_level=True,
    )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Train vs test ROC OVR",
    )


def plot_scoring_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: list,
    subtitle: str = None,
):
    num_classes = y_pred_proba.shape[1]

    assert num_classes == len(
        labels
    ), f"y_pred_proba size {num_classes} does not match labels count {len(labels)}"

    label_binarizer = LabelBinarizer().fit(np.arange(num_classes))
    y_true_ohe = label_binarizer.transform(y_true)

    for class_id, label in enumerate(labels):
        title = f"Probability distribution '{label}' vs rest"

        if subtitle:
            title = f"{title} ({subtitle})"

        fig = px.histogram(
            pd.DataFrame(
                {
                    "prob": y_pred_proba[:, class_id],
                    label: y_true_ohe[:, class_id],
                }
            ),
            x="prob",
            color=label,
            color_discrete_map={1: "green", 0: "red"},
            nbins=50,
            marginal="box",
            title=title,
            barmode="overlay",
        )

        fig.show()
