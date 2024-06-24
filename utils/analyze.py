import numpy as np
from sklearn.metrics import (
    classification_report as sklearn_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    y_true_tr,
    y_proba_tr,
    y_true_val=None,
    y_proba_val=None,
    report=False,
):
    def round(input: float) -> float:
        return np.round(input, 4)

    def collect_metrics(y_true, y_proba, dest: dict):
        dest["roc_auc"] = np.round(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
            ),
            4,
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
        print("\n")
        print("Train:")
        print(sklearn_report(y_true_tr, np.argmax(y_proba_tr, axis=1)))

        if not y_true_val.empty:
            print("Test:")
            print(sklearn_report(y_true_val, np.argmax(y_proba_val, axis=1)))
