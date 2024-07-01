import pandas as pd
from utils.optuna import run_optuna
from utils.optuna_lgb import multiclass_objective
from sklearn.metrics import accuracy_score
import optuna
import json

RANDOM_SEED = 42


with open("./secrets.json", "r", encoding="utf-8") as f:
    secrets = json.load(f)

# setup db access
db_user = secrets["db_user"]
db_password = secrets["db_password"]
db_host = secrets["db_host"]
db_name = secrets["db_name"]
STORAGE = (
    # "sqlite:///optuna_studies.db"
    f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"
)


def train_lgb(
    study_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    cv: bool = False,
):
    study = run_optuna(
        study_name,
        lambda trial: multiclass_objective(
            trial,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metric_fn=accuracy_score,
            num_class=5,
            seed=RANDOM_SEED,
            cv=cv,
        ),
        direction="maximize",
        n_trials=100,
        storage=STORAGE,
        seed=RANDOM_SEED,
    )

    return study


def delete_study(study_name: str):
    optuna.delete_study(
        study_name=study_name,
        storage=STORAGE,
    )


def report(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return {
        "accuracy": accuracy,
    }
