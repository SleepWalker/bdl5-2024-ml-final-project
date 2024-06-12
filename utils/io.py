import json
import pickle
import pandas as pd
from pathlib import Path


def read_json(path: str) -> any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def read_pickle(path: str) -> any:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path: str, data: any):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def run_cached(file: str, fn: callable):
    if Path(file).is_file():
        df = pd.read_parquet(file)
    else:
        df = fn()
        df.to_parquet(file)

    return df
