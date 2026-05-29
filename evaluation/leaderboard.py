import json
import os

import pandas as pd


def build_leaderboard(results_dir="results"):
    rows = []

    if not os.path.isdir(results_dir):
        df = pd.DataFrame(rows)
        print(df)
        return df

    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(results_dir, file), encoding="utf-8") as result_file:
            data = json.load(result_file)

        row = {
            "experiment": file,
            **data["metrics"]
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("safety_accuracy", ascending=False)

    print(df)
    return df