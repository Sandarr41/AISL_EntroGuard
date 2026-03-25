import os
import json
import pandas as pd


def build_leaderboard(results_dir="results"):
    rows = []

    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        data = json.load(open(os.path.join(results_dir, file)))

        row = {
            "experiment": file,
            **data["metrics"]
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("safety_accuracy", ascending=False)

    print(df)
    return df