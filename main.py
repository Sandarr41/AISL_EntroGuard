import yaml
from models.qwen import QwenModel
from guards.entroguard import Entroguard
from ds.loader import load_dataset
from evaluation.runner import run_eval
from evaluation.leaderboard import build_leaderboard


def main():
    config = yaml.safe_load(open("config.yaml"))

    model = QwenModel()
    guard = Entroguard(model=model)

    dataset_name = config["dataset"]
    dataset = load_dataset(dataset_name)

    metrics, by_category = run_eval(
        dataset,
        model,
        guard,
        dataset_name=dataset_name
    )

    print("\n=== METRICS ===")
    print(metrics)

    print("\n=== BY CATEGORY ===")
    for k, v in by_category.items():
        print(k, v)

    print("\n=== LEADERBOARD ===")
    build_leaderboard()


if __name__ == "__main__":
    main()