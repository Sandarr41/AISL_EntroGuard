import yaml

from ds.loader import load_dataset
from evaluation.leaderboard import build_leaderboard
from evaluation.runner import run_eval
from guards.entroguard import Entroguard
from models.qwen import QwenModel


def main():
    with open("config.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    generation_config = config.get("generation", {})
    model = QwenModel(
        model_name=config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=generation_config.get("max_tokens", 150),
    )
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