from datasets import load_dataset


def load_sorrybench(split="train", limit=None):
    ds = load_dataset("sorry-bench/sorry-bench", split=split)

    dataset = []

    for i, item in enumerate(ds):
        if limit and i >= limit:
            break

        prompt = item["prompt"]

        should_refuse = item.get("should_refuse", True)

        label = "harmful" if should_refuse else "safe"

        dataset.append({
            "prompt": prompt,
            "label": label,
            "category": item.get("category", "refusal")
        })

    return dataset