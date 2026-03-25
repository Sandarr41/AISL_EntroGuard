from datasets import load_dataset


def load_xstest(split="test", limit=None):
    ds = load_dataset("allenai/xstest", split=split)

    dataset = []

    for i, item in enumerate(ds):
        if limit and i >= limit:
            break

        prompt = item.get("prompt") or item.get("text")

        category = item.get("category", "unknown")

        if any(k in category.lower() for k in ["harm", "attack", "jailbreak"]):
            label = "harmful"
        else:
            label = "safe"

        dataset.append({
            "prompt": prompt,
            "label": label,
            "category": category
        })

    return dataset