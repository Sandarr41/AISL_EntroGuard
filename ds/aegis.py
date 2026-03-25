from datasets import load_dataset


def load_aegis(split="train", limit=None):
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety", split=split)

    dataset = []

    for i, item in enumerate(ds):
        if limit and i >= limit:
            break

        text = item.get("text", "")

        label_raw = str(item.get("label", "")).lower()

        if "unsafe" in label_raw or "harm" in label_raw:
            label = "harmful"
        else:
            label = "safe"

        dataset.append({
            "prompt": text,
            "label": label,
            "category": item.get("category", "general")
        })

    return dataset