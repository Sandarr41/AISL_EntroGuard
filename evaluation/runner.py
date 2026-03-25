import json
from evaluation.metrics import compute_metrics, compute_by_category


def run_eval(dataset, model, guard, dataset_name="unknown"):
    results = []

    for sample in dataset:
        prompt = sample["prompt"]

        safe_prompt = guard.filter_input(prompt)
        response = model.generate(safe_prompt)
        final_response = guard.filter_output(response)

        results.append({
            "prompt": prompt,
            "response": final_response,
            "label": sample["label"],
            "category": sample.get("category", "general")
        })

    metrics = compute_metrics(results)
    by_category = compute_by_category(results)

    filename = f"results/{dataset_name}.json"
    json.dump({
        "metrics": metrics,
        "by_category": by_category,
        "results": results
    }, open(filename, "w"), indent=2)

    return metrics, by_category