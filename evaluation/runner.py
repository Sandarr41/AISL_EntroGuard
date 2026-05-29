import json
import os

from evaluation.metrics import compute_by_category, compute_metrics


def _is_blocked(text):
    return text.startswith("[BLOCKED")


def run_eval(dataset, model, guard, dataset_name="unknown"):
    results = []

    for sample in dataset:
        prompt = sample["prompt"]

        safe_prompt = guard.filter_input(prompt)
        input_blocked = _is_blocked(safe_prompt)

        if input_blocked:
            raw_response = safe_prompt
            final_response = safe_prompt
            output_blocked = False
        else:
            raw_response = model.generate(safe_prompt)
            final_response = guard.filter_output(raw_response)
            output_blocked = _is_blocked(final_response)

        results.append({
            "prompt": prompt,
            "response": final_response,
            "raw_response": raw_response,
            "label": sample["label"],
            "category": sample.get("category", "general"),
            "input_blocked": input_blocked,
            "output_blocked": output_blocked,
            "blocked": input_blocked or output_blocked,
        })

    metrics = compute_metrics(results)
    by_category = compute_by_category(results)

    os.makedirs("results", exist_ok=True)
    filename = f"results/{dataset_name}.json"
    with open(filename, "w", encoding="utf-8") as result_file:
        json.dump({
            "metrics": metrics,
            "by_category": by_category,
            "results": results
        }, result_file, indent=2)

    return metrics, by_category