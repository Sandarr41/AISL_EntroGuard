"""Агрегирует метрики по сидам (results/exp{1,2_lstm,2_cnn,3}_metrics_seed*.json)
в mean +/- std по каждой защите и метрике, чтобы число в отчёте не было
случайностью одного сида. Запускать после run_multiseed.py, либо вручную
после того как сгенерированы results/exp*_metrics_seed*.json для двух и
более сидов (например, `ENTRO_SEED=7 python evaluate_defense.py` после
обучения и под этим сидом тоже).

Использование:
    python aggregate_results.py

Результаты (один набор на найденное семейство экспериментов, семейство —
exp1 / exp2_lstm / exp2_cnn / exp3):
    results/<family>_metrics_agg.json   -- {"seeds": [...], "metrics": {...}}
    results/<family>_metrics_agg.md
    results/<family>_BLEU-2_agg.png     (matplotlib, если установлен)
"""
import glob
import json
import os
import re
import statistics as stats

import config

FAMILIES = ["exp1", "exp2_lstm", "exp2_cnn", "exp3"]


def _runs_for(family):
    pattern = os.path.join(config.RESULTS_DIR, f"{family}_metrics_seed*.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"_seed(\d+)\.json$", path)
        if not m:
            continue
        with open(path) as f:
            runs.append((int(m.group(1)), json.load(f)))
    return runs


def aggregate(runs):
    """runs: список (seed, {name: {metric: value}}). Возвращает
    {name: {metric: {"mean", "std", "n"}}}. Метрики, отсутствующие в
    некоторых сидах (в норме такого быть не должно), усредняются по тем
    сидам, где они есть."""
    names = runs[0][1].keys()
    agg = {}
    for name in names:
        agg[name] = {}
        metrics = runs[0][1][name].keys()
        for metric in metrics:
            values = [r[name][metric] for _, r in runs if name in r and metric in r[name]]
            if not values:
                continue
            mean = stats.fmean(values)
            std = stats.pstdev(values) if len(values) > 1 else 0.0
            agg[name][metric] = {"mean": mean, "std": std, "n": len(values)}
    return agg


def write_markdown(family, agg, seeds):
    cols = sorted(next(iter(agg.values())).keys())
    lines = [f"| defense ({family}, n={len(seeds)} seeds {seeds}) | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, metrics in agg.items():
        row = [f"{metrics[c]['mean']:.4f} ± {metrics[c]['std']:.4f}" if c in metrics else "-"
               for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    path = os.path.join(config.RESULTS_DIR, f"{family}_metrics_agg.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"записал {path}")


def try_plot(family, agg):
    metric = "BLEU-2" if "BLEU-2" in next(iter(agg.values())) else None
    if metric is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, график пропущен (pip install matplotlib)")
        return

    names = list(agg.keys())
    means = [agg[n][metric]["mean"] for n in names]
    stds = [agg[n][metric]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=stds, capsize=4)
    ax.set_ylabel(f"{metric} (ниже = лучше приватность)")
    ax.set_title(f"{family}: mean ± std по {agg[names[0]][metric]['n']} сидам")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"{family}_{metric}_agg.png")
    fig.savefig(path, dpi=150)
    print(f"записал {path}")


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    any_found = False
    for family in FAMILIES:
        runs = _runs_for(family)
        if len(runs) < 1:
            continue
        any_found = True
        seeds = [s for s, _ in runs]
        if len(runs) == 1:
            print(f"{family}: найден только 1 сид ({seeds[0]}) -- агрегирую всё равно "
                  f"(std=0), но это ещё не настоящая мультисидовая оценка")
        agg = aggregate(runs)

        path = os.path.join(config.RESULTS_DIR, f"{family}_metrics_agg.json")
        with open(path, "w") as f:
            json.dump({"seeds": seeds, "metrics": agg}, f, indent=2)
        print(f"записал {path}  (seeds={seeds})")
        write_markdown(family, agg, seeds)
        try_plot(family, agg)

    if not any_found:
        print("не найдено results/exp*_metrics_seed*.json -- сначала запустите "
              "evaluate_defense.py / evaluate_transfer.py / evaluate_adaptive.py "
              "(напрямую или через run_multiseed.py)")


if __name__ == "__main__":
    main()
