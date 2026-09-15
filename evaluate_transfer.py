"""Эксперимент 2 -- обобщение за пределы семейства архитектур.

Сигнал максимизации энтропии EntroGuard (eq. 8) считается по блочным
logit-lens распределениям transformer-атакующего. Работает ли получившаяся
защита против атакующего с совершенно другой архитектурой? Проверяется на
ДВУХ независимых нетрансформерных семействах -- LSTM (рекуррентность) и
causal-CNN декодере (свёртка) -- каждое в двух условиях:

  zero-shot : переиспользуется генератор, обученный против TRANSFORMER-
              атакующего (train_entroguard.py --target transformer),
              оценивается против атакующего, которого он никогда не видел
              во время обучения.
  native    : отдельный генератор обучается прямо против этого целевого
              атакующего (train_entroguard.py --target lstm|cnn), используя
              его собственные послойные распределения -- тот же рецепт,
              заново применённый нативно к нетрансформерной цели (НЕ путать
              с "адаптивным атакующим" из evaluate_adaptive.py, где вместо
              этого дообучается сам АТАКУЮЩИЙ против фиксированной защиты).

Обе версии сравниваются с теми же бейзлайнами none/gaussian/pgd (против
того же целевого атакующего), так что всё на равных условиях. Один файл
результатов на целевую архитектуру, так как одна таблица на все сразу
была бы почти нечитаемой.

Предпосылки (для каждой целевой архитектуры, которую хотите включить):
    python train_attacker.py   --arch transformer
    python train_attacker.py   --arch lstm | cnn
    python train_entroguard.py --target transformer
    python train_entroguard.py --target lstm | cnn

Результаты (TARGET — lstm или cnn):
    results/exp2_TARGET_metrics_seed<N>.json
    results/exp2_TARGET_table_seed<N>.md
    results/exp2_TARGET_privacy_utility_seed<N>.png   (matplotlib)
"""
import json
import os

import config
import utils
from evaluate_defense import decode_all, embed_all, run_defense
from metrics import privacy_leakage_report, retrieval_recall_overlap
from models import EmbeddingModel, PerturbationGenerator
from train_attacker import build_attacker

TARGET_ARCHS = ["lstm", "cnn"]


def run_for_target(target_arch, emb_model, vocab, test, ret_queries, ret_passages):
    attacker = build_attacker(target_arch, len(vocab)).to(config.device)
    utils.load_checkpoint(attacker, utils.ckpt_name(f"attacker_{target_arch}"))
    attacker.eval()

    gen_zero_shot = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
    utils.load_checkpoint(gen_zero_shot, utils.ckpt_name("entroguard_transformer"))
    gen_zero_shot.eval()

    gen_native = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
    utils.load_checkpoint(gen_native, utils.ckpt_name(f"entroguard_{target_arch}"))
    gen_native.eval()

    # ветка приватности: отложенные предложения PersonaChat (или synthetic),
    # декодируются целевым атакующим после каждой защиты
    e0_priv = embed_all(emb_model, vocab, test)
    input_ids_priv, target_ids_priv = utils.attacker_io(vocab, test)
    refs = [s.split() for s in test]

    # ветка полезности: queries MS MARCO против базы passages MS MARCO
    # (utils.load_retrieval_pool), как и в evaluate_defense.py
    db_embeddings = embed_all(emb_model, vocab, ret_passages)
    e0_util = embed_all(emb_model, vocab, ret_queries)
    input_ids_util, target_ids_util = utils.attacker_io(vocab, ret_queries)

    conditions = [
        ("none", None), ("gaussian", None), ("pgd", None),
        ("entroguard", gen_zero_shot), ("entroguard", gen_native),
    ]
    labels = ["none", "gaussian", "pgd",
              "entroguard_transformer(zero-shot)", f"entroguard_{target_arch}(native)"]

    results = {}
    for (name, generator), label in zip(conditions, labels):
        e_prot_priv, ms_priv = run_defense(name, e0_priv, generator, attacker,
                                            input_ids_priv, target_ids_priv, vocab.pad_id)
        hyps = decode_all(attacker, e_prot_priv, vocab)
        privacy = privacy_leakage_report(hyps, refs)

        e_prot_util, ms_util = run_defense(name, e0_util, generator, attacker,
                                            input_ids_util, target_ids_util, vocab.pad_id)
        utility = retrieval_recall_overlap(db_embeddings, e0_util, e_prot_util, k=config.RETRIEVAL_K)

        results[label] = {**privacy, "retrieval_recall@%d" % config.RETRIEVAL_K: utility,
                           "ms_per_1k_queries": (ms_priv + ms_util) / 2}
        print(f"[{target_arch:>11s}] {label:>36s}: {results[label]}")
    return results


def main():
    utils.set_seed()
    train, val, test, vocab = utils.load_dataset()
    ret_queries, ret_passages = utils.load_retrieval_pool()

    emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                max_len=config.MAX_LEN).to(config.device)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for target_arch in TARGET_ARCHS:
        ckpt = os.path.join(config.CKPT_DIR, utils.ckpt_name(f"attacker_{target_arch}"))
        if not os.path.exists(ckpt):
            print(f"skipping target={target_arch}: {ckpt} not found "
                  f"(train_attacker.py --arch {target_arch} first)")
            continue

        results = run_for_target(target_arch, emb_model, vocab, test, ret_queries, ret_passages)

        with open(os.path.join(config.RESULTS_DIR, utils.result_name(f"exp2_{target_arch}_metrics")), "w") as f:
            json.dump(results, f, indent=2)
        write_markdown_table(results, target_arch)
        try_plot(results, target_arch)


def write_markdown_table(results, target_arch):
    cols = ["BLEU-2", "ROUGE-1", "EMR", f"retrieval_recall@{config.RETRIEVAL_K}", "ms_per_1k_queries"]
    lines = [f"| defense (vs {target_arch}) | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        row = [f"{r[c]:.4f}" if c != "ms_per_1k_queries" else f"{r[c]:.2f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    path = os.path.join(config.RESULTS_DIR, utils.result_name(f"exp2_{target_arch}_table", "md"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def try_plot(results, target_arch):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен, график пропущен (pip install matplotlib)")
        return

    names = list(results.keys())
    bleu = [results[n]["BLEU-2"] for n in names]
    recall = [results[n][f"retrieval_recall@{config.RETRIEVAL_K}"] for n in names]

    x = range(len(names))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar([i - 0.2 for i in x], bleu, width=0.4, label="утечка приватности (BLEU-2, ниже = лучше)")
    ax1.bar([i + 0.2 for i in x], recall, width=0.4, label="полезность (retrieval recall, выше = лучше)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title(f"Эксперимент 2: перенос на {target_arch}-атакующего")
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, utils.result_name(f"exp2_{target_arch}_privacy_utility", "png"))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
