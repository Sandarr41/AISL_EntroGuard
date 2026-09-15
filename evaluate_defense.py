"""Эксперимент 1 -- воспроизводит ключевое сравнение из статьи:
EntroGuard против базовых защит эмбеддингов (без защиты / гауссов шум /
PGD), все оцениваются против ОДНОГО И ТОГО ЖЕ transformer-атакующего и с
ОДНИМ И ТЕМ ЖЕ бюджетом возмущения (epsilon), так что компромисс
приватность-полезность-эффективность измеряется на равных условиях.

Предпосылки:
    python train_attacker.py   --arch transformer
    python train_entroguard.py --target transformer

Результаты:
    results/exp1_metrics_seed<N>.json
    results/exp1_table_seed<N>.md
    results/exp1_privacy_utility_seed<N>.png   (matplotlib)
"""
import json
import os

import torch

import config
import utils
from entroguard import (bound_aware_adaptation, gaussian_noise_defense, pgd_defense,
                         rescale_to_e0_norm)
from metrics import privacy_leakage_report, retrieval_recall_overlap
from models import EmbeddingModel, PerturbationGenerator
from train_attacker import build_attacker


@torch.no_grad()
def embed_all(emb_model, vocab, sentences):
    out = []
    for batch in utils.iterate_batches(sentences, config.BATCH_SIZE, shuffle=False):
        emb_ids = utils.embedding_io(vocab, batch)
        out.append(emb_model(emb_ids, vocab.pad_id))
    return torch.cat(out, dim=0)


def decode_all(attacker, embeddings, vocab):
    hyps = []
    for i in range(0, embeddings.size(0), config.BATCH_SIZE):
        chunk = embeddings[i:i + config.BATCH_SIZE]
        gen = attacker.greedy_decode(chunk, vocab.bos_id, vocab.eos_id, config.ATTACKER_SEQ_LEN)
        hyps.extend(vocab.decode(row.tolist()).split() for row in gen)
    return hyps


def run_defense(name, e0, generator, attacker, input_ids, target_ids, pad_id):
    """Возвращает (protected_embeddings, seconds_per_1k_queries)."""
    with utils.Timer() as t:
        if name == "none":
            e_prot = e0
        elif name == "gaussian":
            e_prot = bound_aware_adaptation(e0, gaussian_noise_defense(e0, config.GAUSSIAN_SIGMA),
                                             epsilon=config.EPSILON)
        elif name == "pgd":
            e_prot = pgd_defense(attacker, e0, target_ids, input_ids, pad_id,
                                  epsilon=config.EPSILON, step_size=config.PGD_STEP_SIZE,
                                  n_steps=config.PGD_STEPS)
        elif name == "entroguard":
            with torch.no_grad():
                e_prot = bound_aware_adaptation(e0, rescale_to_e0_norm(e0, generator(e0)), epsilon=config.EPSILON)
        else:
            raise ValueError(name)
    per_1k = t.elapsed / e0.size(0) * 1000
    return e_prot.detach(), per_1k


def main():
    utils.set_seed()
    train, val, test, vocab = utils.load_dataset()
    ret_queries, ret_passages = utils.load_retrieval_pool()

    emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                max_len=config.MAX_LEN).to(config.device)
    attacker = build_attacker("transformer", len(vocab)).to(config.device)
    utils.load_checkpoint(attacker, utils.ckpt_name("attacker_transformer"))
    attacker.eval()

    generator = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
    utils.load_checkpoint(generator, utils.ckpt_name("entroguard_transformer"))
    generator.eval()

    # ветка приватности: отложенные предложения PersonaChat (или synthetic),
    # декодируются атакующим после каждой защиты
    e0_priv = embed_all(emb_model, vocab, test)
    input_ids_priv, target_ids_priv = utils.attacker_io(vocab, test)
    refs = [s.split() for s in test]

    # ветка полезности: queries MS MARCO защищаются и сопоставляются с базой
    # passages MS MARCO (utils.load_retrieval_pool) -- отдельно от ветки
    # приватности, согласно eq. (5)
    db_embeddings = embed_all(emb_model, vocab, ret_passages)
    e0_util = embed_all(emb_model, vocab, ret_queries)
    input_ids_util, target_ids_util = utils.attacker_io(vocab, ret_queries)

    methods = ["none", "gaussian", "pgd", "entroguard"]
    results = {}
    for name in methods:
        e_prot_priv, ms_priv = run_defense(name, e0_priv, generator, attacker,
                                            input_ids_priv, target_ids_priv, vocab.pad_id)
        hyps = decode_all(attacker, e_prot_priv, vocab)
        privacy = privacy_leakage_report(hyps, refs)

        e_prot_util, ms_util = run_defense(name, e0_util, generator, attacker,
                                            input_ids_util, target_ids_util, vocab.pad_id)
        utility = retrieval_recall_overlap(db_embeddings, e0_util, e_prot_util, k=config.RETRIEVAL_K)

        results[name] = {**privacy, "retrieval_recall@%d" % config.RETRIEVAL_K: utility,
                          "ms_per_1k_queries": (ms_priv + ms_util) / 2}
        print(f"{name:>10s}: {results[name]}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, utils.result_name("exp1_metrics")), "w") as f:
        json.dump(results, f, indent=2)

    write_markdown_table(results)
    try_plot(results)


def write_markdown_table(results):
    cols = ["BLEU-2", "ROUGE-1", "EMR", f"retrieval_recall@{config.RETRIEVAL_K}", "ms_per_1k_queries"]
    lines = ["| defense | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        row = [f"{r[c]:.4f}" if c != "ms_per_1k_queries" else f"{r[c]:.2f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    path = os.path.join(config.RESULTS_DIR, utils.result_name("exp1_table", "md"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


def try_plot(results):
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
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar([i - 0.2 for i in x], bleu, width=0.4, label="утечка приватности (BLEU-2, ниже = лучше)")
    ax1.bar([i + 0.2 for i in x], recall, width=0.4, label="полезность (retrieval recall, выше = лучше)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title("Эксперимент 1: сравнение защит против transformer-атакующего")
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, utils.result_name("exp1_privacy_utility", "png"))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
