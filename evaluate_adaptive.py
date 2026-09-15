"""Эксперимент 3 -- сколько защиты выживает против полностью
информированного (адаптивного) атакующего?

Для каждой защиты из {gaussian, pgd, entroguard} сравниваются:
  static   : атакующий из Эксперимента 1 -- обучен только на ЧИСТЫХ
             эмбеддингах, никогда не видел эту защиту во время обучения.
  adaptive : атакующий из train_adaptive_attacker.py -- дообучен именно
             против ЭТОЙ защиты.

Оба атакуют РОВНО ОДНИ И ТЕ ЖЕ защищённые тестовые эмбеддинги (защищённый
эмбеддинг генерируется один раз на защиту, для pgd — на градиентах
статического атакующего, как и в методологии Эксперимента 1); разрыв
между static и adaptive BLEU-2/ROUGE-1 — это запас прочности защиты
против противника, точно знающего, что развёрнуто. Такую адаптивную
проверку стоит делать всегда, прежде чем доверять цифрам Эксперимента 1
— известно, что оценка только на статическом атакующем завышает
защищённость (Carlini et al., "On Evaluating Adversarial Robustness").

Предпосылки:
    python train_attacker.py          --arch transformer
    python train_entroguard.py        --target transformer   (для entroguard)
    python train_adaptive_attacker.py --defense gaussian
    python train_adaptive_attacker.py --defense pgd
    python train_adaptive_attacker.py --defense entroguard

Результаты:
    results/exp3_metrics_seed<N>.json
    results/exp3_table_seed<N>.md
"""
import json
import os

import config
import utils
from evaluate_defense import decode_all, embed_all, run_defense
from metrics import privacy_leakage_report
from models import EmbeddingModel, PerturbationGenerator
from train_attacker import build_attacker

DEFENSES = ["gaussian", "pgd", "entroguard"]
ARCH = "transformer"


def main():
    utils.set_seed()
    train, val, test, vocab = utils.load_dataset()

    emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                max_len=config.MAX_LEN).to(config.device)

    static_attacker = build_attacker(ARCH, len(vocab)).to(config.device)
    utils.load_checkpoint(static_attacker, utils.ckpt_name(f"attacker_{ARCH}"))
    static_attacker.eval()

    generator = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
    utils.load_checkpoint(generator, utils.ckpt_name(f"entroguard_{ARCH}"))
    generator.eval()

    e0 = embed_all(emb_model, vocab, test)
    input_ids, target_ids = utils.attacker_io(vocab, test)
    refs = [s.split() for s in test]

    results = {}
    for defense in DEFENSES:
        adaptive_ckpt = os.path.join(config.CKPT_DIR, utils.ckpt_name(f"attacker_{ARCH}_adaptive_{defense}"))
        if not os.path.exists(adaptive_ckpt):
            print(f"skipping defense={defense}: {adaptive_ckpt} not found "
                  f"(train_adaptive_attacker.py --defense {defense} first)")
            continue

        # защищённый эмбеддинг генерируется один раз (pgd использует
        # градиенты СТАТИЧЕСКОГО атакующего, как в Эксперименте 1) -- оба
        # атакующих получают один и тот же вход
        e_prot, _ = run_defense(defense, e0, generator, static_attacker, input_ids, target_ids, vocab.pad_id)

        static_hyps = decode_all(static_attacker, e_prot, vocab)
        static_privacy = privacy_leakage_report(static_hyps, refs)

        adaptive_attacker = build_attacker(ARCH, len(vocab)).to(config.device)
        utils.load_checkpoint(adaptive_attacker, utils.ckpt_name(f"attacker_{ARCH}_adaptive_{defense}"))
        adaptive_attacker.eval()
        adaptive_hyps = decode_all(adaptive_attacker, e_prot, vocab)
        adaptive_privacy = privacy_leakage_report(adaptive_hyps, refs)

        results[defense] = {
            "static_BLEU-2": static_privacy["BLEU-2"], "static_ROUGE-1": static_privacy["ROUGE-1"],
            "adaptive_BLEU-2": adaptive_privacy["BLEU-2"], "adaptive_ROUGE-1": adaptive_privacy["ROUGE-1"],
            "delta_BLEU-2": adaptive_privacy["BLEU-2"] - static_privacy["BLEU-2"],
            "delta_ROUGE-1": adaptive_privacy["ROUGE-1"] - static_privacy["ROUGE-1"],
        }
        print(f"{defense:>10s}: {results[defense]}")

    if not results:
        print("nothing to evaluate -- train at least one adaptive attacker first")
        return

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, utils.result_name("exp3_metrics")), "w") as f:
        json.dump(results, f, indent=2)
    write_markdown_table(results)


def write_markdown_table(results):
    cols = ["static_BLEU-2", "adaptive_BLEU-2", "delta_BLEU-2",
            "static_ROUGE-1", "adaptive_ROUGE-1", "delta_ROUGE-1"]
    lines = ["| defense | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        row = [f"{r[c]:+.4f}" if c.startswith("delta") else f"{r[c]:.4f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    path = os.path.join(config.RESULTS_DIR, utils.result_name("exp3_table", "md"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
