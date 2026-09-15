"""Эксперимент 3 -- адаптивный атакующий.

Все атакующие, оценённые до сих пор (evaluate_defense.py,
evaluate_transfer.py), обучались ОДИН РАЗ на чистых эмбеддингах и никогда
не видели защищённый эмбеддинг во время своего обучения -- это
"статический" атакующий, самый слабый разумный противник. Полностью
информированный противник, знающий, какая именно защита развёрнута,
вместо этого переобучился бы (здесь: дообучился) на эмбеддингах,
произведённых ИМЕННО ЭТОЙ защитой. Этот скрипт производит такого
адаптивного атакующего; evaluate_adaptive.py затем сравнивает его со
статическим по каждой защите, чтобы увидеть, сколько защиты выживает
против полностью информированного противника (стандартная практика при
оценке защит -- см. Carlini et al. про оценку adversarial robustness:
всегда включайте адаптивную атаку).

--defense выбирает, к чему адаптируется атакующий:
    gaussian    -> дообучение против gaussian_noise_defense + bound_aware_adaptation
    pgd         -> дообучение против pgd_defense, считаемой на СОБСТВЕННЫХ
                   (прямо сейчас дообучаемых) весах атакующего в каждом
                   батче -- то есть стандартный adversarial training
                   против движущейся цели
    entroguard  -> дообучение против ЗАМОРОЖЕННОГО, уже обученного
                   генератора EntroGuard (развёрнутая защита, известная
                   атакующему)

Стартует от предобученного на чистых эмбеддингах атакующего
(attacker_{arch}) как от инициализации -- он уже умеет декодировать в
целом, ему нужно лишь адаптироваться к распределению защищённых
эмбеддингов, поэтому здесь заметно меньше эпох и меньше LR, чем в
train_attacker.py.

Использование:
    python train_attacker.py --arch transformer      # предпосылка
    python train_entroguard.py --target transformer  # предпосылка для --defense entroguard
    python train_adaptive_attacker.py --defense entroguard
    python train_adaptive_attacker.py --defense gaussian --arch transformer
"""
import argparse
import json
import os

import torch

import config
import utils
from entroguard import (bound_aware_adaptation, cross_entropy_loss,
                         gaussian_noise_defense, pgd_defense, rescale_to_e0_norm)
from metrics import privacy_leakage_report
from models import EmbeddingModel, PerturbationGenerator
from train_attacker import build_attacker


def protect(defense, attacker, generator, e0, input_ids, target_ids, pad_id):
    """Те же три защиты, что и в evaluate_defense.run_defense, но вызываемые
    прямо из цикла обучения. `pgd` намеренно оставлен ВНЕ torch.no_grad()
    (ему нужен autograd через дообучаемого атакующего); остальные обёрнуты,
    так как градиенты им вообще не нужны."""
    if defense == "pgd":
        return pgd_defense(attacker, e0, target_ids, input_ids, pad_id,
                            epsilon=config.EPSILON, step_size=config.PGD_STEP_SIZE,
                            n_steps=config.PGD_STEPS)
    with torch.no_grad():
        if defense == "gaussian":
            return bound_aware_adaptation(e0, gaussian_noise_defense(e0, config.GAUSSIAN_SIGMA),
                                           epsilon=config.EPSILON)
        if defense == "entroguard":
            return bound_aware_adaptation(e0, rescale_to_e0_norm(e0, generator(e0)), epsilon=config.EPSILON)
    raise ValueError(f"unknown defense: {defense}")


def evaluate_adaptive(attacker, generator, defense, emb_model, vocab, sentences):
    attacker.eval()
    hyps, refs = [], []
    for batch in utils.iterate_batches(sentences, config.BATCH_SIZE, shuffle=False):
        emb_ids = utils.embedding_io(vocab, batch)
        with torch.no_grad():
            e0 = emb_model(emb_ids, vocab.pad_id)
        input_ids, target_ids = utils.attacker_io(vocab, batch)
        e_prot = protect(defense, attacker, generator, e0, input_ids, target_ids, vocab.pad_id)
        gen = attacker.greedy_decode(e_prot, vocab.bos_id, vocab.eos_id, config.ATTACKER_SEQ_LEN)
        for i, s in enumerate(batch):
            hyps.append(vocab.decode(gen[i].tolist()).split())
            refs.append(s.split())
    report = privacy_leakage_report(hyps, refs)
    attacker.train()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["transformer", "lstm", "cnn"], default="transformer")
    parser.add_argument("--defense", choices=["gaussian", "pgd", "entroguard"], required=True)
    args = parser.parse_args()

    utils.set_seed()
    train, val, test, vocab = utils.load_dataset()

    emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                max_len=config.MAX_LEN).to(config.device)

    attacker = build_attacker(args.arch, len(vocab)).to(config.device)
    utils.load_checkpoint(attacker, utils.ckpt_name(f"attacker_{args.arch}"))
    # намеренно остаётся обучаемым -- в этом весь смысл адаптации

    generator = None
    if args.defense == "entroguard":
        generator = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
        utils.load_checkpoint(generator, utils.ckpt_name(f"entroguard_{args.arch}"))
        generator.eval()
        for p in generator.parameters():
            p.requires_grad_(False)

    print(f"adapting attacker_{args.arch} against defense={args.defense}, "
          f"params: {utils.count_params(attacker):,}")
    opt = torch.optim.Adam(attacker.parameters(), lr=config.ADAPTIVE_ATTACKER_LR)

    ckpt_name = f"attacker_{args.arch}_adaptive_{args.defense}"
    best_val_bleu = -1.0
    history = []

    for epoch in range(1, config.ADAPTIVE_ATTACKER_EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for batch in utils.iterate_batches(train, config.BATCH_SIZE):
            emb_ids = utils.embedding_io(vocab, batch)
            with torch.no_grad():
                e0 = emb_model(emb_ids, vocab.pad_id)
            input_ids, target_ids = utils.attacker_io(vocab, batch)

            e_prot = protect(args.defense, attacker, generator, e0, input_ids, target_ids, vocab.pad_id)
            logits = attacker(e_prot, input_ids)
            loss = cross_entropy_loss(logits, target_ids, vocab.pad_id)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        val_report = evaluate_adaptive(attacker, generator, args.defense, emb_model, vocab, val)
        print(f"epoch {epoch:3d}  train_ce={total_loss / n_batches:.4f}  "
              f"val_BLEU-2={val_report['BLEU-2']:.4f}  val_ROUGE-1={val_report['ROUGE-1']:.4f}")
        history.append({"epoch": epoch, "train_ce": total_loss / n_batches, **val_report})

        # в отличие от train_attacker.py/train_entroguard.py: сам АТАКУЮЩИЙ
        # хочет МАКСИМИЗИРОВАТЬ свой успех против защиты
        if val_report["BLEU-2"] > best_val_bleu:
            best_val_bleu = val_report["BLEU-2"]
            utils.save_checkpoint(attacker, utils.ckpt_name(ckpt_name))

    utils.load_checkpoint(attacker, utils.ckpt_name(ckpt_name))
    test_report = evaluate_adaptive(attacker, generator, args.defense, emb_model, vocab, test)
    print(f"\n[adaptive attacker vs {args.defense}] TEST -> {test_report}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    hist_path = os.path.join(config.RESULTS_DIR, utils.result_name(f"history_adaptive_{args.arch}_{args.defense}"))
    with open(hist_path, "w") as f:
        json.dump({"arch": args.arch, "defense": args.defense, "seed": config.SEED,
                    "history": history, "test": test_report}, f, indent=2)


if __name__ == "__main__":
    main()
