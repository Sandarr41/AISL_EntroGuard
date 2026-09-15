"""Обучает генератор возмущений EntroGuard G_theta (eq. 8-10) против
предобученного, ЗАМОРОЖЕННОГО атакующего. Оптимизируется только G_theta;
атакующий выступает фиксированным дифференцируемым критиком, дающим
сигнал энтропии / реконструкции.

--target выбирает, против какого семейства атакующих обучается G_theta:
    transformer  -> исходная постановка статьи (Эксперимент 1)
    lstm, cnn    -> нетрансформерные цели, используются в Эксперименте 2,
                    чтобы проверить, обобщается ли рецепт максимизации
                    энтропии как метод обучения за пределы transformer'а,
                    под который он изначально проектировался (в отличие от
                    zero-shot переноса, который просто переиспользует
                    генератор, обученный на transformer'е, как есть)

Использование:
    python train_attacker.py --arch transformer   # предпосылка
    python train_entroguard.py --target transformer
    python train_entroguard.py --target cnn
"""
import argparse
import json
import os

import torch

import config
import utils
from entroguard import (bound_aware_adaptation, cross_entropy_loss,
                         entropy_loss, entroguard_train_loss, rescale_to_e0_norm,
                         similarity_loss)
from metrics import privacy_leakage_report
from models import EmbeddingModel, PerturbationGenerator
from train_attacker import build_attacker


def compute_loss(attacker, arch, e0, e_prime, input_ids, target_ids, pad_id):
    if arch == "transformer":
        return entroguard_train_loss(attacker, e0, e_prime, input_ids, target_ids, pad_id,
                                      alpha=config.ALPHA, beta=config.BETA, gamma=config.GAMMA)
    # lstm / cnn: та же композиция eq.(10), но блочные/слоевые выходы
    # читаются через общий контракт `return_layer_outputs=True` +
    # `layer_distributions(...)`, который реализуют оба нетрансформерных
    # атакующих (TransformerAttacker вместо этого использует
    # `return_block_outputs` + `block_distributions`).
    e_prime = rescale_to_e0_norm(e0, e_prime)
    logits, layer_outs = attacker(e_prime, input_ids, return_layer_outputs=True)
    dists = attacker.layer_distributions(layer_outs)
    l_ent = entropy_loss(dists)
    l_ce = cross_entropy_loss(logits, target_ids, pad_id)
    l_sim = similarity_loss(e0, e_prime).mean()
    loss = config.ALPHA * l_sim - config.BETA * l_ent - config.GAMMA * l_ce
    return loss, {"sim": l_sim.item(), "entropy": l_ent.item(), "ce": l_ce.item()}


@torch.no_grad()
def evaluate(generator, attacker, emb_model, vocab, sentences):
    generator.eval()
    hyps, refs = [], []
    for batch in utils.iterate_batches(sentences, config.BATCH_SIZE, shuffle=False):
        emb_ids = utils.embedding_io(vocab, batch)
        e0 = emb_model(emb_ids, vocab.pad_id)
        e_prime = bound_aware_adaptation(e0, rescale_to_e0_norm(e0, generator(e0)), epsilon=config.EPSILON)
        gen = attacker.greedy_decode(e_prime, vocab.bos_id, vocab.eos_id, config.ATTACKER_SEQ_LEN)
        for i, s in enumerate(batch):
            hyps.append(vocab.decode(gen[i].tolist()).split())
            refs.append(s.split())
    report = privacy_leakage_report(hyps, refs)
    generator.train()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["transformer", "lstm", "cnn"], default="transformer")
    args = parser.parse_args()

    utils.set_seed()
    train, val, test, vocab = utils.load_dataset()

    emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                max_len=config.MAX_LEN).to(config.device)

    attacker = build_attacker(args.target, len(vocab)).to(config.device)
    utils.load_checkpoint(attacker, utils.ckpt_name(f"attacker_{args.target}"))
    attacker.eval()
    for p in attacker.parameters():
        p.requires_grad_(False)

    generator = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
    print(f"generator params: {utils.count_params(generator):,}")
    opt = torch.optim.Adam(generator.parameters(), lr=config.ENTROGUARD_LR)

    best_val_bleu = float("inf")
    history = []
    for epoch in range(1, config.ENTROGUARD_EPOCHS + 1):
        acc = {"sim": 0.0, "entropy": 0.0, "ce": 0.0}
        n_batches = 0
        for batch in utils.iterate_batches(train, config.BATCH_SIZE):
            emb_ids = utils.embedding_io(vocab, batch)
            with torch.no_grad():
                e0 = emb_model(emb_ids, vocab.pad_id)
            input_ids, target_ids = utils.attacker_io(vocab, batch)

            e_prime = generator(e0)
            loss, stats = compute_loss(attacker, args.target, e0, e_prime, input_ids, target_ids, vocab.pad_id)

            opt.zero_grad()
            loss.backward()
            opt.step()

            for k in acc:
                acc[k] += stats[k]
            n_batches += 1

        val_report = evaluate(generator, attacker, emb_model, vocab, val)
        print(f"epoch {epoch:3d}  sim_loss={acc['sim']/n_batches:.4f}  "
              f"entropy={acc['entropy']/n_batches:.4f}  ce={acc['ce']/n_batches:.4f}  "
              f"val_BLEU-2={val_report['BLEU-2']:.4f}  val_ROUGE-1={val_report['ROUGE-1']:.4f}")
        history.append({"epoch": epoch, "sim_loss": acc["sim"] / n_batches,
                         "entropy": acc["entropy"] / n_batches, "ce": acc["ce"] / n_batches,
                         **val_report})

        if val_report["BLEU-2"] < best_val_bleu:
            best_val_bleu = val_report["BLEU-2"]
            utils.save_checkpoint(generator, utils.ckpt_name(f"entroguard_{args.target}"))

    utils.load_checkpoint(generator, utils.ckpt_name(f"entroguard_{args.target}"))
    test_report = evaluate(generator, attacker, emb_model, vocab, test)
    print(f"\n[entroguard vs {args.target}] TEST -> {test_report}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    hist_path = os.path.join(config.RESULTS_DIR, utils.result_name(f"history_entroguard_{args.target}"))
    with open(hist_path, "w") as f:
        json.dump({"target": args.target, "seed": config.SEED, "params": utils.count_params(generator),
                    "history": history, "test": test_report}, f, indent=2)


if __name__ == "__main__":
    main()
