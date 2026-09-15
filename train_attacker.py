"""Предобучает атакующего инверсии эмбеддингов в стиле GEIA (transformer,
LSTM или causal-CNN декодер) на ЧИСТЫХ эмбеддингах от замороженной
"жертвы" EmbeddingModel.

Это тот атакующий, против которого потом оцениваются защиты. Качество его
декодирования на чистых эмбеддингах — это же неявная верхняя граница
утечки приватности "без защиты" из статьи.

Использование:
    python train_attacker.py --arch transformer
    python train_attacker.py --arch lstm
    python train_attacker.py --arch cnn
    ENTRO_SEED=7 python train_attacker.py --arch transformer   # другой сид
"""
import argparse
import json
import os

import torch

import config
import utils
from data import Vocab
from entroguard import cross_entropy_loss
from metrics import privacy_leakage_report
from models import CNNAttacker, EmbeddingModel, LSTMAttacker, TransformerAttacker


def build_attacker(arch, vocab_size):
    if arch == "transformer":
        return TransformerAttacker(
            vocab_size, emb_dim=config.EMB_DIM, d_model=config.D_MODEL,
            nhead=config.NHEAD, nlayers=config.ATTACKER_LAYERS,
            max_len=config.ATTACKER_SEQ_LEN,
        )
    if arch == "lstm":
        return LSTMAttacker(
            vocab_size, emb_dim=config.EMB_DIM, d_model=config.D_MODEL,
            nlayers=config.LSTM_LAYERS, max_len=config.ATTACKER_SEQ_LEN,
        )
    if arch == "cnn":
        return CNNAttacker(
            vocab_size, emb_dim=config.EMB_DIM, d_model=config.D_MODEL,
            nlayers=config.CNN_LAYERS, max_len=config.ATTACKER_SEQ_LEN,
            kernel_size=config.CNN_KERNEL,
        )
    raise ValueError(f"unknown arch: {arch}")


@torch.no_grad()
def evaluate(attacker, emb_model, vocab, sentences):
    attacker.eval()
    hyps, refs, total_loss, n_batches = [], [], 0.0, 0
    for batch in utils.iterate_batches(sentences, config.BATCH_SIZE, shuffle=False):
        emb_ids = utils.embedding_io(vocab, batch)
        e0 = emb_model(emb_ids, vocab.pad_id)
        input_ids, target_ids = utils.attacker_io(vocab, batch)
        logits = attacker(e0, input_ids)
        total_loss += cross_entropy_loss(logits, target_ids, vocab.pad_id).item()
        n_batches += 1

        gen = attacker.greedy_decode(e0, vocab.bos_id, vocab.eos_id, config.ATTACKER_SEQ_LEN)
        for i, s in enumerate(batch):
            hyps.append(vocab.decode(gen[i].tolist()).split())
            refs.append(s.split())
    report = privacy_leakage_report(hyps, refs)
    report["loss"] = total_loss / max(n_batches, 1)
    attacker.train()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["transformer", "lstm", "cnn"], default="transformer")
    args = parser.parse_args()

    utils.set_seed()
    train, val, test, vocab = utils.load_dataset()
    print(f"[{args.arch}] train/val/test = {len(train)}/{len(val)}/{len(test)}  vocab={len(vocab)}")

    emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                max_len=config.MAX_LEN).to(config.device)
    attacker = build_attacker(args.arch, len(vocab)).to(config.device)
    print(f"attacker params: {utils.count_params(attacker):,}")

    opt = torch.optim.Adam(attacker.parameters(), lr=config.ATTACKER_LR)
    best_val = float("inf")
    history = []

    for epoch in range(1, config.ATTACKER_EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for batch in utils.iterate_batches(train, config.BATCH_SIZE):
            emb_ids = utils.embedding_io(vocab, batch)
            with torch.no_grad():
                e0 = emb_model(emb_ids, vocab.pad_id)
            input_ids, target_ids = utils.attacker_io(vocab, batch)

            logits = attacker(e0, input_ids)
            loss = cross_entropy_loss(logits, target_ids, vocab.pad_id)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        val_report = evaluate(attacker, emb_model, vocab, val)
        print(f"epoch {epoch:3d}  train_ce={total_loss / n_batches:.4f}  "
              f"val_ce={val_report['loss']:.4f}  val_BLEU-2={val_report['BLEU-2']:.4f}  "
              f"val_ROUGE-1={val_report['ROUGE-1']:.4f}  val_EMR={val_report['EMR']:.4f}")
        history.append({"epoch": epoch, "train_ce": total_loss / n_batches, **val_report})

        if val_report["loss"] < best_val:
            best_val = val_report["loss"]
            utils.save_checkpoint(attacker, utils.ckpt_name(f"attacker_{args.arch}"))

    utils.load_checkpoint(attacker, utils.ckpt_name(f"attacker_{args.arch}"))
    test_report = evaluate(attacker, emb_model, vocab, test)
    print(f"\n[{args.arch}] TEST (clean embeddings, no defense) -> {test_report}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    hist_path = os.path.join(config.RESULTS_DIR, utils.result_name(f"history_attacker_{args.arch}"))
    with open(hist_path, "w") as f:
        json.dump({"arch": args.arch, "seed": config.SEED, "params": utils.count_params(attacker),
                    "history": history, "test": test_report}, f, indent=2)


if __name__ == "__main__":
    main()
