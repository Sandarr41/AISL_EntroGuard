import os
import random
import time

import numpy as np
import torch

import config
from data import Vocab, build_corpus, load_msmarco, load_personachat


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset():
    """Строит основной корпус (config.DATA_SOURCE) один раз и делит его на
    train/val/test. Детерминировано при фиксированном config.SEED плюс (для
    реальных данных) стабильной ревизии датасета на HF, так что каждый
    скрипт/процесс видит идентичные разбиения и идентичный Vocab без
    сохранения чего-либо на диск.

    Для DATA_SOURCE == "personachat" Vocab строится из объединения текстов
    PersonaChat *и* MS MARCO, хотя как train/val/test возвращается только
    PersonaChat -- тот же замороженный EmbeddingModel потом используется
    для эмбеддинга queries/passages MS MARCO в метрике полезности retrieval
    (см. load_retrieval_pool), и ему тоже нужны эти слова."""
    if config.DATA_SOURCE == "synthetic":
        sentences = build_corpus(config.N_SENTENCES)
        vocab = Vocab(sentences)
    elif config.DATA_SOURCE == "personachat":
        sentences = load_personachat(config.PERSONACHAT_MAX_SENTENCES,
                                      config.PERSONACHAT_DATASET, config.PERSONACHAT_SPLIT)
        ms_queries, ms_passages = load_msmarco(config.MSMARCO_MAX_QUERIES, config.MSMARCO_MAX_PASSAGES,
                                                config.MSMARCO_DATASET, config.MSMARCO_CONFIG, config.MSMARCO_SPLIT)
        vocab = Vocab(sentences + ms_queries + ms_passages, max_size=config.VOCAB_MAX_SIZE)
    else:
        raise ValueError(f"unknown config.DATA_SOURCE: {config.DATA_SOURCE!r}")

    n = len(sentences)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    train = sentences[:n_train]
    val = sentences[n_train:n_train + n_val]
    test = sentences[n_train + n_val:]
    return train, val, test, vocab


def load_retrieval_pool():
    """(queries, passages) для метрики полезности retrieval (eq. 5) --
    реальный MS MARCO при DATA_SOURCE == "personachat" (queries защищаются,
    passages — база для retrieval); при DATA_SOURCE == "synthetic"
    переиспользует основное разбиение (test как queries, train+val как
    база документов), сохраняя прежнее поведение."""
    if config.DATA_SOURCE == "personachat":
        return load_msmarco(config.MSMARCO_MAX_QUERIES, config.MSMARCO_MAX_PASSAGES,
                             config.MSMARCO_DATASET, config.MSMARCO_CONFIG, config.MSMARCO_SPLIT)
    train, val, test, _ = load_dataset()
    return test, train + val


def encode_batch(vocab, sentences, max_len):
    ids = [vocab.encode(s, max_len) for s in sentences]
    return torch.tensor(ids, dtype=torch.long, device=config.device)


def iterate_batches(sentences, batch_size, shuffle=True):
    idx = list(range(len(sentences)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        yield [sentences[j] for j in chunk]


def attacker_io(vocab, sentences):
    """Строит (input_ids, target_ids) для teacher-forcing предсказания
    следующего токена: full = [BOS, w1..wn, EOS, PAD...] длины
    ATTACKER_SEQ_LEN; input — это full[:-1], target — full[1:]."""
    full = encode_batch(vocab, sentences, config.ATTACKER_SEQ_LEN)
    return full[:, :-1], full[:, 1:]


def embedding_io(vocab, sentences):
    return encode_batch(vocab, sentences, config.MAX_LEN)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_tag():
    return f"seed{config.SEED}"


def ckpt_name(base):
    """checkpoints/<base>_seed<N>.pt -- каждый обучающий скрипт называет
    свой чекпоинт именно так, чтобы разные сиды никогда не сталкивались,
    а aggregate_results.py мог собрать их обратно по базовому имени."""
    return f"{base}_{seed_tag()}.pt"


def result_name(base, ext="json"):
    return f"{base}_{seed_tag()}.{ext}"


def save_checkpoint(model, name):
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    path = os.path.join(config.CKPT_DIR, name)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model, name):
    path = os.path.join(config.CKPT_DIR, name)
    model.load_state_dict(torch.load(path, map_location=config.device))
    return model


class Timer:
    """Таймер по настенным часам для сравнения эффективности (throughput,
    а не только приватности/полезности)."""
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
