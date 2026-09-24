"""Интерактивное демо для веб-панели (app.py, вкладка "Демо"): на РЕАЛЬНЫХ
обученных чекпоинтах показывает, что атакующий восстанавливает из
эмбеддинга введённой фразы -- без защиты и после выбранной защиты
(gaussian / pgd / entroguard).

Никакой отдельной "демо-модели" нет: используется тот же frozen
EmbeddingModel + transformer-attacker + entroguard-generator, что и в
evaluate_defense.py (Эксперимент 1), просто на batch=1 и с чекпоинтом
конкретного сида, выбранного в интерфейсе, вместо всего тестового сплита.

Опционально (см. `run(..., use_adaptive=True)`) демо задействует и
Эксперимент 3: рядом со статическим атакующим (никогда не видел эту
защиту) декодирует ТОТ ЖЕ защищённый эмбеддинг АДАПТИВНЫМ атакующим
(attacker_transformer_adaptive_{defense}_seed<N>.pt из
train_adaptive_attacker.py) -- полностью информированным противником,
дообученным именно против выбранной защиты. Разница между ними -- это
"запас прочности" защиты, который не виден при оценке только статическим
атакующим (см. README "Эксперимент 3" и Carlini et al.).

Модели грузятся лениво по первому запросу на каждый сид и кешируются в
памяти процесса (_MODELS) -- иначе загрузка датасета/словаря на каждый
клик была бы заметно медленнее самого инференса.
"""
import os
import threading

import torch
import torch.nn.functional as F

import config
import utils
from data import simple_tokenize
from entroguard import (bound_aware_adaptation, gaussian_noise_defense, pgd_defense,
                         rescale_to_e0_norm)
from metrics import bleu2
from models import EmbeddingModel, PerturbationGenerator
from train_attacker import build_attacker

DEFENSES = ["gaussian", "pgd", "entroguard"]

_LOCK = threading.Lock()
_VOCAB = None
_MODELS = {}  # seed (str) -> {"emb_model", "attacker", "generator"}


def _ckpt_path(base, seed):
    return os.path.join(config.CKPT_DIR, f"{base}_seed{seed}.pt")


def _adaptive_ckpt_path(defense, seed):
    return _ckpt_path(f"attacker_transformer_adaptive_{defense}", seed)


def available_seeds():
    """Сиды, для которых на диске есть чекпоинты И attacker_transformer, И
    entroguard_transformer -- ровно то, что нужно этому демо (Эксперимент 1
    задействует только transformer-атакующего)."""
    if not os.path.isdir(config.CKPT_DIR):
        return []
    prefix, suffix = "attacker_transformer_seed", ".pt"
    seeds = []
    for fname in os.listdir(config.CKPT_DIR):
        if fname.startswith(prefix) and fname.endswith(suffix):
            seed = fname[len(prefix):-len(suffix)]
            if os.path.exists(_ckpt_path("entroguard_transformer", seed)):
                seeds.append(seed)
    return sorted(set(seeds), key=lambda s: (len(s), s))


def available_adaptive_defenses(seed):
    """Защиты, для которых у этого сида есть дообученный (Эксперимент 3)
    адаптивный атакующий -- используется интерфейсом, чтобы не предлагать
    галочку "адаптивный атакующий" там, где чекпоинта ещё нет."""
    seed = str(seed)
    return [d for d in DEFENSES if os.path.exists(_adaptive_ckpt_path(d, seed))]


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        _, _, _, _VOCAB = utils.load_dataset()
    return _VOCAB


def _load_for_seed(seed):
    """Восстанавливает EmbeddingModel с ТЕМ ЖЕ случайным весами, что были у
    неё во время обучения этого сида: EmbeddingModel никогда не обучается
    градиентом (frozen victim), поэтому её веса целиком определяются
    состоянием RNG сразу после utils.set_seed(seed) -- тем же порядком
    вызовов (set_seed -> EmbeddingModel(...) -> build_attacker(...)), что
    и в evaluate_defense.py/train_attacker.py. Если сид перепутать,
    получится другая (тоже валидная, но не та) "жертва", и реальный
    чекпоинт атакующего перестанет ей соответствовать."""
    with _LOCK:
        if seed in _MODELS:
            return _MODELS[seed]
        vocab = _vocab()
        utils.set_seed(int(seed))
        emb_model = EmbeddingModel(len(vocab), d_model=config.D_MODEL, nhead=config.NHEAD,
                                    max_len=config.MAX_LEN).to(config.device)
        attacker = build_attacker("transformer", len(vocab)).to(config.device)
        attacker.load_state_dict(torch.load(_ckpt_path("attacker_transformer", seed),
                                             map_location=config.device))
        attacker.eval()
        generator = PerturbationGenerator(emb_dim=config.EMB_DIM).to(config.device)
        generator.load_state_dict(torch.load(_ckpt_path("entroguard_transformer", seed),
                                              map_location=config.device))
        generator.eval()
        state = {"vocab": vocab, "emb_model": emb_model, "attacker": attacker, "generator": generator,
                  "adaptive": {}}
        _MODELS[seed] = state
        return state


def _load_adaptive_attacker(seed, defense, vocab):
    """Адаптивный атакующий -- в отличие от EmbeddingModel, его веса целиком
    приходят из чекпоинта (обучались градиентом), так что состояние RNG в
    момент конструирования не важно -- инициализация всё равно будет тут же
    перезаписана load_state_dict."""
    state = _MODELS[seed]
    if defense in state["adaptive"]:
        return state["adaptive"][defense]
    path = _adaptive_ckpt_path(defense, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"нет адаптивного атакующего для defense={defense!r}, seed={seed} "
            f"({path}) -- обучите его на вкладке «Пайплайн» (шаг 5a/5b/5c)")
    adaptive = build_attacker("transformer", len(vocab)).to(config.device)
    adaptive.load_state_dict(torch.load(path, map_location=config.device))
    adaptive.eval()
    state["adaptive"][defense] = adaptive
    return adaptive


@torch.no_grad()
def _decode(attacker, embedding, vocab):
    gen = attacker.greedy_decode(embedding, vocab.bos_id, vocab.eos_id, config.ATTACKER_SEQ_LEN)
    return vocab.decode(gen[0].tolist())


def run(text, defense, seed, use_adaptive=False):
    """Прогоняет одну фразу через: encode -> e0 -> атака без защиты ->
    выбранная защита -> атака после защиты. Возвращает всё, что нужно
    интерфейсу, включая слова вне словаря (иначе непонятно, почему
    реконструкция бессвязна для фраз с редкими словами).

    use_adaptive=True (Эксперимент 3) дополнительно декодирует ТОТ ЖЕ
    защищённый эмбеддинг адаптивным (дообученным именно против этой
    защиты) атакующим -- вызывающая сторона сама решает, показывать ли
    его рядом со статическим (см. available_adaptive_defenses)."""
    if defense not in DEFENSES:
        raise ValueError(f"неизвестная защита: {defense!r} (доступны: {DEFENSES})")
    text = (text or "").strip()
    if not text:
        raise ValueError("введите непустую фразу")

    seed = str(seed)
    state = _load_for_seed(seed)
    vocab, emb_model = state["vocab"], state["emb_model"]
    attacker, generator = state["attacker"], state["generator"]

    tokens = simple_tokenize(text)
    clean = " ".join(tokens)
    if not clean:
        raise ValueError("после токенизации фраза пуста (только знаки препинания?)")
    unknown = sorted({w for w in tokens if w not in vocab.stoi})

    emb_ids = utils.embedding_io(vocab, [clean])
    e0 = emb_model(emb_ids, vocab.pad_id)
    input_ids, target_ids = utils.attacker_io(vocab, [clean])

    leaked = _decode(attacker, e0, vocab)

    with utils.Timer() as t:
        if defense == "gaussian":
            e_prot = bound_aware_adaptation(e0, gaussian_noise_defense(e0, config.GAUSSIAN_SIGMA),
                                             epsilon=config.EPSILON)
        elif defense == "pgd":
            e_prot = pgd_defense(attacker, e0, target_ids, input_ids, vocab.pad_id,
                                  epsilon=config.EPSILON, step_size=config.PGD_STEP_SIZE,
                                  n_steps=config.PGD_STEPS)
        else:  # entroguard
            with torch.no_grad():
                e_prot = bound_aware_adaptation(e0, rescale_to_e0_norm(e0, generator(e0)),
                                                 epsilon=config.EPSILON)
    e_prot = e_prot.detach()

    protected = _decode(attacker, e_prot, vocab)
    ref_tokens = clean.split()
    cos_sim = F.cosine_similarity(e0, e_prot, dim=-1).item()
    bleu2_defended = bleu2(protected.split(), ref_tokens)

    result = {
        "tokenized_input": clean,
        "unknown_words": unknown,
        "reconstruction_no_defense": leaked,
        "reconstruction_defended": protected,
        "bleu2_no_defense": bleu2(leaked.split(), ref_tokens),
        "bleu2_defended": bleu2_defended,
        "cosine_similarity": cos_sim,
        "ms": t.elapsed * 1000,
        "defense": defense,
        "seed": seed,
        "adaptive": None,
    }

    if use_adaptive:
        # тот же e_prot, что уже увидел статический атакующий выше -- честное
        # сравнение "два атакующих на одном и том же защищённом входе", как в
        # evaluate_adaptive.py, а не два отдельных случайных прогона.
        adaptive_attacker = _load_adaptive_attacker(seed, defense, vocab)
        adaptive_recon = _decode(adaptive_attacker, e_prot, vocab)
        adaptive_bleu2 = bleu2(adaptive_recon.split(), ref_tokens)
        result["adaptive"] = {
            "reconstruction": adaptive_recon,
            "bleu2": adaptive_bleu2,
            "delta_bleu2": adaptive_bleu2 - bleu2_defended,
        }

    return result
