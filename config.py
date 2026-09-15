"""Центральные гиперпараметры и пути — чтобы все скрипты работали с
одними и теми же настройками. Здесь нет никакой архитектурно-специфичной
магии, просто числа, которые должны оставаться согласованными между
data / train / eval скриптами.

Любое значение ниже можно переопределить для конкретного запуска через
переменную окружения ENTRO_* без правки файла — именно так веб-панель
app.py (и любой человек в терминале) перезапускает скрипт с другим
источником данных или бюджетом эпох:
`ENTRO_DATA_SOURCE=synthetic python train_attacker.py ...`.
"""
import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env(name, default, cast=str):
    val = os.environ.get(f"ENTRO_{name}")
    return default if val is None else cast(val)


# --- данные -------------------------------------------------------------
# "synthetic": офлайн шаблонный корпус (data.build_corpus), сеть не нужна.
# "personachat": реальные PersonaChat (приватность) + реальный MS MARCO
#                (только полезность/retrieval), через Hugging Face
#                `datasets` -- см. NOTES.md "Данные".
DATA_SOURCE = _env("DATA_SOURCE", "personachat")

N_SENTENCES = _env("N_SENTENCES", 900, int)        # используется только при DATA_SOURCE == "synthetic"
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # остаток идёт в test
SEED = _env("SEED", 1024, int)

# список сидов для run_multiseed.py / aggregate_results.py -- через запятую,
# например ENTRO_SEEDS=1024,7,42. Каждый отдельный скрипт смотрит только на
# SEED выше (его подставляет run_multiseed.py в переменные окружения
# подпроцесса); это поле — просто список для батч-прогона.
SEEDS = [int(s) for s in _env("SEEDS", "1024,7,42").split(",") if s.strip()]

PERSONACHAT_DATASET = _env("PERSONACHAT_DATASET", "AlekseyKorshuk/persona-chat")
PERSONACHAT_SPLIT = "train"
PERSONACHAT_MAX_SENTENCES = _env("PERSONACHAT_MAX_SENTENCES", 4000, int)

# старое короткое имя "ms_marco" (без неймспейса) больше не резолвится
# новыми версиями `datasets`/HF hub -- нужен канонический repo id
MSMARCO_DATASET = _env("MSMARCO_DATASET", "microsoft/ms_marco")
MSMARCO_CONFIG = _env("MSMARCO_CONFIG", "v1.1")
MSMARCO_SPLIT = "train"
MSMARCO_MAX_QUERIES = _env("MSMARCO_MAX_QUERIES", 500, int)
MSMARCO_MAX_PASSAGES = _env("MSMARCO_MAX_PASSAGES", 5000, int)

VOCAB_MAX_SIZE = 20000    # ограничивает словарь реальных корпусов по частоте; остальное -> <unk>

# длины последовательностей (см. NOTES.md "Sequence length bookkeeping")
MAX_LEN = 24              # длина, подаваемая в замороженный EmbeddingModel
ATTACKER_SEQ_LEN = MAX_LEN + 1  # BOS+...+EOS+pad, teacher-forcing, сдвиг на 1

# --- размерности моделей --------------------------------------------------
EMB_DIM = 64
D_MODEL = 64
NHEAD = 4
ATTACKER_LAYERS = 3
LSTM_LAYERS = 3
CNN_LAYERS = 3     # дилатированные causal-свёрточные блоки, dilation = 2**i -> рецептивное поле 15 при kernel=3
CNN_KERNEL = 3

# --- обучение -------------------------------------------------------------
BATCH_SIZE = _env("BATCH_SIZE", 64, int)
ATTACKER_EPOCHS = _env("ATTACKER_EPOCHS", 40, int)
ATTACKER_LR = _env("ATTACKER_LR", 3e-4, float)

ENTROGUARD_EPOCHS = _env("ENTROGUARD_EPOCHS", 40, int)
ENTROGUARD_LR = _env("ENTROGUARD_LR", 1e-3, float)
ALPHA = _env("ALPHA", 6.0, float)
BETA = _env("BETA", 1.0, float)
GAMMA = _env("GAMMA", 1.0, float)   # веса из eq.(10)

# бюджет дообучения адаптивного атакующего (Эксперимент 3) -- стартует от
# уже предобученного на чистых эмбеддингах атакующего, поэтому нужно
# заметно меньше эпох, чем при обучении с нуля.
ADAPTIVE_ATTACKER_EPOCHS = _env("ADAPTIVE_ATTACKER_EPOCHS", 20, int)
ADAPTIVE_ATTACKER_LR = _env("ADAPTIVE_ATTACKER_LR", 1e-4, float)

# --- бюджеты защиты / атаки ----------------------------------------------
EPSILON = _env("EPSILON", 0.15, float)          # граница возмущения по косинусному расстоянию (Algorithm 1)
GAUSSIAN_SIGMA = _env("GAUSSIAN_SIGMA", 0.15, float)
PGD_STEPS = _env("PGD_STEPS", 10, int)
PGD_STEP_SIZE = _env("PGD_STEP_SIZE", 0.05, float)

RETRIEVAL_K = _env("RETRIEVAL_K", 5, int)

# --- пути -------------------------------------------------------------
CKPT_DIR = "checkpoints"
RESULTS_DIR = "results"
