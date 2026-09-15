"""Прогоняет весь пайплайн (атакующие, EntroGuard, адаптивные атакующие и
все три эксперимента) по одному разу на каждый сид из config.SEEDS, затем
aggregate_results.py, чтобы собрать результаты по сидам в mean +/- std.
Точечная оценка с одного сида не скажет, реальна ли разница между
защитами или это шум — вот это как раз и отвечает на вопрос.

Каждый шаг — отдельный подпроцесс (как в run_all.py), чтобы состояние
CUDA/Python никогда не протекало между запусками; ENTRO_SEED
выставляется для каждого сида в окружении этого подпроцесса — это всё,
что нужно utils.set_seed() / utils.ckpt_name() / utils.result_name(),
чтобы каждый артефакт был корректно помечен сидом и не сталкивался с
другими.

Это ОЧЕНЬ МНОГО обучения (3 архитектуры атакующего + 3 цели EntroGuard +
3 адаптивных атакующих + 3 оценки, умножить на len(config.SEEDS)) —
ожидайте, что это займёт время даже на GPU. Переопределите эпохи/сиды
через переменные окружения ENTRO_* (см. config.py), если хотите сначала
быстрый прогон меньшего масштаба.

Использование:
    python run_multiseed.py                  # config.SEEDS, по умолчанию "1024,7,42"
    ENTRO_SEEDS=1,2,3 python run_multiseed.py
    ENTRO_SEEDS=1024,7 ENTRO_ATTACKER_EPOCHS=10 ENTRO_ENTROGUARD_EPOCHS=10 python run_multiseed.py
"""
import os
import subprocess
import sys

import config

STEPS = [
    ["train_attacker.py", "--arch", "transformer"],
    ["train_attacker.py", "--arch", "lstm"],
    ["train_attacker.py", "--arch", "cnn"],
    ["train_entroguard.py", "--target", "transformer"],
    ["train_entroguard.py", "--target", "lstm"],
    ["train_entroguard.py", "--target", "cnn"],
    ["train_adaptive_attacker.py", "--defense", "gaussian"],
    ["train_adaptive_attacker.py", "--defense", "pgd"],
    ["train_adaptive_attacker.py", "--defense", "entroguard"],
    ["evaluate_defense.py"],
    ["evaluate_transfer.py"],
    ["evaluate_adaptive.py"],
]


def main():
    print(f"seeds: {config.SEEDS}")
    for seed in config.SEEDS:
        env = os.environ.copy()
        env["ENTRO_SEED"] = str(seed)
        for step in STEPS:
            print(f"\n{'=' * 70}\n>>> seed={seed}  {' '.join(step)}\n{'=' * 70}")
            subprocess.run([sys.executable, *step], check=True, env=env)

    print(f"\n{'=' * 70}\n>>> aggregate_results.py\n{'=' * 70}")
    subprocess.run([sys.executable, "aggregate_results.py"], check=True)


if __name__ == "__main__":
    main()
