"""Раннер для удобства: выполняет весь пайплайн (атакующие, EntroGuard,
адаптивные атакующие, все три эксперимента) для ТЕКУЩЕГО сида
(config.SEED / ENTRO_SEED), по одному подпроцессу на шаг. Эквивалентно
ручному запуску каждого скрипта -- см. NOTES.md, если хотите
запускать/настраивать этапы по отдельности. Для нескольких сидов с
агрегацией mean/std используйте вместо этого run_multiseed.py (те же
шаги, но в цикле по сидам).
"""
import subprocess
import sys


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
    for step in STEPS:
        print(f"\n{'=' * 70}\n>>> {' '.join(step)}\n{'=' * 70}")
        subprocess.run([sys.executable, *step], check=True)


if __name__ == "__main__":
    main()
