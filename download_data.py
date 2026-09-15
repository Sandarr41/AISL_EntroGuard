"""Разовый шаг: скачивает и кэширует PersonaChat + MS MARCO через
библиотеку Hugging Face `datasets`, печатает несколько примеров, чтобы
можно было проверить схему до того, как тратить время на обучение.

Первый запуск требует доступ к сети. После него `datasets` берёт всё из
локального кэша (по умолчанию ~/.cache/huggingface), так что
train_*/evaluate_*.py больше не полезут в сеть, пока кэш не очистят или
не изменятся имена датасетов/сплиты в config.py.

Использование:
    python download_data.py
"""
import config
from data import load_msmarco, load_personachat


def main():
    print(f"скачиваю/кэширую {config.PERSONACHAT_DATASET} (split={config.PERSONACHAT_SPLIT}) ...")
    persona = load_personachat(config.PERSONACHAT_MAX_SENTENCES,
                                config.PERSONACHAT_DATASET, config.PERSONACHAT_SPLIT)
    print(f"  -> {len(persona)} уникальных персона-предложений, например:")
    for s in persona[:5]:
        print("     -", s)

    print(f"\nскачиваю/кэширую {config.MSMARCO_DATASET}/{config.MSMARCO_CONFIG} "
          f"(split={config.MSMARCO_SPLIT}) ...")
    queries, passages = load_msmarco(config.MSMARCO_MAX_QUERIES, config.MSMARCO_MAX_PASSAGES,
                                      config.MSMARCO_DATASET, config.MSMARCO_CONFIG, config.MSMARCO_SPLIT)
    print(f"  -> {len(queries)} queries, {len(passages)} уникальных passages")
    print("     примеры queries:")
    for s in queries[:3]:
        print("     -", s)
    print("     примеры passages:")
    for s in passages[:2]:
        print("     -", s[:120], "..." if len(s) > 120 else "")


if __name__ == "__main__":
    main()
