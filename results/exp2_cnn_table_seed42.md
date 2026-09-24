| defense (vs cnn) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.1170 | 0.4260 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0991 | 0.4036 | 0.0000 | 0.3744 | 7.30 |
| pgd | 0.0149 | 0.2369 | 0.0000 | 0.3640 | 6.88 |
| entroguard_transformer(zero-shot) | 0.0467 | 0.3520 | 0.0000 | 0.2868 | 2.90 |
| entroguard_cnn(native) | 0.0158 | 0.1677 | 0.0000 | 0.2360 | 3.86 |
