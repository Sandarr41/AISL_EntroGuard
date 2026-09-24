| defense (vs lstm) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.2081 | 0.5184 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0984 | 0.4077 | 0.0000 | 0.3792 | 8.65 |
| pgd | 0.0071 | 0.2723 | 0.0000 | 0.3916 | 5.93 |
| entroguard_transformer(zero-shot) | 0.0786 | 0.4188 | 0.0000 | 0.2808 | 2.47 |
| entroguard_lstm(native) | 0.0092 | 0.2059 | 0.0000 | 0.4448 | 4.45 |
