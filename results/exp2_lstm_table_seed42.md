| defense (vs lstm) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.1992 | 0.4924 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0912 | 0.4111 | 0.0000 | 0.3800 | 7.36 |
| pgd | 0.0050 | 0.2404 | 0.0000 | 0.4212 | 6.97 |
| entroguard_transformer(zero-shot) | 0.0681 | 0.3721 | 0.0000 | 0.2860 | 3.12 |
| entroguard_lstm(native) | 0.0078 | 0.2308 | 0.0000 | 0.3676 | 4.06 |
