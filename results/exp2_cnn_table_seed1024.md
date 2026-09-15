| defense (vs cnn) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.1142 | 0.4461 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0847 | 0.4095 | 0.0000 | 0.3724 | 6.59 |
| pgd | 0.0058 | 0.2241 | 0.0000 | 0.3744 | 5.14 |
| entroguard_transformer(zero-shot) | 0.0765 | 0.4159 | 0.0000 | 0.2664 | 1.88 |
| entroguard_cnn(native) | 0.0018 | 0.1562 | 0.0000 | 0.2904 | 2.77 |
