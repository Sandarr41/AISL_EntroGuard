| defense (vs cnn) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.1191 | 0.4447 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0956 | 0.4056 | 0.0000 | 0.3736 | 8.52 |
| pgd | 0.0034 | 0.2356 | 0.0000 | 0.3908 | 6.37 |
| entroguard_transformer(zero-shot) | 0.0661 | 0.3902 | 0.0000 | 0.1816 | 2.66 |
| entroguard_cnn(native) | 0.0209 | 0.2644 | 0.0000 | 0.4452 | 4.97 |
