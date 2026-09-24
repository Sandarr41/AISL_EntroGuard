| defense (vs lstm) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.2057 | 0.4990 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0919 | 0.3870 | 0.0000 | 0.3468 | 8.52 |
| pgd | 0.0050 | 0.2627 | 0.0000 | 0.3996 | 6.06 |
| entroguard_transformer(zero-shot) | 0.0938 | 0.4051 | 0.0000 | 0.1788 | 2.60 |
| entroguard_lstm(native) | 0.0043 | 0.2417 | 0.0000 | 0.3544 | 4.09 |
