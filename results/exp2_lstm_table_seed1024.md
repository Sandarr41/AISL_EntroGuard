| defense (vs lstm) | BLEU-2 | ROUGE-1 | EMR | retrieval_recall@5 | ms_per_1k_queries |
|---|---|---|---|---|---|
| none | 0.2082 | 0.5179 | 0.0000 | 1.0000 | 0.00 |
| gaussian | 0.0984 | 0.4081 | 0.0000 | 0.3792 | 6.49 |
| pgd | 0.0071 | 0.2721 | 0.0000 | 0.3928 | 4.82 |
| entroguard_transformer(zero-shot) | 0.0786 | 0.4187 | 0.0000 | 0.2808 | 1.85 |
| entroguard_lstm(native) | 0.0093 | 0.2067 | 0.0000 | 0.4604 | 3.36 |
