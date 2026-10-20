## External test set (rule-based errors, one per perturbed report; original reports as controls)

| Model | Any-error AUROC | Sensitivity at 0.5 | Specificity at 0.5 | Sensitivity / specificity (val threshold, Se 0.90) | Macro type recall |
|---|---|---|---|---|---|
| BioBERT v1.1 | 0.848 | 0.503 | 0.967 | 0.247 / 0.992 | 0.374 |
| CXR-BERT | 0.866 | 0.524 | 0.974 | 0.216 / 0.995 | 0.392 |
| Qwen3-8B (zero-shot) | 0.566 | 0.945 | 0.188 | – | 0.495 |
| Qwen3-8B (4-shot) | 0.644 | 0.782 | 0.505 | – | 0.493 |

### Per injected type: recall of the injected type / share of reports flagged (threshold 0.5)

| Type | n | BioBERT v1.1 | CXR-BERT | Qwen3-8B (zero-shot) | Qwen3-8B (4-shot) |
|---|---|---|---|---|---|
| Add medical device | 311 | 0.19 / 0.23 | 0.23 / 0.25 | 0.17 / 0.95 | 0.40 / 0.70 |
| Add repetitions | 310 | 0.72 / 0.93 | 0.75 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 |
| Add typo | 310 | 0.45 / 0.50 | 0.50 / 0.56 | 0.41 / 0.92 | 0.53 / 0.81 |
| Change location | 310 | 0.09 / 0.12 | 0.17 / 0.20 | 0.68 / 0.86 | 0.40 / 0.61 |
| Change measurement | 225 | 0.00 / 0.04 | 0.01 / 0.06 | 0.17 / 0.98 | 0.32 / 0.86 |
| Change severity | 311 | 0.20 / 0.33 | 0.33 / 0.47 | 0.82 / 0.91 | 0.37 / 0.59 |
| Change to homophone | 311 | 0.35 / 0.87 | 0.38 / 0.92 | 0.39 / 0.96 | 0.28 / 0.88 |
| False negation | 310 | 0.69 / 0.70 | 0.76 / 0.77 | 0.26 / 0.94 | 0.40 / 0.70 |
| False prediction | 310 | 0.68 / 0.68 | 0.40 / 0.42 | 0.57 / 0.99 | 0.74 / 0.91 |
