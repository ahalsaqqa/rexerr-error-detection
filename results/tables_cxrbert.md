## Table A. Main comparison (official test split, report level, 12 native types)

Mean ± SD over seeds when several seeds were run; 95% CI (patient-level bootstrap) for the first seed.

| Model | Seeds | Macro-F1 | Micro-F1 | Macro-AUROC | Macro-AUPRC | Any-error AUROC | Any-error AUPRC |
|---|---|---|---|---|---|---|---|
| CXR-BERT | 1 | 0.768 (0.754–0.780) | 0.807 (0.799–0.815) | 0.973 (0.971–0.975) | 0.827 (0.811–0.843) | 0.995 (0.993–0.996) | 0.995 (0.993–0.996) |
| BioBERT v1.1 [focal loss] | 1 | 0.752 (0.739–0.766) | 0.793 (0.784–0.801) | 0.970 (0.967–0.972) | 0.822 (0.808–0.836) | 0.992 (0.990–0.994) | 0.993 (0.991–0.994) |
| RadBERT (RoBERTa) | 1 | 0.762 (0.749–0.775) | 0.802 (0.794–0.811) | 0.971 (0.968–0.973) | 0.822 (0.807–0.836) | 0.992 (0.990–0.994) | 0.992 (0.989–0.994) |
| BlueBERT | 1 | 0.757 (0.744–0.771) | 0.793 (0.785–0.802) | 0.969 (0.966–0.972) | 0.816 (0.801–0.832) | 0.992 (0.990–0.994) | 0.993 (0.991–0.995) |
| BioBERT v1.1 | 3 | 0.752 ± 0.003 | 0.791 ± 0.003 | 0.969 ± 0.001 | 0.814 ± 0.003 | 0.991 ± 0.001 | 0.992 ± 0.000 |
| BiomedBERT (PubMedBERT) | 1 | 0.750 (0.736–0.764) | 0.792 (0.784–0.801) | 0.968 (0.965–0.970) | 0.813 (0.798–0.828) | 0.991 (0.988–0.992) | 0.991 (0.989–0.993) |
| Bio+ClinicalBERT | 1 | 0.739 (0.725–0.755) | 0.787 (0.779–0.797) | 0.967 (0.964–0.969) | 0.808 (0.794–0.824) | 0.990 (0.988–0.992) | 0.991 (0.988–0.993) |
| BioBERT v1.1 [weighted BCE] | 1 | 0.705 (0.693–0.715) | 0.727 (0.717–0.736) | 0.968 (0.966–0.971) | 0.805 (0.791–0.821) | 0.991 (0.989–0.993) | 0.991 (0.988–0.993) |
| BERT-base (cased) | 1 | 0.743 (0.729–0.756) | 0.783 (0.774–0.792) | 0.965 (0.962–0.968) | 0.805 (0.791–0.820) | 0.988 (0.986–0.990) | 0.989 (0.987–0.992) |
| BioBERT v1.1 [sentence-level] | 1 | 0.658 (0.643–0.672) | 0.737 (0.728–0.747) | 0.959 (0.956–0.962) | 0.754 (0.739–0.773) | 0.980 (0.977–0.983) | 0.981 (0.978–0.984) |
| BioBERT v1.1 [trained without clean reports] | 1 | 0.635 (0.622–0.649) | 0.641 (0.633–0.649) | 0.913 (0.908–0.918) | 0.681 (0.667–0.701) | 0.925 (0.919–0.932) | 0.922 (0.912–0.930) |
| TF-IDF + LR | 1 | 0.494 (0.478–0.509) | 0.572 (0.561–0.584) | 0.891 (0.886–0.896) | 0.578 (0.565–0.597) | 0.918 (0.911–0.925) | 0.926 (0.919–0.932) |
| Qwen3-8B (zero-shot) | 1 | 0.264 (0.256–0.272) | 0.298 (0.290–0.306) | 0.614 (0.608–0.620) | 0.178 (0.174–0.182) | 0.584 (0.574–0.595) | 0.544 (0.538–0.551) |
| Qwen2.5-7B-Instruct (zero-shot) | 1 | 0.246 (0.236–0.255) | 0.279 (0.269–0.288) | 0.611 (0.603–0.620) | 0.170 (0.166–0.176) | 0.647 (0.633–0.662) | 0.586 (0.576–0.598) |
| Llama-3.1-8B-Instruct (zero-shot) | 1 | 0.238 (0.232–0.243) | 0.242 (0.237–0.248) | 0.634 (0.626–0.641) | 0.169 (0.166–0.173) | 0.514 (0.511–0.518) | 0.506 (0.504–0.508) |
| Report length + LR | 1 | 0.001 (0.000–0.002) | 0.003 (0.001–0.005) | 0.631 (0.623–0.640) | 0.161 (0.158–0.167) | 0.573 (0.569–0.578) | 0.554 (0.551–0.559) |

## Table B. Per-type results of the main model (CXR-BERT, threshold 0.5)

| Error type | Support | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Add medical device | 880 | 0.951 | 0.967 | 0.959 | 0.997 | 0.983 |
| False prediction | 1484 | 0.881 | 0.844 | 0.862 | 0.975 | 0.944 |
| False negation | 1059 | 0.862 | 0.709 | 0.778 | 0.956 | 0.869 |
| Add repetitions | 365 | 0.771 | 0.764 | 0.768 | 0.981 | 0.817 |
| Change name of device | 371 | 0.751 | 0.512 | 0.609 | 0.942 | 0.671 |
| Change position of device | 393 | 0.697 | 0.634 | 0.664 | 0.974 | 0.753 |
| Change severity | 671 | 0.758 | 0.739 | 0.749 | 0.964 | 0.826 |
| Change location | 808 | 0.748 | 0.651 | 0.696 | 0.947 | 0.787 |
| Change measurement | 65 | 0.672 | 0.600 | 0.634 | 0.988 | 0.691 |
| Change to homophone | 118 | 0.833 | 0.847 | 0.840 | 0.993 | 0.813 |
| Add typo | 884 | 0.855 | 0.917 | 0.885 | 0.988 | 0.945 |

## Table C. Six-group results of the main model

| Group | Support | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| False findings / devices | 2536 | 0.962 | 0.960 | 0.961 | 0.992 | 0.990 |
| Incorrect location / position | 1152 | 0.789 | 0.714 | 0.750 | 0.955 | 0.845 |
| Measurement error | 65 | 0.672 | 0.600 | 0.634 | 0.988 | 0.691 |
| Linguistic / transcription error | 1338 | 0.841 | 0.886 | 0.863 | 0.975 | 0.927 |
| Severity misclassification | 671 | 0.758 | 0.739 | 0.749 | 0.964 | 0.826 |

## Table C2. Confusion counts of the main model for every error type (threshold 0.5)

| Error type | TP | FP | FN | TN | Specificity |
|---|---|---|---|---|---|
| Add medical device | 851 | 44 | 29 | 4492 | 0.990 |
| False prediction | 1252 | 169 | 232 | 3763 | 0.957 |
| False negation | 751 | 120 | 308 | 4237 | 0.972 |
| Add repetitions | 279 | 83 | 86 | 4968 | 0.984 |
| Change name of device | 190 | 63 | 181 | 4982 | 0.988 |
| Change position of device | 249 | 108 | 144 | 4915 | 0.978 |
| Change severity | 496 | 158 | 175 | 4587 | 0.967 |
| Change location | 526 | 177 | 282 | 4431 | 0.962 |
| Change measurement | 39 | 19 | 26 | 5332 | 0.996 |
| Change to homophone | 100 | 20 | 18 | 5278 | 0.996 |
| Add typo | 811 | 138 | 73 | 4394 | 0.970 |

## Table C3. Per-type thresholds tuned on the validation split (secondary analysis)

Macro-F1 0.768 (vs. 0.768 at 0.5); micro-F1 0.805 (vs. 0.807).


## Table D. Detecting whether a report contains any error (main model)

| Operating point | Threshold | Sensitivity | Specificity | Precision (test mix) |
|---|---|---|---|---|
| Default 0.5 | 0.500 | 0.966 | 0.975 | 0.974 |
| Sensitivity 0.90 on val | 0.902 | 0.885 | 0.997 | 0.996 |
| Sensitivity 0.95 on val | 0.794 | 0.941 | 0.989 | 0.989 |

## Table E. Expected performance at realistic error prevalence (main model, sensitivity-0.90 operating point)

| Prevalence | PPV | NPV | Alerts / 1000 reports | True alerts / 1000 | Missed errors / 1000 |
|---|---|---|---|---|---|
| 3% | 0.892 | 0.996 | 30 | 27 | 3.4 |
| 5% | 0.934 | 0.994 | 47 | 44 | 5.7 |
| 10% | 0.967 | 0.987 | 92 | 89 | 11.5 |
| 20% | 0.985 | 0.972 | 180 | 177 | 23.0 |

## Table F. Paired bootstrap comparisons against the main model (difference = main − other)

| Other model | Δ Macro-F1 (95% CI) | p | Δ Macro-AUPRC (95% CI) | p |
|---|---|---|---|---|
| BERT-base (cased) | +0.025 (+0.014, +0.036) | 0.000 | +0.022 (+0.012, +0.031) | 0.000 |
| BioBERT v1.1 [focal loss] | +0.015 (+0.005, +0.026) | 0.002 | +0.005 (-0.004, +0.014) | 0.296 |
| BioBERT v1.1 [trained without clean reports] | +0.133 (+0.123, +0.143) | 0.000 | +0.146 (+0.132, +0.157) | 0.000 |
| BioBERT v1.1 | +0.015 (+0.006, +0.025) | 0.000 | +0.010 (+0.001, +0.019) | 0.042 |
| BioBERT v1.1 [weighted BCE] | +0.063 (+0.050, +0.075) | 0.000 | +0.022 (+0.011, +0.030) | 0.000 |
| BiomedBERT (PubMedBERT) | +0.017 (+0.008, +0.026) | 0.000 | +0.014 (+0.005, +0.023) | 0.004 |
| BlueBERT | +0.010 (+0.002, +0.018) | 0.016 | +0.011 (+0.001, +0.021) | 0.026 |
| Bio+ClinicalBERT | +0.028 (+0.016, +0.039) | 0.000 | +0.019 (+0.009, +0.030) | 0.000 |
| Report length + LR | +0.766 (+0.753, +0.779) | 0.000 | +0.666 (+0.649, +0.680) | 0.000 |
| Llama-3.1-8B-Instruct (zero-shot) | +0.530 (+0.516, +0.542) | 0.000 | +0.658 (+0.642, +0.673) | 0.000 |
| Qwen2.5-7B-Instruct (zero-shot) | +0.522 (+0.507, +0.536) | 0.000 | +0.657 (+0.641, +0.672) | 0.000 |
| Qwen3-8B (zero-shot) | +0.503 (+0.488, +0.519) | 0.000 | +0.649 (+0.634, +0.664) | 0.000 |
| RadBERT (RoBERTa) | +0.006 (-0.002, +0.014) | 0.186 | +0.005 (-0.003, +0.014) | 0.280 |
| TF-IDF + LR | +0.274 (+0.257, +0.290) | 0.000 | +0.249 (+0.232, +0.262) | 0.000 |

## Table G. Sentence-level error localization

| Model | Sentences | Localization AUROC | AUPRC | F1 | Type macro-F1 | Report-level macro-F1 (aggregated) |
|---|---|---|---|---|---|---|
| BioBERT v1.1 [sentence-level] | 36719 | 0.956 | 0.879 | 0.760 | 0.613 | 0.658 |
