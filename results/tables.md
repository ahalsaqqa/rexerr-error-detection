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

## Table B. Per-type results of the main model (BioBERT v1.1, threshold 0.5)

| Error type | Support | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Add medical device | 880 | 0.945 | 0.960 | 0.953 | 0.996 | 0.981 |
| False prediction | 1484 | 0.874 | 0.842 | 0.858 | 0.967 | 0.933 |
| False negation | 1059 | 0.850 | 0.665 | 0.746 | 0.948 | 0.851 |
| Add repetitions | 365 | 0.803 | 0.795 | 0.799 | 0.984 | 0.841 |
| Change name of device | 371 | 0.751 | 0.431 | 0.548 | 0.935 | 0.626 |
| Change position of device | 393 | 0.699 | 0.639 | 0.668 | 0.972 | 0.744 |
| Change severity | 671 | 0.748 | 0.681 | 0.713 | 0.956 | 0.801 |
| Change location | 808 | 0.732 | 0.627 | 0.676 | 0.935 | 0.752 |
| Change measurement | 65 | 0.712 | 0.569 | 0.632 | 0.988 | 0.673 |
| Change to homophone | 118 | 0.825 | 0.797 | 0.810 | 0.994 | 0.850 |
| Add typo | 884 | 0.853 | 0.891 | 0.872 | 0.986 | 0.937 |

## Table C. Six-group results of the main model

| Group | Support | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| False findings / devices | 2536 | 0.954 | 0.952 | 0.953 | 0.989 | 0.987 |
| Incorrect location / position | 1152 | 0.778 | 0.703 | 0.739 | 0.945 | 0.822 |
| Measurement error | 65 | 0.712 | 0.569 | 0.632 | 0.988 | 0.673 |
| Linguistic / transcription error | 1338 | 0.852 | 0.880 | 0.865 | 0.974 | 0.923 |
| Severity misclassification | 671 | 0.748 | 0.681 | 0.713 | 0.956 | 0.801 |

## Table C2. Confusion counts of the main model for every error type (threshold 0.5)

| Error type | TP | FP | FN | TN | Specificity |
|---|---|---|---|---|---|
| Add medical device | 845 | 49 | 35 | 4487 | 0.989 |
| False prediction | 1250 | 180 | 234 | 3752 | 0.954 |
| False negation | 704 | 124 | 355 | 4233 | 0.972 |
| Add repetitions | 290 | 71 | 75 | 4980 | 0.986 |
| Change name of device | 160 | 53 | 211 | 4992 | 0.989 |
| Change position of device | 251 | 108 | 142 | 4915 | 0.978 |
| Change severity | 457 | 154 | 214 | 4591 | 0.968 |
| Change location | 507 | 186 | 301 | 4422 | 0.960 |
| Change measurement | 37 | 15 | 28 | 5336 | 0.997 |
| Change to homophone | 94 | 20 | 24 | 5278 | 0.996 |
| Add typo | 788 | 136 | 96 | 4396 | 0.970 |

## Table C3. Per-type thresholds tuned on the validation split (secondary analysis)

Macro-F1 0.747 (vs. 0.752 at 0.5); micro-F1 0.785 (vs. 0.793).


## Table D. Detecting whether a report contains any error (main model)

| Operating point | Threshold | Sensitivity | Specificity | Precision (test mix) |
|---|---|---|---|---|
| Default 0.5 | 0.500 | 0.957 | 0.967 | 0.967 |
| Sensitivity 0.90 on val | 0.889 | 0.882 | 0.992 | 0.991 |
| Sensitivity 0.95 on val | 0.677 | 0.944 | 0.978 | 0.977 |

## Table E. Expected performance at realistic error prevalence (main model, sensitivity-0.90 operating point)

| Prevalence | PPV | NPV | Alerts / 1000 reports | True alerts / 1000 | Missed errors / 1000 |
|---|---|---|---|---|---|
| 3% | 0.779 | 0.996 | 34 | 26 | 3.5 |
| 5% | 0.857 | 0.994 | 51 | 44 | 5.9 |
| 10% | 0.927 | 0.987 | 95 | 88 | 11.8 |
| 20% | 0.966 | 0.971 | 183 | 176 | 23.6 |

## Table F. Paired bootstrap comparisons against the main model (difference = main − other)

| Other model | Δ Macro-F1 (95% CI) | p | Δ Macro-AUPRC (95% CI) | p |
|---|---|---|---|---|
| BERT-base (cased) | +0.010 (+0.002, +0.017) | 0.014 | +0.012 (+0.004, +0.019) | 0.000 |
| BioBERT v1.1 [focal loss] | +0.000 (-0.007, +0.008) | 0.994 | -0.005 (-0.011, +0.000) | 0.070 |
| BioBERT v1.1 [trained without clean reports] | +0.117 (+0.108, +0.127) | 0.000 | +0.136 (+0.122, +0.146) | 0.000 |
| BioBERT v1.1 [weighted BCE] | +0.048 (+0.035, +0.061) | 0.000 | +0.012 (+0.004, +0.019) | 0.000 |
| BiomedBERT (PubMedBERT) | +0.002 (-0.007, +0.011) | 0.724 | +0.004 (-0.004, +0.011) | 0.336 |
| BlueBERT | -0.005 (-0.014, +0.003) | 0.280 | +0.001 (-0.006, +0.009) | 0.726 |
| Bio+ClinicalBERT | +0.013 (+0.003, +0.021) | 0.006 | +0.009 (+0.001, +0.017) | 0.034 |
| CXR-BERT | -0.015 (-0.025, -0.006) | 0.000 | -0.010 (-0.019, -0.001) | 0.042 |
| Report length + LR | +0.751 (+0.737, +0.764) | 0.000 | +0.657 (+0.641, +0.669) | 0.000 |
| Llama-3.1-8B-Instruct (zero-shot) | +0.514 (+0.501, +0.529) | 0.000 | +0.648 (+0.634, +0.663) | 0.000 |
| Qwen2.5-7B-Instruct (zero-shot) | +0.506 (+0.490, +0.523) | 0.000 | +0.647 (+0.631, +0.662) | 0.000 |
| Qwen3-8B (zero-shot) | +0.488 (+0.472, +0.503) | 0.000 | +0.639 (+0.625, +0.654) | 0.000 |
| RadBERT (RoBERTa) | -0.010 (-0.019, -0.000) | 0.050 | -0.004 (-0.013, +0.004) | 0.270 |
| TF-IDF + LR | +0.259 (+0.244, +0.274) | 0.000 | +0.239 (+0.222, +0.251) | 0.000 |

## Table G. Sentence-level error localization

| Model | Sentences | Localization AUROC | AUPRC | F1 | Type macro-F1 | Report-level macro-F1 (aggregated) |
|---|---|---|---|---|---|---|
| BioBERT v1.1 [sentence-level] | 36719 | 0.956 | 0.879 | 0.760 | 0.613 | 0.658 |
