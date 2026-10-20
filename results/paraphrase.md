## Paraphrase control (error-free reports reworded by Qwen3-8B; every flag is a false alarm)

11,681 of 17,336 sentence rewrites (67.4%) passed the content checks; 2,614 of 2,708 reports have at least one. Partial: 2.7 sentences reworded per report; full: 4.5 (69% of sentences). Median share of words changed per reworded sentence: 0.25 (ReXErr erroneous sentences: 0.14).

### Share of reports flagged, threshold 0.5

| Model | Originals | Partially reworded (Δ, 95% CI) | Fully reworded (Δ, 95% CI) | ReXErr error reports |
|---|---|---|---|---|
| BioBERT v1.1 | 0.035 | 0.069 (+0.034; +0.026 to +0.044) | 0.095 (+0.060; +0.050 to +0.072) | 0.953 |
| CXR-BERT | 0.023 | 0.049 (+0.025; +0.018 to +0.033) | 0.068 (+0.045; +0.036 to +0.054) | 0.968 |
| RadBERT (RoBERTa) | 0.028 | 0.047 (+0.019; +0.013 to +0.026) | 0.075 (+0.047; +0.037 to +0.057) | 0.959 |

### Share of reports flagged, threshold val_sens_0.90

| Model | Originals | Partially reworded (Δ, 95% CI) | Fully reworded (Δ, 95% CI) | ReXErr error reports |
|---|---|---|---|---|
| BioBERT v1.1 | 0.009 | 0.019 (+0.010; +0.006 to +0.015) | 0.028 (+0.020; +0.014 to +0.026) | 0.889 |
| CXR-BERT | 0.005 | 0.010 (+0.004; +0.001 to +0.007) | 0.016 (+0.011; +0.007 to +0.016) | 0.887 |
| RadBERT (RoBERTa) | 0.006 | 0.012 (+0.006; +0.002 to +0.010) | 0.018 (+0.012; +0.008 to +0.017) | 0.886 |

### Most frequent types predicted on fully reworded reports (share at 0.5)

- BioBERT v1.1: False prediction 0.067, False negation 0.017, Change location 0.010 (originals re-scored with max |Δ| = 0.0147)
- CXR-BERT: False prediction 0.039, False negation 0.014, Change location 0.008 (originals re-scored with max |Δ| = 0.0123)
- RadBERT (RoBERTa): False prediction 0.041, False negation 0.015, Change location 0.009 (originals re-scored with max |Δ| = 0.0250)
