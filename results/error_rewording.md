## Error-rewording control (erroneous sentences reworded by Qwen3-8B, errors kept)

4,423 of 6,609 erroneous sentences (66.9%) were reworded with all checks passed; 2,395 error-injected reports have at least one (mean 1.8 reworded sentences). Typos, homophones and repetitions were not reworded.

| Model | Flagged at 0.5: before → after (Δ, 95% CI) | Flagged at validation threshold | Macro type recall (reworded types) |
|---|---|---|---|
| BioBERT v1.1 | 0.966 → 0.958 (-0.008; -0.013 to -0.004) | 0.897 → 0.879 (-0.018; -0.028 to -0.008) | 0.677 → 0.659 |
| CXR-BERT | 0.976 → 0.969 (-0.007; -0.012 to -0.002) | 0.909 → 0.892 (-0.017; -0.026 to -0.008) | 0.708 → 0.690 |

### Type recall before → after

| Type | n | BioBERT v1.1 | CXR-BERT |
|---|---|---|---|
| Add medical device | 813 | 0.96 → 0.96 | 0.96 → 0.96 |
| Add repetitions | 302 | 0.80 → 0.81 | 0.80 → 0.80 |
| Add typo | 779 | 0.90 → 0.90 | 0.93 → 0.93 |
| Change location | 751 | 0.63 → 0.60 | 0.66 → 0.63 |
| Change measurement | 64 | 0.58 → 0.53 | 0.59 → 0.56 |
| Change name of device | 350 | 0.43 → 0.41 | 0.50 → 0.48 |
| Change position of device | 383 | 0.63 → 0.67 | 0.65 → 0.70 |
| Change severity | 586 | 0.69 → 0.67 | 0.73 → 0.71 |
| Change to homophone | 100 | 0.83 → 0.84 | 0.83 → 0.84 |
| False negation | 992 | 0.67 → 0.60 | 0.72 → 0.64 |
| False prediction | 1351 | 0.84 → 0.82 | 0.84 → 0.83 |
