# Error detection in chest X-ray reports: a patient-disjoint benchmark on ReXErr-v1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23811898.svg)](https://doi.org/10.5281/zenodo.23811898)

Code for the manuscript *Detecting Synthetic Errors in Chest X-Ray Reports: A Patient-Disjoint Benchmark with
Original-Report Controls* (under review; first submitted as "... with Error-Free Controls").

The repository trains and evaluates report-level error detectors on [ReXErr-v1](https://physionet.org/content/rexerr-v1/1.0.0/),
a public collection of MIMIC-CXR chest X-ray reports with GPT-4o-injected errors, under a protocol designed to resemble
deployment:

| Choice | This benchmark | Pitfall it avoids |
|---|---|---|
| Splits | Official ReXErr / MIMIC-CXR train, validation and test splits (patient-disjoint) | Random report-level splits that put the same patients in training and test sets |
| Negatives | The original report of every study is an original-report control (no injected error; it may contain natural errors) | Test sets in which every report contains an error, so false alarms are never measured |
| Labels | 12 native error types from the sentence-level annotations (11 have positives); six-group view as a secondary analysis | Sampled error types that are not always present in the generated report |
| Text | Case preserved; whitespace collapsed | Formatting differences between original and GPT-4o-edited reports acting as a shortcut |
| Baselines | Report length + LR, TF-IDF + LR, seven encoders (BioBERT, BERT, Bio+ClinicalBERT, BiomedBERT, BlueBERT, CXR-BERT, RadBERT), three local zero-shot LLMs (Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B) | Results reported without reference points |
| Statistics | Three seeds for every encoder, patient-level bootstrap 95% CIs, paired bootstrap tests (10,000 resamples, Holm over all 21 encoder pairs), learning-rate check, AUPRC, validation-tuned thresholds, weighted-BCE and focal-loss variants, any-error detection, expected PPV at 3–20% error prevalence | Single runs and prevalence-blind metrics |
| Labels, checked | Blinded audit of 250 sentence-level labels and 63 control sentences by two consultant radiologists | Unverified automatic labels |

## Results

Official test split (5,416 reports: 2,708 error-injected reports and their 2,708 original versions), F1 at a threshold of
0.5, macro averages over the 11 error types with positives. Confidence intervals, paired bootstrap tests, per-type
results and the secondary analyses are in [`results/tables.md`](results/tables.md); the prediction files of every
run are in `results/runs` (no report text).

| Model | Macro-F1 | Macro-AUPRC | Any-error AUROC |
|---|---|---|---|
| RadBERT | 0.764 ± 0.004 | 0.820 ± 0.004 | 0.992 ± 0.000 |
| CXR-BERT | 0.759 ± 0.009 | 0.819 ± 0.008 | 0.994 ± 0.001 |
| BlueBERT | 0.754 ± 0.003 | 0.814 ± 0.002 | 0.992 ± 0.000 |
| BioBERT v1.1 | 0.752 ± 0.003 | 0.814 ± 0.003 | 0.991 ± 0.001 |
| BiomedBERT | 0.749 ± 0.005 | 0.813 ± 0.004 | 0.991 ± 0.000 |
| Bio+ClinicalBERT | 0.746 ± 0.006 | 0.808 ± 0.001 | 0.990 ± 0.000 |
| BERT-base-cased | 0.745 ± 0.002 | 0.806 ± 0.005 | 0.989 ± 0.001 |
| TF-IDF + LR | 0.494 | 0.578 | 0.918 |
| Report length + LR | 0.001 | 0.161 | 0.573 |

Encoders: mean ± SD over seeds 0–2. The LLMs answer with a binary list of error types, so they are reported at their
single operating point instead of with AUROC/AUPRC:

| LLM (zero-shot / 4 examples) | Macro-F1 | Any-error sensitivity | Any-error specificity |
|---|---|---|---|
| Qwen3-8B | 0.264 / 0.295 | 0.980 / 0.896 | 0.187 / 0.505 |
| Qwen2.5-7B-Instruct | 0.246 / 0.278 | 0.895 / 0.773 | 0.399 / 0.746 |
| Llama-3.1-8B-Instruct | 0.238 / 0.253 | 0.987 / 0.968 | 0.041 / 0.230 |

**Encoder comparisons.** `scripts/pairwise_encoders.py` compares all 21 encoder pairs on seed-averaged macro-F1 with a
patient-level paired bootstrap (10,000 resamples) and Holm's adjustment: RadBERT is better than four encoders and
CXR-BERT than two; the other 15 differences, including RadBERT versus CXR-BERT, are not significant
([`results/pairwise_encoders.json`](results/pairwise_encoders.json)). `scripts/revision_tables.py` gives per-type
metrics and micro, weighted and rare-type-free averages ([`results/revision_tables.json`](results/revision_tables.json)).

**Learning rate.** Every encoder was also trained once at 5×10⁻⁵ (and BioBERT at 1×10⁻⁵), prediction files in
`results/lr_runs`. Every encoder gained 0.013–0.023 macro-F1 and the ranking was largely preserved (Spearman ρ = 0.93):
`scripts/lr_sensitivity.py`, [`results/lr_sensitivity.json`](results/lr_sensitivity.json). Two runs repeated on
different GPUs (T4 float16 and A100 bfloat16) differed by 0.002 macro-F1 (`results/hardware_repro`).

BioBERT trained without original-report controls flags 80% of the original reports (any-error specificity 0.199), although
on error-injected reports alone it is close to BioBERT trained with controls (macro-F1 0.743 vs. 0.758).

**Cross-generator test set.** `scripts/make_external.py` gives each original test report one error produced by a
deterministic rule (laterality, severity, measurement, negation, homophone, typo, repetition or an inserted device), so
the error generator differs from GPT-4o. BioBERT and CXR-BERT trained on ReXErr flag only about half of these reports
(sensitivity 0.50–0.52 at unchanged specificity; any-error AUROC 0.85–0.87 instead of 0.99): see
[`results/external.md`](results/external.md). The external set is rebuilt from ReXErr and not distributed.

**Paraphrase control.** `scripts/paraphrase.py` rewords every sentence of the original test reports with Qwen3-8B, `scripts/make_paraphrase_control.py` keeps only rewrites that preserve numbers, sides, negation, uncertainty, severity, devices and placeholders, and builds partially and fully reworded original reports, and `scripts/score_encoder.py` scores them with encoders saved by `train_encoder.py`. Rewording raised the share of original reports flagged by RadBERT, CXR-BERT and BioBERT only from 2.3–3.5% to 6.8–9.5% (threshold 0.5), against 95–97% of ReXErr error reports: see [`results/paraphrase.md`](results/paraphrase.md). The reworded reports are not distributed.

**Error-rewording control.** `scripts/paraphrase.py --target errors` rewords the erroneous sentences of the error-injected test reports with Qwen3-8B (typos, homophones and repetitions are left as they are), `scripts/make_paraphrase_control.py --target errors` keeps the rewrites that pass the same checks, and `scripts/evaluate_error_rewording.py` compares detection before and after rewording. The share of error reports flagged at 0.5 fell only from 0.966 to 0.958 (BioBERT) and from 0.976 to 0.969 (CXR-BERT): see [`results/error_rewording.md`](results/error_rewording.md). The reworded reports are not distributed.

**Operating points and comparisons.** `scripts/operating_points.py` gives sensitivity by the number of annotated error types, PPV projections with bootstrap CIs (also for reports with a single error and for reworded reports) and single-error results on the cross-generator set ([`results/operating_points.json`](results/operating_points.json)); `scripts/holm_family.py` adjusts the seven encoder comparisons with Holm's method ([`results/holm.json`](results/holm.json)); `scripts/audit_extra.py` adds dataset checks such as deleted erroneous sentences and phrasing traces of inserted devices ([`results/audit_extra.json`](results/audit_extra.json)).

**Human audit.** `scripts/label_audit.py make` draws a stratified, blinded sample of 250 test sentences (18 per error type, 42 sentences labeled unchanged or neutral whose text changed, and 10 attention checks), and `scripts/rewrite_audit.py make` draws 63 sentences from the rewording controls and the cross-generator set; `scripts/audit_reader_package.py` builds the reader workbooks and an Arabic plain-language guide. Two consultant radiologists (22 and 20 years of experience) filled them in independently and blinded. They confirmed an error in 97.5% and 96.5% of the sentences labeled erroneous (agreement κ = 0.92), assigned the labeled type to 80% and 82% of them, found no changed meaning in any of 30 reworded original sentences, and could not recognize inserted devices or findings, laterality swaps or changed measurements without the original report (κ = 0.94). `analyze` compares the filled workbooks with the answer keys, which are now published together with the random sampling seeds (`results/label_audit_key.csv`, `results/rewrite_audit_key.csv`, `*_seed.txt`). Results: [`results/human_audit.json`](results/human_audit.json); the readers' answers without names or comments: `results/human_audit_answers/`. The workbooks contain report text and are not distributed.

## Data

ReXErr-v1 is open access under the Open Data Commons Attribution License v1.0. No report text is stored in this
repository; prediction files contain only study/patient identifiers, labels and probabilities.

```bash
bash scripts/download_data.sh data   # downloads and verifies ReXErr-v1 (87 MB zip)
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest            # unit tests on synthetic data, no download needed
```

## Reproducing the experiments

CPU steps:

```bash
python scripts/audit.py --data_dir data --out results/audit.json --tokenizer dmis-lab/biobert-v1.1
python scripts/baselines.py --data_dir data --model length
python scripts/baselines.py --data_dir data --model tfidf        # needs > 24 GB RAM; --max_train_studies 100000 otherwise
```

GPU steps (one run per command; checkpoints allow resuming):

```bash
python scripts/train_encoder.py --data_dir data --model dmis-lab/biobert-v1.1 --seed 0 --out_dir results/runs/biobert_s0
python scripts/train_encoder.py --data_dir data --model dmis-lab/biobert-v1.1 --task sentence \
    --max_train_studies 40000 --epochs 1 --out_dir results/runs/biobert_sentence_s0
python scripts/llm_zeroshot.py --data_dir data --model Qwen/Qwen2.5-7B-Instruct --n_studies 0 \
    --out_dir results/runs/llm_qwen2.5-7b
```

The full set of runs (three seeds of every encoder, localization, the no-control and class-imbalance ablations, the
learning-rate check and the LLM baselines) is scripted in [`notebooks/run_gpu_experiments.ipynb`](notebooks/run_gpu_experiments.ipynb), which runs on
Kaggle (GPU T4 x2) or Google Colab, embeds the code, downloads the data and writes `rexerr_results.zip` with the
prediction files. On Kaggle the runs are split over three versions to stay within the 12-hour session limit (see the
first cell).

Evaluation and figures:

```bash
python scripts/evaluate.py --runs_dir results/runs --out_dir results --main biobert --data_dir data
python scripts/figures.py --results results --runs_dir results/runs
```

`evaluate.py` writes `summary.json` (all metrics, confidence intervals, seed aggregates, paired comparisons and
prevalence analysis), `tables.md` and `error_examples.csv`.

## Layout

```
rexerr/                    taxonomy (labels.py), task construction (data.py), metrics, prediction I/O
scripts/audit.py           dataset audit: splits, leakage, label agreement, formatting, lengths
scripts/baselines.py       report-length and TF-IDF logistic regression
scripts/train_encoder.py   fine-tuning at report or sentence level (Hugging Face Trainer, resumable)
scripts/llm_zeroshot.py    local zero-shot LLM baseline (bf16, or 4-bit on GPUs under 20 GB)
scripts/evaluate.py        single evaluation path for every run
scripts/figures.py         figures
scripts/pairwise_encoders.py, scripts/lr_sensitivity.py, scripts/revision_tables.py   revision analyses
scripts/label_audit.py, scripts/rewrite_audit.py, scripts/audit_reader_package.py     human audit
scripts/build_notebook.py  regenerates the self-contained notebook from the sources
tests/                     unit tests
```

## Citation

Until the paper is published, please cite the archived code:

Alsbakhi M, Alsaqqa A, Lubbad M, AbuSamra AA. Code and predictions for: Detecting Synthetic Errors in Chest X-Ray Reports: A
Patient-Disjoint Benchmark with Original-Report Controls. Zenodo; 2026. https://doi.org/10.5281/zenodo.23811898

## License

Code: MIT (see [LICENSE](LICENSE)). Data: ReXErr-v1 and MIMIC-CXR remain under their own licenses and terms.
