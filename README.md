# Error detection in chest X-ray reports: a patient-disjoint benchmark on ReXErr-v1

Code for the manuscript *Multi-Label Error Detection in Chest X-Ray Reports with Fine-Tuned Biomedical Transformers:
A Patient-Disjoint Benchmark with Error-Free Controls* (under review).

The repository trains and evaluates report-level error detectors on [ReXErr-v1](https://physionet.org/content/rexerr-v1/1.0.0/),
a public collection of MIMIC-CXR chest X-ray reports with GPT-4o-injected errors, under a protocol designed to resemble
deployment:

| Choice | This benchmark | Pitfall it avoids |
|---|---|---|
| Splits | Official ReXErr / MIMIC-CXR train, validation and test splits (patient-disjoint) | Random report-level splits that put the same patients in training and test sets |
| Negatives | The original report of every study is an error-free control | Test sets in which every report contains an error, so false alarms are never measured |
| Labels | 12 native error types from the sentence-level annotations (11 have positives); six-group view as a secondary analysis | Sampled error types that are not always present in the generated report |
| Text | Case preserved; whitespace collapsed | Formatting differences between original and GPT-4o-edited reports acting as a shortcut |
| Baselines | Report length + LR, TF-IDF + LR, seven encoders (BioBERT, BERT, Bio+ClinicalBERT, BiomedBERT, BlueBERT, CXR-BERT, RadBERT), three local zero-shot LLMs (Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B) | Results reported without reference points |
| Statistics | Seeds, patient-level bootstrap 95% CIs, paired bootstrap tests, AUPRC, validation-tuned thresholds, weighted-BCE and focal-loss variants, any-error detection, expected PPV at 3–20% error prevalence | Single runs and prevalence-blind metrics |

## Results

Official test split (5,416 reports: 2,708 error-injected reports and their 2,708 original versions), F1 at a threshold of
0.5, macro averages over the 11 error types with positives. Confidence intervals, paired bootstrap tests, per-type
results and the secondary analyses are in [`results/tables.md`](results/tables.md); the prediction files of every
run are in `results/runs` (no report text).

| Model | Macro-F1 | Macro-AUPRC | Any-error AUROC |
|---|---|---|---|
| RadBERT (3 seeds) | 0.764 ± 0.004 | 0.820 ± 0.004 | 0.992 ± 0.000 |
| CXR-BERT (3 seeds) | 0.759 ± 0.009 | 0.819 ± 0.008 | 0.994 ± 0.001 |
| BlueBERT | 0.757 | 0.816 | 0.992 |
| BioBERT v1.1 (3 seeds) | 0.752 ± 0.003 | 0.814 ± 0.003 | 0.991 ± 0.001 |
| BiomedBERT | 0.750 | 0.813 | 0.991 |
| BERT-base-cased | 0.743 | 0.805 | 0.988 |
| Bio+ClinicalBERT | 0.739 | 0.808 | 0.990 |
| TF-IDF + LR | 0.494 | 0.578 | 0.918 |
| Qwen3-8B (4-shot / zero-shot) | 0.295 / 0.264 | 0.192 / 0.178 | 0.700 / 0.584 |
| Qwen2.5-7B-Instruct (4-shot / zero-shot) | 0.278 / 0.246 | 0.194 / 0.170 | 0.759 / 0.647 |
| Llama-3.1-8B-Instruct (4-shot / zero-shot) | 0.253 / 0.238 | 0.179 / 0.169 | 0.599 / 0.514 |
| Report length + LR | 0.001 | 0.161 | 0.573 |

Seed-averaged paired comparisons are in [`results/seed_comparison.md`](results/seed_comparison.md).

BioBERT trained without error-free controls flags 80% of the original reports (any-error specificity 0.199), although
on error-injected reports alone it is close to BioBERT trained with controls (macro-F1 0.743 vs. 0.758).

**Cross-generator test set.** `scripts/make_external.py` gives each original test report one error produced by a
deterministic rule (laterality, severity, measurement, negation, homophone, typo, repetition or an inserted device), so
the error generator differs from GPT-4o. BioBERT and CXR-BERT trained on ReXErr flag only about half of these reports
(sensitivity 0.50–0.52 at unchanged specificity; any-error AUROC 0.85–0.87 instead of 0.99): see
[`results/external.md`](results/external.md). The external set is rebuilt from ReXErr and not distributed.

**Paraphrase control.** `scripts/paraphrase.py` rewords every sentence of the original test reports with Qwen3-8B, `scripts/make_paraphrase_control.py` keeps only rewrites that preserve numbers, sides, negation, uncertainty, severity, devices and placeholders, and builds partially and fully reworded error-free reports, and `scripts/score_encoder.py` scores them with encoders saved by `train_encoder.py`. Rewording raised the share of error-free reports flagged by RadBERT, CXR-BERT and BioBERT only from 2.3–3.5% to 6.8–9.5% (threshold 0.5), against 95–97% of ReXErr error reports: see [`results/paraphrase.md`](results/paraphrase.md). The reworded reports are not distributed.

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

The full set of runs (three BioBERT seeds, six comparison encoders, localization, the no-control and class-imbalance
ablations and the LLM baselines) is scripted in [`notebooks/run_gpu_experiments.ipynb`](notebooks/run_gpu_experiments.ipynb), which runs on
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
scripts/build_notebook.py  regenerates the self-contained notebook from the sources
tests/                     unit tests
```

## Citation

Citation details will be added on publication.

## License

Code: MIT (see [LICENSE](LICENSE)). Data: ReXErr-v1 and MIMIC-CXR remain under their own licenses and terms.
