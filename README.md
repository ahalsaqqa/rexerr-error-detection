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
