"""Build notebooks/run_gpu_experiments.ipynb: a single self-contained notebook for Colab or Kaggle.

The notebook embeds every source file, so it needs no access to this repository.
usage: python scripts/build_notebook.py
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCES = sorted(glob.glob(os.path.join(ROOT, "rexerr", "*.py"))) + [
    os.path.join(ROOT, "scripts", f) for f in ("audit.py", "baselines.py", "train_encoder.py",
                                                 "llm_zeroshot.py", "evaluate.py")]

INTRO = """# ReXErr error detection — GPU experiments

**Kaggle (recommended):** Settings → Accelerator **GPU T4 x2**, Internet **on** (needs a verified phone number), then
**Save Version → Save & Run All (Commit)**; the run continues in the background and its files appear in the Output tab.
Each Kaggle session is limited to 12 hours, so run the groups in separate versions:

| Version | `RUN_GROUPS` | Accelerator | Approx. time |
|---|---|---|---|
| 1 | `["main", "encoders"]` — BioBERT × 3 seeds; BERT-base-cased, Bio+ClinicalBERT, BiomedBERT, BlueBERT, CXR-BERT, RadBERT | GPU T4 x2 | ~6.5 h |
| 2 | `["extras"]` — localization, no-control ablation, weighted-BCE and focal-loss BioBERT, Qwen2.5-7B, Llama-3.1-8B and Qwen3-8B zero-shot | GPU T4 x2 | ~4 h |
| 3 | `["tfidf"]` — full-data TF-IDF baseline (needs > 24 GB RAM, no GPU) | None (CPU) | ~1.5 h |

Each version writes `rexerr_results.zip` (prediction files only, no report text) to its Output tab. Runs already finished
in an earlier version are skipped if that version's output is added as an input (Add Input → Your Work).

**Colab:** choose a GPU runtime and Run all; results go to Google Drive and re-running resumes where it stopped.
Times are estimates.
"""

CONFIG = """# Choose which groups to run in this session: "main", "encoders", "extras", "tfidf".
RUN_GROUPS = ["main", "encoders"]
EPOCHS = 2
BATCH_SIZE = 32            # effective batch, identical on every GPU setup so runs are comparable
UNASSIGN_WHEN_DONE = True  # Colab only: release the runtime after the last cell
# Optional "owner/slug" of a Kaggle dataset holding the ReXErr-v1 zip; downloaded with the KAGGLE_API_TOKEN
# secret (Colab: key icon) or environment variable instead of the slower PhysioNet server.
KAGGLE_DATASET = ""
"""

ENV = """import glob, os, shutil, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/rexerr_experiments"
    CACHE = WORK                 # keep the 87 MB archive on Drive between sessions
elif IN_KAGGLE:
    WORK = "/kaggle/working/rexerr_experiments"  # saved as notebook output
    CACHE = "/tmp"
else:
    WORK = os.path.abspath("rexerr_experiments")
    CACHE = WORK
DATA = "/tmp/rexerr_data"        # local disk, not saved: fast reads and a small output
CODE, RUNS = f"{WORK}/code", f"{WORK}/runs"
for d in (CODE + "/rexerr", CODE + "/scripts", DATA, RUNS):
    os.makedirs(d, exist_ok=True)
if IN_KAGGLE:  # reuse runs finished in earlier versions whose output was added as an input
    for f in glob.glob("/kaggle/input/**/runs/*/preds_test.npz", recursive=True):
        src = os.path.dirname(f)
        dst = f"{RUNS}/{os.path.basename(src)}"
        if not os.path.exists(f"{dst}/preds_test.npz"):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print("reusing finished run:", os.path.basename(src))
import torch
N_GPU = max(1, torch.cuda.device_count())  # T4 x2: per-device batch is halved to keep the effective batch at 32
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
GPU_MEM_TOTAL = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("WORK =", WORK)
print(f"GPU: {GPU_NAME} x{torch.cuda.device_count()} ({GPU_MEM_TOTAL / 1e9:.0f} GB in total)")
if not torch.cuda.is_available() and set(RUN_GROUPS) - {"tfidf"}:
    print("NO GPU FOUND - Kaggle: Settings > Accelerator > GPU T4 x2; Colab: Runtime > Change runtime type")
"""

INSTALL = """!pip -q install "transformers==5.20.1" "accelerate>=1.0" "scikit-learn>=1.4" "pandas>=2.0" "scipy" "bitsandbytes>=0.45"
"""

DOWNLOAD = """import glob, os, shutil, subprocess, time, zipfile
import requests

def kaggle_token():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token and IN_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("KAGGLE_API_TOKEN")
        except Exception:
            token = None
    return token

def fetch_from_kaggle(dataset, token, dest):
    # requests drops the Authorization header on the redirect to the storage server
    with requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{dataset}",
                      headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Kaggle download failed with HTTP {r.status_code}")
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f, 1 << 20)

def fetch_from_physionet(dest):
    url = "https://physionet.org/content/rexerr-v1/get-zip/1.0.0/"
    for attempt in range(40):  # PhysioNet often resets or stalls long downloads; curl -C - resumes
        print("downloading ReXErr-v1 from PhysioNet, attempt", attempt + 1, flush=True)
        if subprocess.run(f'curl -sS -L -C - --speed-limit 2000 --speed-time 120 -o "{dest}" "{url}"',
                          shell=True).returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError("PhysioNet download failed")

def download_rexerr():
    if os.path.exists(f"{DATA}/ReXErr-report-level/ReXErr-report-level_test.csv"):
        print("data already present"); return
    # Kaggle: a copy of ReXErr-v1 added as an input (Add Input > Datasets) is used instead of downloading
    mirror = glob.glob("/kaggle/input/**/ReXErr-report-level/ReXErr-report-level_test.csv", recursive=True)
    if mirror:
        root = os.path.dirname(os.path.dirname(mirror[0]))
        for d in ("ReXErr-report-level", "ReXErr-sentence-level"):
            shutil.copytree(f"{root}/{d}", f"{DATA}/{d}", dirs_exist_ok=True)
        print("data copied from", root); return
    zips = [z for z in glob.glob("/kaggle/input/**/*.zip", recursive=True)  # not rexerr_results.zip of earlier runs
            if any("ReXErr-report-level/" in n for n in zipfile.ZipFile(z).namelist())]
    saved = f"{CACHE}/rexerr-v1-1.0.0.zip"  # verified archive kept between sessions (on Drive in Colab)
    if zips:
        archive = zips[0]
    elif os.path.exists(saved + ".done") and os.path.exists(saved):  # the marker alone is not enough
        archive = saved
    else:
        archive = "/tmp/rexerr-download.zip"  # local disk: resuming a partial file on Drive leaves duplicates
        token = kaggle_token()
        if KAGGLE_DATASET and token:
            print("downloading ReXErr-v1 from Kaggle dataset", KAGGLE_DATASET, flush=True)
            fetch_from_kaggle(KAGGLE_DATASET, token, archive)
        else:
            fetch_from_physionet(archive)
        subprocess.run(f'unzip -tq "{archive}"', shell=True, check=True)
        if os.path.dirname(saved) != "/tmp":
            shutil.copy(archive, saved)
            open(saved + ".done", "w").close()
    shutil.rmtree("/tmp/rexerr_unzip", ignore_errors=True)
    subprocess.run(f'unzip -q -o "{archive}" -d /tmp/rexerr_unzip', shell=True, check=True)
    for root, dirs, files in os.walk("/tmp/rexerr_unzip"):  # the zip has one top-level folder
        if "ReXErr-report-level" in dirs:
            subprocess.run(f'cp -r "{root}"/* "{DATA}"/', shell=True, check=True); break
    print(sorted(os.listdir(DATA)))
download_rexerr()
"""

RUNNER = """import json, os, shlex, subprocess, time
FAILED = []

def run(cmd, name=None):
    # A failed run is recorded and the notebook carries on, so the results of the other runs are still packaged.
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "UNEXPECTED" in line or "can be ignored" in line:
            continue
        print(line, end="", flush=True)
    if p.wait() != 0:
        FAILED.append(name or cmd)
        print(f"[failed] {name or cmd}", flush=True)
        return False
    return True

def train(name, model, extra=""):
    out = f"{RUNS}/{name}"
    if os.path.exists(f"{out}/preds_test.npz"):
        print(f"[skip] {name} already finished"); return
    t = time.time()
    run(f"python scripts/train_encoder.py --data_dir {shlex.quote(DATA)} --out_dir {shlex.quote(out)} "
        f"--model {model} --epochs {EPOCHS} --batch_size {max(1, BATCH_SIZE // N_GPU)} {extra}", name)
    print(f"[time] {name}: {(time.time() - t) / 60:.1f} min")

run(f"python scripts/audit.py --data_dir {shlex.quote(DATA)} --out {shlex.quote(WORK)}/audit.json "
    f"--tokenizer dmis-lab/biobert-v1.1 > /dev/null", "audit")
"""

MAIN = """if "main" in RUN_GROUPS:
    for seed in (0, 1, 2):
        train(f"biobert_s{seed}", "dmis-lab/biobert-v1.1", f"--seed {seed}")
"""

ENCODERS = """if "encoders" in RUN_GROUPS:
    train("bert-base-cased_s0", "google-bert/bert-base-cased")
    train("clinicalbert_s0", "emilyalsentzer/Bio_ClinicalBERT")
    train("biomedbert_s0", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    train("cxrbert_s0", "microsoft/BiomedVLP-CXR-BERT-general")
    train("bluebert_s0", "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
    train("radbert_s0", "UCSD-VA-health/RadBERT-RoBERTa-4m")
"""

EXTRAS = """if "extras" in RUN_GROUPS:
    # error localization: (sentence, report) pairs from a patient-level subset of the training split
    train("biobert_sentence_s0", "dmis-lab/biobert-v1.1", "--task sentence --max_train_studies 40000 --epochs 1")
    # ablation: no error-free reports during training
    train("biobert_noclean_s0", "dmis-lab/biobert-v1.1", "--include_clean 0")
    # class-imbalance strategies
    train("biobert_weighted_s0", "dmis-lab/biobert-v1.1", "--loss weighted")
    train("biobert_focal_s0", "dmis-lab/biobert-v1.1", "--loss focal")
    # zero-shot LLMs; whole test split when the GPUs hold the model in 16-bit (>= 20 GB in total, e.g. T4 x2)
    big = GPU_MEM_TOTAL >= 20e9
    for key, model in (("llm_qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct"),
                       ("llm_llama3.1-8b", "unsloth/Meta-Llama-3.1-8B-Instruct"),  # ungated copy of Meta's weights
                       ("llm_qwen3-8b", "Qwen/Qwen3-8B")):
        out = f"{RUNS}/{key}"
        if not os.path.exists(f"{out}/preds_test.npz"):
            run(f"python scripts/llm_zeroshot.py --data_dir {shlex.quote(DATA)} --out_dir {shlex.quote(out)} "
                f"--model {model} --n_studies {0 if big else 500} --batch_size {16 if big else 8}", key)
"""

TFIDF = """if "tfidf" in RUN_GROUPS:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if os.path.exists(f"{RUNS}/tfidf_sentence_clean1/preds_test.npz"):
        print("[skip] tfidf already finished")
    elif ram > 24e9:
        run(f"python scripts/baselines.py --data_dir {shlex.quote(DATA)} --runs_dir {shlex.quote(RUNS)} --model tfidf",
            "tfidf")
    else:
        print(f"only {ram / 1e9:.0f} GB RAM: full-data TF-IDF needs > 24 GB (Kaggle CPU sessions have 30 GB)")
"""

PACKAGE = """import glob, os, shutil, zipfile
zpath = f"{WORK}/rexerr_results.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    if os.path.exists(f"{WORK}/audit.json"):
        z.write(f"{WORK}/audit.json", "audit.json")
    for f in glob.glob(f"{RUNS}/*/*"):
        if os.path.basename(f) in ("preds_val.npz", "preds_test.npz", "meta.json", "raw_outputs.jsonl"):
            z.write(f, os.path.relpath(f, WORK))
print("finished runs:", sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{RUNS}/*/preds_test.npz")))
print("failed runs:", FAILED or "none")
print("wrote", zpath, f"({os.path.getsize(zpath) / 1e6:.1f} MB)")
if IN_KAGGLE:
    shutil.copy(zpath, "/kaggle/working/rexerr_results.zip")
    print("Download rexerr_results.zip from the Output tab of this version.")
elif IN_COLAB:
    print("Get it from Google Drive: MyDrive/rexerr_experiments/rexerr_results.zip")
    if UNASSIGN_WHEN_DONE:
        from google.colab import runtime
        runtime.unassign()  # stop consuming compute units
"""

def cell(kind, src):
    lines = src.splitlines(keepends=True)
    cell.n = getattr(cell, "n", 0) + 1
    c = {"cell_type": kind, "id": f"cell-{cell.n:02d}", "metadata": {}, "source": lines}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main():
    cells = [cell("markdown", INTRO), cell("code", CONFIG), cell("code", ENV), cell("code", INSTALL),
             cell("markdown", "### Source code (embedded)\n")]
    for path in SOURCES:
        rel = os.path.relpath(path, ROOT)
        with open(path) as f:
            body = f.read()
        cells.append(cell("code", f"%%writefile {{CODE}}/{rel}\n" + body))
    cells += [cell("markdown", "### Data (ReXErr-v1, open access, ODC-By 1.0)\n"), cell("code", DOWNLOAD),
              cell("markdown", "### Experiments\n"), cell("code", RUNNER), cell("code", MAIN), cell("code", ENCODERS),
              cell("code", EXTRAS), cell("code", TFIDF), cell("markdown", "### Package the results\n"),
              cell("code", PACKAGE)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                                  "isGpuEnabled": True, "language": "python",
                                                  "sourceType": "notebook", "dataSources": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, "notebooks", "run_gpu_experiments.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    main()
