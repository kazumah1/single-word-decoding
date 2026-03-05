# Setup Notes — Sentence Decoding Pipeline

Complete record of every step, code change, and fix required to run the sentence decoding
pipeline from scratch on a fresh machine.

**Environment:** Python 3.13, macOS Apple Silicon (MPS GPU), no pre-installed deps.
**Dataset:** Gwilliams2022 (MEG-MASC), subject sub-01.
**Run date:** 2026-03-04.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Dataset: Gwilliams2022](#2-dataset-gwilliams2022)
3. [Environment setup](#3-environment-setup)
4. [Install local packages](#4-install-local-packages)
5. [Install extra dependencies](#5-install-extra-dependencies)
6. [Pin x-transformers to 1.26.0](#6-pin-x-transformers-to-1260)
7. [Pin torchmetrics to 1.5.2](#7-pin-torchmetrics-to-152)
8. [Build kenlm from source](#8-build-kenlm-from-source)
9. [Create required directories](#9-create-required-directories)
10. [Download data](#10-download-data)
11. [Code change: set dataset to Gwilliams2022](#11-code-change-set-dataset-to-gwilliams2022)
12. [Code change: inference_mode=False on Trainer](#12-code-change-inference_modefalse-on-trainer)
13. [Code change: allow UninitializedParameter in checkpoints](#13-code-change-allow-uninitializedparameter-in-checkpoints)
14. [Run the pipeline](#14-run-the-pipeline)
15. [Results](#15-results)
16. [Model architecture summary](#16-model-architecture-summary)

---

## 1. Project overview

The pipeline decodes sentences from MEG brain recordings. Given a MEG segment recorded
while a subject listened to a spoken word/sentence, the model predicts the corresponding
text embedding. At test time it retrieves the closest sentence from a candidate set.

**Repo structure:**
- `neuralset/` — MEG dataset loaders (local package, editable install required)
- `neuraltrain/` — model training utilities (local package, editable install required)
- `sentence_decoding/` — main experiment: model, loss, metrics, callbacks, grids
- `download_gwilliams_subject.py` — script to download a single subject from OSF (created during this session)

**Key config file:** `sentence_decoding/grids/defaults.py`
**Test grid entry point:** `sentence_decoding/grids/test.py`

**Training config (from defaults.py):**
| Parameter | Value |
|---|---|
| Epochs | 50 (early stop patience=10) |
| Learning rate | 1e-4 |
| Batch size | 128 |
| MEG frequency | 50 Hz |
| MEG filter | 0.1–40 Hz |
| Segment duration | 3.0 s |
| Text features | T5-large, layer 50%, trigger aggregation |
| Loss | SigLip (contrastive sigmoid) |
| Monitor metric | val_retrieval_acc10_size=all_macro_0 |

---

## 2. Dataset: Gwilliams2022

**Full name:** MEG-MASC (Magneto-EncephaloGraphy for Mapping Auditory Sentence Comprehension)

**Why this dataset:**
- English language (confirmed; code sets `event["language"] = "english"`)
- 208-channel KIT MEG system at 1000 Hz
- 27 subjects, 2 sessions each, ~2 hours of naturalistic spoken stories per subject
- BIDS format, publicly available on OSF
- High SNR, large vocabulary → best dataset in the repo for this task

**Citation:** Gwilliams et al. (2023). *MEG-MASC: a high-quality magneto-encephalography
dataset for evaluating systems that map perception to language.* Scientific Data.

**OSF repositories (3 repos split by subject):**
- `ag3kj` — contains sub-01 through sub-09 (approximately)
- `h2tzn` — contains remaining subjects
- `u5327` — contains stimuli shared across all subjects

**Data structure (BIDS):**
```
gwilliams2022/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── ses-0/
│   │   └── meg/  (.fif files, events, channels, coordsystem)
│   └── ses-1/
└── stimuli/
    └── audio/  (.wav files of stories)
```

**Size:** sub-01 alone = ~4.3 GB, 117 files (only in OSF repo `ag3kj`).
Full dataset (27 subjects) ≈ 300+ GB across all three OSF repos.

---

## 3. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # or: .venv/bin/activate.fish
```

All subsequent `pip install` commands assume the venv is active or use `.venv/bin/pip`.

---

## 4. Install local packages

**Problem:** Standard `pip install -e .` does NOT correctly expose submodules for
`neuralset` and `neuraltrain` under Python 3.13. Imports like
`from neuralset import SegmentDataset` fail with `AttributeError`.

**Fix:** Use `editable_mode=strict` which creates proper import symlinks:

```bash
pip install --config-settings editable_mode=strict -e neuralset/
pip install --config-settings editable_mode=strict -e neuraltrain/
pip install -e sentence_decoding/
```

Note: `sentence_decoding` works with the standard editable install.

---

## 5. Install extra dependencies

These are not declared in any `pyproject.toml` but are required at runtime:

```bash
pip install lightning pytorch-lightning torchvision x-transformers wandb
```

---

## 6. Pin x-transformers to 1.26.0

**Error encountered:**
```
TypeError: Encoder.__init__() got an unexpected keyword argument 'resi_dual'
```

**Root cause:** `neuraltrain/neuraltrain/models/transformer.py` passes `resi_dual=...`
to `x_transformers.Encoder`. This kwarg was removed in x-transformers versions after
1.26.0.

**Fix:** Pin to the last compatible version:

```bash
pip install "x-transformers==1.26.0"
```

**Note:** You will see harmless `FutureWarning` about `torch.cuda.amp.autocast` — this
is a known deprecation in x-transformers 1.26.0 that does not affect functionality.

---

## 7. Pin torchmetrics to 1.5.2

**Error encountered:**
```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
  File "torchmetrics/functional/text/bleu.py", line 88, in _bleu_score_update
    preds_len += len(pred)
```

**Root cause:** Lightning 2.x runs validation inside `torch.inference_mode()` by
default. Metric state tensors reset within that context become "inference tensors".
When `on_validation_epoch_end` runs (partially outside the inference_mode context),
the inplace `+=` in torchmetrics BLEU fails. This bug exists in both 1.8.2 and 1.5.2
but is avoided when combined with fix #8 below (`inference_mode=False`).

**Fix:**

```bash
pip install "torchmetrics==1.5.2"
```

---

## 8. Build kenlm from source

**Error encountered:**
```
ModuleNotFoundError: No module named 'kenlm'
```
Followed by build failure when installing from PyPI:
```
error: use of undeclared identifier '_PyLong_AsByteArray'
error: use of undeclared identifier '_PyGen_SetStopIterationValue'
```

**Root cause:** The PyPI `kenlm` wheel uses deprecated C Python APIs that were removed
in Python 3.13.

**Fix:** Clone the repo, regenerate the Cython bindings with a modern Cython version,
then build manually:

```bash
git clone https://github.com/kpu/kenlm
cd kenlm/python
pip install cython
cython kenlm.pyx --cplus    # regenerates kenlm.cpp compatible with Python 3.13
cd ..
pip install .
cd ..
```

---

## 9. Create required directories

**Error encountered:**
```
pydantic_core._pydantic_core.ValidationError:
  Value error, Infra folder parent .../sentence_results/cache needs to exist
  Value error, No folder/file named projects
```

**Fix:**

```bash
mkdir -p sentence_results/cache
mkdir -p projects/
```

These must exist relative to wherever you run the pipeline from (i.e., the `Code/`
root directory).

---

## 10. Download data

A custom download script `download_gwilliams_subject.py` was created to download
a single subject from the three Gwilliams2022 OSF repos without pulling the entire
~300 GB dataset.

**What it downloads:**
- `sub-01/` folder from all OSF repos where it exists
- `stimuli/` folder (audio files needed for feature extraction)
- Top-level metadata files (`dataset_description.json`, `participants.tsv`, etc.)

**Usage:**

```bash
.venv/bin/python download_gwilliams_subject.py
```

Data is saved to `neural_data/gwilliams2022/` (~4.3 GB, 117 files).
Only OSF repo `ag3kj` contains sub-01; `h2tzn` and `u5327` contain other subjects
and stimuli respectively. The script skips already-downloaded files so it is safe
to re-run if interrupted.

---

## 11. Code change: set dataset to Gwilliams2022

**File:** `sentence_decoding/grids/test.py`, line 33

**Error without this change:**
```
AssertionError: run study.download() first [Armeni2022 not downloaded]
```

The default test grid targets `Armeni2022` which is not downloaded.

**Change:**
```python
# Before:
params = {"data.dataset": "Armeni2022"}

# After:
params = {"data.dataset": "Gwilliams2022"}
```

---

## 12. Code change: `inference_mode=False` on Trainer

**File:** `sentence_decoding/main.py`, inside the `pl.Trainer(...)` constructor (~line 432)

**Error without this change:**
```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```
(same root cause as fix #7 — this code change is what actually resolves it)

**Change:** Add `inference_mode=False` to the Trainer:

```python
self._trainer = pl.Trainer(
    gradient_clip_val=self.trainer_config.gradient_clip_val,
    devices=self.infra.gpus_per_node,
    limit_train_batches=None,
    max_epochs=self.trainer_config.n_epochs,
    enable_progress_bar=True,
    log_every_n_steps=20,
    fast_dev_run=self.trainer_config.fast_dev_run,
    logger=self._logger,
    callbacks=callbacks,
    inference_mode=False,  # Required: torchmetrics BLEU does inplace += on metric state tensors
)
```

**Why:** `inference_mode=False` makes Lightning use `torch.no_grad()` instead of
`torch.inference_mode()` during validation. `torch.no_grad()` does not freeze tensors
as "inference tensors", so the inplace update in torchmetrics BLEU succeeds.

---

## 13. Code change: allow `UninitializedParameter` in checkpoint loading

**File:** `sentence_decoding/main.py`, after the existing `from torch import nn` import

**Error without this change:**
```
_pickle.UnpicklingError: Weights only load failed.
  WeightsUnpickler error: Unsupported global: GLOBAL torch.nn.parameter.UninitializedParameter
  was not an allowed global by default.
```

This error occurs at the **test phase**, after training completes, when Lightning loads
the best checkpoint via `trainer.test(..., ckpt_path="best")`.

**Root cause:** PyTorch 2.6 changed `torch.load()` to default to `weights_only=True`.
The saved checkpoint contains `UninitializedParameter` (used by lazy-initialized layers
in the model), which is not in PyTorch's default safe globals list.

**Change:** Add before any model/trainer code runs:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader

# PyTorch 2.6+ changed weights_only=True as default; allow UninitializedParameter in checkpoints
torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
```

---

## 14. Run the pipeline

```bash
DATAPATH=/path/to/Code/neural_data \
SAVEPATH=/path/to/Code/sentence_results \
WANDB_MODE=disabled \
.venv/bin/python -m sentence_decoding.grids.test
```
