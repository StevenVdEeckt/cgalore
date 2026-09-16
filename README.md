# CGaLore: Curvature-Guided GaLore for Memory-Efficient Continual Adaptation of ASR Foundation Models

This repository contains supplementary material for the paper:

**CGaLore: Curvature-Guided GaLore for Memory-Efficient Continual Adaptation of ASR Foundation Models**, accepted at the **2026 IEEE Spoken Language Technology Workshop (SLT 2026)**, Palermo, Italy, December 2026.

The repository contains the code, configuration files, and data split metadata required to reproduce the CGaLore and GaLore experiments reported in the paper.

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{vander-eeckt2026cgalore,
  title     = {{CGaLore}: Curvature-Guided {GaLore} for Memory-Efficient Continual Adaptation of {ASR} Foundation Models},
  author    = {{Vander Eeckt}, Steven and {Van hamme}, Hugo},
  booktitle = {2026 IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026},
  month     = dec,
  address   = {Palermo, Italy}
}
```

## Overview

The repository is organized into three main parts:

```text
.
├── espnet/   # ESPnet additions and minimal modifications required for CGaLore
├── conf/     # Training configurations used in the experiments
└── data/     # Utterance- and speaker-level split definitions
```

The `espnet/` directory contains the files that need to be added to, or minimally modified in, a standard [ESPnet](https://github.com/espnet/espnet) installation. The `conf/` directory contains the training configurations used for CGaLore, GaLore, full fine-tuning, and the parameter-efficient baselines. The `data/` directory contains the task and memory-set definitions used in the experiments.

## Code

CGaLore is implemented on top of ESPnet. The files under `espnet/` are intended to be combined with the corresponding files from a standard ESPnet checkout.

For exact reproducibility, we recommend using the same ESPnet revision as used for the experiments and applying the files in this repository on top of that version. The upstream ESPnet repository is available at:

https://github.com/espnet/espnet

The relevant structure is:

```text
espnet/
└── espnet2/
    ├── bin/
    │   └── s2t_consolidate.py
    ├── legacy/
    │   └── nets/
    │       └── pytorch_backend/
    │           ├── consolidate3.py
    │           └── continual_learning3.py
    ├── optimizers/
    │   ├── galore_adamw.py
    │   └── named_optimizer.py
    ├── s2t/
    │   └── espnet_model.py
    └── tasks/
        ├── abs_task.py
        └── s2t.py
```

### Added files

- **`espnet2/optimizers/galore_adamw.py`**  
  Implements GaLore and CGaLore with AdamW. CGaLore uses KFAC curvature factors to construct the projection subspace. The implementation supports the projection variants used in the paper and periodically refreshes the low-rank basis.
- **`espnet2/optimizers/named_optimizer.py`**  
  Defines a lightweight optimizer interface indicating that an optimizer requires named parameters. This is needed because CGaLore associates each trainable weight matrix with its corresponding stored KFAC factors.
- **`espnet2/bin/s2t_consolidate.py`**  
  Entry point used to compute the curvature statistics required by CGaLore before adapting to a new task.
- **`espnet2/legacy/nets/pytorch_backend/consolidate3.py`**  
  Computes and stores the Kronecker-factored curvature statistics used by CGaLore. For each selected linear layer, it accumulates the input covariance and output-gradient covariance factors.
- **`espnet2/legacy/nets/pytorch_backend/continual_learning3.py`**  
  Contains the continual-learning utility used by the configurations, `FineTuningLinear`, which selects the parameters (weight matrices of `torch.nn.Linear` modules) that remain trainable during adaptation.

### Minimally modified ESPnet files

- **`espnet2/tasks/abs_task.py`**  
  Registers `GaLoreAdamW` as an ESPnet optimizer and adds support for optimizers that require `model.named_parameters()` rather than only `model.parameters()`.
- **`espnet2/tasks/s2t.py`**  
  Registers and constructs `FineTuningLinear` and passes the resulting continual-learning object to the S2T model.
- **`espnet2/s2t/espnet_model.py`**  
  Kept identical to the standard ESPnet implementation except for the minimal support needed to pass and store the continual-learning object in the model.

All remaining ESPnet functionality is taken from the standard ESPnet codebase.

### Baseline implementations

The baseline implementations are available in the repositories accompanying the corresponding earlier work:

- **PECL baselines (LoRA, BiLoRA, CSSVD):**  https://github.com/StevenVdEeckt/pecl-for-asr
- **SVR:**  https://github.com/StevenVdEeckt/efficient-rehearsal-for-cl-in-asr
- **IHR:**  https://github.com/StevenVdEeckt/inverse-hessian-regularization

The configuration files used for these baselines are nevertheless included in this repository so that the experimental settings reported in the CGaLore paper are explicit.

## Configuration files

Training configurations are provided under `conf/`:

```text
conf/
├── fft_baselines/
├── galore/
└── pecl_baselines/
```

### `conf/galore/`

```text
train_asr_owsmv32_small_cgalore.yaml
train_asr_owsmv32_small_galore.yaml
```

These are the main configurations for CGaLore and standard GaLore. They specify the adapted OWSM v3.2 small model, optimizer settings, projection rank, projection refresh interval, and, for CGaLore, the stored KFAC statistics.

### `conf/fft_baselines/`

```text
train_asr_owsmv32_small_finetuning_linear.yaml
train_asr_owsmv32_small_svr_50.yaml
```

These contain the full fine-tuning and SVR baseline settings used in the experiments.

### `conf/pecl_baselines/`

```text
train_asr_owsmv32_small_bilora.yaml
train_asr_owsmv32_small_cssvd.yaml
train_asr_owsmv32_small_lora_fta.yaml
```

These contain the parameter-efficient continual-learning baseline configurations used for comparison.

## Data splits

The `data/` directory contains the task definitions and split metadata used in the experiments:

```text
data/
├── exp1/
├── exp2/
├── heldout_tasks/
└── initial_tasks/
```

- **`exp1/`** contains the task and memory-set definitions for Experiment 1.
- **`exp2/`** contains the task and memory-set definitions for Experiment 2.
- **`initial_tasks/`** contains the splits for the tasks used to establish the initial multilingual model state.
- **`heldout_tasks/`** contains the held-out evaluation task definitions.

Within these directories, each task contains the relevant dataset splits. Depending on the task, this can include `train`, `dev`, and/or `test`; a `test` split is always provided for evaluation.

Each split contains:

```text
list_of_utterances.txt
list_of_speakers.txt
```

where:

- `list_of_utterances.txt` lists the utterance IDs belonging to the split;
- `list_of_speakers.txt` lists the speakers represented in the split.

The experiment directories also contain the memory-set definitions used by rehearsal-based baselines where applicable.

## Results

All reported adaptation results are based on single training runs. The utterance-level significance tests assess differences in recognition errors, but do not capture variability across independent training runs.
