# MTS-DDI

MTS-DDI is a multi-task drug–drug interaction (DDI) predictor. It combines a molecular-graph encoder with 3,200-dimensional Signaturizer features through a learnable gated fusion layer, then predicts both DDI existence and interaction severity (Minor / Moderate / Major).

This repository is the compact reproducibility code release. It contains the model implementation, preprocessing and training entry points, evaluation code, environment files and reproduction notes. Training data, the 2026 external cohort, generated results and trained model weights are intentionally withheld from this initial package.

## Graphical abstract

![MTS-DDI graphical abstract](figures/Graphical_Abstract.png)

## Repository contents

```text
config.py, config_cold.py       Model/training configuration
src/                            Model, dataset, preprocessing and training code
evaluate_2026.py                2026 external evaluation entry point
trained_models/                 Placeholder for weights released later
data/                           Placeholder for authorized local data
environment.yml                 Conda environment (Python 3.9, PyTorch, PyG, RDKit)
requirements.txt                Python packages installed after the conda environment
```

Exploratory plots, SLURM logs, exploratory notebooks, analysis intermediates, generated results and checkpoints are not included.

## Installation

```bash
bash setup_env.sh
conda activate mts-ddi
```

The environment file targets PyTorch 2.1 with CUDA 11.8. For a CPU-only machine, remove `pytorch-cuda=11.8` and add the appropriate CPU PyTorch option before creating the environment.

The Signaturizer model directories are not bundled. Place the local A1–E5 model directories under `external/biosignature/`, or set:

```bash
export MTS_DDI_BIO_MODEL_DIR=/path/to/biosignature
```

## Reproducing training and evaluation

Training and evaluation require the restricted DDI files, the 2026 cohort, local Signaturizer models and trained checkpoints. These files are not part of this initial code release. After they are made available locally, the entry points are:

```bash
python src/train_cold.py
python evaluate_2026.py
```

The main protocol is scaffold-based cold-start five-fold training with a GATv2 molecular encoder, gated structure–bio fusion, OHNM binary loss and weighted severity loss. See [REPRODUCTION.md](REPRODUCTION.md) for the protocol and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the release policy.

## Data and model availability

The DDI data, the 2026 external data, derived preprocessing caches, generated evaluation results and trained model weights are withheld from this initial GitHub package. **They will be uploaded after the accompanying manuscript is accepted**, subject to the licenses and redistribution conditions of the underlying sources. The release notes will then document versions, access instructions, checksums and required attribution.

## Citation

A citation entry will be added with the final author list, DOI and version after publication.

## Method overview

![Fig. 1. MTS-DDI method overview](figures/Fig1.png)

Figure 1 summarizes the curated DDI data construction, multimodal representation learning, gated fusion, multi-task prediction and scaffold-based evaluation workflow.

