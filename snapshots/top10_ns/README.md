Image Matching Challenge 2025
=============================

1st Place Solution:
- https://www.kaggle.com/competitions/image-matching-challenge-2025/discussion/583058


This repository contains my IMC2023-IMC2025 submission code.

> [!WARNING]
> Please note that code from previous years may not work correctly.

> [!WARNING]
> This repository includes various implementations adapted from other repositories. For licenses, please refer to the respective repositories.

### TODO

- [ ] Rearrange files and directories, keeping flat-layout style (i.e., Move all sub-packages and modules to the `imc25` package).
- [ ] For packaging, Use `pyproject.toml` instead of `setup.py`.
- [ ] Apply ruff to all files.
- [x] Disable to import QuadTreeMatcher to reduce dependencies.


## Setup

### Dependencies

Install dependencies in pyproject.toml. Here is an example using uv:

```bash
# Install Python-3.11 if needed.
$ uv python install 3.11

# Use Python-3.11, aligning with the Python version in the Kaggle notebook.
$ uv python pin 3.11
$ python --version
Python 3.11.11

# Install dependencies with pyproject.toml.
# Actually, this command will fail because pre-built ASMK and curope packages is not in bundle/oss
$ uv sync
$ . .venv/bin/activate
```

#### Install python packages with compiled extensions

- ASMK: https://github.com/naver/mast3r (Get Started > Installation > 3)
- curope: https://github.com/naver/mast3r (Get Started > Installation > 4)

Build the packages as *.whl file and move them into bundle/oss before `uv sync`.


### Data

Place the IMC25 dataset directory (or a symbolic link to it) at the root of this repository.

```bash
$ tree -L 2 data/
data/
├── sample_submission.csv
├── test
│   ├── ETs
│   └── stairs
├── train
│   ├── ETs
│   ├── amy_gardens
│   ├── fbk_vineyard
│   ├── imc2023_haiper
│   ├── imc2023_heritage
│   ├── imc2023_theather_imc2024_church
│   ├── imc2024_dioscuri_baalshamin
│   ├── imc2024_lizard_pond
│   ├── pt_brandenburg_british_buckingham
│   ├── pt_piazzasanmarco_grandplace
│   ├── pt_sacrecoeur_trevi_tajmahal
│   ├── pt_stpeters_stpauls
│   └── stairs
├── train_labels.csv
└── train_thresholds.csv
```


### Models

Specify the path for each model in conf/models.yaml. The format is as follows:

```
MODEL_NAME:
  local: path/to/model/on/your/system
  kernel: path/to/model/on/kaggle/notebook
```

For IMC25, you need to define the paths for at least the following models:

```
MAGICLEAP_SUPERPOINT:
  local: extra/pretrained_models/research_only/magicleap/superpoint_v1.pth
  kernel: ../input/imc2023models/research_only/magicleap/superpoint_v1.pth

DINOV2_BASE:
  local: extra/kagglehub/dinov2-base
  kernel: /kaggle/input/dinov2/pytorch/base/1

ALIKED_LIGHTGLUE_N16:
  local: extra/kagglehub/aliked-n16/aliked-n16.pth
  kernel: /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth

MAST3R:
  local: extra/pretrained_models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
  kernel: ../input/imc2025models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

MAST3R_RETRIEVAL:
  local: extra/pretrained_models/mast3r_retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
  kernel: ../input/imc2025models/mast3r_retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth

MAST3R_RETRIEVAL_CODEBOOK:
  local: extra/pretrained_models/mast3r_retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
  kernel: ../input/imc2025models/mast3r_retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

DAD:
  local: extra/pretrained_models/dad/dad.pth
  kernel: ../input/imc2025models/dad/dad.pth

ISC:
  local: extra/pretrained_models/isc/isc_ft_v107.pth.tar
  kernel: ../input/imc2025models/isc/isc_ft_v107.pth.tar
```

You will need to download the weights for these models from their official repositories:

- SuperPoint: https://github.com/magicleap/SuperGluePretrainedNetwork
- ALIKED: https://github.com/cvg/LightGlue
- MASt3R: https://github.com/naver/mast3r
- DaD: https://github.com/parskatt/dad
- ISC: https://github.com/lyakaap/ISC21-Descriptor-Track-1st

You can select `local`/`kernel` to be used with `--env-name` of `evaluate_imc2025.py` and `kernel.py`. `local` will be used as default.


## Evaluation

```bash
# Use only 2025 datasets
$ python evaluate_imc2025.py -c path/to/config.yaml -p imc2025

# Use only ETs
$ python evaluate_imc2025.py -c path/to/config.yaml --datasets ETs
```

### Run evaluation with the best submission

```bash
$ python evaluate_imc2025.py -c conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-c.yaml --datasets ETs

ETs: score=59.46% (mAA=42.31%, clusterness=100.00%)
Average over all datasets: score=59.46% (mAA=42.31%, clusterness=100.00%)
```

## Submission

### Environment variables

```bash
# Limit the number of threads for image preprocessing
$ export OMP_NUM_THREADS=1
$ export MKL_NUM_THREADS=1

# Force to disable internet access
$ export HF_HUB_OFFLINE=1
$ export TRANSFORMERS_OFFLINE=1

# Set if an error occurred when using deterministic settings
$ export CUBLAS_WORKSPACE_CONFIG=':4096:8'

# You can change the default paths using environment variables below
$ export DEFAULT_DATASET_DIR=data
$ export DEFAULT_MODEL_LIST_PATH=conf/models.yaml
```

### Multi-GPUs

```bash
# Run the submission script (2 GPUs)
$ torchrun --nnodes 1 --nproc_per_node 2 --standalone -m ns64_imc2025lib.kernel -c config.yaml --env-name kernel --dist --kaggle-submit
```

See the kaggle notebook in detail.
