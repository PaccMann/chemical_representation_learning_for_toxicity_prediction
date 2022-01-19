[![Build Status](https://github.com/PaccMann/toxsmi/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/toxsmi/actions/workflows/build.yml)

# toxsmi

PyTorch implementation of `toxsmi`, a package for toxicity prediction models
from SMILES.

## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements.

Create a conda environment:

```sh
conda env create -f conda.yml
```

Activate the environment:

```sh
conda activate toxsmi_test
```

Install in editable mode for development:

```sh
pip install -e .
```

## Example usage

In the `scripts` directory is a training script [train_tox.py](./scripts/train_tox.py) that makes use
of `toxsmi`.

Download sample data from the Tox21 database and store it in a folder called `data`
[here](https://ibm.box.com/s/kahxnlg2k2s0x3z0r5fa6y67tmfhs6or).


```console
(toxsmi) $ python3 scripts/train_tox.py data/tox21_train.csv \
data/tox21_score.csv data/tox21.smi data/smiles_language_tox21.pkl \
models params/mca.json test --embedding_path data/smiles_vae_embeddings.pkl
```

Type `python scripts/train_tox.py -h` for further help.

