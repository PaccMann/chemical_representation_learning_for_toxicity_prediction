[![Build Status](https://github.com/PaccMann/toxsmi/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/toxsmi/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# toxsmi

PyTorch implementation of `toxsmi`, a package for toxicity prediction models
from SMILES.

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 

Set up as follows:

```sh
conda env create -f conda.yml
conda activate toxsmi
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

## Attention visualization
ToxSmi uses a self-attention mechanism that can highlight chemical motifs used for the predictions.
In [notebooks/toxicity_attention.ipynb](notebooks/toxicity_attention.ipynb) we share a tutorial on how to create such plots:
![Attention](assets/attention.gif "toxicophore attention")


## Citation
If you use `toxsmi` in your projects, please (temporarily) cite the following (full paper in review):

```bib
@inproceedings{markert2020chemical,
  title={Chemical representation learning for toxicity prediction},
  author={Markert, Greta and Born, Jannis and Manica, Matteo and Schneider, Gisbert and Rodriguez Martinez, M},
  booktitle={PharML Workshop at ECML-PKDD (European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases)},
  year={2020}
}
```