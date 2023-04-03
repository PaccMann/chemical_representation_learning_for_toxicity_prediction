[![Build Status](https://github.com/PaccMann/toxsmi/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/toxsmi/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Gradio demo](https://img.shields.io/website-up-down-green-red/https/hf.space/gradioiframe/GT4SD/molecular_properties/+.svg?label=demo%20status)](https://huggingface.co/spaces/GT4SD/molecular_properties)

# Chemical Representation Learning for Toxicity Prediction


PyTorch implementation related to the paper *Chemical Representation Learning for Toxicity Prediction* ([Born et al, 2023, *Digital Discovery*](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g)).
## Training your own model

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 

### Setup
```sh
conda env create -f conda.yml
conda activate toxsmi
pip install -e .
```

### Start a training

In the `scripts` directory is a training script [train_tox.py](./scripts/train_tox.py).

Download sample data from the Tox21 database and store it in a folder called `data`
[here](https://ibm.box.com/s/kahxnlg2k2s0x3z0r5fa6y67tmfhs6or). 

```console
(toxsmi) $ python3 scripts/train_tox.py data/tox21_train.csv \
data/tox21_score.csv data/tox21.smi data/smiles_language_tox21.pkl \
models params/mca.json test --embedding_path data/smiles_vae_embeddings.pkl
```

Type `python scripts/train_tox.py -h` for further help.

## Inference (using our pretrained models)
Several of our trained models are available via the [GT4SD](https://github.com/GT4SD), the Generative Toolkit for Scientific Discovery. See the paper [here](https://arxiv.org/abs/2207.03928).
We recommend to use [GT4SD](https://github.com/GT4SD/gt4sd-core) for inference. Once you install that library, use as follows:
```py
from gt4sd.properties import PropertyPredictorRegistry
tox21 = PropertyPredictorRegistry.get_property_predictor('tox21', {'algorithm_version': 'v0'})
tox21('CCO')
```

The other models are the SIDER model and the ClinTox model from the [MoleculeNet](https://moleculenet.org/datasets-1) benchmark:
```py
from gt4sd.properties import PropertyPredictorRegistry
sider = PropertyPredictorRegistry.get_property_predictor('sider', {'algorithm_version': 'v0'})
clintox = PropertyPredictorRegistry.get_property_predictor('clintox', {'algorithm_version': 'v0'})
print(f"SIDE effect predictions: {sider('CCO')}")
print(f"Clinical toxicitiy predictions: {clintox('CCO')}")
```

## Attention visualization
The model uses a self-attention mechanism that can highlight chemical motifs used for the predictions.
In [notebooks/toxicity_attention.ipynb](notebooks/toxicity_attention.ipynb) we share a tutorial on how to create such plots:
![Attention](assets/attention.gif "toxicophore attention")


## Citation
If you use this code in your projects, please cite the following:

```bib
@article{born2023chemical,
    author = {Born, Jannis and Markert, Greta and Janakarajan, Nikita and Kimber, Talia B. and Volkamer, Andrea and Martínez, María Rodríguez and Manica, Matteo},
    title = {Chemical representation learning for toxicity prediction},
    journal = {Digital Discovery},
    year = {2023},
    pages = {-},
    publisher = {RSC},
    doi = {10.1039/D2DD00099G},
    url = {http://dx.doi.org/10.1039/D2DD00099G}
}
```
