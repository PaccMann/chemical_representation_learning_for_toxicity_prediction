#!/usr/bin/env python3
"""Train DeepChem baselines on toxsmi prediction"""
import argparse
import json
import logging
import os
import sys
from importlib import reload

import deepchem as dc
import numpy as np
from deepchem.molnet import load_tox21, load_toxcast

# setup logging
logging.shutdown()
reload(logging)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset', type=str,
    help='Name of the deepchem dataset (toxcast or tox21)'
)
parser.add_argument(
    'featurizer', type=str,
    help='Type of features used for prediction'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
# yapf: enable


def main(dataset, featurizer, model_path, params_filepath, training_name):

    log = logging.getLogger('deepchem_baseline')
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    # Prepare the dataset
    log.info('Start data preprocessing...')

    if dataset == 'tox21':
        loader = load_tox21
    elif dataset == 'toxcast':
        loader = load_toxcast
    else:
        raise ValueError('Unknown dataset type, choose from {toxcast, tox21}.')

    tasks, data, transformers = loader(featurizer=featurizer, reload=False)
    train_data, validation_data, _ = data

    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode='classification'
    )

    model = dc.models.tensorgraph.fcnet.MultitaskClassifier(
        len(tasks),
        train_data.get_data_shape()[0],
        layer_sizes=params.get(
            'dense_layers_sizes',
            [2048, 1024],
        ),
        dropout=params.get('dropout', 0.5),
        n_classes=2
    )

    # Fit trained model
    log.info('Training about to start...')
    model.fit(train_data, nb_epoch=params['epochs'])
    #model.save_checkpoint(model_dir=os.path.join(model_dir, 'weights'))

    log.info('Evaluating model...')
    train_scores = model.evaluate(train_data, [metric], transformers)
    valid_scores = model.evaluate(validation_data, [metric], transformers)

    log.info(f'Train scores = {train_scores}')

    log.info(f'Validation scores = {valid_scores}')
    log.info('Done with training, models saved, shutting down.')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.dataset, args.featurizer, args.model_path, args.params_filepath,
        args.training_name
    )
