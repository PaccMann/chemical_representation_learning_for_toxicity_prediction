#!/usr/bin/env python3
"""Test toxsmi predictor."""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import AnnotatedDataset, SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from toxsmi.models import MODEL_FACTORY
from toxsmi.utils import disable_rdkit_logging
from paccmann_predictor.utils.interpret import (
    monte_carlo_dropout, test_time_augmentation
)

# setup logging
# yapf: disable
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument(
    'model_path', type=str,
    help='Path to the trained model'
)
parser.add_argument(
    'smi_filepath', type=str,
    help='Path to the SMILES data (.smi)'
)
parser.add_argument(
    'labels_filepath', type=str,
    help='Path to the test toxicity scores (.csv)'
)
parser.add_argument(
    'output_folder', type=str,
    help='Directory where the output .csv will be stored.'
)
parser.add_argument(
    '-s', '--smiles_language_filepath', type=str, default='.',
    help='Path to a pickle of a SMILES language object.'
)
parser.add_argument(
    '-m', '--model_id', type=str, default='mca',
    help='ID for model factory'
)
parser.add_argument(
    '-c', '--confidence', default=True, action='store_true',
    help='Compute sample-wise confidences'
)
# yapf: enable


def main(
    model_path, smi_filepath, labels_filepath, output_folder,
    smiles_language_filepath, model_id, confidence
):

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('eval_toxicity')
    logger.setLevel(logging.INFO)
    disable_rdkit_logging()

    # Process parameter file:
    params = {}
    with open(os.path.join(model_path, 'model_params.json'), 'r') as fp:
        params.update(json.load(fp))

    # Create model directory
    os.makedirs(output_folder, exist_ok=True)

    if model_id not in MODEL_FACTORY.keys():
        raise KeyError(
            f'Model ID: Pass one of {MODEL_FACTORY.keys()}, not {model_id}'
        )

    device = get_device()
    weights_path = os.path.join(
        model_path, 'weights', f'best_ROC-AUC_{model_id}.pt'
    )

    # Restore model
    model = MODEL_FACTORY[model_id](params).to(device)
    if os.path.isfile(weights_path):
        try:
            model.load(weights_path, map_location=device)
        except Exception:
            logger.error(f'Error in model restoring from {weights_path}')
    else:
        logger.info(
            f'Did not find weights at {weights_path}, '
            f'name weights: "best_ROC-AUC_{model_id}.pt".'
        )
    model.eval()

    logger.info('Model restored. Model specs & parameters follow')
    for name, param in model.named_parameters():
        logger.info((name, param.shape))

    # Load language
    if smiles_language_filepath == '.':
        smiles_language_filepath = os.path.join(
            model_path, 'smiles_language.pkl'
        )
    smiles_language = SMILESLanguage.load(smiles_language_filepath)

    # Assemble datasets
    smiles_dataset = SMILESDataset(
        smi_filepath,
        smiles_language=smiles_language,
        padding_length=params.get('smiles_padding_length', None),
        padding=params.get('padd_smiles', True),
        add_start_and_stop=params.get('add_start_stop_token', True),
        augment=params.get('augment_test_smiles', False),
        canonical=params.get('test_canonical', False),
        kekulize=params.get('test_kekulize', False),
        all_bonds_explicit=params.get('test_all_bonds_explicit', False),
        all_hs_explicit=params.get('test_all_hs_explicit', False),
        randomize=False,
        remove_bonddir=params.get('test_remove_bonddir', False),
        remove_chirality=params.get('test_remove_chirality', False),
        selfies=params.get('selfies', False),
        sanitize=params.get('test_sanitize', False),
    )
    logger.info(
        f'SMILES Padding length is {smiles_dataset._dataset.padding_length}.'
        'Consider setting manually if this looks wrong.'
    )
    dataset = AnnotatedDataset(
        annotations_filepath=labels_filepath,
        dataset=smiles_dataset,
        device=device
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    if confidence:
        smiles_aleatoric_dataset = SMILESDataset(
            smi_filepath,
            smiles_language=smiles_language,
            padding_length=params.get('smiles_padding_length', None),
            padding=params.get('padd_smiles', True),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=True,  # Natively true for aleatoric uncertainity estimate
            canonical=params.get('test_canonical', False),
            kekulize=params.get('test_kekulize', False),
            all_bonds_explicit=params.get('test_all_bonds_explicit', False),
            all_hs_explicit=params.get('test_all_hs_explicit', False),
            remove_bonddir=params.get('test_remove_bonddir', False),
            remove_chirality=params.get('test_remove_chirality', False),
            selfies=params.get('selfies', False),
            sanitize=params.get('test_sanitize', False),
        )
        ale_dataset = AnnotatedDataset(
            annotations_filepath=labels_filepath,
            dataset=smiles_aleatoric_dataset,
            device=device
        )
        ale_loader = torch.utils.data.DataLoader(
            dataset=ale_dataset,
            batch_size=10,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        smiles_epi_dataset = SMILESDataset(
            smi_filepath,
            smiles_language=smiles_language,
            padding_length=params.get('smiles_padding_length', None),
            padding=params.get('padd_smiles', True),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=False,  # Natively false for epistemic uncertainity estimate
            canonical=params.get('test_canonical', False),
            kekulize=params.get('test_kekulize', False),
            all_bonds_explicit=params.get('test_all_bonds_explicit', False),
            all_hs_explicit=params.get('test_all_hs_explicit', False),
            remove_bonddir=params.get('test_remove_bonddir', False),
            remove_chirality=params.get('test_remove_chirality', False),
            selfies=params.get('selfies', False),
            sanitize=params.get('test_sanitize', False),
        )
        epi_dataset = AnnotatedDataset(
            annotations_filepath=labels_filepath,
            dataset=smiles_epi_dataset,
            device=device
        )
        epi_loader = torch.utils.data.DataLoader(
            dataset=epi_dataset,
            batch_size=10,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

    logger.info(
        f'Device for data loader is {dataset.device} and for '
        f'model is {device}'
    )
    # Start evaluation
    logger.info('Evaluation about to start...\n')

    preds, labels, attention_scores, smiles = [], [], [], []
    for idx, (smiles_batch, labels_batch) in enumerate(loader):
        pred, pred_dict = model(smiles_batch.to(device))
        preds.extend(pred.detach().squeeze().tolist())
        attention_scores.extend(
            torch.stack(pred_dict['smiles_attention'], dim=1).detach()
        )
        smiles.extend(
            [
                smiles_language.token_indexes_to_smiles(s.tolist())
                for s in smiles_batch
            ]
        )
        labels.extend(labels_batch.squeeze().tolist())
    # Scores are now 3D: num_samples x num_att_layers x padding_length
    attention = torch.stack(attention_scores, dim=0).numpy()

    if confidence:
        # Compute uncertainity estimates and save them
        epistemic_conf, epistemic_pred = monte_carlo_dropout(
            model, regime='loader', loader=epi_loader
        )
        aleatoric_conf, aleatoric_pred = test_time_augmentation(
            model, regime='loader', loader=ale_loader
        )
        epi_conf_df = pd.DataFrame(
            data=epistemic_conf.numpy(),
            columns=[
                f'epistemic_conf_{i}' for i in range(epistemic_conf.shape[1])
            ],
        )
        ale_conf_df = pd.DataFrame(
            data=aleatoric_conf.numpy(),
            columns=[
                f'aleatoric_conf_{i}' for i in range(aleatoric_conf.shape[1])
            ],
        )
        epi_pred_df = pd.DataFrame(
            data=epistemic_pred.numpy(),
            columns=[
                f'epistemic_pred_{i}' for i in range(epistemic_pred.shape[1])
            ],
        )
        ale_pred_df = pd.DataFrame(
            data=aleatoric_pred.numpy(),
            columns=[
                f'aleatoric_pred_{i}' for i in range(aleatoric_pred.shape[1])
            ],
        )

    logger.info(f'Shape of attention scores {attention.shape}.')
    np.save(os.path.join(output_folder, 'attention_raw.npy'), attention)
    attention_avg = np.mean(attention, axis=1)
    att_df = pd.DataFrame(
        data=attention_avg,
        columns=[f'att_idx_{i}' for i in range(attention_avg.shape[1])],
    )
    pred_df = pd.DataFrame(
        data=preds,
        columns=[f'pred_{i}' for i in range(len(preds[0]))],
    )
    lab_df = pd.DataFrame(
        data=labels,
        columns=[f'label_{i}' for i in range(len(labels[0]))],
    )
    df = pd.concat([pred_df, lab_df], axis=1)
    if confidence:
        df = pd.concat(
            [df, epi_conf_df, ale_conf_df, epi_pred_df, ale_pred_df], axis=1
        )
    df = pd.concat([df, att_df], axis=1)
    df.insert(0, 'SMILES', smiles)
    df.to_csv(os.path.join(output_folder, 'results.csv'), index=False)

    logger.info('Done, shutting down.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.model_path, args.smi_filepath, args.labels_filepath,
        args.output_folder, args.smiles_language_filepath, args.model_id,
        args.confidence
    )
