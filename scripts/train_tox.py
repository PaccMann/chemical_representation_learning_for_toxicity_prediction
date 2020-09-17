#!/usr/bin/env python3
"""Train toxsmi predictor."""
import argparse
import json
import logging
import os
import sys
from time import time

import numpy as np
import torch
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.interpret import (
    monte_carlo_dropout, test_time_augmentation
)
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import AnnotatedDataset, SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import SMILESToMorganFingerprints
from pytoda.transforms import Compose, ToTensor
from toxsmi.models import MODEL_FACTORY
from toxsmi.utils import disable_rdkit_logging

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument(
    'train_scores_filepath',
    type=str,
    help='Path to the training toxicity scores (.csv)'
)
parser.add_argument(
    'test_scores_filepath',
    type=str,
    help='Path to the test toxicity scores (.csv)'
)
parser.add_argument(
    'smi_filepath', type=str, help='Path to the SMILES data (.smi)'
)
parser.add_argument(
    'smiles_language_filepath',
    type=str,
    help='Path to a pickle object a SMILES language object.'
)
parser.add_argument(
    'model_path', type=str, help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str, help='Path to the parameter file.'
)
parser.add_argument('training_name', type=str, help='Name for the training.')
parser.add_argument(
    '--embedding_path', type=str, default=None,
    help='Optional path to a pickle object of a pretrained embedding.'
)


def main(
    train_scores_filepath, test_scores_filepath, smi_filepath,
    smiles_language_filepath, model_path, params_filepath, training_name,
    embedding_path=None
):

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(f'{training_name}')
    logger.setLevel(logging.INFO)
    disable_rdkit_logging()

    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    if embedding_path:
        params['embedding_path'] = embedding_path

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    logger.info('Start data preprocessing...')
    smiles_language = SMILESLanguage.load(smiles_language_filepath)

    # Prepare FP processing
    if params.get('model_fn', 'mca') == 'dense':
        morgan_transform = Compose(
            [
                SMILESToMorganFingerprints(
                    radius=params.get('fp_radius', 2),
                    bits=params.get('num_drug_features', 512),
                    chirality=params.get('fp_chirality', True)
                ),
                ToTensor(get_device())
            ]
        )

        def smiles_tensor_batch_to_fp(smiles):
            """ To abuse SMILES dataset for FP usage"""
            out = torch.Tensor(
                smiles.shape[0], params.get('num_drug_features', 256)
            )
            for ind, tensor in enumerate(smiles):
                smiles = smiles_language.token_indexes_to_smiles(
                    tensor.tolist()
                )
                out[ind, :] = torch.squeeze(morgan_transform(smiles))
            return out

    # Assemble datasets
    smiles_dataset = SMILESDataset(
        smi_filepath,
        smiles_language=smiles_language,
        padding_length=params.get('smiles_padding_length', None),
        padding=params.get('padd_smiles', True),
        add_start_and_stop=params.get('add_start_stop_token', True),
        augment=params.get('augment_smiles', False),
        canonical=params.get('canonical', False),
        kekulize=params.get('kekulize', False),
        all_bonds_explicit=params.get('all_bonds_explicit', False),
        all_hs_explicit=params.get('all_hs_explicit', False),
        randomize=params.get('randomize', False),
        remove_bonddir=params.get('remove_bonddir', False),
        remove_chirality=params.get('remove_chirality', False),
        selfies=params.get('selfies', False),
        sanitize=params.get('sanitize', True)
    )

    train_dataset = AnnotatedDataset(
        annotations_filepath=train_scores_filepath,
        dataset=smiles_dataset,
        device=get_device()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )

    if (
        params.get('uncertainty', True)
        and params.get('augment_test_data', False)
    ):
        raise ValueError(
            'Epistemic uncertainty evaluation not supported if augmentation '
            'is enabled for test data.'
        )

    # Generally, if sanitize is True molecules are de-kekulized. Augmentation
    # preserves the "kekulization", so if it is used, test data should be
    # sanitized or canonicalized.
    smiles_test_dataset = SMILESDataset(
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
        sanitize=params.get('test_sanitize', False)
    )
    logger.info(smiles_dataset._dataset.transform)
    logger.info(smiles_test_dataset._dataset.transform)

    test_dataset = AnnotatedDataset(
        annotations_filepath=test_scores_filepath,
        dataset=smiles_test_dataset,
        device=get_device()
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=params.get('num_workers', 0)
    )

    if params.get('confidence', False):
        smiles_conf_dataset = SMILESDataset(
            smi_filepath,
            smiles_language=smiles_language,
            padding_length=params.get('smiles_padding_length', None),
            padding=params.get('padd_smiles', True),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=True,  # Natively true for epistemic uncertainity estimate
            canonical=params.get('canonical', False),
            kekulize=params.get('kekulize', False),
            all_bonds_explicit=params.get('all_bonds_explicit', False),
            all_hs_explicit=params.get('all_hs_explicit', False),
            randomize=params.get('randomize', False),
            remove_bonddir=params.get('remove_bonddir', False),
            remove_chirality=params.get('remove_chirality', False),
            selfies=params.get('selfies', False)
        )
        conf_dataset = AnnotatedDataset(
            annotations_filepath=test_scores_filepath,
            dataset=smiles_conf_dataset,
            device=get_device()
        )
        conf_loader = torch.utils.data.DataLoader(
            dataset=conf_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=params.get('num_workers', 0)
        )

    if not params.get('embedding', 'learned') == 'pretrained':
        params.update(
            {'smiles_vocabulary_size': smiles_language.number_of_tokens}
        )

    device = get_device()
    logger.info(
        f'Device for data loader is {train_dataset.device} and for '
        f'model is {device}'
    )
    save_top_model = os.path.join(model_dir, 'weights/{}_{}_{}.pt')

    model = MODEL_FACTORY[params.get('model_fn', 'mca')](params).to(device)
    logger.info(model)

    logger.info(f'Parameters follow')
    for name, param in model.named_parameters():
        logger.info((name, param.shape))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({'number_of_parameters': num_params})
    logger.info(f'Number of parameters {num_params}')

    # Define optimizer
    optimizer = (
        OPTIMIZER_FACTORY[params.get('optimizer', 'adam')]
        (model.parameters(), lr=params.get('lr', 0.00001))
    )
    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp)

    # Start training
    logger.info('Training about to start...\n')
    t = time()
    min_loss, max_roc_auc = 1000000, 0
    max_precision_recall_score = 0

    for epoch in range(params['epochs']):

        model.train()
        logger.info(params_filepath.split('/')[-1])
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, y) in enumerate(train_loader):

            smiles = torch.squeeze(smiles.to(device))
            # Transform smiles to FP if needed
            if params.get('model_fn', 'mca') == 'dense':
                smiles = smiles_tensor_batch_to_fp(smiles).to(device)

            y_hat, pred_dict = model(smiles)

            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            '\t **** TRAINING ****   '
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f'loss: {train_loss / len(train_loader):.5f}. '
            f'This took {time() - t:.1f} secs.'
        )
        t = time()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (smiles, y) in enumerate(test_loader):

                smiles = torch.squeeze(smiles.to(device))
                # Transform smiles to FP if needed
                if params.get('model_fn', 'mca') == 'dense':
                    smiles = smiles_tensor_batch_to_fp(smiles).to(device)

                y_hat, pred_dict = model(smiles)
                predictions.append(y_hat)
                # Copy y tensor since loss function applies downstream
                #   modification
                labels.append(y.clone())
                loss = model.loss(y_hat, y.to(device))
                test_loss += loss.item()

        predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
        labels = torch.cat(labels, dim=0).flatten().cpu().numpy()

        # Remove NaNs from labels to compute scores
        predictions = predictions[~np.isnan(labels)]
        labels = labels[~np.isnan(labels)]
        logger.info('length labels:', len(labels), labels)
        logger.info('length predictions:', len(predictions), predictions)
        test_loss_a = test_loss / len(test_loader)
        fpr, tpr, _ = roc_curve(labels, predictions)
        test_roc_auc_a = auc(fpr, tpr)

        # calculations for visualization plot
        precision, recall, _ = precision_recall_curve(labels, predictions)
        # score for precision vs accuracy
        test_precision_recall_score = average_precision_score(
            labels, predictions
        )

        logger.info(
            f"\t **** TEST **** Epoch [{epoch + 1}/{params['epochs']}], "
            f'loss: {test_loss_a:.5f}, , roc_auc: {test_roc_auc_a:.5f}, '
            f'avg precision-recall score: {test_precision_recall_score:.5f}'
        )
        info = {
            'test_auc': test_roc_auc_a,
            'train_loss': train_loss / len(train_loader),
            'test_loss': test_loss_a,
            'test_auc': test_roc_auc_a,
            'best_test_auc': max_roc_auc,
            'test_precision_recall_score': test_precision_recall_score,
            'best_precision_recall_score': max_precision_recall_score,
        }

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, params.get('model_fn', 'mca')))
            if typ == 'best':
                logger.info(
                    f'\t New best performance in {metric}'
                    f' with value : {val:.7f} in epoch: {epoch+1}'
                )

        if test_roc_auc_a > max_roc_auc:
            max_roc_auc = test_roc_auc_a
            info.update({'best_test_auc': max_roc_auc})
            save(save_top_model, 'ROC-AUC', 'best', max_roc_auc)
            np.save(
                os.path.join(model_dir, 'results', 'best_predictions.npy'),
                predictions
            )
            with open(
                os.path.join(model_dir, 'results', 'metrics.json'), 'w'
            ) as f:
                json.dump(info, f)
            if params.get('confidence', False):
                # Compute uncertainity estimates and save them
                epistemic_conf = monte_carlo_dropout(
                    model, regime='loader', loader=conf_loader
                )
                aleatoric_conf = test_time_augmentation(
                    model, regime='loader', loader=conf_loader
                )

                np.save(
                    os.path.join(model_dir, 'results', 'epistemic_conf.npy'),
                    epistemic_conf
                )
                np.save(
                    os.path.join(model_dir, 'results', 'aleatoric_conf.npy'),
                    aleatoric_conf
                )

        if test_precision_recall_score > max_precision_recall_score:
            max_precision_recall_score = test_precision_recall_score
            info.update(
                {'best_precision_recall_score': max_precision_recall_score}
            )
            save(
                save_top_model, 'precision-recall score', 'best',
                max_precision_recall_score
            )

        if test_loss_a < min_loss:
            min_loss = test_loss_a
            save(save_top_model, 'loss', 'best', min_loss)
            ep_loss = epoch

        if (epoch + 1) % params.get('save_model', 100) == 0:
            save(save_top_model, 'epoch', str(epoch))

    logger.info(
        'Overall best performances are: \n \t'
        f'Loss = {min_loss:.4f} in epoch {ep_loss} '
    )
    save(save_top_model, 'training', 'done')
    logger.info('Done with training, models saved, shutting down.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.train_scores_filepath, args.test_scores_filepath,
        args.smi_filepath, args.smiles_language_filepath, args.model_path,
        args.params_filepath, args.training_name, args.embedding_path
    )
