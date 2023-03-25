#!/usr/bin/env python3
"""Train toxsmi predictor."""
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from time import time

import numpy as np

# Ensure Ubuntu/rdkit compatibility
import torch
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.interpret import (
    monte_carlo_dropout,
    test_time_augmentation,
)
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import AnnotatedDataset, SMILESTokenizerDataset
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.smiles.transforms import SMILESToMorganFingerprints
from pytoda.transforms import Compose, ToTensor
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from toxsmi.models import MODEL_FACTORY
from toxsmi.utils import disable_rdkit_logging
from toxsmi.utils.performance import PerformanceLogger

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    "-train_scores_filepath",
    type=str,
    help="Path to the training toxicity scores (.csv)",
)
parser.add_argument(
    "--test",
    "-test_scores_filepath",
    type=str,
    help="Path to the test toxicity scores (.csv)",
)
parser.add_argument(
    "--smi", "-smi_filepath", type=str, help="Path to the SMILES data (.smi)"
)
parser.add_argument(
    "--language",
    "-smiles_language_filepath",
    type=str,
    help="Path to a pickle object a SMILES language object.",
)
parser.add_argument(
    "--model", "-model_path", type=str, help="Directory where the model will be stored."
)
parser.add_argument(
    "--params", "-params_filepath", type=str, help="Path to the parameter file."
)
parser.add_argument("--name", "-training_name", type=str, help="Name for the training.")
parser.add_argument(
    "--embedding",
    "-embedding_path",
    type=str,
    default=None,
    help="Optional path to a pickle object of a pretrained embedding.",
)


def main(
    train_scores_filepath,
    test_scores_filepath,
    smi_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
    training_name,
    embedding_path=None,
):

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(f"{training_name}")
    logger.setLevel(logging.INFO)
    disable_rdkit_logging()

    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    if embedding_path:
        params["embedding_path"] = embedding_path

    # Create model directory and dump files
    print(model_path, training_name)
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp, indent=4)

    logger.info("Start data preprocessing...")
    smiles_language = SMILESTokenizer(
        vocab_file=smiles_language_filepath,  # if None, new language is created
        padding_length=params.get("padding_length", None),
        randomize=False,
        add_start_and_stop=params.get("start_stop_token", True),
        padding=params.get("padding", True),
        augment=params.get("augment_smiles", False),
        canonical=params.get("canonical", False),
        kekulize=params.get("kekulize", False),
        all_bonds_explicit=params.get("bonds_explicit", False),
        all_hs_explicit=params.get("all_hs_explicit", False),
        remove_bonddir=params.get("remove_bonddir", False),
        remove_chirality=params.get("remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("sanitize", False),
    )
    test_smiles_language = deepcopy(smiles_language)
    test_smiles_language.set_smiles_transforms(
        augment=False,
        canonical=params.get(
            "test_smiles_canonical", params.get("augment_smiles", False)
        ),
    )

    # Prepare FP processing
    if params.get("model_fn", "mca") == "dense":
        # NOTE: Might not work out of the box with pytoda >0.1.1
        morgan_transform = Compose(
            [
                SMILESToMorganFingerprints(
                    radius=params.get("fp_radius", 2),
                    bits=params.get("num_drug_features", 512),
                    chirality=params.get("fp_chirality", True),
                ),
                ToTensor(get_device()),
            ]
        )

        def smiles_tensor_batch_to_fp(smiles):
            """To abuse SMILES dataset for FP usage"""
            out = torch.Tensor(smiles.shape[0], params.get("num_drug_features", 256))
            for ind, tensor in enumerate(smiles):
                smiles = smiles_language.token_indexes_to_smiles(tensor.tolist())
                out[ind, :] = torch.squeeze(morgan_transform(smiles))
            return out

    # Assemble datasets
    smiles_dataset = SMILESTokenizerDataset(
        smi_filepath, smiles_language=smiles_language
    )

    # include arg label_columns if data file has any unwanted columns (such as index) to be ignored.
    train_dataset = AnnotatedDataset(
        annotations_filepath=train_scores_filepath, dataset=smiles_dataset
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params.get("num_workers", 0),
    )

    if params.get("uncertainty", True) and params.get("augment_test_data", False):
        raise ValueError(
            "Epistemic uncertainty evaluation not supported if augmentation "
            "is not enabled for test data."
        )

    # Generally, if sanitize is True molecules are de-kekulized. Augmentation
    # preserves the "kekulization", so if it is used, test data should be
    # sanitized or canonicalized.
    smiles_test_dataset = SMILESTokenizerDataset(
        smi_filepath, smiles_language=test_smiles_language
    )

    logger.info("storing languages")
    os.makedirs(os.path.join(model_dir, "smiles_language"), exist_ok=True)
    smiles_language.save_pretrained(os.path.join(model_dir, "smiles_language"))

    logger.info(
        f"Language: {smiles_language.transform_smiles} and {smiles_language.transform_encoding}"
    )
    logger.info(
        f"Test language: {test_smiles_language.transform_smiles} and {test_smiles_language.transform_encoding}"
    )

    # include arg label_columns if data file has any unwanted columns (such as index) to be ignored.
    test_dataset = AnnotatedDataset(
        annotations_filepath=test_scores_filepath, dataset=smiles_test_dataset
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=params.get("num_workers", 0),
    )

    if params.get("confidence", False):
        conf_smiles_language = deepcopy(test_smiles_language)
        conf_smiles_language.set_smiles_transforms(
            augment=True,  # natively true
            canonical=False,
            kekulize=params.get("kekulize", False),
            all_bonds_explicit=params.get("bonds_explicit", False),
            all_hs_explicit=params.get("all_hs_explicit", False),
            remove_bonddir=params.get("remove_bonddir", False),
            remove_chirality=params.get("remove_chirality", False),
            selfies=params.get("selfies", False),
            sanitize=params.get("sanitize", False),
        )
        smiles_conf_dataset = SMILESTokenizerDataset(
            smi_filepath, smiles_language=conf_smiles_language
        )
        conf_dataset = AnnotatedDataset(
            annotations_filepath=test_scores_filepath, dataset=smiles_conf_dataset
        )
        conf_loader = torch.utils.data.DataLoader(
            dataset=conf_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=params.get("num_workers", 0),
        )

    if not params.get("embedding", "learned") == "pretrained":
        params.update({"smiles_vocabulary_size": smiles_language.number_of_tokens})

    device = get_device()
    logger.info(f"Device is {device}")

    model = MODEL_FACTORY[params.get("model_fn", "mca")](params).to(device)
    logger.info(model)
    logger.info(model.loss_fn.class_weights)

    logger.info("Parameters follow")
    for name, param in model.named_parameters():
        logger.info((name, param.shape))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({"number_of_parameters": num_params})
    logger.info(f"Number of parameters {num_params}")

    # Define optimizer
    optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "adam")](
        model.parameters(), lr=params.get("lr", 0.00001)
    )
    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp)

    # Start training
    logger.info("Training about to start...\n")
    t = time()
    save_top_model = os.path.join(model_path, "weights/{}_{}.pt")
    min_loss = 1000000

    # Set up the performance logger
    task = "regression" if "cross" not in params["loss_fn"] else "binary_classification"
    performer = PerformanceLogger(
        model_path=model_dir,
        task=task,
        epochs=params["epochs"],
        train_batches=len(train_loader),
        test_batches=len(test_loader),
    )

    for epoch in range(params["epochs"]):

        performer.epoch += 1
        model.train()
        logger.info(params_filepath.split("/")[-1])
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, y) in enumerate(train_loader):
            smiles = torch.squeeze(smiles.to(device))
            # Transform smiles to FP if needed
            if params.get("model_fn", "mca") == "dense":
                smiles = smiles_tensor_batch_to_fp(smiles).to(device)

            y_hat, pred_dict = model(smiles)

            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
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
                if params.get("model_fn", "mca") == "dense":
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

        # performance.update
        best = performer.report(labels, predictions, test_loss, model)

        if best and params.get("confidence", False):
            # Compute uncertainity estimates and save them
            epistemic_conf = monte_carlo_dropout(
                model, regime="loader", loader=conf_loader
            )
            aleatoric_conf = test_time_augmentation(
                model, regime="loader", loader=conf_loader
            )
            np.save(
                os.path.join(model_dir, "results", f"{best}_epistemic_conf.npy"),
                epistemic_conf,
            )
            np.save(
                os.path.join(model_dir, "results", f"{best}_aleatoric_conf.npy"),
                aleatoric_conf,
            )

        if (epoch + 1) % params.get("save_model", 100) == 0:
            performer.save(model, "epoch", str(epoch))

    performer.final_report()
    performer.save(model, "training", "done")
    logger.info("Done with training, models saved, shutting down.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.train,
        args.test,
        args.smi,
        args.language,
        args.model,
        args.params,
        args.name,
        args.embedding,
    )
