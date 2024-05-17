#! /usr/bin/env python3
"""Test toxsmi predictor."""
import argparse
import glob
import json
import logging
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from paccmann_predictor.utils.interpret import (
    monte_carlo_dropout,
    test_time_augmentation,
)
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import AnnotatedDataset, SMILESTokenizerDataset
from pytoda.smiles.smiles_language import SMILESTokenizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_curve,
)
from toxsmi.models import MODEL_FACTORY
from toxsmi.utils import disable_rdkit_logging
from toxsmi.utils.performance import PerformanceLogger
from tqdm import tqdm

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", "-model", type=str, help="Path to the trained model"
)
parser.add_argument(
    "--smi_filepath", "-smi", type=str, help="Path to the SMILES data (.smi)"
)
parser.add_argument(
    "--label_filepath", "-labels", type=str, help="Path to the test scores (.csv)"
)
parser.add_argument(
    "--checkpoint_name",
    "-checkpoint",
    type=str,
    help="The name of the checkpoint file. Will pick the first checkpoint that contains the string",
)
parser.add_argument(
    "--confidence",
    "-c",
    default=False,
    action="store_true",
    help="Compute sample-wise confidences",
)
parser.add_argument(
    "--threshold_binarization",
    "-threshold",
    default=3.5,
    type=int,
    help="Only used for regression task, for pseudo-binarizing labels to get classification-alike metrics",
)


def main(
    model_path,
    smi_filepath,
    label_filepath,
    checkpoint_name,
    confidence,
    threshold_binarization,
):

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("eval_toxicity")
    logger.setLevel(logging.INFO)
    disable_rdkit_logging()

    # Process parameter file:
    params = {}
    with open(os.path.join(model_path, "model_params.json"), "r") as fp:
        params.update(json.load(fp))

    # Create model directory
    output_folder = os.path.join(model_path, "results")
    os.makedirs(output_folder, exist_ok=True)

    device = get_device()

    weights_paths = glob.glob(
        os.path.join(model_path, "weights", f"*{checkpoint_name}*")
    )
    if len(weights_paths) == 0:
        raise FileNotFoundError(
            f"No weights called {checkpoint_name} at {os.path.join(model_path, 'weights')}."
        )
    elif len(weights_paths) > 1:
        logger.info(
            f"Multiple paths for checkpoint {checkpoint_name} found: "
            f"{weights_paths}. Will load first one!"
        )
    weights_path = weights_paths[0]
    result_prefix = os.path.join(
        output_folder,
        f"{label_filepath.split('/')[-1].split('.')[0]}_{checkpoint_name}",
    )

    # Restore model
    model = MODEL_FACTORY["mca"](params).to(device)
    try:
        model.load(weights_path, map_location=device)
    except Exception:
        raise ValueError(f"Error in model restoring from {weights_path}")

    model.eval()
    logger.info("Model restored. Model specs & parameters follow")
    for name, param in model.named_parameters():
        logger.debug((name, param.shape))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters = {num_params}")
    smiles_language = SMILESTokenizer(
        vocab_file=os.path.join(model_path, "smiles_language"),
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
    # Natively false for inference
    smiles_language.set_smiles_transforms(augment=False)

    # Assemble datasets
    smiles_dataset = SMILESTokenizerDataset(
        smi_filepath, smiles_language=smiles_language
    )
    num_tasks = params["num_tasks"]
    label_columns = params.get("label_columns", list(range(num_tasks)))
    dataset = AnnotatedDataset(
        annotations_filepath=label_filepath,
        dataset=smiles_dataset,
        label_columns=label_columns,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    if confidence:
        aug_smiles_language = deepcopy(smiles_language)
        aug_smiles_language.set_smiles_transforms(augment=True, canonical=False)
        ale_dataset = SMILESTokenizerDataset(
            smi_filepath, smiles_language=aug_smiles_language
        )
        ale_dataset = AnnotatedDataset(
            annotations_filepath=label_filepath,
            dataset=ale_dataset,
            label_columns=label_columns,
        )
        ale_loader = torch.utils.data.DataLoader(
            dataset=ale_dataset,
            batch_size=params.get("batch_size", 32),
            shuffle=False,
            drop_last=False,
            num_workers=params.get("num_workers", 0),
        )

    task = "regression" if "cross" not in params["loss_fn"] else "binary_classification"
    test_df = pd.read_csv(label_filepath)
    task_names = list(test_df.columns[label_columns])
    performer = PerformanceLogger(
        model_path=model_path,
        task=task,
        epochs=0,
        train_batches=0,
        test_batches=len(loader),
        task_names=task_names,
    )

    # Start evaluation
    logger.info("Evaluation about to start...\n")

    preds, labels, attention_scores, smiles = [], [], [], []
    for x, y in tqdm(loader, total=len(loader)):
        yhat, pred_dict = model(torch.squeeze(x.to(device)))

        preds.append(yhat.detach())
        # Copy y tensor since loss function applies downstream modification
        labels.append(y.clone())
        attention_scores.extend(
            torch.stack(pred_dict["smiles_attention"], dim=1).detach().cpu()
        )
        smiles.extend(
            [smiles_language.token_indexes_to_smiles(smi.tolist()) for smi in x]
        )
    # Scores are now 3D: num_samples x num_att_layers x padding_length
    attention = torch.stack(attention_scores, dim=0).numpy()
    logger.info(f"Shape of attention scores {attention.shape}.")
    # np.save(os.path.join(output_folder, "attention_raw.npy"), attention)
    # np.save(os.path.join(output_folder, "encodings.npy"), encodings)
    attention_avg = np.mean(attention, axis=1)
    att_df = pd.DataFrame(
        data=attention_avg,
        columns=[f"att_idx_{i}" for i in range(attention_avg.shape[1])],
    )

    predictions = torch.cat(preds, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    pred_df = pd.DataFrame(
        data=predictions,
        columns=[f"pred_{i}" for i in range(num_tasks)],
    )
    lab_df = pd.DataFrame(
        data=labels,
        columns=[f"label_{i}" for i in range(num_tasks)],
    )
    df = pd.concat([pred_df, lab_df], axis=1)

    if confidence:
        # Compute uncertainity estimates and save them
        epistemic_conf, epistemic_pred = monte_carlo_dropout(
            model, regime="loader", loader=loader
        )
        aleatoric_conf, aleatoric_pred = test_time_augmentation(
            model, regime="loader", loader=ale_loader
        )
        epi_conf_df = pd.DataFrame(
            data=epistemic_conf.numpy(),
            columns=[f"epistemic_conf_{i}" for i in range(epistemic_conf.shape[1])],
        )
        ale_conf_df = pd.DataFrame(
            data=aleatoric_conf.numpy(),
            columns=[f"aleatoric_conf_{i}" for i in range(aleatoric_conf.shape[1])],
        )
        epi_pred_df = pd.DataFrame(
            data=epistemic_pred.numpy(),
            columns=[f"epistemic_pred_{i}" for i in range(epistemic_pred.shape[1])],
        )
        ale_pred_df = pd.DataFrame(
            data=aleatoric_pred.numpy(),
            columns=[f"aleatoric_pred_{i}" for i in range(aleatoric_pred.shape[1])],
        )
        df = pd.concat([df, epi_conf_df, ale_conf_df, epi_pred_df, ale_pred_df], axis=1)
    # df = pd.concat([df, att_df], axis=1)
    # Add ligand/task information
    df.insert(0, "SMILES", smiles)
    df.to_csv(result_prefix + "_results.csv", index=False)

    flat_df = pd.DataFrame(
        {
            "MoleculeID": list(np.repeat(test_df.mol_id, num_tasks)),
            "SMILES": list(np.repeat(smiles, num_tasks)),
            "Task": task_names * len(smiles),
            "Prediction": predictions.flatten(),
            "Label": labels.flatten(),
        }
    )
    flat_df.to_csv(result_prefix + "_results_flat.csv", index=False)
    result = performer.inference_report(labels, predictions)

    with open(result_prefix + "_metrics.json", "w") as f:
        json.dump(result, f, indent=4)

    if task == "regression":
        logger.info(
            f"Binarizing regression task with threshold {threshold_binarization}"
        )
        class_result = performer.inference_report_binarized_regression(
            labels, predictions, threshold=threshold_binarization
        )
        with open(result_prefix + "_classification_metrics.json", "w") as f:
            json.dump(class_result, f, indent=4)

    logger.info("Done, shutting down.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.model_path,
        args.smi_filepath,
        args.label_filepath,
        args.checkpoint_name,
        args.confidence,
        args.threshold_binarization,
    )
