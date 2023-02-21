""""Utils for model finetuning"""
import logging
from collections import OrderedDict

import torch.nn as nn
from paccmann_predictor.utils.layers import dense_layer
from paccmann_predictor.utils.utils import get_device

from toxsmi.models import MCAMultiTask
from toxsmi.utils.hyperparams import ACTIVATION_FN_FACTORY
from toxsmi.utils.layers import EnsembleLayer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def update_mca_model(model: MCAMultiTask, params: dict) -> MCAMultiTask:
    """
    Receives a pretrained model (instance of MCAMultiTask), modifies it and returns
    the updated object

    Args:
        model (MCAMultiTask): Pretrained model to be modified.
        params (dict): Hyperparameter file for the modifications. Needs to include:
            - number_of_tunable_layers (how many layers should not be frozen. If
                number exceeds number of existing layers, all layers are tuned.)
            - fresh_dense_sizes (a list of fresh dense layers to be plugged in at
                the end).
            - num_tasks (number of classfication tasks being performed).

    Returns:
        MCAMultiTask: Modified model for finetune
    """

    if not isinstance(model, MCAMultiTask):
        raise TypeError(f"Wrong model type, was {type(model)}, not MCAMultiTask.")

    # Freeze the correct layers and add new ones
    # Not strictly speaking all layers, but all param matrices, gradient-req or not.
    num_layers = len(["" for p in model.parameters()])
    num_to_tune = params["number_of_tunable_layers"]
    if num_to_tune > num_layers:
        logger.warning(
            f"Model has {num_layers} tunable layers. Given # is larger: {num_to_tune}."
        )
        num_to_tune = num_layers
    fresh_sizes = params["fresh_dense_sizes"]
    logger.info(
        f"Model has {num_layers} layers. {num_to_tune} will be finetuned, "
        f"{len(fresh_sizes)} fresh ones will be added (sizes: {fresh_sizes})."
    )
    # Count the ensemble layers (will be replaced anyways)
    num_ensemble_layers = len(
        list(filter(lambda tpl: "ensemble" in tpl[0], model.named_parameters()))
    )
    # Freeze the right layers
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx < num_layers - num_to_tune - num_ensemble_layers:
            param.requires_grad = False

    # Add more dense layers
    fresh_sizes.insert(0, model.hidden_sizes[-1])
    model.dense_layers = nn.Sequential(
        model.dense_layers,
        nn.Sequential(
            OrderedDict(
                [
                    (
                        "fresh_dense_{}".format(ind),
                        dense_layer(
                            fresh_sizes[ind],
                            fresh_sizes[ind + 1],
                            act_fn=ACTIVATION_FN_FACTORY[
                                params.get("activation_fn", "relu")
                            ],
                            dropout=params.get("dropout", 0.5),
                            batch_norm=params.get("batch_norm", True),
                        ).to(get_device()),
                    )
                    for ind in range(len(fresh_sizes) - 1)
                ]
            )
        ),
    )

    # Replace final layer
    model.num_tasks = params["num_tasks"]
    loss = params.get("loss_fn", "binary_cross_entropy_ignore_nan_and_sum")
    final_activation = (
        ACTIVATION_FN_FACTORY["sigmoid"]
        if "cross" in loss
        else ACTIVATION_FN_FACTORY["none"]
    )

    model.final_dense = EnsembleLayer(
        typ=params.get("ensemble", "score"),
        input_size=fresh_sizes[-1],
        output_size=model.num_tasks,
        ensemble_size=params.get("ensemble_size", 5),
        fn=final_activation,
    )

    return model
