import pickle
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from brc_pytorch.layers import (
    BistableRecurrentCell,
    MultiLayerBase,
    NeuromodulatedBistableRecurrentCell,
)
from paccmann_predictor.utils.hyperparams import ACTIVATION_FN_FACTORY
from paccmann_predictor.utils.layers import dense_layer
from paccmann_predictor.utils.utils import get_device

from toxsmi.utils.hyperparams import LOSS_FN_FACTORY
from toxsmi.utils.layers import EnsembleLayer


class RNN(nn.Module):
    """
    Class for several flavors of RNN models for (multilabel) multiclass classification
    """

    rnn_cell_factory = {
        "RNN": nn.RNNCell,
        "GRU": nn.GRUCell,
        "LSTM": nn.LSTMCell,
        "BRC": BistableRecurrentCell,
        "nBRC": NeuromodulatedBistableRecurrentCell,
    }

    def __init__(self, params: dict, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense encoder.

        Obligatory items in params:
            rnn_cell_type (str): One of ['Vanilla', 'LSTM', 'GRU', 'BRC', 'nBRC'].
            smiles_vocabulary_size (int): Size of SMILES vocabulary.
            smiles_embedding_size (int): Size of embedding space for SMILES (ignored
                if embedding == 'learned').

            All other parameters are optional and have defaults.
        """
        super(RNN, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.num_tasks = params.get("num_tasks", 12)

        # RNN Parameters
        self.bidirectional = params.get("bidirectional", False)
        self.num_directions = 2 if self.bidirectional else 1
        self.n_layers = params.get("n_layers", 2)
        self.rnn_cell_size = params.get("rnn_cell_size", 64)
        self.rnn_cell_type = params["rnn_cell_type"]

        if params["rnn_cell_type"] not in self.rnn_cell_factory.keys():
            raise ValueError(
                f"RNN Cell type {self.rnn_cell_type} not supported"
                f"Choose from {self.rnn_cell_factory.keys()}"
            )
        self.rnn_cell = self.rnn_cell_factory[self.rnn_cell_type]

        self.dropout = params.get("dropout", 0.5)
        self.use_batch_norm = self.params.get("batch_norm", True)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get("activation_fn", "relu")]

        # Build the model
        # First the embeddings
        if params.get("embedding", "learned") == "learned":

            self.smiles_embedding = nn.Embedding(
                self.params["smiles_vocabulary_size"],
                self.params["smiles_embedding_size"],
                scale_grad_by_freq=params.get("embed_scale_grad", False),
            )
        elif params.get("embedding", "learned") == "one_hot":
            self.smiles_embedding = nn.Embedding(
                self.params["smiles_vocabulary_size"],
                self.params["smiles_vocabulary_size"],
            )
            # Plug in one hot-vectors and freeze weights
            self.smiles_embedding.load_state_dict(
                {
                    "weight": torch.nn.functional.one_hot(
                        torch.arange(self.params["smiles_vocabulary_size"])
                    )
                }
            )
            self.smiles_embedding.weight.requires_grad = False

        elif params.get("embedding", "learned") == "pretrained":
            # Load the pretrained embeddings
            try:
                with open(params["embedding_path"], "rb") as f:
                    embeddings = pickle.load(f)
            except KeyError:
                raise KeyError("Path for embeddings is missing in params.")

            # Plug into layer
            self.smiles_embedding = nn.Embedding(
                embeddings.shape[0], embeddings.shape[1]
            )
            self.smiles_embedding.load_state_dict({"weight": torch.Tensor(embeddings)})
            if params.get("fix_embeddings", True):
                self.smiles_embedding.weight.requires_grad = False

        else:
            raise ValueError(f"Unknown embedding type: {params['embedding']}")

        self.smiles_embedding_size = self.smiles_embedding.weight.shape[1]
        # Input sizes need to be doubled in case of bidirectional network
        self.input_sizes = [self.smiles_embedding_size] + [
            self.rnn_cell_size * self.num_directions
        ] * (self.n_layers - 1)

        self.hidden_sizes = [self.num_directions * self.rnn_cell_size] + params.get(
            "stacked_hidden_sizes", [1024, 512]
        )

        # Set up RNN
        self.recurrent_layers = [
            self.rnn_cell(input_size, self.rnn_cell_size)
            for input_size in self.input_sizes
        ]

        self.rnn = MultiLayerBase(
            self.rnn_cell_type,
            self.recurrent_layers,
            self.rnn_cell_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            return_sequences=False,
            device=self.device,
        )

        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense_{}".format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        if params.get("ensemble", "None") not in ["score", "prob", "None"]:
            raise NotImplementedError(
                "Choose ensemble type from ['score', 'prob', 'None']"
            )

        self.final_dense = EnsembleLayer(
            typ=params.get("ensemble", "score"),
            input_size=self.hidden_sizes[-1],
            output_size=self.num_tasks,
            ensemble_size=params.get("ensemble_size", 5),
            fn=ACTIVATION_FN_FACTORY["sigmoid"],
        )

        self.loss_fn = LOSS_FN_FACTORY[
            params.get("loss_fn", "binary_cross_entropy_ignore_nan_and_sum")
        ]
        # Set class weights manually
        if "binary_cross_entropy_ignore_nan" in params.get(
            "loss_fn", "binary_cross_entropy_ignore_nan_and_sum"
        ):
            self.loss_fn.class_weights = params.get("class_weights", [1, 1])

    def forward(self, smiles: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the RNN.

        Args:
            smiles (torch.Tensor): type int and shape: [batch_size, seq_length]

        Returns:
            (torch.Tensor, dict): predictions, prediction_dict

            predictions are of shape `[bs, num_tasks]`.
        """
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # SMILES RNN pass. Output is num_directions x batch_size x rnn_cell_size
        encodings = self.rnn(embedded_smiles)

        if self.rnn_cell_type in ["LSTM"]:
            # Discard cell state
            encodings = encodings[0]
        encodings = encodings.permute(1, 2, 0).reshape(
            -1, self.rnn_cell_size * self.num_directions
        )
        encodings = self.dense_layers(encodings)
        predictions = self.final_dense(encodings)
        prediction_dict = {"predictions": predictions}
        return predictions, prediction_dict

    def loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(yhat, y)

    def load(self, path: str, *args, **kwargs) -> None:
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path: str, *args, **kwargs) -> None:
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
