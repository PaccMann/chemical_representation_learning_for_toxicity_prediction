import pickle
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from paccmann_predictor.utils.layers import (
    alpha_projection,
    convolutional_layer,
    dense_layer,
    smiles_projection,
)
from paccmann_predictor.utils.utils import get_device
from toxsmi.utils.hyperparams import ACTIVATION_FN_FACTORY, LOSS_FN_FACTORY
from toxsmi.utils.layers import EnsembleLayer


class MCAMultiTask(nn.Module):
    """
    Multiscale Convolutional Attentive Encoder.
    This is the MCA model similiar to the one presented in publication in
    Molecular Pharmaceutics https://arxiv.org/abs/1904.11223.
    Differences:
        - uses self instead of context attention since input is unimodal.
        - MultiLabel classification implementation (sigmoidal in last layer)
    """

    def __init__(self, params: dict, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense encoder.

        Items in params:
            smiles_embedding_size (int): dimension of tokens' embedding.
            smiles_vocabulary_size (int): size of the tokens vocabulary.
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Defaults to 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Defaults to True.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Defaults to 0.5.
            filters (list[int], optional): Numbers of filters to learn per
                convolutional layer. Defaults to [64, 64, 64].
            kernel_sizes (list[list[int]], optional): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['smiles_embedding_size']],
                    [5, params['smiles_embedding_size']],
                    [11, params['smiles_embedding_size']]
                ]
                NOTE: The kernel sizes should match the dimensionality of the
                smiles_embedding_size, so if the latter is 8, the images are
                t x 8, then treat the 8 embedding dimensions like channels
                in an RGB image.
            multiheads (list[int], optional): Amount of attentive multiheads
                per SMILES embedding. Should have len(filters)+1.
                Defaults to [4, 4, 4, 4].
            stacked_dense_hidden_sizes (list[int], optional): Sizes of the
                hidden dense layers. Defaults to [1024, 512].
            smiles_attention_size (int, optional): size of the attentive layer
                for the smiles sequence. Defaults to 64.

        Example params:
        ```
        {
            "smiles_attention_size": 8,
            "smiles_vocabulary_size": 28,
            "smiles_embedding_size": 8,
            "filters": [128, 128],
            "kernel_sizes": [[3, 8], [5, 8]],
            "multiheads":[4, 4, 4]
            "stacked_dense_hidden_sizes": [1024, 512]
        }
        ```
        """
        super(MCAMultiTask, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.num_tasks = params.get("num_tasks", 12)
        self.smiles_attention_size = params.get("smiles_attention_size", 64)

        # Model architecture (hyperparameter)
        self.multiheads = params.get("multiheads", [4, 4, 4, 4])
        self.filters = params.get("filters", [64, 64, 64])

        self.dropout = params.get("dropout", 0.5)
        self.use_batch_norm = self.params.get("batch_norm", True)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get("activation_fn", "relu")]

        # Build the model. First the embeddings
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

        self.kernel_sizes = params.get(
            "kernel_sizes",
            [
                [3, self.smiles_embedding.weight.shape[1]],
                [5, self.smiles_embedding.weight.shape[1]],
                [11, self.smiles_embedding.weight.shape[1]],
            ],
        )

        self.hidden_sizes = [
            self.multiheads[0] * self.smiles_embedding.weight.shape[1]
            + sum([h * f for h, f in zip(self.multiheads[1:], self.filters)])
        ] + params.get("stacked_hidden_sizes", [1024, 512])

        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError("Length of filter and kernel size lists do not match.")
        if len(self.filters) + 1 != len(self.multiheads):
            raise ValueError("Length of filter and multihead lists do not match")

        self.convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"convolutional_{index}",
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.filters, self.kernel_sizes)
                    )
                ]
            )
        )

        smiles_hidden_sizes = [self.smiles_embedding.weight.shape[1]] + self.filters
        self.smiles_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"smiles_projection_{self.multiheads[0]*layer+index}",
                        smiles_projection(
                            smiles_hidden_sizes[layer], self.smiles_attention_size
                        ),
                    )
                    for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )
        self.alpha_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"alpha_projection_{self.multiheads[0]*layer+index}",
                        alpha_projection(self.smiles_attention_size),
                    )
                    for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])

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
        if params.get("ensemble", "None") == "None":
            params["ensemble_size"] = 1

        self.loss_name = params.get(
            "loss_fn", "binary_cross_entropy_ignore_nan_and_sum"
        )

        final_activation = (
            ACTIVATION_FN_FACTORY["sigmoid"]
            if "cross" in self.loss_name
            else ACTIVATION_FN_FACTORY["none"]
        )
        self.final_dense = EnsembleLayer(
            typ=params.get("ensemble", "score"),
            input_size=self.hidden_sizes[-1],
            output_size=self.num_tasks,
            ensemble_size=params.get("ensemble_size", 5),
            fn=final_activation,
        )

        self.loss_fn = LOSS_FN_FACTORY[self.loss_name]
        # Set class weights manually
        if "binary_cross_entropy_ignore_nan" in params.get(
            "loss_fn", "binary_cross_entropy_ignore_nan_and_sum"
        ):
            self.loss_fn.class_weights = params.get("class_weights", [1, 1])

        self.to(self.device)

    def forward(self, smiles: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the MCA.

        Args:
            smiles (torch.Tensor): type int and shape: [batch_size, seq_length]

        Returns:
            (torch.Tensor, dict): predictions, prediction_dict

            predictions are toxicity predictions of shape `[bs, num_tasks]`.
            prediction_dict includes the prediction and attention weights.
        """

        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # SMILES Convolutions. Unsqueeze has shape batch_size x 1 x T x H.
        encoded_smiles = [embedded_smiles] + [
            self.convolutional_layers[ind](torch.unsqueeze(embedded_smiles, 1)).permute(
                0, 2, 1
            )
            for ind in range(len(self.convolutional_layers))
        ]

        # NOTE: SMILES Self Attention mechanism (see )
        smiles_alphas, encodings = [], []
        for layer in range(len(self.multiheads)):
            for head in range(self.multiheads[layer]):
                ind = self.multiheads[0] * layer + head
                smiles_alphas.append(
                    self.alpha_projections[ind](
                        self.smiles_projections[ind](encoded_smiles[layer])
                    )
                )
                # Sequence is always reduced.
                encodings.append(
                    torch.sum(
                        encoded_smiles[layer] * torch.unsqueeze(smiles_alphas[-1], -1),
                        1,
                    )
                )
        encodings = torch.cat(encodings, dim=1)

        # Apply batch normalization if specified
        if self.use_batch_norm:
            encodings = self.batch_norm(encodings)
        for dl in self.dense_layers:
            encodings = dl(encodings)

        predictions = self.final_dense(encodings)
        prediction_dict = {
            "smiles_attention": smiles_alphas,
            "toxicities": predictions,
            "encodings": encodings,
        }
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
