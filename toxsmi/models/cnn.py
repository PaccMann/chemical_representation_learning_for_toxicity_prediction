import torch
import torch.nn as nn
from paccmann_predictor.utils.hyperparams import ACTIVATION_FN_FACTORY
from paccmann_predictor.utils.layers import convolutional_layer
from paccmann_predictor.utils.utils import get_device

from toxsmi.utils.hyperparams import LOSS_FN_FACTORY


class CNN(nn.Module):
    """This is a simple model for stacked convolutions on SMILES"""

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense Decoder.
                TODO params should become actual arguments (use **params).
        Items in params:
            filters (list[int], optional): Numbers of filters to learn per
                convolutional layer.
            kernel_sizes (list[list[int]], optional): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['smiles_embedding_size']],
                    [5, params['smiles_embedding_size']],
                    [11, params['smiles_embedding_size']]
                ]
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Defaults to 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Defaults to False.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Defaults to 0.0.
            *args, **kwargs: positional and keyword arguments are ignored.
        """

        super(CNN, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get("loss_fn", "cnn")]

        self.kernel_sizes = params.get(
            "kernel_sizes",
            [
                [3, params["smiles_embedding_size"]],
                [5, params["smiles_embedding_size"]],
                [11, params["smiles_embedding_size"]],
            ],
        )

        self.num_filters = [1] + params.get("num_filters", [10, 20, 50])

        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError("Length of filter and kernel size lists do not match.")

        self.smiles_embedding = nn.Embedding(
            self.params["smiles_vocabulary_size"],
            self.params["smiles_embedding_size"],
            scale_grad_by_freq=params.get("embed_scale_grad", False),
        )

        self.dropout = params.get("dropout", 0.0)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get("activation_fn", "relu")]

        self.conv_layers = [
            convolutional_layer(
                self.num_filters[layer],
                self.num_filters[layer + 1],
                self.kernel_sizes[layer],
            )
            for layer in range(len(self.channel_inputs) - 1)
        ]

        self.final_dense = nn.Linear(
            (self.num_filters[-1] * self.params["smiles_embedding_size"]),
            self.num_tasks,
        )
        self.final_act_fn = ACTIVATION_FN_FACTORY["sigmoid"]

    def forward(self, x):
        """Forward pass through the cnn model.
        Args:
            x (torch.Tensor) of type int and shape `[batch_size, seq_length].
        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict
            predictions are toxicity predictions of shape `[bs, num_tasks]`.
        """

        inputs = x.float()

        for conv_layer in self.conv_layers:
            inputs = conv_layer(inputs, self.act_fn)

        predictions = self.final_act_fn(self.final_dense(inputs))
        prediction_dict = {"Toxicity scores": predictions}
        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
