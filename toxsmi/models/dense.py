import torch
import torch.nn as nn
from paccmann_predictor.utils.hyperparams import ACTIVATION_FN_FACTORY
from paccmann_predictor.utils.layers import dense_layer
from paccmann_predictor.utils.utils import get_device

from toxsmi.utils.hyperparams import LOSS_FN_FACTORY
from toxsmi.utils.layers import EnsembleLayer


class Dense(nn.Module):
    """This is a Dense model for validation. To be trained on fingerprints"""

    def __init__(self, params, *args, **kwargs):
        """Constructor.
        Args:
            params (dict): A dictionary containing the parameter to built the
                dense Decoder.
        Items in params:
            dense_sizes (list[int]): Number of neurons in the hidden layers.
            num_drug_features (int, optional): Number of features for molecule.
                Defaults to 512 (bits fingerprint).
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Defaults to 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Defaults to False.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Defaults to 0.0.
            *args, **kwargs: positional and keyword arguments are ignored.
        """
        super(Dense, self).__init__(*args, **kwargs)

        self.device = get_device()
        self.params = params
        self.num_drug_features = params.get("num_drug_features", 512)
        self.num_tasks = params.get("num_tasks", 12)
        self.hidden_sizes = params.get(
            "stacked_dense_hidden_sizes", [self.num_drug_features, 5000, 1000, 500]
        )
        self.dropout = params.get("dropout", 0.0)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get("activation_fn", "relu")]
        self.dense_layers = nn.ModuleList(
            [
                dense_layer(
                    self.hidden_sizes[ind],
                    self.hidden_sizes[ind + 1],
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=self.params.get("batch_norm", True),
                ).to(self.device)
                for ind in range(len(self.hidden_sizes) - 1)
            ]
        )

        self.final_dense = EnsembleLayer(
            typ=params.get("ensemble", "score"),
            input_size=self.hidden_sizes[-1],
            output_size=self.num_tasks,
            ensemble_size=params.get("ensemble_size", 5),
            fn=ACTIVATION_FN_FACTORY["sigmoid"],
        ).to(self.device)
        self.loss_fn = LOSS_FN_FACTORY[
            params.get("loss_fn", "binary_cross_entropy_ignore_nan_and_sum")
        ]

    def forward(self, x):
        """Forward pass through the dense model.

        Args:
            x (torch.Tensor) of type int and shape `[batch_size, 512 (bits).
        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions are toxicity predictions of shape `[bs, num_tasks]`.
        """
        inputs = x.float()

        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
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
