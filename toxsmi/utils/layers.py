from typing import Callable

import torch
import torch.nn as nn
from paccmann_predictor.utils.utils import get_device

DEVICE = get_device()


class EnsembleLayer(nn.Module):
    """
    Following Lee at al (2015) we implement probability and score averaging
    model ensembles.
    """

    def __init__(
        self,
        typ: str,
        input_size: int,
        output_size: int,
        ensemble_size: int = 5,
        fn: Callable = nn.ReLU(),
    ):
        """
        Args:
            typ: from {'prob', 'score'} depending on whether the
                ensemble includes the activation function ('prob').
            input_size: amount of input neurons
            output_size: amount of output neurons (# tasks/classes)
            ensemble_size: amount of parallel ensemble learners
            act_fn: activation function used

        """
        super(EnsembleLayer, self).__init__()

        self.type = typ
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.act_fn = fn

        if typ == "prob":
            self.ensemble = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(input_size, output_size), fn).to(DEVICE)
                    for _ in range(ensemble_size)
                ]
            )

        elif typ == "score":
            self.ensemble = nn.ModuleList(
                [
                    nn.Linear(input_size, output_size).to(DEVICE)
                    for _ in range(ensemble_size)
                ]
            )

        else:
            raise NotImplementedError("Choose type from {'score', 'prob'}")

    def forward(self, x):
        """Run forward pass through model ensemble

        Arguments:
            x {torch.Tensor} -- shape: batch_size x input_size

        Returns:
            torch.Tensor -- shape: batch_size x output_size
        """

        dist = [e(x) for e in self.ensemble]
        output = torch.mean(torch.stack(dist), dim=0)
        if self.type == "score":
            output = self.act_fn(output)

        return output
