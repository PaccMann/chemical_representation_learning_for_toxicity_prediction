import logging
from typing import Iterable

import torch
import torch.nn as nn
from paccmann_predictor.utils.utils import get_device

logger = logging.getLogger(__name__)

DEVICE = get_device()


class BCEIgnoreNaN(nn.Module):
    """Wrapper for BCE function that ignores NaNs"""

    def __init__(self, reduction: str, class_weights: tuple = (1, 1)) -> None:
        """

        Args:
            reduction (str): Reduction applied in loss function. Either sum or mean.
            class_weights (tuple, optional): Class weights for loss function.
                Defaults to (1, 1), i.e. equal class weighhts.
        """
        super(BCEIgnoreNaN, self).__init__()
        self.loss = nn.BCELoss(reduction="none")

        if reduction != "sum" and reduction != "mean":
            raise ValueError(f"Chose reduction type as mean or sum, not {reduction}")
        self.reduction = reduction

        if not isinstance(class_weights, Iterable):
            raise TypeError(f"Pass iterable for weights, not: {type(class_weights)}")
        if not len(class_weights) == 2:
            raise ValueError(f"Class weight len should be 2, not: {len(class_weights)}")
        if not all(w > 0 for w in class_weights):
            raise ValueError(f"All weigths should be positive not: {class_weights}")

        self.class_weights = class_weights
        logger.info(f"Class weights are {class_weights}.")

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            y (torch.Tensor): Labels (1D or 2D).
            yhat (torch.Tensor): Predictions (1D or 2D).

        NOTE: This function has side effects (in-place modification of labels).
        This is needed since deepcopying the tensor destroys gradient flow. Just be
        aware that repeated calls of this function with the same variables lead to
        different results if and only if at least one nan is in y.

        Returns:
            torch.Tensor: BCE loss
        """

        # Find position of NaNs, set them to 0 to well-define BCE loss and
        # then filter them out
        nans = ~torch.isnan(y)
        y[y != y] = 0
        loss = self.loss(yhat, y) * nans.type(torch.float32).to(DEVICE)

        # Set a tensor with class weights, equal shape to labels.
        # NaNs are 0 in y now, but since loss is 0, downstream calc is unaffected.
        weight_tensor = torch.ones(y.shape)
        weight_tensor[y == 0.0] = self.class_weights[0]
        weight_tensor[y == 1.0] = self.class_weights[1]

        out = loss * weight_tensor.to(DEVICE)

        if self.reduction == "mean":
            return torch.mean(out)
        elif self.reduction == "sum":
            return torch.sum(out)


NAN_LOSS_FACTORY = {
    "mse": nn.MSELoss(reduction="none"),
    "l1": nn.L1Loss(reduction="none"),
    "binary_cross_entropy": nn.BCELoss(reduction="none"),
}


class GenericIgnoreNaN(nn.Module):
    """Wrapper for MSE loss function that ignores NaNs"""

    def __init__(
        self, loss: str, reduction: str, class_weights: tuple = (1, 1)
    ) -> None:
        """

        Args:
            reduction (str): Reduction applied in loss function. Either sum or mean.
            class_weights (tuple, optional): Class weights for loss function.
                Defaults to (1, 1), i.e. equal class weighhts.
        """
        super(GenericIgnoreNaN, self).__init__()
        if loss not in NAN_LOSS_FACTORY.keys():
            raise ValueError(f"Unknown loss function {loss}")
        self.loss = NAN_LOSS_FACTORY[loss]

        if reduction != "sum" and reduction != "mean":
            raise ValueError(f"Chose reduction type as mean or sum, not {reduction}")
        self.reduction = reduction

        self.regression = loss != "binary_cross_entropy"

        if self.regression:
            # No class weights needed
            logger.info(f"Class weights {class_weights} are ignored.")
            return
        if not isinstance(class_weights, Iterable):
            raise TypeError(f"Pass iterable for weights, not: {type(class_weights)}")
        if not len(class_weights) == 2:
            raise ValueError(f"Class weight len should be 2, not: {len(class_weights)}")
        if not all(w > 0 for w in class_weights):
            raise ValueError(f"All weigths should be positive not: {class_weights}")

        self.class_weights = class_weights
        logger.info(f"Class weights are {class_weights}.")

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            y (torch.Tensor): Labels (1D or 2D).
            yhat (torch.Tensor): Predictions (1D or 2D).

        NOTE: This function has side effects (in-place modification of labels).
        This is needed since deepcopying the tensor destroys gradient flow. Just be
        aware that repeated calls of this function with the same variables lead to
        different results if and only if at least one nan is in y.

        Returns:
            torch.Tensor: BCE loss
        """

        # Find position of NaNs, set them to 0 to well-define BCE loss and
        # then filter them out
        nans = ~torch.isnan(y)
        y[y != y] = 0
        loss = self.loss(yhat, y) * nans.type(torch.float32).to(DEVICE)

        if self.regression:
            out = loss
        else:
            # Set a tensor with class weights, equal shape to labels.
            # NaNs are 0 in y now, but since loss is 0, downstream calc is unaffected.
            weight_tensor = torch.ones(y.shape)
            weight_tensor[y == 0.0] = self.class_weights[0]
            weight_tensor[y == 1.0] = self.class_weights[1]
            out = loss * weight_tensor.to(DEVICE)

        if self.reduction == "mean":
            return torch.mean(out)
        elif self.reduction == "sum":
            return torch.sum(out)
