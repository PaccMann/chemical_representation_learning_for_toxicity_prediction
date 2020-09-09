import torch
import torch.nn as nn
from paccmann_predictor.utils.utils import get_device

DEVICE = get_device()


class BCEIgnoreNaN(nn.Module):
    """Wrapper for BCE function that ignores NaNs"""

    def __init__(self, reduction):
        super(BCEIgnoreNaN, self).__init__()
        self.loss = nn.BCELoss(reduction='none')
        self.reduction = reduction
        if reduction != 'sum' and reduction != 'mean':
            raise ValueError('Reduction type is mean or sum')

    def forward(self, yhat, y):
        """
        Arguments:
            y (torch.Tensor): Labels
            yhat (torch.Tensor): Predictions

        Returns:
            torch.Tensor: BCE loss
        """

        # Find position of NaNs, set them to 0 to well-define BCE loss and
        # then filter them out

        nans = ~torch.isnan(y)
        y[y != y] = 0
        out = self.loss(yhat, y) * nans.type(torch.float32).to(DEVICE)

        if self.reduction == 'mean':
            return torch.mean(out)
        elif self.reduction == 'sum':
            return torch.sum(out)
