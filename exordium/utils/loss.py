from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from exordium import PathType


def kde(samples: np.ndarray = np.random.normal(0.5, 0.15, 6000).clip(0,1),
        min_value: float | None = 0.,
        max_value: float | None = 1.,
        number_of_steps: int = 1000,
        fig_path: PathType | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Kernel Density Estimation.

    Args:
        samples (np.ndarray | None, optional): input samples of shape (N,). Defaults to np.random.normal(0.5, 0.15, 6000).clip(0,1).
        number_of_steps (int, optional): number of steps. Defaults to 1000.
        fig_path (str | None, optional): path to the output figure. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: the x, y values of kde.
    """
    density = gaussian_kde(samples)
    density.covariance_factor = lambda : .5
    density._compute_covariance()
    min_value = min_value or samples.min()
    max_value = max_value or samples.max()
    x_vals = np.linspace(min_value, max_value, number_of_steps)
    y_vals = density(x_vals)

    if fig_path:
        plt.hist(samples, bins=number_of_steps, density=True)
        plt.plot(x_vals, y_vals)
        plt.savefig(str(fig_path))

    return x_vals, y_vals


class InvKDEWeightedLoss():

    def __init__(self, gt: np.ndarray,
                       gt_min: float | None = 0.,
                       gt_max: float | None = 1.,
                       weight_min: float = 0.2,
                       weight_max: float = 1.,
                       number_of_steps: int = 1000,
                       reduction: str = 'mean'):
        """Losses weighted with inverted kernel density estimation of ground truth.

        Args:
            gt (np.ndarray): ground truth of shape (N, M).
            gt_min (float | None, optional): minimum value of the ground truth target-wise. None means it is calculated from gt. Defaults to 0.
            gt_max (float | None, optional): maximum value of the ground truth target-wise. None means it is calculated from gt. Defaults to 1.
            weight_min (float, optional): minimum weight value. Defaults to 0.2.
            weight_max (float, optional): maximum weight value. Defaults to 1.
            number_of_steps (int, optional): number of steps. Defaults to 1000.
            reduction (str, optional): reduction method of output. Only 'mean' and None is supported. Defaults to 'mean'.
        """
        if gt.ndim != 2:
            raise ValueError(f'Invalid gt argument. Expected shape is (N, M) got instead {gt.shape}.')

        if weight_max <= weight_min:
            raise ValueError(f'Value of max {weight_max} is expected to be higher than the value of min {weight_min}.')

        self.gt = gt
        self.num_sample, self.num_target = self.gt.shape
        self.gt_min = self.gt.min(axis=1) if gt_min is None else torch.ones(size=(self.num_target,)) * gt_min
        self.gt_max = self.gt.max(axis=1) if gt_max is None else torch.ones(size=(self.num_target,)) * gt_max
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.number_of_steps = number_of_steps
        self.reduction = reduction

        self.weights = []
        for i in range(self.num_target):
            _, y_vals = kde(gt[:,i], min_value=self.gt_min[i], max_value=self.gt_max[i], number_of_steps=number_of_steps)
            self.weights.append(torch.from_numpy(self._init_weights(y_vals)))

    def _init_weights(self, y_vals: np.ndarray) -> np.ndarray:
        y_vals *= -1
        y_vals_normed = np.interp(y_vals, (y_vals.min(), y_vals.max()), (self.weight_min, self.weight_max))
        return y_vals_normed

    def bell(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            diff = y_true - y_pred
            sqr_error = torch.pow(diff, 2)
            abs_error = torch.abs(diff)
            scale = 2 * pow(9, 2)
            bell_loss = 300 * (1 - torch.exp(-(sqr_error/scale)))
            inv_pdf_x = [torch.linspace(self.gt_min[t], self.gt_max[t], self.number_of_steps, device=y_true.device)
                         for t in range(y_true.shape[1])]
            w_ind_t = lambda t_val, t_ind: torch.argmin(torch.abs(t_val-inv_pdf_x[t_ind]))

            W = torch.zeros_like(y_true, device=y_true.device)
            for t in range(y_true.shape[1]):
                for n in range(y_true.shape[0]):
                    W[n, t] = self.weights[t][w_ind_t(y_true[n, t], t)]

            wbell = W * (bell_loss + sqr_error + abs_error)

            match self.reduction:
                case 'mean':
                    return wbell.mean()
                case _:
                    return wbell
        return loss


def bell_l2_l1_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Bell + L2 + L1 loss.

    Args:
        y_pred (torch.Tensor): prediction of shape (N, M).
        y_true (torch.Tensor): ground truth of shape (N, M).

    Returns:
        torch.Tensor: scalar loss value
    """
    diff = y_true - y_pred
    sqr_error = torch.pow(diff, 2)
    abs_error = torch.abs(diff)
    scale = 2 * pow(9, 2)
    bell_loss = 300 * (1 - torch.exp(-(sqr_error/scale)))
    return torch.mean(bell_loss) + torch.mean(sqr_error) + torch.mean(abs_error)


class BellL2L1Loss(nn.Module):

    def __init__(self, reduction: Callable | None = torch.mean):
        super(BellL2L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """Bell + L2 + L1 loss.

        Args:
            y_pred (torch.Tensor): prediction of shape (N, M).
            y_true (torch.Tensor): ground truth of shape (N, M).

        Returns:
            torch.Tensor: scalar loss value
        """
        diff = y_true - y_pred # (N,M)
        sqr_error = torch.pow(diff, 2) # (N,M)
        abs_error = torch.abs(diff) # (N,M)
        scale = 2 * pow(9, 2) # ()
        bell_loss = 300 * (1 - torch.exp(-(sqr_error/scale))) # (N,M)
        belll2l1 = torch.mean(bell_loss, dim=1) + torch.mean(sqr_error, dim=1) + torch.mean(abs_error, dim=1)

        if self.reduction:
            return self.reduction(belll2l1) # ()

        return belll2l1 # (M,)


def ecl1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Error Consistent MAE loss function.

    Implementation is based on paper:
    Süleyman Aslan, Uğur Güdükbay, Hamdi Dibeklioğlu; Multimodal assessment of apparent personality
    using feature attention and error consistency constraint, Image and Vision Computing, Volume 110, 104163, 2021

    Args:
        y_pred (torch.Tensor): prediction of shape (N, M).
        y_true (torch.Tensor): ground truth of shape (N, M).

    Returns:
        torch.Tensor: scalar loss value.
    """
    diff = y_true - y_pred # (N,M)
    abs_error = torch.abs(diff) # (N,M)
    mae = torch.mean(abs_error) # ()
    mae_traits = torch.mean(abs_error, dim=1) # (M,)
    ec = 0.5 * torch.sqrt(torch.sum(torch.pow(mae_traits-mae, 2))) # ()
    return mae + ec # ()