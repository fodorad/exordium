import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt


def kde(samples: np.ndarray = None, min: float = None, max: float = None, n_step: int = 100, verbose: bool = False, fig_path: str = 'test.png'):
    if samples is None:
        samples = np.random.normal(0.5, 0.15, 6000).clip(0,1)

    if min is None:
        min = samples.min()

    if max is None:
        max = samples.max()

    density = gaussian_kde(samples)
    density.covariance_factor = lambda : .5
    density._compute_covariance()
    x_vals = np.linspace(min, max, n_step)
    y_vals = density(x_vals)

    if verbose:
        plt.hist(samples, bins=n_step, density=True)
        plt.plot(x_vals, y_vals)
        plt.savefig(fig_path)

    return x_vals, y_vals


class InvKDEWeightedLoss():
    # Losses weighted with inverted kernel density estimation of ground truth

    def __init__(self, gt: np.ndarray, 
                       gt_min: float = 0, 
                       gt_max: float = 1,
                       weight_min: float = 0.2, 
                       weight_max: float = 1.,
                       n_step: int = 1000,
                       reduction: str = 'mean') -> None:
        assert gt.ndim == 2, f'Given gt is expected to be a vector of shape (N,M). Instead it has a shape of {gt.shape}'
        assert weight_max > weight_min, 'Max is expected to be higher than min.'

        self.gt = gt
        self.num_sample, self.num_target = self.gt.shape

        if gt_min is None:
            gt_min = self.gt.min(axis=1)
        else:
            gt_min = torch.ones(size=(self.num_target,)) * gt_min
        if gt_max is None:
            gt_max = self.gt.max(axis=1)
        else:
            gt_max = torch.ones(size=(self.num_target,)) * gt_max

        self.gt_min = gt_min
        self.gt_max = gt_max
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.n_step = n_step
        self.reduction = reduction

        self.weights = []
        for i in range(self.num_target):
            _, y_vals = kde(gt[:,i], min=self.gt_min[i], max=self.gt_max[i], n_step=n_step)
            self.weights.append(self._init_weights(y_vals))


    def _init_weights(self, y_vals):
        y_vals *= -1
        y_vals_normed = np.interp(y_vals, (y_vals.min(), y_vals.max()), (self.weight_min, self.weight_max))
        return torch.from_numpy(y_vals_normed)


    def bell(self):

        def loss(y_pred, y_true):
            diff = y_true - y_pred
            sqr_error = torch.pow(diff, 2)
            abs_error = torch.abs(diff)
            scale = 2 * pow(9, 2)
            bell_loss = 300 * (1 - torch.exp(-(sqr_error/scale)))
            inv_pdf_x = [torch.linspace(self.gt_min[t], self.gt_max[t], self.n_step, device=y_true.device) 
                         for t in range(y_true.shape[1])]
            w_ind_t = lambda t_val, t_ind: torch.argmin((t_val-inv_pdf_x[t_ind]).abs())

            W = torch.zeros_like(y_true, device=y_true.device)
            for t in range(y_true.shape[1]):
                for n in range(y_true.shape[0]):
                    W[n, t] = self.weights[t][w_ind_t(y_true[n, t], t)]

            wbell = W*(bell_loss+sqr_error+abs_error)

            if self.reduction == 'none':
                return wbell

            return wbell.mean()

        return loss


def bell_l2_l1_loss(y_pred, y_true):
    diff = y_true - y_pred
    sqr_error = torch.pow(diff, 2)
    abs_error = torch.abs(diff)
    scale = 2 * pow(9, 2)
    bell_loss = 300 * (1 - torch.exp(-(sqr_error/scale)))
    return torch.mean(bell_loss) + torch.mean(sqr_error) + torch.mean(abs_error)


class BellL2L1Loss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(BellL2L1Loss, self).__init__()
        assert reduction in ['none', 'mean']
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred # (N,M)
        sqr_error = torch.pow(diff, 2) # (N,M)
        abs_error = torch.abs(diff) # (N,M)
        scale = 2 * pow(9, 2) # ()
        bell_loss = 300 * (1 - torch.exp(-(sqr_error/scale))) # (N,M)
        belll2l1 = torch.mean(bell_loss, dim=1) + torch.mean(sqr_error, dim=1) + torch.mean(abs_error, dim=1)

        if self.reduction == 'none':
            return belll2l1 # (M,)

        return torch.mean(belll2l1) # ()


def ec_mae(y_pred, y_true):
    # error consistent mae based on paper
    # 2021: Multimodal assessment of apparent personality using feature attention and error consistency constraint
    diff = y_true - y_pred
    abs_error = torch.abs(diff)
    mae = torch.mean(abs_error)
    mae_traits = torch.mean(abs_error, dim=1)
    ec = 0.5 * torch.sqrt(torch.sum(torch.pow(mae_traits-mae, 2)))
    return mae_traits + ec


class ECL1Loss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(ECL1Loss, self).__init__()
        assert reduction in ['none', 'mean']
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred # (N,M)
        abs_error = torch.abs(diff) # (N,M)
        mae = torch.mean(abs_error) # ()
        mae_traits = torch.mean(abs_error, dim=1) # (M,)
        ec = 0.5 * torch.sqrt(torch.sum(torch.pow(mae_traits-mae, 2))) # ()

        if self.reduction == 'none': 
            return mae_traits + ec # (M,)
        
        return mae + ec # ()


if __name__ == '__main__':
    # kde(min=0., max=1., verbose=True)
    # gt = np.random.normal(0.5, 0.15, 6000).clip(0,1) # (6000,)
    gt = np.stack([
        np.random.normal(0.3, 0.15, 6000).clip(0,1),
        np.random.normal(0.5, 0.12, 6000).clip(0,1),
        np.random.normal(0.4, 0.13, 6000).clip(0,1),
        np.random.normal(0.7, 0.11, 6000).clip(0,1),
        np.random.normal(0.6, 0.12, 6000).clip(0,1)
    ], axis=1) # (6000,5)
    print('gt:', gt.shape)

    obj = InvKDEWeightedLoss(gt)
    loss = obj.bell()

    print('bell:', loss(torch.from_numpy(np.zeros((10,5))), 
                        torch.from_numpy(np.zeros((10,5)))))

    print('ecmae:', ec_mae(torch.from_numpy(np.zeros((10,5))), 
                           torch.from_numpy(np.zeros((10,5)))))
