import torch


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
    bell_loss = 300 * (1 - torch.exp(-(sqr_error / scale)))
    return torch.mean(bell_loss) + torch.mean(sqr_error) + torch.mean(abs_error)


def ecl1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Error Consistent MAE loss function.

    Implementation is based on paper:
        Süleyman Aslan, Uğur Güdükbay, Hamdi Dibeklioğlu; Multimodal assessment of apparent
        personality using feature attention and error consistency constraint,
        Image and Vision Computing, Volume 110, 104163, 2021

    Args:
        y_pred (torch.Tensor): prediction of shape (N, M).
        y_true (torch.Tensor): ground truth of shape (N, M).

    Returns:
        torch.Tensor: scalar loss value.

    """
    diff = y_true - y_pred  # (N,M)
    abs_error = torch.abs(diff)  # (N,M)
    mae = torch.mean(abs_error)  # ()
    mae_traits = torch.mean(abs_error, dim=1)  # (M,)
    ec = 0.5 * torch.sqrt(torch.sum(torch.pow(mae_traits - mae, 2)))  # ()
    return mae + ec  # ()
