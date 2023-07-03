from pathlib import Path
import torch
import numpy as np
from scipy import stats
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    auc,
    precision_recall_curve,
    confusion_matrix,
    roc_curve,
)


def format(x):
    return np.around(x, decimals=3)


def correlation_analysis(x1: np.ndarray,
                         x2: np.ndarray,
                         output_path: str | Path = "correlation.png"):
    corr, p_pval = stats.pearsonr(x1, x2)
    rho, s_pval = stats.spearmanr(x1, x2)

    print("pearsonr:", corr, "pval:", p_pval)
    print("spearmanr:", rho, "pval:", s_pval)

    if output_path is not None:
        plt.figure()
        plt.scatter(x1, x2, alpha=0.3)
        plt.title(
            f"pearsonr: {np.round(corr, decimals=3)}, p-value: {p_pval}\nspearmanr: {np.round(rho, decimals=3)}, p-value: {s_pval}"
        )
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.savefig(output_path)
        plt.close()

    return corr, p_pval, rho, s_pval


def binary_classification_plots(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    output_dir: str | Path | None = ".",
) -> tuple[float, float]:
    """Generate binary classification plots

    In general, if the positive class is rare or the cost of false negatives is high, then the precision-recall curve may be more relevant.
    On the other hand, if the negative class is rare or the cost of false positives is high, then the ROC curve may be more relevant.
    The F1-threshold that optimizes F1 score can be found using either curve.

    Args:
        y_true (np.ndarray | torch.Tensor): binary annotation. np.ndarray of shape (batch_size, 1)
        y_prob (np.ndarray | torch.Tensor): after sigmoid values, probabilities. Values are in the range [0..1]. np.ndarray of shape (batch_size, 1)
        output_dir (str | Path, optional): path to the saved figures. Defaults to '.'.

    Returns:
        (float): optimal F1 treshold

    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().squeeze().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().squeeze().numpy()

    y_true = y_true.astype(int)

    if y_true.ndim == 2 and y_true.shape[1] != 1:
        y_true = np.nanmax(y_true, axis=1)

    if y_prob.ndim == 2 and y_prob.shape[1] != 1:
        y_prob = np.nanmax(y_prob, axis=1)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    #############
    # ROC curve #
    #############
    # calculate the false positive rate (FPR) and true positive rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    optimal_roc_idx = np.nanargmax(tpr - fpr)
    optimal_roc_threshold = thresholds[optimal_roc_idx]

    if output_dir is not None:
        # plot the ROC curve
        plt.figure()
        plt.plot(fpr,
                 tpr,
                 color="darkorange",
                 lw=2,
                 label="ROC curve (area=%0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.plot(
            fpr[optimal_roc_idx],
            tpr[optimal_roc_idx],
            "ro",
            label="optimal (thr=%0.2f)" % optimal_roc_threshold,
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(output_dir / "fig_roc.png")
        plt.close()

    #############
    # PR  curve #
    #############
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    prc_auc = auc(recall, precision)
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_prc_idx = np.nanargmax(f1_scores)
    optimal_prc_threshold = thresholds[optimal_prc_idx]

    if output_dir is not None:
        plt.step(
            recall,
            precision,
            color="b",
            alpha=0.2,
            where="post",
            label="PRC curve (auc=%0.2f)" % prc_auc,
        )
        plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        # alternative solution
        # optimal_prc_idx = np.argmax(recall + precision)
        # optimal_prc_threshold = thresholds[optimal_prc_idx]
        # plt.plot(recall[optimal_prc_idx], precision[optimal_prc_idx],
        #         'ro', label='optimal (thr=%0.2f)' % optimal_prc_threshold)
        plt.plot(
            recall[optimal_prc_idx],
            precision[optimal_prc_idx],
            "ro",
            label="optimal (thr=%0.2f)" % optimal_prc_threshold,
        )
        plt.plot([0, 1], [no_skill, no_skill],
                 linestyle="--",
                 color="blue",
                 label="No Skill")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f"Precision-Recall curve")
        plt.legend(loc="lower right")
        plt.savefig(output_dir / "fig_prc.png")
        plt.close()

    return optimal_roc_threshold, optimal_prc_threshold


def binary_classification_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    thr: float = 0.5,
):
    # default: y_prob is probability, after sigmoid values

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().squeeze().numpy()

    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().squeeze().numpy()

    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= thr).astype(float)

    assert (isinstance(y_true, np.ndarray) and y_true.ndim
            == 1), f"Invalid ground truth format: {y_true.shape}"
    assert (isinstance(y_prob, np.ndarray)
            and y_prob.ndim == 1), f"Invalid prediction format: {y_prob.shape}"
    assert (
        y_true.shape == y_prob.shape
    ), f"y_true and y_prob shape should be the same ({y_true.shape} vs {y_prob.shape})"

    y_true_unique, y_true_support = np.unique(y_true, return_counts=True)
    y_pred_unique, y_pred_support = np.unique(y_pred, return_counts=True)

    assert all([elem in [0, 1] for elem in y_true_unique
                ]), f"Ground truth should be binary: {y_true_unique}"
    assert all([elem in [0, 1] for elem in y_pred_unique
                ]), f"Prediction should be binary: {y_pred_unique}"

    metrics = {}

    metrics["Thr"] = format(thr)
    metrics["Accuracy"] = format(accuracy_score(y_true, y_pred))
    # It is calculated as the average of recall scores for each class, weighted by the number of samples in each class.
    # it is not fair to use it in case of sequence-level calculations: there can be missing classes (no blink = full 0)
    # metrics["BalancedAccuracy"] = format(
    #    balanced_accuracy_score(y_true, y_pred)
    # )

    metrics["ConfusionMatrix"] = confusion_matrix(y_true,
                                                  y_pred,
                                                  labels=[0, 1]).ravel()
    metrics["NormalizedConfusionMatrix"] = confusion_matrix(y_true,
                                                            y_pred,
                                                            normalize="true",
                                                            labels=[0, 1
                                                                    ]).ravel()
    metrics["SupportTrue"] = y_true_support
    metrics["SupportPred"] = y_pred_support

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["TP"] = tp
    metrics["TN"] = tn
    metrics["FP"] = fp
    metrics["FN"] = fn

    metrics["Support"] = tp + tn + fp + fn

    if (
            tp == 0
    ):  # without TP, we can just drop the sample from further calculations instead of including 0s
        metrics["Precision"] = np.nan
        metrics["Recall"] = np.nan
        metrics["F1"] = np.nan
        metrics["AP"] = np.nan
    else:
        metrics["Precision"] = format(
            precision_score(y_true,
                            y_pred,
                            pos_label=1,
                            average="binary",
                            zero_division="warn"))
        # True Positive Rate, Sensitivity, Hit Rate
        metrics["Recall"] = format(
            recall_score(y_true,
                         y_pred,
                         pos_label=1,
                         average="binary",
                         zero_division="warn"))
        metrics["F1"] = format(
            f1_score(y_true,
                     y_pred,
                     pos_label=1,
                     average="binary",
                     zero_division="warn"))
        metrics["AP"] = format(
            average_precision_score(y_true, y_prob, pos_label=1))

    # True Negative Rate, Specificity, Selectivity
    Specificity = tn / (tn + fp) if tn != 0 or tn + fp != 0 else np.nan
    metrics["Specificity"] = format(Specificity)

    return metrics


def blinking_metrics_with_probs(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thr: float = 0.5,
        mask_gt: bool = True) -> tuple[dict, list[dict]]:
    """Blinking metrics with probabilities as predictions

    Expected inputs:
        y_true and y_prob are in [0,1]: for every sample/timestamp there is an annotated ground truth and a prediction (ideal use case)
        y_true is in [0, 1] and y_prob is in [0, 1, np.nan]: failed detection of the face or eyes may lead to np.nan instead of non-blink (x<thr)
                                                             to measure the blink detector's performance, only those cases are considered in the
                                                             metrics, that are not np.nan in the pred. Ground truth is masked in these cases.

    Notation:
        y_raw: raw outputs from a network, logits
        y_prob: sigmoid is applied to the logits, probabilities
        y_pred: thresholding is applied to the probabilities, binary
        y_true: ground truth labels, binary

    Args:
        y_true (np.ndarray): ground truth sequences with shape (number_of_samples, number_of_timestamp)
        y_prob (np.ndarray): prediction probability sequences with shape (number_of_samples, number_of_timestamp)
        thr (float): threshold to convert probabilities to binary values (Default: 0.5)
        mask_gt (bool): ignore the gt values as well, where there are np.nan values in the preds/probs (the pipeline failed)
                        to measure the performance of the blink detector only instead of the full pipeline (Default: True)
    """
    assert (
        isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
        and y_true.ndim == 2 and y_prob.ndim == 2
    ), f"Expected shape for y_true and y_prob are (N,T) got instead {y_true.shape} and {y_prob.shape}"
    assert (
        isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
        and y_true.shape == y_prob.shape
    ), f"Shape of y_true and y_prob should be the same, got instead {y_true.shape} and {y_prob.shape}"

    # nans are handled differently with float and float64
    y_prob = y_prob.astype(float)
    y_true = y_true.astype(float)

    # y_pred = (y_prob >= thr).astype(int)  # np.nan > thr is also False

    # mask = np.isnan(y_prob)
    # if mask_gt:
    #    nans = np.full_like(y_pred, fill_value=np.nan, dtype=float)
    #    y_pred = np.where(mask, nans, y_pred)
    #    y_true = np.where(mask, nans, y_pred)
    # TODO binary classification metrics (in sklearn) do not handle np.nan values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_prob)):
        raise NotImplementedError(
            "binary classification metrics in sklearn do not handle np.nan values"
        )

    seq_metrics = [
        sequence_level_metrics(y_true[i, :], y_prob[i, :], thr=thr)
        for i in range(y_true.shape[0])
    ]
    sam_metrics = sample_level_metrics(y_true, y_prob, thr=thr)

    sam_metrics["s_mean_IoU"] = format(
        np.nanmean([elem["IoU"] for elem in seq_metrics]))
    sam_metrics["s_std_IoU"] = format(
        np.nanstd([elem["IoU"] for elem in seq_metrics]))

    sam_metrics["s_IoU_TP"] = format(
        np.nansum([elem["IoU_TP"] for elem in seq_metrics]))

    sam_metrics["s_Dilate3_IoU"] = format(
        np.nanmean([elem["Dilate3_IoU"] for elem in seq_metrics]))
    sam_metrics["s_Dilate3_IoU_TP"] = format(
        np.nansum([elem["Dilate3_IoU_TP"] for elem in seq_metrics]))

    sam_metrics["s_TP"] = np.sum([elem["TP"] for elem in seq_metrics])
    sam_metrics["s_TN"] = np.sum([elem["TN"] for elem in seq_metrics])
    sam_metrics["s_FP"] = np.sum([elem["FP"] for elem in seq_metrics])
    sam_metrics["s_FN"] = np.sum([elem["FN"] for elem in seq_metrics])
    sam_metrics["s_Precision"] = format(
        np.nanmean([elem["Precision"] for elem in seq_metrics]))
    sam_metrics["s_Recall"] = format(
        np.nanmean([elem["Recall"] for elem in seq_metrics]))
    sam_metrics["s_F1"] = format(
        np.nanmean([elem["F1"] for elem in seq_metrics]))
    return sam_metrics, seq_metrics


def blinking_metrics_with_probs_v3(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_prob_cls: np.ndarray,
        thr: float = 0.5) -> tuple[dict, list[dict]]:
    """Blinking metrics with probabilities as predictions

    Expected inputs:
        y_true and y_prob are in [0,1]: for every sample/timestamp there is an annotated ground truth and a prediction (ideal use case)
        y_true is in [0, 1] and y_prob is in [0, 1, np.nan]: failed detection of the face or eyes may lead to np.nan instead of non-blink (x<thr)
                                                             to measure the blink detector's performance, only those cases are considered in the
                                                             metrics, that are not np.nan in the pred. Ground truth is masked in these cases.

    Notation:
        y_raw: raw outputs from a network, logits
        y_prob: sigmoid is applied to the logits, probabilities
        y_prob_cls: sigmoid is applied to the logit, probability
        y_pred: thresholding is applied to the probabilities, binary
        y_true: ground truth labels, binary

    Args:
        y_true (np.ndarray): ground truth sequences with shape (number_of_samples, number_of_timestamp)
        y_prob (np.ndarray): prediction probability sequences with shape (number_of_samples, number_of_timestamp)
        y_prob_cls (np.ndarray): prediction probability sequences with shape (number_of_samples,)
        thr (float): threshold to convert probabilities to binary values (Default: 0.5)
        mask_gt (bool): ignore the gt values as well, where there are np.nan values in the preds/probs (the pipeline failed)
                        to measure the performance of the blink detector only instead of the full pipeline (Default: True)
    """
    assert (
        isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
        and y_true.ndim == 2 and y_prob.ndim == 2
    ), f"Expected shape for y_true and y_prob are (N,T) got instead {y_true.shape} and {y_prob.shape}"
    assert (
        isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
        and y_true.shape == y_prob.shape
    ), f"Shape of y_true and y_prob should be the same, got instead {y_true.shape} and {y_prob.shape}"
    assert isinstance(y_prob_cls, np.ndarray) and y_prob_cls.shape == (
        y_prob.shape[0],
    ), f"Shape of y_prob_cls should be (N,), got instead {y_prob_cls.shape}"

    # nans are handled differently with float and float64
    y_prob = y_prob.astype(float)
    y_true = y_true.astype(float)

    # TODO binary classification metrics (in sklearn) do not handle np.nan values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_prob)):
        raise NotImplementedError(
            "binary classification metrics in sklearn do not handle np.nan values"
        )

    seq_metrics = [
        binary_classification_metrics(y_true[i, :], y_prob[i, :], thr=thr)
        for i in range(y_true.shape[0])
    ]
    sam_metrics = sample_level_metrics_v2(y_true, y_prob_cls, thr=thr)

    sam_metrics["s_TP"] = np.sum([elem["TP"] for elem in seq_metrics])
    sam_metrics["s_TN"] = np.sum([elem["TN"] for elem in seq_metrics])
    sam_metrics["s_FP"] = np.sum([elem["FP"] for elem in seq_metrics])
    sam_metrics["s_FN"] = np.sum([elem["FN"] for elem in seq_metrics])
    sam_metrics["s_Precision"] = format(
        sam_metrics["s_TP"] / (sam_metrics["s_TP"] + sam_metrics["s_FP"]))
    sam_metrics["s_Recall"] = format(
        sam_metrics["s_TP"] / (sam_metrics["s_TP"] + sam_metrics["s_FN"]))
    sam_metrics["s_F1"] = format(
        2 * (sam_metrics["s_Precision"] * sam_metrics["s_Recall"]) /
        (sam_metrics["s_Precision"] + sam_metrics["s_Recall"]))
    sam_metrics["sw_Precision"] = format(
        np.nanmean([elem["Precision"] for elem in seq_metrics]))
    sam_metrics["sw_Recall"] = format(
        np.nanmean([elem["Recall"] for elem in seq_metrics]))
    sam_metrics["sw_F1"] = format(
        np.nanmean([elem["F1"] for elem in seq_metrics]))

    return sam_metrics, seq_metrics


def blinking_metrics_with_probs_v2(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_prob_cls: np.ndarray,
        thr: float = 0.5) -> tuple[dict, list[dict]]:
    """Blinking metrics with probabilities as predictions

    Expected inputs:
        y_true and y_prob are in [0,1]: for every sample/timestamp there is an annotated ground truth and a prediction (ideal use case)
        y_true is in [0, 1] and y_prob is in [0, 1, np.nan]: failed detection of the face or eyes may lead to np.nan instead of non-blink (x<thr)
                                                             to measure the blink detector's performance, only those cases are considered in the
                                                             metrics, that are not np.nan in the pred. Ground truth is masked in these cases.

    Notation:
        y_raw: raw outputs from a network, logits
        y_prob: sigmoid is applied to the logits, probabilities
        y_prob_cls: sigmoid is applied to the logit, probability
        y_pred: thresholding is applied to the probabilities, binary
        y_true: ground truth labels, binary

    Args:
        y_true (np.ndarray): ground truth sequences with shape (number_of_samples, number_of_timestamp)
        y_prob (np.ndarray): prediction probability sequences with shape (number_of_samples, number_of_timestamp)
        y_prob_cls (np.ndarray): prediction probability sequences with shape (number_of_samples,)
        thr (float): threshold to convert probabilities to binary values (Default: 0.5)
        mask_gt (bool): ignore the gt values as well, where there are np.nan values in the preds/probs (the pipeline failed)
                        to measure the performance of the blink detector only instead of the full pipeline (Default: True)
    """
    assert (
        isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
        and y_true.ndim == 2 and y_prob.ndim == 2
    ), f"Expected shape for y_true and y_prob are (N,T) got instead {y_true.shape} and {y_prob.shape}"
    assert (
        isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
        and y_true.shape == y_prob.shape
    ), f"Shape of y_true and y_prob should be the same, got instead {y_true.shape} and {y_prob.shape}"
    assert isinstance(y_prob_cls, np.ndarray) and y_prob_cls.shape == (
        y_prob.shape[0],
    ), f"Shape of y_prob_cls should be (N,), got instead {y_prob_cls.shape}"

    # nans are handled differently with float and float64
    y_prob = y_prob.astype(float)
    y_true = y_true.astype(float)

    # TODO binary classification metrics (in sklearn) do not handle np.nan values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_prob)):
        raise NotImplementedError(
            "binary classification metrics in sklearn do not handle np.nan values"
        )

    seq_metrics = [
        binary_classification_metrics(y_true[i, :], y_prob[i, :], thr=thr)
        for i in range(y_true.shape[0])
    ]
    sam_metrics = sample_level_metrics_v2(y_true, y_prob_cls,
                                          thr=thr)  # weighted cls head
    sam_max_metrics = sample_level_metrics(y_true, y_prob,
                                           thr=thr)  # max aggregated from seq

    sam_metrics["m_TP"] = sam_max_metrics["TP"]
    sam_metrics["m_TN"] = sam_max_metrics["TN"]
    sam_metrics["m_FP"] = sam_max_metrics["FP"]
    sam_metrics["m_FN"] = sam_max_metrics["FN"]
    sam_metrics["m_Precision"] = format(sam_max_metrics["Precision"])
    sam_metrics["m_Recall"] = format(sam_max_metrics["Recall"])
    sam_metrics["m_F1"] = format(sam_max_metrics["F1"])

    sam_metrics["s_TP"] = np.sum([elem["TP"] for elem in seq_metrics])
    sam_metrics["s_TN"] = np.sum([elem["TN"] for elem in seq_metrics])
    sam_metrics["s_FP"] = np.sum([elem["FP"] for elem in seq_metrics])
    sam_metrics["s_FN"] = np.sum([elem["FN"] for elem in seq_metrics])
    sam_metrics["s_Precision"] = format(
        sam_metrics["s_TP"] / (sam_metrics["s_TP"] + sam_metrics["s_FP"]))
    sam_metrics["s_Recall"] = format(
        sam_metrics["s_TP"] / (sam_metrics["s_TP"] + sam_metrics["s_FN"]))
    sam_metrics["s_F1"] = format(
        2 * (sam_metrics["s_Precision"] * sam_metrics["s_Recall"]) /
        (sam_metrics["s_Precision"] + sam_metrics["s_Recall"]))
    sam_metrics["sw_Precision"] = format(
        np.nanmean([elem["Precision"] for elem in seq_metrics]))
    sam_metrics["sw_Recall"] = format(
        np.nanmean([elem["Recall"] for elem in seq_metrics]))
    sam_metrics["sw_F1"] = format(
        np.nanmean([elem["F1"] for elem in seq_metrics]))

    return sam_metrics, seq_metrics


def sequence_level_metrics(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           thr: float = 0.5) -> dict:
    """Metrics calculated for a single binary sequence ground truth and prediction

    Note:
        y_pred: thresholding is applied to the probabilities, binary

    Args:
        y_prob: sigmoid is applied to the logits, probabilities
        y_true: ground truth labels, binary

    """
    assert (
        y_true.ndim == 1 and y_prob.ndim == 1
    ), f"Expected shape for y_true and y_prob are (T,) got instead {y_true.shape} and {y_prob.shape}"

    y_pred = (y_prob >= thr).astype(float)

    metrics = binary_classification_metrics(y_true=y_true,
                                            y_prob=y_prob,
                                            thr=thr)

    iou = format(blink_iou(y_true, y_pred))
    dilate3_iou = format(blink_iou(y_true, dilate(y_pred, 3)))

    metrics["IoU"] = iou
    metrics["IoU_TP"] = iou > 0.2 if not np.isnan(iou) else np.nan
    metrics["Dilate3_IoU"] = dilate3_iou
    metrics["Dilate3_IoU_TP"] = dilate3_iou > 0 if not np.isnan(
        dilate3_iou) else np.nan

    return metrics


def sample_level_metrics(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         thr: float = 0.5) -> dict:
    y_true_sam = np.nanmax(y_true, axis=1)
    y_prob_sam = np.nanmax(y_prob, axis=1)
    metrics = binary_classification_metrics(y_true=y_true_sam,
                                            y_prob=y_prob_sam,
                                            thr=thr)
    return metrics


def sample_level_metrics_v2(y_true: np.ndarray,
                            y_prob_cls: np.ndarray,
                            thr: float = 0.5) -> dict:
    y_true_sam = np.nanmax(y_true, axis=1)
    metrics = binary_classification_metrics(y_true=y_true_sam,
                                            y_prob=y_prob_cls,
                                            thr=thr)
    return metrics


def blink_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # intersection over union metric, which measures the overlap between the ground truth and the prediction blink intervals
    # y_true and y_pred both are binary vectors
    # 0: there is no overlap between the two regions
    # 1: perfect overlap
    # 0 if there is gt, but no prediction
    # np.nan if there is no gt only prediction
    assert (isinstance(y_true, np.ndarray) and y_true.ndim
            == 1), f"Invalid ground truth format: {y_true.shape}"
    assert (isinstance(y_pred, np.ndarray)
            and y_pred.ndim == 1), f"Invalid prediction format: {y_pred.shape}"
    assert all([
        elem in [0, 1] for elem in np.unique(y_true)
    ]), f"Ground truth should be binary. Got instead: {np.unique(y_true)}"
    assert all([
        elem in [0, 1] for elem in np.unique(y_pred)
    ]), f"Prediction should be binary. Got instead: {np.unique(y_pred)}"

    if not any(y_true):  # no gt, there is pred
        return np.nan

    if any(y_true) and not any(y_pred):  # there is gt, and no pred
        return 0

    y_true_ind = np.where(y_true == 1)[0]  # filter np.nan values
    y_pred_ind = np.where(y_pred == 1)[0]  # filter np.nan values
    _min = np.min(np.concatenate([y_true_ind, y_pred_ind]))
    _max = np.max(np.concatenate([y_true_ind, y_pred_ind])) + 1
    intersection = np.sum(np.logical_and(y_true[_min:_max], y_pred[_min:_max]))
    union = _max - _min
    return intersection / union


def dilate(vec: np.ndarray, dilation: int = 3) -> np.ndarray:
    """Dilation operation for vectors

    Example:
        dilation = 1: [0,0,0,0,1,0,0,0] -> [0,0,0,1,1,1,0,0]
        dilation = 2: [0,0,0,0,1,0,0,0] -> [0,0,1,1,1,1,1,0]
        dilation = 3: [0,0,0,0,0,0,1,0] -> [0,0,0,1,1,1,1,1]
    """
    assert vec.ndim == 1, f"Expected shape is (N,) got instead {vec.shape}"
    assert all(
        [elem in {0, 1} for elem in np.unique(vec)]
    ), f"Binary vector is expected, got instead vector with the following elements: {np.unique(vec)}"

    vec = vec.copy()
    indices = np.where(vec == 1)[0]  # filter np.nan values

    for index in indices:
        start = max(0, index - dilation)
        end = index + dilation + 1
        vec[start:end] = 1

    return vec


if __name__ == "__main__":

    y_true = np.array([1, 0, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 0])
    metrics = binary_classification_metrics(y_true, y_pred)
    print(metrics)

    y_true = np.array([1, 0, 0, 0, 1, 1])
    y_probs = np.array([0.7, 0.1, 0.5, 0.45, 0.8, 0.3])
    binary_classification_plots(y_true, y_probs, "tmp")
    exit()

    y_true = np.array([[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0]])
    sam_metrics, seq_metrics = _blinking_metrics(y_true, y_pred)
    print(sam_metrics, "\n", seq_metrics)

    y_true = np.array([[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0]])
    sam_metrics, seq_metrics = blinking_metrics_with_probs(y_true,
                                                           y_pred,
                                                           thr=0.5)
    print(sam_metrics, "\n", seq_metrics)

    y_true = np.array([[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0]])
    y_pred = np.array([[0, 0, np.nan, 1, 1, 0, 0], [0, 0, 1, 0, 1, np.nan, 0]])
    sam_metrics, seq_metrics = blinking_metrics_with_probs(y_true,
                                                           y_pred,
                                                           thr=0.5)
    print(sam_metrics, "\n", seq_metrics)
