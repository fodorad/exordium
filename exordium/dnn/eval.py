import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score, 
                             accuracy_score,
                             balanced_accuracy_score,
                             average_precision_score, 
                             auc, 
                             precision_recall_curve, 
                             confusion_matrix,
                             roc_curve, 
                             roc_auc_score)


def binary_classification_plots(y_true: list | np.ndarray | torch.Tensor, y_probs: list | np.ndarray | torch.Tensor, output_dir: str | Path = '.'):
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().squeeze().numpy()
    if isinstance(y_probs, torch.Tensor): y_probs = y_probs.detach().cpu().squeeze().numpy()

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ns_fpr, ns_tpr, _ = roc_curve(y_true, np.zeros_like(y_true))
    fpr, tpr, ft_thresholds = roc_curve(y_true=y_true, y_score=y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    optimal_roc_idx = np.argmax(tpr - fpr)
    optimal_roc_threshold = ft_thresholds[optimal_roc_idx]

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', color='blue', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve (AUC: {roc_auc:.3f}, optimal thr: {optimal_roc_threshold:.2f})')
    plt.legend()
    plt.savefig(output_dir / 'fig_roc.png')
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    optimal_prc_idx = np.argmax(recall + precision)
    optimal_prc_threshold = thresholds[optimal_prc_idx]
    prc_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_probs >= 0.5)

    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='blue', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Model')
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title(f'PRC curve (F1: {f1:.3f}, AUC: {prc_auc:.3f}, optimal thr: {optimal_prc_threshold:.2f})')
    plt.legend()
    plt.savefig(output_dir / 'fig_prc.png')
    plt.close()

    thresholds = np.arange(0.0, 1.0, 0.001)
    f1scores = np.zeros(shape=(len(thresholds)))
    # Sweep across the thresholds and calculate the f1-score for those predictions.
    # The idea here is that the threshold with the highest f1-score is ideal
    for index, elem in enumerate(thresholds):
        # Corrected probabilities
        y_pred = (y_probs > elem).astype('int')
        # Calculate the f-score
        f1scores[index] = f1_score(y_true, y_pred)
    # Find the optimal threshold
    f1score_argmax_index = np.argmax(f1scores)
    optimal_f1score_threshold = thresholds[f1score_argmax_index]
    f1score_at_optimal_threshold = f1scores[f1score_argmax_index]
    plt.plot(thresholds, f1scores)
    plt.plot(optimal_f1score_threshold, f1score_at_optimal_threshold, color="g", marker="o", markersize=5)
    plt.title(f"Optimal Threshold: {optimal_f1score_threshold:.2f}")
    plt.savefig(output_dir / f"fig_f1_threshold.png")
    plt.close()


def binary_classification_metrics(y_true: list | np.ndarray | torch.Tensor, y_pred: list | np.ndarray | torch.Tensor):

    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().squeeze().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().squeeze().numpy()

    format = lambda x: np.around(x, decimals=4)
    y_true = np.rint(np.array(y_true))
    y_pred = np.rint(np.array(y_pred))

    assert isinstance(y_true, np.ndarray) and y_true.ndim == 1, f'Invalid ground truth format: {y_true.shape}'
    assert isinstance(y_pred, np.ndarray) and y_pred.ndim == 1, f'Invalid prediction format: {y_pred.shape}'
    assert y_true.shape == y_pred.shape, f'y_true and y_pred shape should be the same ({y_true.shape} vs {y_pred.shape})'
    
    y_true_unique, y_true_support = np.unique(y_true, return_counts=True)
    y_pred_unique, y_pred_support = np.unique(y_pred, return_counts=True)

    #print('true:', y_true_unique, y_true_support)
    #print('pred:', y_pred_unique, y_pred_support)

    if len(y_true_unique) <= 1: return None # Invalid input, skip calculations

    assert all([elem in [0, 1] for elem in y_true_unique]), f'Ground truth should be binary: {y_true_unique}'
    assert all([elem in [0, 1] for elem in y_pred_unique]), f'Prediction should be binary: {y_pred_unique}'

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    Specificity = tn / (tn + fp) 
    
    metrics = {}
    metrics['Accuracy'] = format(accuracy_score(y_true, y_pred))
    metrics['BalancedAccuracy'] = format(balanced_accuracy_score(y_true, y_pred))
    metrics['ConfusionMatrix'] = confusion_matrix(y_true, y_pred)
    metrics['SupportTrue'] = y_true_support
    metrics['SupportPred'] = y_pred_support
    metrics['Precision'] = format(precision_score(y_true, y_pred, average='binary'))
    metrics['Recall'] = format(recall_score(y_true, y_pred, average='binary')) # True Positive Rate, Sensitivity, Hit Rate
    metrics['Specificity'] = format(Specificity) # True Negative Rate, Specificity, Selectivity
    metrics['F1'] = format(f1_score(y_true, y_pred, average='binary'))
    metrics['AP'] = format(average_precision_score(y_true, y_pred))
    return metrics


if __name__ == "__main__":
    y_true = [1, 0, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 0]
    metrics = binary_classification_metrics(y_true, y_pred)
    print(metrics)

    y_true = [1, 0, 0, 0, 1, 1]
    y_probs = [0.7, 0.1, 0.5, 0.45, 0.8, 0.3]
    binary_classification_plots(y_true, y_probs, 'tmp')
