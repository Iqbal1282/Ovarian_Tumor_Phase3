import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import os
import numpy as np
import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


def plot_multilabel_roc_curve(y_true, y_probs, class_names=None, fold_idx=None, wandb_logger=None):
    n_classes = y_true.shape[1]
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    acc_dict = {}

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        y_pred = (y_probs[:, i] >= 0.5).astype(int)
        acc = accuracy_score(y_true[:, i], y_pred)

        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        roc_auc_dict[i] = roc_auc
        acc_dict[i] = acc

        class_label = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr, tpr, lw=2, label=f"{class_label} (AUC = {roc_auc:.2f}, Acc = {acc:.2f})")

    # Reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-Label ROC Curve - Fold {fold_idx}' if fold_idx is not None else 'Multi-Label ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save & log
    os.makedirs("plots", exist_ok=True)
    img_path = f'plots/multilabel_roc_fold_{fold_idx}.png' if fold_idx is not None else 'plots/multilabel_roc.png'
    plt.savefig(img_path)
    plt.close()

    if wandb_logger:
        log_dict = {f'class_{i}_auc': roc_auc_dict[i] for i in range(n_classes)}
        log_dict.update({f'class_{i}_accuracy': acc_dict[i] for i in range(n_classes)})
        log_dict[f'Multilabel ROC Curve Fold {fold_idx}'] = wandb.Image(img_path)
        wandb_logger.experiment.log(log_dict)

    return fpr_dict, tpr_dict, roc_auc_dict, acc_dict


# def plot_roc_curve_multilabel(y_true, y_probs, num_classes, fold_idx):
#     fpr, tpr, roc_auc = {}, {}, {}
#     for i in range(num_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

#     plt.figure()
#     for i in range(num_classes):
#         plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.title(f'ROC Curve (Fold {fold_idx})')
#     plt.legend()
#     path = f"plots/roc_curve_fold_{fold_idx}.png"
#     plt.savefig(path)
#     plt.close()
#     return fpr, tpr, roc_auc, path

# ROC Plotting Utility
def plot_roc_curve_multilabel(y_true, y_probs, num_classes):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve - Validation')
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    path = "plots/roc_curve_validation.png"
    plt.savefig(path)
    plt.close()
    return roc_auc, path



def plot_roc_curve_multilabel(y_true, y_probs, num_classes):
    fpr, tpr, roc_auc = {}, {}, {}

    # Store all interpolated TPRs for averaging
    all_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Interpolate TPR to a common FPR basis for mean ROC
        interp_tpr = np.interp(all_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0  # Ensure the start is 0
        interp_tprs.append(interp_tpr)

    # Calculate mean TPR and AUC
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the end is 1
    mean_auc = auc(all_fpr, mean_tpr)

    # Plot individual class ROC curves
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Plot mean ROC curve
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', linewidth=2,
             label=f'Mean ROC (AUC = {mean_auc:.2f})')

    # Plot random line
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation')
    plt.legend(loc='lower right')

    # Save plot
    os.makedirs("plots", exist_ok=True)
    path = "plots/roc_curve_validation.png"
    plt.savefig(path)
    plt.close()

    return roc_auc, path

def compute_weighted_accuracy(y_probs, y_true, threshold=0.5):
    """
    Computes average accuracy across all labels (classes).
    Args:
        y_probs (torch.Tensor): shape (batch_size, num_classes), probabilities from sigmoid.
        y_true (torch.Tensor): shape (batch_size, num_classes), binary ground truth labels.
        threshold (float): threshold to convert probabilities to binary predictions.

    Returns:
        float: weighted (mean) accuracy across all classes.
    """
    # Binarize predictions
    y_pred = (y_probs >= threshold).int()
    y_true = y_true.int()

    correct = (y_pred == y_true).float()
    # Compute accuracy for each class
    class_accuracy = correct.sum(dim=0) / y_true.shape[0]
    # Return mean accuracy across classes
    return class_accuracy.mean().item()