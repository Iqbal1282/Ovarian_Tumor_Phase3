import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import os
import numpy as np
import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import torch.nn.functional as F
import torch

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





def contrastive_loss(us_emb, pa_emb, temperature=0.07):
    """
    us_emb: [B, D]
    pa_emb: [B, D]
    """
    # Normalize
    us_emb = F.normalize(us_emb, dim=1)
    pa_emb = F.normalize(pa_emb, dim=1)

    # Similarity matrix
    logits = torch.matmul(us_emb, pa_emb.t()) / temperature     # [B, B]

    labels = torch.arange(us_emb.size(0), device=us_emb.device)

    # InfoNCE symmetric loss (US->PA and PA->US)
    loss_us_to_pa = F.cross_entropy(logits, labels)
    loss_pa_to_us = F.cross_entropy(logits.t(), labels)

    return (loss_us_to_pa + loss_pa_to_us) / 2


def contrastive_loss_3modal(us_emb, pa_emb, rad_emb, temperature=0.07):
    """
    us_emb:  [B, D]
    pa_emb:  [B, D]
    rad_emb: [B, D]
    """
    # if us_emb is not dim 2 , reshape
    if us_emb.dim() > 2:
        us_emb = us_emb.reshape(us_emb.size(0), -1)
    if pa_emb.dim() > 2:
        pa_emb = pa_emb.reshape(pa_emb.size(0), -1)
    if rad_emb.dim() > 2:
        rad_emb = rad_emb.reshape(rad_emb.size(0), -1)

    # Normalize all modality embeddings
    us = F.normalize(us_emb, dim=1)
    pa = F.normalize(pa_emb, dim=1)
    rad = F.normalize(rad_emb, dim=1)

    labels = torch.arange(us.size(0), device=us.device)

    # --- helper function ---
    def single_info_nce(a, b):
        # a: [B, D], b: [B, D]
        sim = torch.matmul(a, b.t()) / temperature    # [B, B]
        return F.cross_entropy(sim, labels)

    # US ↔ PA
    L_up = single_info_nce(us, pa)
    L_pu = single_info_nce(pa, us)

    # US ↔ RAD
    L_ur = single_info_nce(us, rad)
    L_ru = single_info_nce(rad, us)

    # PA ↔ RAD
    L_pr = single_info_nce(pa, rad)
    L_rp = single_info_nce(rad, pa)

    # Final symmetric loss across all pairs
    return (L_up + L_pu + L_ur + L_ru + L_pr + L_rp) / 6
