import torch
import wandb
import numpy as np
import random
import re
import subprocess
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from fusion_models import MultiClassificationTorch_Imagenet, MultiClassificationTorch
from dataset_baseline import  MMotu_Classificaiton_Dataset
from utils import plot_roc_curve_multilabel, compute_weighted_accuracy
from tqdm import tqdm
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"



def main():
    # --- Setup ---
    SEED = 42
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

    max_epochs = 2000
    batch_size = 12
    num_classes = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2

    # Git Info
    try:
        commit_string = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()
        commit_string = re.sub(r'\W+', '_', commit_string)
        commit_log = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        commit_string, commit_log = "no_commit", "0000"
        print(f"Git commit fetch failed: {e}")

    project_title = "Imagenet and fusion Ovarian Cancer Classification MMOTU"
    experiment_name = f"{commit_string}_{commit_log}"
    train_config = {
        "batch_size": batch_size,
        "radiomics": False,
        "encoder_checkpoint": "normtverskyloss_binary_segmentation",
        "input_dim": '256x256:64',
        "model_type": "MultiLabelClassificationTorch",
        "info": "No test set or fold"
    }



    # WandB Init
    run = wandb.init(project=project_title, name=experiment_name, config=train_config)

    # Dataset and DataLoader
    train_dataset =  MMotu_Classificaiton_Dataset(phase='train') # , radiomics_dir=False)
    val_dataset =  MMotu_Classificaiton_Dataset(phase='val') #, radiomics_dir=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


    # Model
    model =  MultiClassificationTorch(input_dim= 64, num_classes= 8,  
                                    encoder_weight_path = r"checkpoints/normtverskyloss_binary_segmentation/a56e77a/best-checkpoint-epoch=77-validation/loss=0.2544.ckpt", 
                                    sdf_model_path= r"checkpoints/deeplabv3_sdf_randomcrop/model_20250711_201243/epoch_84",
                                    radiomics= False).to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    # #optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(),  lr=0.01, momentum=0.9, weight_decay=0.0005)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    # accuracy_metric = MultilabelAccuracy(num_labels=num_classes).to(device)
    # auc_metric = MultilabelAUROC(num_labels=num_classes).to(device)

    accuracy_metric = MultilabelAccuracy(num_labels=8, average=None).to(device)
    auc_metric = MultilabelAUROC(num_labels=8, average=None).to(device)

    best_val_auc = -1
    best_model_state = None

    # --- Training Loop ---
    for epoch in tqdm(range(max_epochs), leave=False):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if len(batch) == 2:
                x, y = batch
                loss = model.compute_loss(x.to(device), y.to(device))
            else:
                x, x2, y = batch
                loss = model.compute_loss(x.to(device), y.to(device), x2.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({f"train/loss": avg_train_loss, "epoch": epoch})
        wandb.log({f"train/lr": scheduler.get_last_lr()[0], "epoch": epoch})

        # --- Validation ---
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores, _= model(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = model(x, x2)

                probs = torch.sigmoid(scores)
                y_probs.append(probs)
                y_true.append(y)

                accuracy_metric.update(probs, y.int())
                auc_metric.update(probs, y.int())

        y_true = torch.cat(y_true)
        y_probs = torch.cat(y_probs)

        # --- Compute top-1 and top-2 accuracy ---
        topk = torch.topk(y_probs, k=2, dim=1)
        top1_preds = topk.indices[:, 0]
        top2_preds = topk.indices
        targets = y_true.argmax(dim=1) if y_true.ndim > 1 else y_true

        top1_acc = (top1_preds == targets).float().mean().item()
        top2_acc = (top2_preds == targets.unsqueeze(1)).any(dim=1).float().mean().item()

        wandb.log({
            "val/top1_acc": top1_acc,
            "val/top2_acc": top2_acc,
            "epoch": epoch
        })


        val_auc_per_class = auc_metric.compute()
        val_acc_per_class = accuracy_metric.compute()

        for i in range(num_classes):
            wandb.log({
                f"val/auc/class_{i}": val_auc_per_class[i].item(),
                f"val/acc/class_{i}": val_acc_per_class[i].item(),
                "epoch": epoch
            })

        val_auc = val_auc_per_class.mean().item()
        val_acc = val_acc_per_class.mean().item()
        val_wacc = compute_weighted_accuracy(y_probs, y_true)

        wandb.log({
            "val/auc_mean": val_auc,
            "val/acc_mean": val_acc,
            "val/weighted_accuracy": val_wacc,
            "epoch": epoch
        })

        accuracy_metric.reset()
        auc_metric.reset()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()

    # # --- Final ROC Curve Plot ---
    # model.load_state_dict(best_model_state)
    # model.eval()
    # y_true, y_probs = model.predict_on_loader(val_loader)
    # roc_auc_dict, roc_path = plot_roc_curve_multilabel(y_true, y_probs, num_classes)
    # wandb.log({"Final ROC Curve (Val)": wandb.Image(roc_path)})

    # Create a directory for saving checkpoints if it doesn't exist
    checkpoint_dir = f"checkpoints/imagenet_fusion_model_mmotu/{commit_log}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': best_model_state,
        'best_val_auc': best_val_auc
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))


    # run.finish()
    # --- Final Evaluation After Loading Best Model ---
    model.load_state_dict(best_model_state)
    model.eval()
    y_true, y_probs = model.predict_on_loader(val_loader)

    roc_auc_dict, roc_path = plot_roc_curve_multilabel(y_true, y_probs, num_classes)
    wandb.log({"Final ROC Curve (Val)": wandb.Image(roc_path)})

    # Move to same device
    y_probs_tensor = torch.tensor(y_probs).to(device)
    y_true_tensor = torch.tensor(y_true).to(device)

    # --- Top-K Accuracy ---
    topk = torch.topk(y_probs_tensor, k=2, dim=1)
    top1_preds = topk.indices[:, 0]
    top2_preds = topk.indices
    targets = y_true_tensor.argmax(dim=1) if y_true_tensor.ndim > 1 else y_true_tensor

    top1_acc = (top1_preds == targets).float().mean().item()
    top2_acc = (top2_preds == targets.unsqueeze(1)).any(dim=1).float().mean().item()

    # --- Metric Calculation ---
    accuracy_metric.update(y_probs_tensor, y_true_tensor.int())
    auc_metric.update(y_probs_tensor, y_true_tensor.int())

    final_auc_per_class = auc_metric.compute()
    final_acc_per_class = accuracy_metric.compute()


    # --- Log to WandB ---
    table_data = []
    for i in range(num_classes):
        table_data.append([
            f"Class {i}",
            final_auc_per_class[i].item(),
            final_acc_per_class[i].item()
        ])

    # Create and log WandB table
    metrics_table = wandb.Table(columns=["Class", "AUC", "Accuracy"], data=table_data)
    wandb.log({"Final Per-Class Metrics Table": metrics_table})

    final_auc = final_auc_per_class.mean().item()
    final_acc = final_acc_per_class.mean().item()
    final_wacc = compute_weighted_accuracy(y_probs_tensor, y_true_tensor)



    summary_table = wandb.Table(columns=["Metric", "Value"], data=[
        ["AUC Mean", final_auc],
        ["Accuracy Mean", final_acc],
        ["Weighted Accuracy", final_wacc],
        ["Top-1 Accuracy", top1_acc],
        ["Top-2 Accuracy", top2_acc],
    ])

    wandb.log({
        "Final Performance Metrics": summary_table
    })

    accuracy_metric.reset()
    auc_metric.reset()

    run.finish()

if __name__ == "__main__":
    main()
