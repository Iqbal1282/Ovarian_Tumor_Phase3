import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from datetime import datetime
from multi_head_models2 import BinaryClassification
#from dataset_washu2 import Classificaiton_Dataset
from dataset import MMotu_Classificaiton_Dataset 

import subprocess
import re
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils import plot_multilabel_roc_curve
import numpy as np
import random 

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED) ; random.seed(SEED); 

max_epochs = 250
min_epochs = 1
batch_size = 32
check_val_every_n_epoch = 3
num_workers = 0
k_fold = 5

try: 
    # Get the latest Git commit message
    commit_string = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()
    commit_string = re.sub(r'\W+', '_', commit_string) 
    commit_log = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    
except Exception as e:
    print(f"An unexpected error occurred: {e}")

project_title = "Ovarian Cancer Classification MMOTU"
Experiment_Group = f"Exp4:{commit_string}_{commit_log}"
train_config = {
        "k_fold": k_fold,
        "batch_size": batch_size,
        "radiomics": False,
        "encoder_checkpoint": "normtverskyloss_binary_segmentation",
        "input_dim": 64,
        "loss_fn": "different loss functions experiements, where score is 2 score2 is 0.5 weight",
        "model_type": "BinaryClassification",
        "info": "Foun encoder median Score experiment",
        "info2": "patient greater than 120 are considered in testset"
    }

fold = 1
# Initialize WandB Logger
run_name = f'Fold_{fold}' #{commit_log}"_"{commit_string}"_{datetime.now()}'
# wandb_logger = WandbLogger(
#         log_model=False, project=project_title, name=run_name, 
#     )

wandb_logger = WandbLogger(
    log_model=False,
    project=project_title,
    name=run_name,
    group= Experiment_Group,
    tags=[f"fold_{fold}", "radiomics=False", f"commit_{commit_log}", "On Reviewed Cleaned Data", commit_string]
)

wandb_logger.experiment.config.update(train_config)


# trainDataset = Classificaiton_Dataset(phase = 'train', k_fold = k_fold, fold = fold, radiomics_dir= False) # r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
# valDataset = Classificaiton_Dataset(phase = 'val', k_fold = k_fold, fold = fold, radiomics_dir= False) #r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
# #testDataset =  Classificaiton_Dataset_test(phase = 'test', k_fold = k_fold, fold = fold, radiomics_dir= r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
# #testDataset = Classificaiton_Dataset(phase = 'val', k_fold = k_fold, fold = 0, radiomics_dir= False) #r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
# testDataset = Classificaiton_Dataset(phase = 'test', radiomics_dir= False)

trainDataset = MMotu_Classificaiton_Dataset(phase = 'train')
valDataset = MMotu_Classificaiton_Dataset(phase = "val")


train_loader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last= True,
            #persistent_workers=True,
        )

val_loader = DataLoader(
            valDataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last= True,
            #persistent_workers=True,
        )

# Initialize Callbacks
early_stopping = EarlyStopping(monitor="validation/loss", patience=100, mode="min")
checkpoint_callback = ModelCheckpoint(
        monitor="validation/combined_score",
        mode="max",
        dirpath=f"checkpoints/washu2_classification/{commit_log}/{fold}",
        filename="best-checkpoint-{epoch:02d}-{validation/auc:.4f}",
        save_top_k=1,
    )


# Initialize Trainer
trainer = pl.Trainer(
        logger=wandb_logger,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu",
    )


model = BinaryClassification(input_dim= 64, num_classes= 8,  encoder_weight_path = r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt", radiomics= False)
#trainer.fit(model, train_loader, val_loader)


# Get predictions on the test set for ROC curve
# Get predictions on the test set
y_true, y_probs = model.get_predictions_on_loader(val_loader)
#Load best model from checkpoint after training
best_model_path = checkpoint_callback.best_model_path
#best_model_path = r"checkpoints\washu2_classification\f1e8fe5\1\best-checkpoint-epoch=122-validation\auc=0.8749.ckpt"
best_model = BinaryClassification.load_from_checkpoint(
    best_model_path,
    input_dim=64,
    num_classes=8,
    encoder_weight_path=r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt"
)
best_model.eval()
best_model.freeze()

# # Get predictions using the best model
# y_true, y_probs = best_model.get_predictions_on_loader(test_loader)

# Plot and log the ROC for this fold
plot_multilabel_roc_curve(y_true, y_probs, fold_idx=fold + 1, wandb_logger=wandb_logger)
# Finish WandB Run
wandb_logger.experiment.unwatch(model)
wandb.finish()

