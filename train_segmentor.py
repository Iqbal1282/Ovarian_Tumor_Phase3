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
from models import Compress_Segmentor
from dataset import SegmentImageDataset
from dataset import transform_img, transform_mask
import subprocess
import re 

# # Initialize WandB
# wandb.init(project="MRI_Poject", entity="iqbal1282")

max_epochs = 100
min_epochs = 1
batch_size = 32 
check_val_every_n_epoch = 3
num_workers = 0


try: 
    # Get the latest Git commit message
    commit_string = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()
    commit_string = re.sub(r'\W+', '_', commit_string) 
    commit_log = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    # Initialize WandB Logger
    run_name = f'Segmentation_log_"{commit_log}"_commit_"{commit_string}"_{datetime.now()}'
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Initialize WandB Logger
#run_name = f"Segmentation_intial_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
wandb_logger = WandbLogger(
    log_model=False, project="Ovarian_Tumor_Segmentation", name=run_name
)


trainDataset = SegmentImageDataset(phase = 'train', img_transform=transform_img, mask_transform=transform_mask)
testDataset = SegmentImageDataset(phase = 'test',  img_transform=transform_img, mask_transform=transform_mask)

train_loader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            #persistent_workers=True,
        )
test_loader = DataLoader(
            testDataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            #persistent_workers=True,
        )



# WandB Logger
#wandb_logger = WandbLogger(project="pytorch-lightning-mnist")

# Initialize Callbacks
early_stopping = EarlyStopping(monitor="validation/loss", patience=40, mode="min")
checkpoint_callback = ModelCheckpoint(
        monitor="validation/loss",
        dirpath=f"checkpoints/normtverskyloss_binary_segmentation/{commit_log}/",
        filename="best-checkpoint-{epoch:02d}-{validation/loss:.4f}",
        save_top_k=1,
        mode="min",
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


model = Compress_Segmentor()
trainer.fit(model, train_loader, test_loader)

# Finish WandB Run
wandb_logger.experiment.unwatch(model)
wandb.finish()
