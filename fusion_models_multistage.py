import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F 
import torch 
import torchvision.utils as vutils
import wandb
import torch
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchvision.models import resnet18 , ResNet18_Weights
import torchvision 


import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_sigmoid = torch.clamp(inputs_sigmoid, self.eps, 1 - self.eps)

        if self.clip is not None and self.clip > 0:
            inputs_sigmoid = (inputs_sigmoid - self.clip).clamp(min=0, max=1)

        targets = targets.float()
        loss_pos = targets * torch.log(inputs_sigmoid) * (1 - inputs_sigmoid) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(1 - inputs_sigmoid) * inputs_sigmoid ** self.gamma_neg
        loss = -loss_pos - loss_neg
        return loss.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class SDFModel(nn.Module):
    def __init__(self):
        super(SDFModel, self).__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.backbone(x)        # Output shape: (B, 1, H, W)
        x = self.activation(x)      # Output in [-1, 1]
        return x
    

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # (B, M, D)
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        return attn_out.mean(dim=1)  # (B, D)


class MultiModalFusionBlock(nn.Module):
    def __init__(self, in_dim, fusion_dim, num_modalities):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(in_dim, fusion_dim) for _ in range(num_modalities)
        ])
        self.attn_fusion = AttentionFusion(fusion_dim, num_modalities)

    def forward(self, features):  # List of (B, C)
        proj_feats = [proj(f) for proj, f in zip(self.projectors, features)]
        fused = self.attn_fusion(torch.stack(proj_feats, dim=1))  # (B, fusion_dim)
        return fused


class MultiModalCancerClassifierMultiStage(nn.Module):
    def __init__(self, out_dim=1, fusion_dim=256, backbone_name='resnet18', dropout_prob=0.0):
        super().__init__()
        self.num_modalities = 3
        self.dropout_prob = dropout_prob

        # Initialize modality-specific ResNet backbones
        self.backbones = nn.ModuleList([
            timm.create_model(backbone_name, pretrained=True, features_only=True)
            for _ in range(self.num_modalities)
        ])
        self.num_stages = len(self.backbones[0].feature_info)

        # Fusion blocks at different stages
        self.fusion_blocks = nn.ModuleList([
            MultiModalFusionBlock(
                in_dim=self.backbones[0].feature_info[i]['num_chs'],
                fusion_dim=fusion_dim,
                num_modalities=self.num_modalities
            ) for i in range(self.num_stages)
        ])

        # Skip connection projection
        self.skip_project = nn.Linear(self.num_stages * fusion_dim, fusion_dim)

        # # Classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(fusion_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, out_dim)
        # )

        # Deeper classifier for richer features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

    def forward(self, imgs):  # imgs: list of 3 tensors, each (B, 1, H, W)
        B = imgs[0].shape[0]
        device = imgs[0].device
        stage_outputs = [[] for _ in range(self.num_stages)]

        # Pass each modality through its backbone
        for i in range(self.num_modalities):
            x = imgs[i]

            if self.training and random.random() < self.dropout_prob and i != 0:
                for s in range(self.num_stages):
                    stage_outputs[s].append(torch.zeros(B, self.backbones[0].feature_info[s]['num_chs'], device=device))
                continue

            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            feats = self.backbones[i](x)  # List of feature maps at each stage
            for s in range(self.num_stages):
                # Global average pooling each stage output
                pooled_feat = F.adaptive_avg_pool2d(feats[s], 1).squeeze(-1).squeeze(-1)
                stage_outputs[s].append(pooled_feat)

        # Fuse at each stage and collect for skip connection
        fused_features = []
        for s in range(self.num_stages):
            fused_feat = self.fusion_blocks[s](stage_outputs[s])  # (B, fusion_dim)
            fused_features.append(fused_feat)

        # Skip connection via concatenation → projection
        fused_cat = torch.cat(fused_features, dim=1)  # (B, num_stages * fusion_dim)
        final_rep = self.skip_project(fused_cat)      # (B, fusion_dim)

        return self.classifier(final_rep)

class MultiClassificationTorch(nn.Module): 
    def __init__(self, input_dim=64, num_classes=8, radiomics=False, radiomics_dim=463, 
                 encoder_weight_path=None, sdf_model_path=None):
        super().__init__()

        self.input_size = input_dim
        self.num_classes = num_classes
        self.radiomics = radiomics

        self.sdf_model = SDFModel()
        self.sdf_model.load_state_dict(torch.load(sdf_model_path))
        for p in self.sdf_model.parameters(): 
            p.requires_grad = False

        self.fusion_model = MultiModalCancerClassifierMultiStage(out_dim=num_classes, backbone_name= "resnet50", dropout_prob= 0)

        # Losses for multi-label (use BCE with logits)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0] * num_classes))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x_sdf = self.sdf_model(x.mean(dim = 1, keepdim = True))
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.7, -0.6).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.45).item()
        center_thresh = torch.empty(1).uniform_(0.1, 0.25).item()

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask = (x_sdf < center_thresh)

        x3 = x * boundary_mask
        x4 = x * center_mask

        output = self.fusion_model([x, x3, x4])  # Output: (B, num_classes)

        return output

    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score = self.forward(x)
            loss = 0.5 * self.loss_fn(score, y) + 0.5 * self.loss_fn2(score, y)

        return loss

    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores = self.forward(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores = self.forward(x, x2)

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()

if __name__ == "__main__":
    # model = MultiModalCancerClassifierWithAttention()
    # img1 = torch.randn(8, 1, 256, 256)
    # img2 = torch.randn(8, 1, 256, 256)
    # img3 = torch.randn(8, 1, 256, 256)
    # img4 = torch.randn(8, 1, 256, 256)

    # output = model([img1, img2, img3]) #, img4])  # shape: (8,)

    # print(output)



    model = MultiClassificationTorch(input_dim= 64, num_classes= 8,  
                                 encoder_weight_path = r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt", 
                                 sdf_model_path= r"checkpoints\deeplabv3_sdf_randomcrop\model_20250711_201243\epoch_84",
                                 radiomics= False)
    
    #model = MultiClassificationTorch_Imagenet()

    print(model)
    model.eval()
    print(model(torch.randn(1, 3,384, 384))) #, torch.randn(1, 1,256, 256)).shape)
