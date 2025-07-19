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
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List

# Dummy Attention blocks (replace with actual implementations)
class AttentionFusion(nn.Module):
    def __init__(self, dim, num_inputs):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim * num_inputs, dim), nn.ReLU())
    def forward(self, x):
        B, N, D = x.shape
        return self.fc(x.view(B, -1))

class CrossAttention(nn.Module):
    def __init__(self, dim, num_inputs):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
    def forward(self, x):
        # x: (B, N, D)
        out, _ = self.attn(x, x, x)
        return out.mean(dim=1)  # reduce across modalities
    
class SEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.ReLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid()
        )

        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        se_weight = self.se(x)
        x = x * se_weight  # Channel-wise reweighting
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerFFNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.final_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x + residual  # Add & Norm
        x = self.norm2(x)
        return self.final_classifier(x)


# Main model
class MultiStageProgressiveFusionModel(nn.Module):
    def __init__(self,
                 out_dim=1,
                 fusion_dim=256,
                 backbone_name='resnet50',
                 dropout_prob=0.0,
                 num_modalities=3,
                 fusion_start_stage=2,
                 fusion_block_type= 'crossattention', # 'attention',  # or
                 use_auxiliary_heads=True, classifier_type = "se"):
        super().__init__()

        self.num_modalities = num_modalities
        self.dropout_prob = dropout_prob
        self.fusion_start_stage = fusion_start_stage
        self.use_auxiliary_heads = use_auxiliary_heads

        # Create backbone for each modality
        self.backbones = nn.ModuleList([
            timm.create_model(backbone_name, pretrained=True, features_only=True)
            for _ in range(self.num_modalities)
        ])
        self.num_stages = len(self.backbones[0].feature_info)

        # Projection layers for each stage of each modality
        self.stage_projs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(info['num_chs'], fusion_dim, kernel_size=1)
                for info in self.backbones[0].feature_info
            ]) for _ in range(self.num_modalities)
        ])

        # Select fusion block type
        fusion_class = AttentionFusion if fusion_block_type == 'attention' else CrossAttention

        # Fusion blocks from fusion_start_stage onwards
        self.fusion_blocks = nn.ModuleList([
            fusion_class(fusion_dim, num_modalities + (1 if i > fusion_start_stage else 0))
            if i >= fusion_start_stage else None
            for i in range(self.num_stages)
        ])

        # Optional auxiliary heads for stage-wise loss
        if self.use_auxiliary_heads:
            self.auxiliary_heads = nn.ModuleList([
                nn.Linear(fusion_dim, out_dim) if i >= fusion_start_stage else None
                for i in range(self.num_stages)
            ])

        # # Final classifier

        hidden_dim = 256

        if classifier_type == 'se':
            self.classifier = SEClassifier(fusion_dim * (self.num_stages - fusion_start_stage), hidden_dim, out_dim)
        elif classifier_type == 'transformer':
            self.classifier = TransformerFFNClassifier(fusion_dim * (self.num_stages - fusion_start_stage), hidden_dim, out_dim)
        else:
            self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * (self.num_stages - fusion_start_stage), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, imgs: List[torch.Tensor]):
        B = imgs[0].shape[0]
        device = imgs[0].device
        stage_outputs = []
        aux_outputs = []

        # Extract features for each modality
        modality_feats = []
        for i in range(self.num_modalities):
            x = imgs[i]
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            feats = self.backbones[i](x)
            proj_feats = [self.stage_projs[i][j](f) for j, f in enumerate(feats)]
            modality_feats.append(proj_feats)

        prev_fused = None
        for stage in range(self.num_stages):
            if stage < self.fusion_start_stage:
                continue  # Skip early stages

            # Gather features for current stage
            stage_feats = []
            for i in range(self.num_modalities):
                pooled = F.adaptive_avg_pool2d(modality_feats[i][stage], 1).squeeze(-1).squeeze(-1)
                stage_feats.append(pooled)

            if prev_fused is not None:
                stage_feats.append(prev_fused)

            fused_input = torch.stack(stage_feats, dim=1)  # (B, M+1, D)
            fused = self.fusion_blocks[stage](fused_input)  # (B, D)
            prev_fused = fused
            stage_outputs.append(fused)

            if self.use_auxiliary_heads:
                aux_head = self.auxiliary_heads[stage]
                if aux_head is not None:
                    aux_outputs.append(aux_head(fused))

        # Final classification
        fusion_vector = torch.cat(stage_outputs, dim=1)
        out = self.classifier(fusion_vector)

        if self.use_auxiliary_heads:
            return out, aux_outputs  # main + stage-wise outputs
        return out

    

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

        self.fusion_model = MultiStageProgressiveFusionModel(out_dim=num_classes, backbone_name= "resnet50", dropout_prob= 0)

        # Losses for multi-label (use BCE with logits)
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0] * num_classes))

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
            # score = self.forward(x)
            # loss = 0.5 * self.loss_fn(score, y) + 0.5 * self.loss_fn2(score, y)

            main_out, aux_outs = self.forward(x)
            loss = self.loss_fn(main_out, y)
            for aux_out in aux_outs:
                loss += self.loss_fn(aux_out, y) * 0.1 #aux_loss_weight

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
                    scores, _ = self.forward(x)
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
