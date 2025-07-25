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
    
# class AttentionFusion(nn.Module):
#     def __init__(self, embed_dim, num_modalities=4):
#         super().__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.scale = embed_dim ** 0.5

#     def forward(self, x):  # x: (B, M, D) where M=modalities, D=embed_dim
#         Q = self.query(x)  # (B, M, D)
#         K = self.key(x)    # (B, M, D)
#         V = self.value(x)  # (B, M, D)

#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, M, M)
#         attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, M, M)
#         fused = torch.matmul(attn_weights, V)                            # (B, M, D)

#         # Optionally pool across modalities (e.g., mean)
#         return fused.mean(dim=1)  # (B, D)

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # (B, M, D)
        Q = self.query(x)  # (B, M, D)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, M, M)
        attn_weights = self.softmax(attn_scores)  # (B, M, M)
        fused = torch.bmm(attn_weights, V)  # (B, M, D)
        fused = fused.mean(dim=1)  # Aggregate across modalities
        return fused


# class MultiModalCancerClassifierWithAttention(nn.Module):
#     def __init__(self, out_dim=1, fusion_dim=None, backbone_name='resnet18', dropout_prob=0.0, use_layernorm=True):
#         super().__init__()
#         self.num_modalities = 3
#         self.dropout_prob = dropout_prob
#         self.use_layernorm = use_layernorm

#         # best backbone with .94 auc 
#         best_backbone_checkpoint_path = "checkpoints/imagenet_model_mmotu/1548ce4/best_model.pth"


#         # # Create modality-specific backbones
#         # self.backbones = nn.ModuleList([
#         #     timm.create_model(backbone_name, pretrained=True, num_classes=out_dim)
#         #     for _ in range(self.num_modalities)
#         # ])
        

#         # Initialize modality-specific backbones
#         self.backbones = nn.ModuleList()
#         for i in range(self.num_modalities):
#             backbone = timm.create_model(backbone_name, pretrained=False, num_classes=out_dim)
#             checkpoint_path = best_backbone_checkpoint_path
#             self._load_backbone_weights(backbone, checkpoint_path)
#             if i == 0:
#                 self._freeze_backbone(backbone)  # Freeze backbone 1
#             self.backbones.append(backbone)

#         # Set a sensible default fusion dim

#         self.backbone_out_dim = out_dim #self.backbones[0].num_features

#         if fusion_dim is None:
#             fusion_dim = min(512, self.backbone_out_dim)  # Keep dimension smaller for fusion stability

#         # Projection layers
#         self.projs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.backbone_out_dim, fusion_dim),
#                 nn.LayerNorm(fusion_dim) if use_layernorm else nn.Identity(),
#                 nn.ReLU()
#             )
#             for _ in range(self.num_modalities)
#         ])

#         # Fusion attention
#         self.attn_fusion = AttentionFusion(embed_dim=fusion_dim, num_modalities=self.num_modalities)

#         # Deeper classifier for richer features
#         self.classifier = nn.Sequential(
#             nn.Linear(fusion_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, out_dim)
#         )
#     def _load_backbone_weights(self, model, checkpoint_path):
#         try:
#             state_dict = torch.load(checkpoint_path, map_location='cpu')
#             if 'state_dict' in state_dict:
#                 state_dict = state_dict['state_dict']  # For checkpoints saved with torch.save({"state_dict": model.state_dict()})
#             # Remove any 'module.' prefix if the model was saved using DataParallel
#             new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#             model.load_state_dict(new_state_dict, strict=False)
#             print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
#         except Exception as e:
#             print(f"[ERROR] Failed to load checkpoint from {checkpoint_path}: {e}")
#     def _freeze_backbone(self, model):
#         for param in model.parameters():
#             param.requires_grad = False
#         print("[INFO] Backbone 1 frozen (no gradient updates).")

#     def forward(self, imgs):  # imgs: list of 3 tensors
#         B = imgs[0].shape[0]
#         device = imgs[0].device

#         fused_feats = []
#         for i in range(self.num_modalities):
#             x = imgs[i]

#             # Modality dropout
#             if self.training and random.random() < self.dropout_prob and i != 0:
#                 fused_feats.append(torch.zeros(B, self.projs[i][0].out_features, device=device))
#                 continue

#             # Convert grayscale to RGB
#             if x.shape[1] == 1:
#                 x = x.repeat(1, 3, 1, 1)

#             feat = self.backbones[i](x)  # (B, backbone_out_dim)
#             proj_feat = self.projs[i](feat)  # (B, fusion_dim)
#             fused_feats.append(proj_feat)

#         fused_stack = torch.stack(fused_feats, dim=1)  # (B, M, fusion_dim)
#         fused_output = self.attn_fusion(fused_stack)   # (B, fusion_dim)
#         out = self.classifier(fused_output)            # (B, out_dim)
#         return out


class MultiModalClassifierWithLogitFusion(nn.Module):
    def __init__(self, out_dim=1, fusion_dim=64, backbone_name='resnet18', dropout_prob=0.1, use_layernorm=True):
        super().__init__()
        self.num_modalities = 3
        self.dropout_prob = dropout_prob
        self.use_layernorm = use_layernorm

        # Assume 8-class classifier for each backbone
        self.backbone_num_classes = 8

        # Checkpoint paths per modality
        self.checkpoints = [
            "checkpoints/imagenet_model_mmotu/1548ce4/best_model.pth",
            "checkpoints/imagenet_model_mmotu/1548ce4/best_model.pth",
            "checkpoints/imagenet_model_mmotu/1548ce4/best_model.pth"
        ]

        self.backbones = nn.ModuleList()
        for i in range(self.num_modalities):
            backbone = timm.create_model(backbone_name, pretrained=False, num_classes=self.backbone_num_classes)
            self._load_backbone_weights(backbone, self.checkpoints[i])
            #if i == 0:
            #    self._freeze_backbone(backbone)
            self.backbones.append(backbone)

        # Project 8-class logits to fusion_dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone_num_classes, fusion_dim),
                nn.LayerNorm(fusion_dim) if use_layernorm else nn.Identity(),
                nn.GELU()
            ) for _ in range(self.num_modalities)
        ])

        # Transformer-based attention fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=4, dim_feedforward=2*fusion_dim, dropout=0.1, batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

    def _load_backbone_weights(self, model, checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load weights from {checkpoint_path}: {e}")

    def _freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False
        print("[INFO] Backbone 1 frozen.")

    def forward(self, imgs):  # imgs: list of 3 modality tensors
        B = imgs[0].shape[0]
        device = imgs[0].device
        fused_feats = []
        logits_list = []

        for i in range(self.num_modalities):
            x = imgs[i]
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            # Random modality dropout (optional)
            if self.training and random.random() < self.dropout_prob and i != 0:
                fused_feats.append(torch.zeros(B, self.projs[i][0].out_features, device=device))
                continue

            logits = self.backbones[i](x)  # Output shape: (B, 8)
            logits_list.append(logits)
            proj_feat = self.projs[i](logits)  # Project to fusion_dim
            fused_feats.append(proj_feat)

        fused_stack = torch.stack(fused_feats, dim=1)  # Shape: (B, M, fusion_dim)
        fused_output = self.fusion_transformer(fused_stack)  # Shape: (B, M, fusion_dim)

        # Aggregate over modalities
        fused_final = fused_output.mean(dim=1)  # (B, fusion_dim)
        out = self.classifier(fused_final)      # (B, out_dim)
        return out, logits_list

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

        self.fusion_model = MultiModalClassifierWithLogitFusion(out_dim=num_classes, backbone_name= "resnet50", dropout_prob= 0)

        # Losses for multi-label (use BCE with logits)
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0] * num_classes))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x_sdf = self.sdf_model(x.mean(dim = 1, keepdim = True))
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.8, -0.75).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.45).item()
        center_thresh = torch.empty(1).uniform_(0.1, 0.25).item()

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask = (x_sdf < center_thresh)

        x3 = x * boundary_mask
        x4 = x * center_mask

        output, auxiliary_heads = self.fusion_model([x, x3, x4])  # Output: (B, num_classes)

        return output, auxiliary_heads 

    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score, aux_outs = self.forward(x)
            loss = self.loss_fn(score, y) #+ 0.5 * self.loss_fn2(score, y)

            for aux_out in aux_outs:
                loss += self.loss_fn(aux_out, y)  #aux_loss_weight

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
                    scores,_ = self.forward(x, x2)

                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    
class MultiClassificationTorch_Imagenet(nn.Module): 
    def __init__(self, num_classes=8, backbone_name = 'resnet50'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=8)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0] * num_classes))

    def forward(self, x, x2_radiomics=None):
        x = self.backbone(x)
        return x

    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()  # Ensure targets are float for BCE loss
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y) + sum(self.loss_fn(t, y) for t in tails)
        else:
            score = self.forward(x)
            loss = self.loss_fn(score, y) #+ 0.5 * self.loss_fn2(score, y)

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
    
class BinaryClassificationTorch(nn.Module):
    def __init__(self, input_dim=64, output_size = 5, num_classes=1, radiomics=False, radiomics_dim=463,
                 encoder_weight_path=None, sdf_model_path=None):
        super().__init__()

        self.input_size = input_dim
        self.hidden_sizes = [512, 128, 64, 32]
        self.hidden_sizes2 = [64, 32]
        self.output_size = output_size 
        self.radiomics = radiomics

        self.sdf_model = SDFModel()
        self.sdf_model.load_state_dict(torch.load(sdf_model_path))
        for p in self.sdf_model.parameters(): p.requires_grad = False

        self.fusion_model = MultiModalCancerClassifierWithAttention()

        self.loss_fn = FocalLoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x_sdf = self.sdf_model(x)
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.45, -0.15).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.65).item()
        center_thresh = torch.empty(1).uniform_(0.3, 0.35).item()

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask = (x_sdf < center_thresh)

        x3 = x * boundary_mask
        x4 = x * center_mask

        output = self.fusion_model([x, x3, x4])

        return output


    def compute_loss(self, x, y, x2_rad=None):
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y.float()) + sum(self.loss_fn(t, y.float()) for t in tails)
        else:
            score = self.forward(x)
            loss = (self.loss_fn(score, y.float()) * 0.5 +
                    self.loss_fn2(score, y.float()) * 0.5)
                   
        return loss

    def predict_on_loader(self, dataloader):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device  # Automatically detect model's device

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
                                 encoder_weight_path = r"checkpoints/normtverskyloss_binary_segmentation/a56e77a/best-checkpoint-epoch=77-validation/loss=0.2544.ckpt", 
                                 sdf_model_path= r"checkpoints/deeplabv3_sdf_randomcrop/model_20250711_201243/epoch_84",
                                 radiomics= False)
    
    #model = MultiClassificationTorch_Imagenet()

    print(model)
    model.eval()
    print(model(torch.randn(1, 3,384, 384))) #, torch.randn(1, 1,256, 256)).shape)
