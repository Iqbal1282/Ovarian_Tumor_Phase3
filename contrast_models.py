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
from einops import rearrange, repeat
import timm
from utils import contrastive_loss_3modal


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

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities=4):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):  # x: (B, M, D) where M=modalities, D=embed_dim
        Q = self.query(x)  # (B, M, D)
        K = self.key(x)    # (B, M, D)
        V = self.value(x)  # (B, M, D)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, M, M)
        attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, M, M)
        fused = torch.matmul(attn_weights, V)                            # (B, M, D)

        # Optionally pool across modalities (e.g., mean)
        return fused.mean(dim=1)  # (B, D)

class MultiModalCancerClassifierWithAttention(nn.Module):
    def __init__(self, out_dim=1, fusion_dim=256, backbone_name='resnet18', dropout_prob=0.25):
        super().__init__()
        self.num_modalities = 3
        self.dropout_prob = dropout_prob

        # Independent backbones
        self.backbones = nn.ModuleList([
            timm.create_model(backbone_name, pretrained=True, num_classes=0)
            for _ in range(self.num_modalities)
        ])
        self.backbone_out_dim = self.backbones[0].num_features

        # Project each modality to common fusion_dim
        self.projs = nn.ModuleList([
            nn.Linear(self.backbone_out_dim, fusion_dim)
            for _ in range(self.num_modalities)
        ])

        # Fusion module: attention-based
        self.attn_fusion = AttentionFusion(embed_dim=fusion_dim, num_modalities=self.num_modalities)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

    def forward(self, imgs):  # imgs: list of 4 tensors, each (B, 1, 256, 256)
        B = imgs[0].shape[0]
        device = imgs[0].device

        fused_feats = []
        for i in range(self.num_modalities):
            x = imgs[i]

            # Modality dropout (like CoAtNet)
            if self.training and random.random() < self.dropout_prob:
                # Replace with zero vector
                fused_feats.append(torch.zeros(B, self.projs[i].out_features, device=device))
                continue

            # Convert grayscale → RGB
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            feat = self.backbones[i](x)         # (B, backbone_out_dim)
            proj_feat = self.projs[i](feat)     # (B, fusion_dim)
            fused_feats.append(proj_feat)

        # Stack and fuse: shape (B, M, D)
        fused_stack = torch.stack(fused_feats, dim=1)  # (B, 4, fusion_dim)
        fused_output = self.attn_fusion(fused_stack)   # (B, fusion_dim)

        out = self.classifier(fused_output)            # (B, 1)
        return out.squeeze()

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
        # check if x.shape[1] == 1
        # then repeat to 3 channels
        if x.shape[1] == 3:
            # Convert to grayscale by averaging channels
            x = x.mean(dim=1, keepdim=True)
        x = self.backbone(x)        # Output shape: (B, 1, H, W)
        x = self.activation(x)      # Output in [-1, 1]
        return x
    

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
        self.loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x_sdf = self.sdf_model(x)
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.45, -0.15).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.65).item()
        center_thresh = torch.empty(1).uniform_(0.1, 0.25).item()

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
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # x: [B, 1, H, W] → [B, embed_dim, H//P, W//P] → [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        attn = self.pool(x).view(b, c)
        attn = self.fc(attn).view(b, c, 1, 1)
        return x * attn

class AdvancedPatchEmbed(nn.Module):
    """Deep, multi-scale patch embedder with residual connections"""
    def __init__(self, img_size=448, patch_size=32, in_chans=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Multi-scale stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, padding=1, groups=embed_dim // 4, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.block1 = self._make_block(embed_dim // 2, embed_dim // 2, dilation=1)
        self.block2 = self._make_block(embed_dim // 2, embed_dim, dilation=2)
        self.block3 = self._make_block(embed_dim, embed_dim, dilation=4)
        
        # Projection shortcuts for channel changes
        self.shortcut2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, bias=False)
        
        # SE block and final layers
        self.se = SEBlock(embed_dim, reduction=16)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        
    def _make_block(self, in_dim, out_dim, dilation):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=dilation, dilation=dilation, groups=in_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        # Block 1: identity residual (same channels)
        x = F.relu(self.block1(x) + x)
        
        # Block 2: projection residual (channels increase)
        x = F.relu(self.block2(x) + self.shortcut2(x))
        
        # Block 3: identity residual (same channels)
        x = F.relu(self.block3(x) + x)
        
        # Channel attention and projection
        x = self.se(x)
        x = self.proj(x)
        
        # Convert to patch sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.norm(x)

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for modality embeddings"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeds_list):
        """
        Args:
            embeds_list: List of modality embeddings [B, D] for each modality
        Returns:
            contrastive loss value
        """
        if len(embeds_list) < 2:
            return torch.tensor(0.0, device=embeds_list[0].device)
            
        batch_size = embeds_list[0].shape[0]
        device = embeds_list[0].device
        
        # Normalize embeddings
        normalized_embeds = [F.normalize(e, dim=-1) for e in embeds_list]
        
        total_loss = 0.0
        num_pairs = 0
        
        # Contrast every pair of modalities
        for i in range(len(normalized_embeds)):
            for j in range(i + 1, len(normalized_embeds)):
                # Positive pairs: same sample across modalities
                # Negative pairs: all other samples
                logits = torch.mm(normalized_embeds[i], normalized_embeds[j].T) / self.temperature
                
                # Labels: positive pairs are on the diagonal
                labels = torch.arange(batch_size, dtype=torch.long, device=device)
                
                # Symmetric InfoNCE loss
                loss_i = F.cross_entropy(logits, labels)
                loss_j = F.cross_entropy(logits.T, labels)
                
                total_loss += (loss_i + loss_j) / 2
                num_pairs += 1
        
        return total_loss / num_pairs
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange

# ------------------------------------------------------------------
#  A deeper hierarchical patch embedder  (4-stage mini-ViT stem)
# ------------------------------------------------------------------
class DeepPatchEmbed(nn.Module):
    """
    Hierarchical patch embedder whose *output* spatial size is exactly
    (H // patch_size) × (W // patch_size), so that
    N = (H // patch_size) * (W // patch_size)  matches the value
    computed in the main model (used for pos_embed, etc.).
    patch_size must be a power of two and ≥ 2.
    """
    def __init__(self, img_size=448, patch_size=32, embed_dim=256,
                 act=nn.GELU, drop_path=0.0):
        super().__init__()
        assert (patch_size & (patch_size - 1)) == 0, "patch_size must be power of 2"
        self.patch_size = patch_size
        self.num_stages = (patch_size & -patch_size).bit_length() - 1   # log2(patch_size)
        # number of 2× down-sampling stages needed
        dims = [embed_dim // (2 ** (self.num_stages - i)) for i in range(self.num_stages)]
        dims[-1] = embed_dim   # exact last dim

        layers = []
        in_ch = 1
        for i, d in enumerate(dims):
            layers += [nn.Conv2d(in_ch, d, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(1, d), act(),
                       Block(d, mlp_ratio=3, act=act,
                             drop_path=drop_path if isinstance(drop_path, float) else drop_path[i])]
            in_ch = d
        self.stem = nn.Sequential(*layers)
        # final 1×1 projection (already embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)

        # compute num_patches once
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.stem(x)          # B, C, H/ps, W/ps
        x = self.proj(x)
        return rearrange(x, 'b c h w -> b (h w) c')

# ------------------------------------------------------------------
#  Basic residual block used inside each stage
# ------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, act=nn.GELU, drop_path=0.):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.GroupNorm(1, dim)
        self.conv1 = nn.Conv2d(dim, mlp_dim, 1)
        self.act   = act()
        self.conv2 = nn.Conv2d(mlp_dim, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return shortcut + self.drop_path(x)

# ------------------------------------------------------------------
#  Replace the shallow CNNs inside your original class
# ------------------------------------------------------------------
class ThreeModalTransformerClassifier(nn.Module):
    def __init__(self,
                 img_size=448,
                 patch_size=32,
                 embed_dim=256,
                 num_heads=4,
                 num_layers=6,
                 num_classes=8,
                 dropout=0.1,
                 common_root_patcher=False,
                 deep_embed_drop_path=0.0):
        super().__init__()

        # --- your frozen SDF model remains ---
        self.sdf_model = SDFModel()
        sdf_model_path = r"./checkpoints/sdf_model/epoch_84"
        self.sdf_model.load_state_dict(torch.load(sdf_model_path, map_location='cpu'))
        for p in self.sdf_model.parameters():
            p.requires_grad = False

        # --- bookkeeping ---
        self.common_root_patcher = common_root_patcher
        self.patch_dim = (img_size // patch_size) ** 2
        self.patch_embed_dim = embed_dim

        # --- choose patcher ---
        if common_root_patcher:
            self.common_patcher = PatchEmbed(img_size=img_size,
                                             patch_size=patch_size,
                                             in_chans=1,
                                             embed_dim=embed_dim)
        else:
            # NEW: deeper embedder for each modality
            self.so2_embed = DeepPatchEmbed(img_size=img_size,
                                patch_size=patch_size,
                                embed_dim=embed_dim,
                                drop_path=deep_embed_drop_path)
            self.thb_embed = DeepPatchEmbed(img_size=img_size,
                                            patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            drop_path=deep_embed_drop_path)
            self.us_embed  = DeepPatchEmbed(img_size=img_size,
                                            patch_size=patch_size,
                                            embed_dim=embed_dim,
                                            drop_path=deep_embed_drop_path)

        # --- everything below is unchanged ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + 3 * self.patch_dim, embed_dim))
        self.dropout   = nn.Dropout(dropout)
        self.modality_tokens = nn.Parameter(torch.randn(3, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.cls_return = False
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    # ------------- forward and helpers stay identical -------------
    def forward(self, x, return_patches=False):
        # if x has 3 channels, convert to grayscale
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        x_sdf = self.sdf_model(x)
        x_sdf = self.normalize_sdf(x_sdf)

        if self.training:
            lower_thresh = torch.empty(1).uniform_(-0.45, -0.15).item()
            upper_thresh = torch.empty(1).uniform_(0.35, 0.65).item()
            center_thresh = torch.empty(1).uniform_(0.1, 0.25).item()
        else:
            lower_thresh, upper_thresh, center_thresh = -0.3, 0.5, 0.2

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask   = (x_sdf < center_thresh)
        so2 = x * boundary_mask
        thb = x * center_mask
        B = so2.size(0)

        if self.common_root_patcher:
            so2_patches = self.common_patcher(so2)
            thb_patches = self.common_patcher(thb)
            us_patches  = self.common_patcher(x)
        else:
            so2_patches = self.so2_embed(so2)
            thb_patches = self.thb_embed(thb)
            us_patches  = self.us_embed(x)

        so2_patches += self.modality_tokens[0]
        thb_patches += self.modality_tokens[1]
        us_patches  += self.modality_tokens[2]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, so2_patches, thb_patches, us_patches], dim=1)
        x += self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x) if self.training else x
        x = self.transformer(x)
        cls_output = self.norm(x[:, 0])

        if return_patches:
            return cls_output, self.head(cls_output), [so2_patches, thb_patches, us_patches]
        
        if self.cls_return:
            return cls_output, self.head(cls_output)
        
        return self.head(cls_output)

    # ------------- normalize_sdf / compute_loss / predict_on_loader -------------
    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    # (keep compute_loss & predict_on_loader exactly as before)
    def compute_loss(self, x, y, x2_rad=None):
        y = y.float()
        score = self.forward(x, return_patches=True)
        loss = self.loss_fn(score[1], y) + 0.1 * contrastive_loss_3modal(*score[2])
        return loss

    def predict_on_loader(self, dataloader, threshold=0.5):
        self.eval()
        all_probs, all_targets = [], []
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                scores = self.forward(x)
                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())
        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()

if __name__ == "__main__":
    model = ThreeModalTransformerClassifier(num_classes=5)

    x = torch.randn(2, 1, 448, 448)  # Example input

    # During inference
    logits = model(x)

    print("Logits shape:", logits.shape)  # Should be [2, num_classes]
