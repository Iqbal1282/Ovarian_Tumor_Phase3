import torch 
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # Probability of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Ensure predictions are in [0,1]
        intersection = torch.sum(preds * targets)
        dice_score = (2.0 * intersection + self.smooth) / (torch.sum(preds) + torch.sum(targets) + self.smooth)
        return 1 - dice_score  # Minimize Dice Loss

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.focal(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for False Negatives
        self.beta = beta  # Weight for False Positives
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        tp = torch.sum(preds * targets)
        fp = torch.sum(preds * (1 - targets))
        fn = torch.sum((1 - preds) * targets)

        tversky_score = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky_score
    

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        return self.alpha * self.focal(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)
