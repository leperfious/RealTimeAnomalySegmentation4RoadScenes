import torch
import torch.nn as nn
import torch.nn.functional as F

# logits - raw outputs from the network without softmax applied
# targets - ground truth class indices, with the shape of [B, H, W]
# CrossEntropyLoss applies softmax by itself.
# criterion = nn.CrossEntropyLoss()
# Here log_softmax being applied manually 
class CrossEntropyLoss(torch.nn.Module): # Cross Entropy Loss

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)



# designed for class imbalance, it down-weights easy examples and 
# focuses training on hard negatives
class FocalLoss(nn.Module): # Focal loss

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets = F.one_hot(targets, num_classes=outputs.size(1)).float()
        targets = targets.permute(0,3,1,2)
        p_t = (outputs * targets).sum(dim=1)
        loss = -self.alpha * (1 - p_t)**self.gamma * torch.log(p_t)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ___________________________________________________________________________________________________________




# it replaces logits with negative distances to enforce large margins 
# and calibrated confidence for OOD detection
class IsoMaxPlusLoss(nn.Module): # Enhanced Isotropy Maximization loss
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLoss, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, outputs, targets):
        distances = -outputs
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = torch.gather(probabilities_for_training, 1, targets.unsqueeze(1))
        probabilities_at_targets = torch.clamp(probabilities_at_targets, min=1e-7)
        loss = -torch.log(probabilities_at_targets).mean()
        return loss




# it normalizes logits before softmax, improving calibration and OOD detection
class LogitNormLoss(nn.Module): # Logit Normalization loss

    def __init__(self, t=0.01):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, outputs, target):
        norms = torch.norm(outputs, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(outputs, norms) / self.t
        return F.cross_entropy(logit_norm, target)
    




# _____________________________________________________________________________________________________________




# Enhanced Isotropy Maximization loss + Focal loss
class IsoMaxPlus_Focal_loss(nn.Module): 
    
    def __init__(self, w_iso=0.5, w_focal=0.5):
        super().__init__()
        self.iso_max = IsoMaxPlusLoss()
        self.focal_loss = FocalLoss()
        
        #weights
        self.w_iso = w_iso
        self.w_focal = w_focal
    
    def forward(self, outputs, targets):
        loss_iso = self.iso_max(outputs, targets)
        loss_focal = self.focal_loss(outputs, targets)
        return self.w_iso * loss_iso + self.w_focal * loss_focal




# Enhanced Isotropy Maximization loss + Cross Entropy L:oss
class IsoMaxPlus_CE_loss(nn.Module): 
    def __init__(self, w_iso=0.5, w_ce=0.5, weight=None):
        super().__init__()
        self.iso_max = IsoMaxPlusLoss()
        self.cross_entropy_loss = CrossEntropyLoss(weight)
        self.w_iso = w_iso
        self.w_ce = w_ce

    def forward(self, outputs, targets):
        loss_iso = self.iso_max(outputs, targets)
        loss_ce = self.cross_entropy_loss(outputs, targets)
        return self.w_iso * loss_iso + self.w_ce * loss_ce
    
    


# Logit Normalization loss + Focal loss
class LogitNorm_Focal_loss(nn.Module): 
    def __init__(self, w_logit=0.5, w_focal=0.5):
        super().__init__()
        self.logit_norm = LogitNormLoss()
        self.focal_loss = FocalLoss()
        self.w_logit = w_logit
        self.w_focal = w_focal

    def forward(self, outputs, targets):
        loss_logit = self.logit_norm(outputs, targets)
        loss_focal = self.focal_loss(outputs, targets)
        return self.w_logit * loss_logit + self.w_focal * loss_focal





# Logit Normalization loss + Cross Entropy loss
class LogitNorm_CE_loss(nn.Module): 
    def __init__(self, w_logit=0.5, w_ce=0.5, weight=None):
        super().__init__()
        self.logit_norm = LogitNormLoss()
        self.cross_entropy_loss = CrossEntropyLoss(weight)
        self.w_logit = w_logit
        self.w_ce = w_ce

    def forward(self, outputs, targets):
        loss_logit = self.logit_norm(outputs, targets)
        loss_ce = self.cross_entropy_loss(outputs, targets)
        return self.w_logit * loss_logit + self.w_ce * loss_ce

