import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module): # Focal loss

    def __init__(self, alpha=None, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        
        
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1) # changes N,C,H,W to N, C, H*W
            input = input.transpose(1,2) # changes N, C, H*W to N, H*W, C
            input = input.contiguous().view(-1, input.size(2)) # N, H*W, C to N*H*W, C
        target = target.view(-1,1)
        
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        
        
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()        
        else:
            return loss.sum()

# ___________________________________________________________________________________________________________




# it replaces logits with negative distances to enforce large margins 
# and calibrated confidence for OOD detection
# class IsoMaxPlusLoss(nn.Module): # Enhanced Isotropy Maximization loss
#     def __init__(self, entropic_scale=10.0):
#         super(IsoMaxPlusLoss, self).__init__()
#         self.entropic_scale = entropic_scale

#     def forward(self, outputs, targets):
#         distances = -outputs
#         probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
#         probabilities_at_targets = torch.gather(probabilities_for_training, 1, targets.unsqueeze(1))
#         probabilities_at_targets = torch.clamp(probabilities_at_targets, min=1e-7)
#         loss = -torch.log(probabilities_at_targets).mean()
#         return loss


class IsoMaxPlusLoss(nn.Module):
    def __init__(self, num_features, num_classes, temperature=1.0, entropic_scale=10.0):
        super(IsoMaxPlusLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.entropic_scale = entropic_scale

        # Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.ones(1))  # scalar learnable

    def forward(self, features, targets):
        """
        features: B x C x H x W (output from decoder)
        targets:  B x H x W (ground truth)
        """
        B, C, H, W = features.size()

        # Flatten spatial dimensions
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets = targets.view(-1)  # [B*H*W]

        # Normalize and compute distances
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        distances = torch.cdist(features, prototypes, p=2)  # [B*H*W, num_classes]

        # Scale and negate distances → logits
        logits = -torch.abs(self.distance_scale) * distances / self.temperature

        # Entropic softmax loss
        probs = F.softmax(-self.entropic_scale * distances, dim=1).clamp(1e-6, 1. - 1e-6)
        probs_at_target = probs[torch.arange(len(targets)), targets]
        loss = -torch.log(probs_at_target).mean()

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
    
    def __init__(self, num_features, num_classes, w_iso=0.5, w_focal=0.5):
        super().__init__()
        self.iso_max = IsoMaxPlusLoss(num_features, num_classes)
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
    def __init__(self,num_features, num_classes, w_iso=0.5, w_ce=0.5, weight=None):
        super().__init__()
        self.iso_max = IsoMaxPlusLoss(num_features, num_classes)
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

