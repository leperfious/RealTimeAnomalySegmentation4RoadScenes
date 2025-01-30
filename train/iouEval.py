import torch

class iouEval:
    """IoU Evaluation for Semantic Segmentation"""
    
    def __init__(self, nClasses, ignoreIndex=19, weights=None):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # Ignore label handling
        self.weights = weights if weights is not None else torch.ones(nClasses).double()
        self.reset()

    def reset(self):
        """Reset accumulators for new IoU computation"""
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):
        """Compute IoU with class weights applied"""
        if x.is_cuda or y.is_cuda:
            x = x.cuda()
            y = y.cuda()

        num_classes = self.nClasses

        # Ensure `x` and `y` are one-hot encoded
        x_onehot = torch.nn.functional.one_hot(x.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        y_onehot = torch.nn.functional.one_hot(y.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.ignoreIndex != -1:
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        # Compute TP, FP, FN
        tp = torch.sum(x_onehot * y_onehot, dim=(0, 2, 3))
        fp = torch.sum(x_onehot * (1 - y_onehot - ignores), dim=(0, 2, 3))
        fn = torch.sum((1 - x_onehot) * y_onehot, dim=(0, 2, 3))

        # Accumulate results
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        """Calculate weighted IoU"""
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-10  # Prevent division by zero
        
        iou = num / den  # Per-class IoU
        weighted_iou = (iou * self.weights[:self.nClasses - 1]).sum() / self.weights[:self.nClasses - 1].sum()

        return weighted_iou, iou  # Returns mean IoU and per-class IoU


class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# ✅ Fix: Color Formatting for IoU Scores
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if val < 0.20:
        return colors.RED
    elif val < 0.40:
        return colors.YELLOW
    elif val < 0.60:
        return colors.BLUE
    elif val < 0.80:
        return colors.CYAN
    else:
        return colors.GREEN