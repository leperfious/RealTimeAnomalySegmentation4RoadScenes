# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:
    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):
        """Compute IoU ensuring tensor shapes match."""
        if x.is_cuda or y.is_cuda:
            x = x.cuda()
            y = y.cuda()

        num_classes = self.nClasses

        # ✅ **Ensure `x` is one-hot encoded correctly**
        x = x.max(1)[1]  # Get predicted class indices
        x_onehot = torch.nn.functional.one_hot(x, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # ✅ **Ensure `y` is formatted correctly**
        y = y.squeeze(1)  # Ensure it's (B, H, W)
        y_onehot = torch.nn.functional.one_hot(y.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        # ✅ **Ensure shapes match after one-hot encoding**
        assert x_onehot.shape == y_onehot.shape, f"x_onehot {x_onehot.shape} vs y_onehot {y_onehot.shape}"

        # ✅ **Fix Ignore Index Handling**
        if self.ignoreIndex != -1 and self.ignoreIndex < num_classes:
            ignore_mask = (y == self.ignoreIndex).unsqueeze(1).expand_as(y_onehot)
            x_onehot[ignore_mask] = 0  # Remove ignored pixels
            y_onehot[ignore_mask] = 0

        # ✅ **Compute True Positives (TP), False Positives (FP), False Negatives (FN)**
        tpmult = x_onehot * y_onehot
        tp = torch.sum(tpmult, dim=(0, 2, 3))

        fpmult = x_onehot * (1 - y_onehot)
        fp = torch.sum(fpmult, dim=(0, 2, 3))

        fnmult = (1 - x_onehot) * y_onehot
        fn = torch.sum(fnmult, dim=(0, 2, 3))

        # ✅ **Store results**
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou  # Returns mean IoU and per-class IoU

# Class for colors
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

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN