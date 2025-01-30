import torch

class iouEval:
    def __init__(self, nClasses, ignoreIndex=19, weights=None):
        """IoU Evaluation with optional class weighting."""
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1
        self.weights = weights if weights is not None else torch.ones(nClasses).double()
        self.reset()

    def reset(self):
        """Reset TP, FP, FN counters."""
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):
        """Compute IoU with class weights applied."""
        if x.is_cuda or y.is_cuda:
            x = x.cuda()
            y = y.cuda()

        num_classes = self.nClasses

        # ✅ Ensure `x` and `y` are LONG tensors
        x = x.long()
        y = y.long()

        # ✅ Ensure `y` has 4D shape (batch, 1, H, W)
        if y.dim() == 3:
            y = y.unsqueeze(1)  # Expand to (B, 1, H, W)

        # ✅ Ensure `x` and `y` have the same spatial dimensions
        assert x.shape[-2:] == y.shape[-2:], f"Shape mismatch: x {x.shape[-2:]} vs y {y.shape[-2:]}"

        # ✅ Convert to one-hot
        x_onehot = torch.nn.functional.one_hot(x.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        y_onehot = torch.nn.functional.one_hot(y.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

        # ✅ Ensure tensors match after one-hot encoding
        assert x_onehot.shape == y_onehot.shape, f"x_onehot {x_onehot.shape} vs y_onehot {y_onehot.shape}"

        # ✅ Handle ignoreIndex properly
        if self.ignoreIndex != -1 and self.ignoreIndex < num_classes:
            ignores = y_onehot[:, self.ignoreIndex, :, :].unsqueeze(1)  # Shape: (B, 1, H, W)
            ignores = ignores.expand(-1, num_classes, -1, -1)  # Broadcast over all classes
            x_onehot = x_onehot * (1 - ignores)  # Ignore regions are set to 0
            y_onehot = y_onehot * (1 - ignores)
        else:
            ignores = torch.zeros_like(y_onehot)

        # Compute TP, FP, FN
        tp = torch.sum(x_onehot * y_onehot, dim=(0, 2, 3))
        fp = torch.sum(x_onehot * (1 - y_onehot - ignores), dim=(0, 2, 3))
        fn = torch.sum((1 - x_onehot) * y_onehot, dim=(0, 2, 3))

        # Apply class weighting
        self.tp += (tp.double().cpu() * self.weights[: self.nClasses])
        self.fp += (fp.double().cpu() * self.weights[: self.nClasses])
        self.fn += (fn.double().cpu() * self.weights[: self.nClasses])

    def getIoU(self):
        """Compute mean IoU and per-class IoU."""
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