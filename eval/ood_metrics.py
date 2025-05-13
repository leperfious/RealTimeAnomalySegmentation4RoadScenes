import numpy as np
from sklearn.metrics import roc_curve


def fpr_at_95_tpr(predictions, labels):
    """
    Computes the False Positive Rate when True Positive Rate is at 95%.
    :param predictions: numpy array of predicted anomaly scores
    :param labels: numpy array of ground truth binary labels (1 = OOD, 0 = in-distribution)
    :return: FPR at 95% TPR
    """
    if np.all(labels == 0) or np.all(labels == 1):
        print("Only one class present in ground truth. FPR@95TPR is not defined.")
        return np.nan

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    try:
        return fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        print("TPR never reaches 95%")
        return 1.0
