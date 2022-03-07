import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


def evaluate_classification(preds: np.array, targets: np.array) -> dict:
    num_classes = preds.shape[1]
    num_samples = preds.shape[0]
    argmax = np.argmax(preds, axis=1)
    accuracy = np.sum(argmax == targets) / num_samples
    ret = {
        "accuracy": accuracy,
    }
    ret["matthews_corr"] = matthews_corrcoef(targets, argmax)
    if num_classes == 2:
        f1 = f1_score(targets, argmax, average='binary')
        ret["f1"] = f1
    return ret
