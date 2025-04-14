from sklearn.metrics import auc
import torch
import torchmetrics.functional.classification as tmc

preds = torch.tensor([
    [0.75, 0.05, 0.05, 0.05, 0.05],
    [0.05, 0.75, 0.05, 0.05, 0.05],
    [0.05, 0.05, 0.75, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.75, 0.04]])
target = torch.tensor([0, 1, 3, 2])

fprs, tprs, thresholds = tmc.multiclass_roc(preds, target, num_classes=5)
print(tprs)
auroc = torch.tensor([round(auc(fpr, tpr), 4) for tpr, fpr in zip(torch.stack(tprs), torch.stack(fprs))])
print(auroc)
print(tmc.multiclass_auroc(preds, target, num_classes=5, average=None))
print(tmc.multiclass_precision(preds, target, num_classes=5, average=None))
print(tmc.multiclass_recall(preds, target, num_classes=5, average=None))
print(tmc.multiclass_specificity(preds, target, num_classes=5, average=None))