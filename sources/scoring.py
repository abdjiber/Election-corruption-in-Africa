import numpy as np
from sklearn import metrics


def scores(y_val, preds, preds_prob):
    acc = np.round(metrics.accuracy_score(y_val, preds), 2)
    recall = np.round(metrics.recall_score(y_val, preds, average=None), 2)
    auc = np.round(metrics.roc_auc_score(y_val, preds_prob), 2)
    print('Accuracy:', acc)
    print('Recall:', recall)
    print('AUC:', auc)
    return acc, recall, auc
