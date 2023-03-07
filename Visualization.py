import torch
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

from sklearn import metrics


def opt_threshold(thresholds, fpr_precision, tpr_recall, metric=None):
    if type(metric) == str:
        if metric == 'g-mean':
            metric_values = np.sqrt(tpr_recall * (1 - fpr_precision))
        elif metric == 'youdenj':
            metric_values = tpr_recall - fpr_precision
        elif metric == 'f-score':
            metric_values = (2 * fpr_precision * tpr_recall) / (fpr_precision + tpr_recall)
        else:
            return 0.5
    elif type(metric) == float:
        return metric
    else:
        return 0.5

    index = np.argmax(metric_values)
    threshold_opt = thresholds[index]

    return threshold_opt, index


def threshold(x, th):
    if x > th:
        return 1
    else:
        return 0


def best_auroc(results):
    old_max = 0.0
    row_idx = 0
    new_max = 0.0
    i = 0

    for res in results['AUROC Classifier']:
        new_max = np.max(res)
        if new_max > old_max:
            old_max = new_max
            row_idx = i
        i += 1

    max_idx = results['AUROC Classifier'][row_idx].index(old_max)
    print(old_max)

    return new_max, row_idx, max_idx


def plot_auroc_curve(results, best_auroc_row):
    AUROC = results['AUROC Classifier'][best_auroc_row]
    epochs = list(range(1, len(AUROC) + 1))
    plt.plot(AUROC)
    plt.plot(AUROC.index(max(AUROC)), max(AUROC), '-o', markersize=10)
    plt.xlabel('epoch')
    plt.ylabel('AUC')
    plt.title(results['Models'][best_auroc_row])
    plt.show()


def compute_roc_curve(y_test, prob, max_auroc):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=max_auroc)
    # display.plot()
    plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.2f)" % max_auroc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Receiver operating characteristic")

    gmean_opt_theshod, gmean_index = opt_threshold(thresholds, fpr, tpr, 'g-mean')
    youdenj_opt_threshold, youdenj_index = opt_threshold(thresholds, fpr, tpr, 'youdenj')

    plt.plot(fpr[gmean_index], tpr[gmean_index], '-o', color='tab:orange', markersize=10)
    plt.text(fpr[gmean_index] - 0.1, tpr[gmean_index] - 0.05, 'G-mean threshold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.plot(fpr[youdenj_index], tpr[youdenj_index], '-o', color='tab:orange', markersize=10)
    plt.text(fpr[youdenj_index] - 0.1, tpr[youdenj_index] + 0.02, 'Youden J threshold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds, gmean_opt_theshod, youdenj_opt_threshold


def compute_precision_recall_curve(y_test, prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, prob)
    # display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
    # display.plot()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    fscore_opt_theshod, fscore_index = opt_threshold(thresholds, precision, recall, 'f-score')

    plt.plot(recall[fscore_index], precision[fscore_index], '-o', markersize=10)
    plt.text(recall[fscore_index] - 0.1, precision[fscore_index] + 0.01, 'F-score threshold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()

    return precision, recall, thresholds, fscore_opt_theshod


def compute_confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    display = metrics.ConfusionMatrixDisplay(cm)
    display.plot()
    plt.title('Confusion Matrix with {} + oversampling'.format(results['Models'][best_auroc_row]))
    plt.show()


def plot_emb_losses(results, best_auroc_row):
    plt.plot(results['Losses'][best_auroc_row])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('{} + oversampling Embedding Losses'.format(results['Models'][best_auroc_row]))
    plt.show()


def plot_auc_loss(results, best_auroc_row):
    plt.plot(results['AUC VGAE'][best_auroc_row])
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC VGAE')
    plt.show()


def plot_clf_losses(results, best_auroc_row):
    plt.plot(results['Train Losses Classifier'][best_auroc_row], label='Train', color='tab:blue', alpha=0.5)
    plt.plot(results['Test Losses Classifier'][best_auroc_row], label='Test', color='tab:orange', alpha=0.5)
    plt.plot(gaussian_filter1d(results['Train Losses Classifier'][best_auroc_row], 6), label='Train Smooth',
             color='tab:blue')
    plt.plot(gaussian_filter1d(results['Test Losses Classifier'][best_auroc_row], 6), label='Test Smooth',
             color='tab:orange')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss with {} + oversampling'.format(results['Models'][best_auroc_row]))
    plt.legend(['Train', 'Test', 'Train Smooth', 'Test Smooth'])
    plt.show()

data_path = '/content/drive/MyDrive/ML/'
results = torch.load(data_path + 'grid search/TransE_oversampling.pt')

best_auroc_value, best_auroc_row, best_auroc_index = best_auroc(results)
plot_auroc_curve(results, best_auroc_row)
y_test = results['y Test'][best_auroc_row][0].cpu().detach().numpy()
prob = results['Predictions'][best_auroc_row][best_auroc_index]

fpr, tpr, roc_thresholds, gmean_opt_theshod, youdenj_opt_threshold = compute_roc_curve(y_test, prob, best_auroc_value)
precision, recall, pr_thresholds, fscore_opt_theshod = compute_precision_recall_curve(y_test, prob)

# threshold_opt = opt_threshold(roc_thresholds, fpr, tpr, precision, recall, 'g-mean')
y_pred = pd.Series(prob).apply(partial(threshold, th=gmean_opt_theshod)).to_numpy()

compute_confusion_matrix(y_test, y_pred)
plot_emb_losses(results, best_auroc_row)
plot_auc_loss(results, best_auroc_row)
plot_clf_losses(results, best_auroc_row)

print(metrics.classification_report(y_test, y_pred))
