import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


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
    # print(old_max)

    return new_max, row_idx, max_idx

def make_roc_curves(results):
    for r in results:
        best_auroc_value, best_auroc_row, best_auroc_index = best_auroc(r)
        y_test = r['y Test'][best_auroc_row][0].cpu().detach().numpy()
        prob = r['Predictions'][best_auroc_row][best_auroc_index]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, prob)
        plt.plot(fpr, tpr)

        gmean_opt_theshod, gmean_index = opt_threshold(thresholds, fpr, tpr, 'g-mean')
        youdenj_opt_threshold, youdenj_index = opt_threshold(thresholds, fpr, tpr, 'youdenj')

        plt.plot(fpr[gmean_index], tpr[gmean_index], '-o', color='tab:brown', markersize=5)
        plt.plot(fpr[youdenj_index], tpr[youdenj_index], '-o', color='tab:gray', markersize=5)

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    te = mlines.Line2D([], [] , color='tab:blue', label='TransE')
    te_os = mlines.Line2D([], [] , color='tab:orange', label='TransE OS')
    dm = mlines.Line2D([], [] , color='tab:green', label='DistMult')
    dm_os = mlines.Line2D([], [] , color='tab:red', label='DistMult OS')
    gmean = mlines.Line2D([], [] , color='tab:brown', marker='o', label='G-Mean')
    youdenj = mlines.Line2D([], [] , color='tab:gray', marker='o', label='Youden J')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Receiver operating characteristic")
    plt.legend(handles=[te, te_os, dm, dm_os, gmean, youdenj])
    plt.show()

def make_pr_curves(results):
    for r in results:
        best_auroc_value, best_auroc_row, best_auroc_index = best_auroc(r)
        y_test = r['y Test'][best_auroc_row][0].cpu().detach().numpy()
        prob = r['Predictions'][best_auroc_row][best_auroc_index]
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, prob)
        plt.plot(recall, precision)

        fscore_opt_theshod, fscore_index = opt_threshold(thresholds, precision, recall, 'f-score')

        plt.plot(recall[fscore_index], precision[fscore_index], '-o', color='tab:brown', markersize=5)

    te = mlines.Line2D([], [] , color='tab:blue', label='TransE')
    te_os = mlines.Line2D([], [] , color='tab:orange', label='TransE OS')
    dm = mlines.Line2D([], [] , color='tab:green', label='DistMult')
    dm_os = mlines.Line2D([], [] , color='tab:red', label='DistMult OS')
    fscore = mlines.Line2D([], [] , color='tab:brown', marker='o', label='F-Score')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(handles=[te, te_os, dm, dm_os, fscore])
    plt.show()

def plot_auroc_curve(results):
    for r in results:
        best_auroc_value, best_auroc_row, best_auroc_index = best_auroc(r)
        AUROC = r['AUROC Classifier'][best_auroc_row]
        epochs = list(range(1, len(AUROC) + 1))
        plt.plot(AUROC)
        plt.plot(AUROC.index(max(AUROC)), max(AUROC), '-o', color='tab:brown', markersize=10)

    te = mlines.Line2D([], [] , color='tab:blue', label='TransE')
    te_os = mlines.Line2D([], [] , color='tab:orange', label='TransE OS')
    dm = mlines.Line2D([], [] , color='tab:green', label='DistMult')
    dm_os = mlines.Line2D([], [] , color='tab:red', label='DistMult OS')
    max_auroc = mlines.Line2D([], [] , color='tab:brown', marker='o', label='Best AUROC')

    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC Classifier values')
    plt.legend(handles=[te, te_os, dm, dm_os, max_auroc])
    plt.show()

def plot_auc_vgae(results):
    for r in results:
        best_auroc_value, best_auroc_row, best_auroc_index = best_auroc(r)
        AUROC = r['AUC VGAE'][best_auroc_row]
        plt.plot(AUROC)
        plt.plot(AUROC.index(max(AUROC)), max(AUROC), '-o', color='tab:brown', markersize=10)

    te = mlines.Line2D([], [] , color='tab:blue', label='TransE')
    te_os = mlines.Line2D([], [] , color='tab:orange', label='TransE OS')
    dm = mlines.Line2D([], [] , color='tab:green', label='DistMult')
    dm_os = mlines.Line2D([], [] , color='tab:red', label='DistMult OS')
    max_auroc = mlines.Line2D([], [] , color='tab:brown', marker='o', label='Best AUROC')

    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC VGAE values')
    plt.legend(handles=[te, te_os, dm, dm_os, max_auroc])
    plt.show()


data_path = '/content/drive/MyDrive/ML/'
results_TE = torch.load(data_path + 'grid search/TransE_no_oversampling.pt')
results_TE_oversampling = torch.load(data_path + 'grid search/TransE_oversampling.pt')
results_DM = torch.load(data_path + 'grid search/DistMult_no_oversampling.pt')
results_DM_oversampling = torch.load(data_path + 'grid search/DistMult_oversampling.pt')

results = [results_TE, results_TE_oversampling, results_DM, results_DM_oversampling]

make_roc_curves(results)
make_pr_curves(results)
plot_auroc_curve(results)
plot_auc_vgae(results)
