import numpy as np
import pandas as pd
import math
import random
import re

from DatasetLoader import DatasetLoader
from TransE import TransE
from DistMult import DistMult
from KnowledgeGraph import KnowledgeGraph
from EncoderVGAE import Encoder_VGAE
from BPR import BPRLoss, Classifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, roc_curve
from imblearn.over_sampling import SMOTE
import torch
import time
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import VGAE


def run_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    # data.edge_index = None
    attr = data.edge_attr

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    attr = attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_pos_edge_attr = attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_post_edge_attr = attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_attr = attr[n_v + n_t:]

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)),
                         min(n_v + n_t, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index, train_pos_edge_attr)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    # print('Training loss', loss.item())


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index, train_pos_edge_attr)
    return model.test(z, pos_edge_index, neg_edge_index)


# Parameters
data_path = '/content/drive/MyDrive/Colab Notebooks/BDA/'
model_emb = ['TransE']
margin = 4
regul = 50
l3_regul = 0
threshold_auroc = 60
embedding = [200]
batch_dim = [50]
learning_rates = [0.005]
epochs_num = [15]
epochs_vgae_num = [25]
epochs_clf = [300]
mode_distmult = ['normal']
tot_combinations_distmult = len(embedding) * len(batch_dim) * len(learning_rates) * len(epochs_num) \
                            * len(epochs_vgae_num) * len(epochs_clf) * len(mode_distmult)
tot_combinations_distmult = 0
tot_combinations_transe = len(embedding) * len(batch_dim) * len(learning_rates) * len(epochs_num) \
                            * len(epochs_vgae_num) * len(epochs_clf)

#tot_combinations_transe = 0
tot_combinations = tot_combinations_transe + tot_combinations_distmult
print(tot_combinations)

dataset_loader = DatasetLoader(data_path)
edge_index = dataset_loader.load_data()

le = LabelEncoder()
le.fit(np.concatenate((edge_index['node1'], edge_index['node2'])))

edge_index['node1id'] = le.transform(edge_index['node1'])
edge_index['node2id'] = le.transform(edge_index['node2'])
len(le.classes_)
edge_attr_dict = {'gene-drug': 0, 'gene-gene': 1, 'bait-gene': 2, 'gene-phenotype': 3, 'drug-phenotype': 4}
edge_index['typeid'] = edge_index['type'].apply(lambda x: edge_attr_dict[x])
edge = torch.tensor(edge_index[['node1id', 'node2id']].values, dtype=torch.long)
edge_attr = torch.tensor(edge_index['typeid'].values, dtype=torch.long)

kg = KnowledgeGraph(edge_index, edge_attr_dict)
kg.triples()

models = []
batch_list = []
learning_rate_list = []
entity_embedding_list = []
relation_embedding_list = []
epoch_loss_list = []
epoch_num_list = []
emb_size_list = []
epochs_vgae_list = []
auc_list = []
ap_list = []
encoded_embeddings_list = []
drug_labels_lists = []
indices_lists = []
types_lists = []
X_train_embeddings_list = []
X_test_embeddings_list = []
y_train_embeddings_list = []
y_test_embeddings_list = []
indices_train_lists = []
indices_test_lists = []
prob_lists = []
AUROC_list = []
AUPRC_list = []
AUROC_lr_list = []
AUPRC_lr_list = []
AUROC_xgboost_list = []
AUPRC_xgboost_list = []
AUROC_rf_list = []
AUPRC_rf_list = []
AUROC_svm_list = []
AUPRC_svm_list = []
loss_clf_list = []
test_loss_clf_list = []
topk_drugs_list = []
emb_size_vgae_clf = []
epochs_clf_list = []
mode_train_list = []  # TransE NON ha mode

test_n = 1
for mode in model_emb:
    for batch_size in batch_dim:
        for learning_rate in learning_rates:
            for epochs in epochs_num:
                for emb_size in embedding:
                    i = 0
                    while i < len(mode_distmult):
                        start_time = time.monotonic()

                        # Embedding
                        if mode == 'TransE':
                            transe = TransE(kg, num_of_dimension=emb_size, batch_size=batch_size, num_of_epochs=epochs,
                                            margin=margin, norm1=regul)
                            entity_emb, relation_emb, epoch_losses = transe.train()
                            i = len(mode_distmult)

                        elif mode == 'DistMult':
                            dist = DistMult(kg, dim=emb_size, batch_size=batch_size, epsilon=None, margin=None,
                                            mode=mode_distmult[i],
                                            neg_ent=1, neg_rel=0, num_of_epochs=epochs, regul_rate=regul,
                                            l3_regul_rate=l3_regul, learning_rate=learning_rate)
                            entity_emb, relation_emb, epoch_losses = dist.train()
                            i = i + 1

                        entity_emb_df = pd.DataFrame(entity_emb)
                        entity_emb_df.index = kg.entities
                        node_feature = torch.tensor(entity_emb, dtype=torch.float)
                        data = Data(x=node_feature, edge_index=edge.t().contiguous(), edge_attr=edge_attr)

                        data_split = train_test_split_edges(data, test_ratio=0.1, val_ratio=0)
                        x, train_pos_edge_index, train_pos_edge_attr = data_split.x, data_split.train_pos_edge_index, data_split.train_pos_edge_attr
                        train_pos_edge_index, train_pos_edge_attr = add_remaining_self_loops(train_pos_edge_index,
                                                                                             train_pos_edge_attr)
                        pd.Series(train_pos_edge_attr.cpu().numpy()).value_counts()
                        x, train_pos_edge_index, train_pos_edge_attr = Variable(x), Variable(
                            train_pos_edge_index), Variable(train_pos_edge_attr)

                        # print('\nVGAE - mode: ', mode,'\n')
                        # setting VGAE embedding size
                        emb_size_vgae = round(emb_size - emb_size / 3)

                        # VGAE encoding
                        for epochs_vgae in epochs_vgae_num:
                            auc_epochs = []
                            ap_epochs = []
                            encoded_embeddings = []
                            drug_labels_list = []
                            indices_list = []
                            X_train_embeddings = []
                            X_test_embeddings = []
                            y_train_embeddings = []
                            y_test_embeddings = []
                            indices_train_list = []
                            indices_test_list = []
                            types_list = []
                            model = VGAE(Encoder_VGAE(node_feature.shape[1], emb_size_vgae))

                            optimizer = torch.optim.Adam(model.parameters())
                            model.test(x, data_split.test_pos_edge_index, data_split.test_neg_edge_index)

                            best_auc_vgae = 0
                            for epoch in range(0, epochs_vgae):
                                train()
                                auc, ap = test(data_split.test_pos_edge_index, data_split.test_neg_edge_index)
                                auc_epochs.append(auc)
                                ap_epochs.append(ap)
                                if auc > best_auc_vgae:
                                    best_auc_vgae = auc

                                    torch.save(model.state_dict(), data_path + 'best_vgae.pt')

                            model.load_state_dict(torch.load(data_path + 'best_vgae.pt'))
                            model.eval()
                            z = model.encode(x, data.edge_index, data.edge_attr)
                            z_np = z.squeeze().detach().cpu().numpy()
                            encoded_embeddings.append(z_np)

                            topk = 300
                            types = np.array([item.split('_')[0] for item in le.classes_])
                            types_list.append(types)
                            # label
                            trials = pd.read_excel(data_path + 'data/All_trails_5_24.xlsx', header=1, index_col=0)
                            trials_drug = set([drug.strip().upper() for lst in trials.loc[
                                trials['study_category'].apply(lambda x: 'drug' in x.lower()), 'intervention'].apply(
                                lambda x: re.split(r'[+|/|,]',
                                                   x.replace(' vs. ', '/').replace(' vs ', '/').replace(' or ',
                                                                                                        '/').replace(
                                                       ' with and without ', '/').replace(' /wo ', '/').replace(' /w ',
                                                                                                                '/').replace(
                                                       ' and ', '/').replace(' - ', '/').replace(' (', '/').replace(
                                                       ') ', '/'))).values for drug in lst])
                            drug_labels = [1 if drug.split('_')[1] in trials_drug else 0 for drug in
                                           le.classes_[types == 'drug']]

                            seed = 70

                            # oversampling
                            oversampling = SMOTE()
                            # print('Original dataset shape %s' % Counter(drug_labels))
                            # umbalanced_z_np = z_np
                            # z_np, drug_labels = oversampling.fit_resample(z_np[types == 'drug'], drug_labels)
                            indices = np.arange(len(drug_labels))
                            indices_list.append(indices)
                            # print('Oversampled dataset shape %s' % Counter(drug_labels))

                            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                                z_np[types == 'drug'], drug_labels,
                                indices, test_size=0.2,
                                random_state=seed, )
                            # X_train, y_train = oversampling.fit_resample(X_train, y_train)
                            # X_train, y_train = shuffle(X_train, y_train)

                            drug_labels_list.append(drug_labels)
                            indices_train_list.append(indices_train)
                            indices_test_list.append(indices_test)

                            _X_train, _y_train = Variable(torch.tensor(X_train, dtype=torch.float)), \
                                                 Variable(torch.tensor(y_train, dtype=torch.float))
                            _X_test, _y_test = Variable(torch.tensor(X_test, dtype=torch.float)), \
                                               Variable(torch.tensor(y_test, dtype=torch.float))

                            X_train_embeddings.append(_X_train)
                            X_test_embeddings.append(_X_test)
                            y_train_embeddings.append(_y_train)
                            y_test_embeddings.append(_y_test)

                            # classification
                            emb_size_clf = emb_size_vgae
                            clf = Classifier(emb_size_clf)
                            optimizer = torch.optim.Adam(clf.parameters())
                            criterion = BPRLoss(num_neg_samples=15)
                            best_auprc = 0
                            best_auroc = 0
                            for epochs_classifier in epochs_clf:
                                AUROC_epochs = []
                                AUPRC_epochs = []
                                loss_clf = []
                                test_loss_clf = []
                                prob_list = []

                                for epoch in range(epochs_classifier):
                                    clf.train()
                                    optimizer.zero_grad()
                                    out = clf(_X_train)
                                    loss = criterion(out.squeeze(), _y_train)
                                    loss.backward()
                                    optimizer.step()
                                    loss_clf.append(loss.item())

                                    clf.eval()

                                    test_loss_clf.append(criterion(clf(_X_test).squeeze(), _y_test).item())
                                    prob = torch.sigmoid(clf(_X_test)).cpu().detach().numpy().squeeze()
                                    prob_list.append(prob)
                                    AUROC_epochs.append(metrics.roc_auc_score(y_test, prob))
                                    AUPRC_epochs.append(metrics.average_precision_score(y_test, prob))

                                    auroc = metrics.roc_auc_score(y_test, prob)

                                    if auroc > best_auroc:
                                        best_auroc = auroc
                                        print(best_auroc)
                                        torch.save(clf, data_path + 'best_clf.pt')

                                clf.load_state_dict(torch.load(data_path + 'best_clf.pt').state_dict())
                                clf.eval()

                                prob = torch.sigmoid(clf(_X_test)).cpu().detach().numpy().squeeze()
                                prob_list.append(prob)

                                top_items_idx = np.argsort(-clf(torch.tensor(z_np[types == 'drug'],
                                                                             dtype=torch.float)).squeeze().detach().cpu().numpy())

                                num_top = 300

                                topk_drugs = pd.DataFrame([(rank + 1, drug.split('_')[1]) for rank, drug in enumerate(
                                    le.inverse_transform((types == 'drug').nonzero()[0][top_items_idx])[:num_top + 1])],
                                                          columns=['rank', 'drug'])
                                topk_drugs['under_trials'] = topk_drugs['drug'].isin(trials_drug).astype(int)
                                # print('\nTop', num_top, 'drugs for Covid treatment (', mode, '): \n', topk_drugs)

                                end_time = time.monotonic()

                                mins, secs = run_time(start_time, end_time)
                                print(f'Time: {mins}m {secs}s')

                                models.append(mode)

                                if mode == 'TransE':
                                    mode_train_list.append('False')
                                elif mode == 'DistMult':
                                    mode_train_list.append(mode_distmult[i - 1])

                                batch_list.append(batch_size)
                                learning_rate_list.append(learning_rate)
                                entity_embedding_list.append(entity_emb)
                                relation_embedding_list.append(relation_emb)
                                epoch_loss_list.append(epoch_losses)
                                epoch_num_list.append(epochs)
                                emb_size_list.append(emb_size)
                                emb_size_vgae_clf.append(round(emb_size - emb_size / 3))

                                epochs_vgae_list.append(epochs_vgae)
                                auc_list.append(auc_epochs)
                                ap_list.append(ap_epochs)
                                encoded_embeddings_list.append(encoded_embeddings)
                                drug_labels_lists.append(drug_labels_list)
                                indices_lists.append(indices_list)
                                X_train_embeddings_list.append(X_train_embeddings)
                                X_test_embeddings_list.append(X_test_embeddings)
                                y_train_embeddings_list.append(y_train_embeddings)
                                y_test_embeddings_list.append(y_test_embeddings)
                                indices_train_lists.append(indices_train_list)
                                indices_test_lists.append(indices_test_list)
                                types_lists.append(types_list)
                                prob_lists.append(prob_list)

                                epochs_clf_list.append(epochs_classifier)
                                loss_clf_list.append(loss_clf)
                                test_loss_clf_list.append(test_loss_clf)
                                AUROC_list.append(AUROC_epochs)
                                AUPRC_list.append(AUPRC_epochs)
                                topk_drugs_list.append(topk_drugs)

                                # TRADITIONAL METHODS
                                clf_lr = LogisticRegression().fit(X_train, y_train)
                                AUROC_logit = roc_auc_score(y_test, clf_lr.predict_proba(X_test)[:, 1])
                                AUPRC_logit = average_precision_score(y_test, clf_lr.predict_proba(X_test)[:, 1])

                                AUROC_lr_list.append(AUROC_logit)
                                AUPRC_lr_list.append(AUPRC_logit)

                                clf_gb = GradientBoostingClassifier().fit(X_train, y_train)
                                AUROC_XGBoost = roc_auc_score(y_test, clf_gb.predict_proba(X_test)[:, 1])
                                AUPRC_XGBoost = average_precision_score(y_test, clf_gb.predict_proba(X_test)[:, 1])

                                AUROC_xgboost_list.append(AUROC_XGBoost)
                                AUPRC_xgboost_list.append(AUPRC_XGBoost)

                                clf_rf = RandomForestClassifier().fit(X_train, y_train)
                                AUROC_rf = roc_auc_score(y_test, clf_rf.predict_proba(X_test)[:, 1])
                                AUPRC_rf = average_precision_score(y_test, clf_rf.predict_proba(X_test)[:, 1])

                                AUROC_rf_list.append(AUROC_rf)
                                AUPRC_rf_list.append(AUPRC_rf)

                                clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True)).fit(
                                    X_train, y_train)
                                AUROC_svm = roc_auc_score(y_test, clf_svc.predict_proba(X_test)[:, 1])
                                AUPRC_svm = average_precision_score(y_test, clf_svc.predict_proba(X_test)[:, 1])

                                AUROC_svm_list.append(AUROC_svm)
                                AUPRC_svm_list.append(AUPRC_svm)

                                print('TEST #', test_n, '/', tot_combinations)
                                print(
                                    '-------------------------------------------------------------------------------------------------------')
                                test_n += 1

                            results = pd.DataFrame({'Models': models,
                                                    'Model mode': mode_train_list,
                                                    'Regularization': regul,
                                                    'l3 regularization': l3_regul,
                                                    'Embedding size': emb_size_list,
                                                    'Entity embeddings': entity_embedding_list,
                                                    'Relation embeddings': relation_embedding_list,
                                                    'Epochs': epoch_num_list,
                                                    'Losses': epoch_loss_list,
                                                    'Batch size': batch_list,
                                                    'Learning rate': learning_rate_list,
                                                    'VGAE and Classifier emb size': emb_size_vgae_clf,
                                                    'Epochs VGAE': epochs_vgae_list,
                                                    'AUC VGAE': auc_list,
                                                    'AP VGAE': ap_list,
                                                    'Latent Vector': encoded_embeddings_list,
                                                    'Drug Labels': drug_labels_lists,
                                                    'Drug Indices': indices_lists,
                                                    'X Train': X_train_embeddings_list,
                                                    'X Test': X_test_embeddings_list,
                                                    'y Train': y_train_embeddings_list,
                                                    'y Test': y_test_embeddings_list,
                                                    'Train Indices': indices_train_lists,
                                                    'Test Indices': indices_test_lists,
                                                    'Types': types_lists,
                                                    'Predictions': prob_lists,
                                                    'Epochs Classifier': epochs_clf_list,
                                                    'Train Losses Classifier': loss_clf_list,
                                                    'Test Losses Classifier': test_loss_clf_list,
                                                    'AUROC Classifier': AUROC_list,
                                                    'AUPRC Classifier': AUPRC_list,
                                                    'Logistic Regression AUROC': AUROC_lr_list,
                                                    'Logistic Regression AUPRC': AUPRC_lr_list,
                                                    'XGBoost AUROC': AUROC_xgboost_list,
                                                    'XGBoost AUPRC': AUPRC_xgboost_list,
                                                    'Random Forest AUROC': AUROC_rf_list,
                                                    'Random Forest AUPRC': AUPRC_rf_list,
                                                    'SVC AUROC': AUROC_svm_list,
                                                    'SVC AUPRC': AUPRC_svm_list,
                                                    'Top drugs': topk_drugs_list})
                            torch.save(results, data_path + 'grid search/TransE_os_06-11-22.pt')

results = pd.DataFrame({'Models': models,
                        'Model mode': mode_train_list,
                        'Regularization': regul,
                        'l3 regularization': l3_regul,
                        'Embedding size': emb_size_list,
                        'Entity embeddings': entity_embedding_list,
                        'Relation embeddings': relation_embedding_list,
                        'Epochs': epoch_num_list,
                        'Losses': epoch_loss_list,
                        'Batch size': batch_list,
                        'Learning rate': learning_rate_list,
                        'VGAE and Classifier emb size': emb_size_vgae_clf,
                        'Epochs VGAE': epochs_vgae_list,
                        'AUC VGAE': auc_list,
                        'AP VGAE': ap_list,
                        'Latent Vector': encoded_embeddings_list,
                        'Drug Labels': drug_labels_lists,
                        'Drug Indices': indices_lists,
                        'X Train': X_train_embeddings_list,
                        'X Test': X_test_embeddings_list,
                        'y Train': y_train_embeddings_list,
                        'y Test': y_test_embeddings_list,
                        'Train Indices': indices_train_lists,
                        'Test Indices': indices_test_lists,
                        'Types': types_lists,
                        'Predictions': prob_lists,
                        'Epochs Classifier': epochs_clf_list,
                        'Train Losses Classifier': loss_clf_list,
                        'Test Losses Classifier': test_loss_clf_list,
                        'AUROC Classifier': AUROC_list,
                        'AUPRC Classifier': AUPRC_list,
                        'Logistic Regression AUROC': AUROC_lr_list,
                        'Logistic Regression AUPRC': AUPRC_lr_list,
                        'XGBoost AUROC': AUROC_xgboost_list,
                        'XGBoost AUPRC': AUPRC_xgboost_list,
                        'Random Forest AUROC': AUROC_rf_list,
                        'Random Forest AUPRC': AUPRC_rf_list,
                        'SVC AUROC': AUROC_svm_list,
                        'SVC AUPRC': AUPRC_svm_list,
                        'Top drugs': topk_drugs_list})

torch.save(results, data_path + 'grid search/TransE_os_06-11-22.pt')
