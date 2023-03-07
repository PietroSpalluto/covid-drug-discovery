import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import random


class DistMult(nn.Module):

    def __init__(self, kg, dim=50, batch_size=100, learning_rate=0.01, epsilon=None, margin=None, mode='normal',
                 neg_ent=1, neg_rel=0, num_of_epochs=10, filter_flag=True, regul_rate=0, l3_regul_rate=0.0):
        super(DistMult, self).__init__()
        self.h_of_tr = {}
        self.t_of_hr = {}
        self.r_of_ht = {}
        self.ent_tot = kg.n_entity
        self.rel_tot = kg.n_relation
        self.triples = kg.triples_set
        self.dim = dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel
        self.regul_rate = regul_rate
        self.l3_regul_rate = l3_regul_rate
        self.margin = margin
        self.epsilon = epsilon
        self.filter_flag = filter_flag
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.data, self.data_corr = self.corrupt_triples()
        self.loss_sp = nn.Softplus()
        self.array_loss = []
        self.mode = mode

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def corrupt_triples(self):

        batch_h = np.array([item[0] for item in self.triples]).reshape(-1, 1)
        batch_t = np.array([item[1] for item in self.triples]).reshape(-1, 1)
        batch_r = np.array([item[2] for item in self.triples]).reshape(-1, 1)
        # Time to add triples for neg samples
        batch_h = np.repeat(batch_h, 1 + self.neg_ent + self.neg_rel, axis=-1)
        batch_t = np.repeat(batch_t, 1 + self.neg_ent + self.neg_rel, axis=-1)
        batch_r = np.repeat(batch_r, 1 + self.neg_ent + self.neg_rel, axis=-1)

        for h, t, r in self.triples:
            if (h, r) not in self.t_of_hr:
                self.t_of_hr[(h, r)] = []
            self.t_of_hr[(h, r)].append(t)
            if (t, r) not in self.h_of_tr:
                self.h_of_tr[(t, r)] = []
            self.h_of_tr[(t, r)].append(h)

        for index, item in enumerate(self.triples):
            last = 1
            if self.neg_ent > 0:
                neg_head, neg_tail = self.normal_batch(item[0], item[1], item[2], self.neg_ent)
                if len(neg_head) > 0:
                    batch_h[index][last:last + len(neg_head)] = neg_head
                    last += len(neg_head)
                if len(neg_tail) > 0:
                    batch_t[index][last:last + len(neg_tail)] = neg_tail
                    last += len(neg_tail)
            if self.neg_rel > 0:
                neg_rel = self.rel_batch(item[0], item[1], item[2], self.neg_rel)
                batch_r[index][last:last + len(neg_rel)] = neg_rel
        batch_h = batch_h.transpose()
        batch_t = batch_t.transpose()
        batch_r = batch_r.transpose()
        data = pd.DataFrame({'head': batch_h[0], 'tail': batch_t[0], 'relation': batch_r[0]})
        data_corr = pd.DataFrame({'head_corr': batch_h[1], 'tail_corr': batch_t[1], 'relation_corr': batch_r[1]})

        data_tuple = [tuple(i) for i in data.values.tolist()]
        data_corr_tuple = [tuple(i) for i in data_corr.values.tolist()]

        return data_tuple, data_corr_tuple

    def normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        prob = 0.5
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.corrupt_entity(t, r, self.h_of_tr[(t, r)], num_max=(neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)

        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.corrupt_entity(h, r, self.t_of_hr[(h, r)], num_max=(neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]

    def corrupt_entity(self, t, r, entity, num_max=1):
        tmp = torch.randint(low=0, high=self.ent_tot, size=(num_max,)).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, entity, assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def _calc(self, h, t, r):
        if self.mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if self.mode == 'head_batch':
            score = h * (r * t)
        else:
            score = (h * r) * t
        score = torch.sum(score, -1).flatten()
        return score

    def score(self, data: torch.tensor):
        batch_h = data[0]
        batch_t = data[1]
        batch_r = data[2]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r)
        return score

    def regularization(self, data):
        batch_h = data[0]
        batch_t = data[1]
        batch_r = data[2]
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
        return regul

    def l3_regularization(self):
        return self.ent_embeddings.weight.norm(p=3) ** 3 + self.rel_embeddings.weight.norm(p=3) ** 3

    def forward(self, data: torch.tensor):
        score = -self.score(data)
        var = int(score.size(dim=0) / 2)
        if var < self.batch_size:
            p_score = score[:var]
            p_score = p_score.view(-1, var).permute(1, 0)
            n_score = score[var:]
            n_score = n_score.view(-1, var).permute(1, 0)
        else:
            p_score = score[:self.batch_size]
            p_score = p_score.view(-1, self.batch_size).permute(1, 0)
            n_score = score[self.batch_size:]
            n_score = n_score.view(-1, self.batch_size).permute(1, 0)
        loss_res = self.SoftPlusLoss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res += self.regul_rate * self.regularization(data)
        if self.l3_regul_rate != 0:
            loss_res += self.l3_regul_rate * self.l3_regularization()
        return loss_res

    def SoftPlusLoss(self, p_score, n_score):
        return (self.loss_sp(-p_score).mean() + self.loss_sp(n_score).mean()) / 2

    def train(self):
        train_dl = DataLoader(self.data, batch_size=self.batch_size)
        train_dl_corr = DataLoader(self.data_corr, batch_size=self.batch_size)
        params = list(self.ent_embeddings.parameters()) + list(self.rel_embeddings.parameters())
        optimizer = optim.SGD(params, lr=self.learning_rate)

        for epoch in range(self.num_of_epochs):
            epoch_loss = 0
            for batch, corr_batch in zip(tqdm(train_dl), tqdm(train_dl_corr)):
                # mini_batch = batch[0]
                mini_batch = batch
                corr_mini_batch = corr_batch
                tot_batch = [torch.cat([mini_batch[0], corr_mini_batch[0]]),
                             torch.cat([mini_batch[1], corr_mini_batch[1]]),
                             torch.cat([mini_batch[2], corr_mini_batch[2]])]

                optimizer.zero_grad()

                loss = self.forward(tot_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.array_loss.append(epoch_loss)
            # print('Epoch ', epoch, ' ', epoch_loss)
            # print(self.ent_embeddings.weight.data.numpy())

        return self.ent_embeddings.weight.data.numpy(), self.rel_embeddings.weight.data.numpy(), self.array_loss
