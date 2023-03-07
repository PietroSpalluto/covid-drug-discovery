import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, WeightedRandomSampler


class Classifier(nn.Module):
    def __init__(self, embedding_dim=50):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        # print(x)
        residual1 = x
        x = F.dropout(x, training=self.training)
        x = self.bn(F.dropout(F.relu(self.fc1(x)), training=self.training))
        x += residual1
        return self.fc2(x)


class BPRLoss(nn.Module):
    def __init__(self, num_neg_samples):
        super(BPRLoss, self).__init__()
        self.num_neg_samples = num_neg_samples

    def forward(self, output, label):
        positive_output = output[label == 1]
        negative_output = output[label != 1]

        # print(f'tot output {output}')
        # print(f'pos output {positive_output}')
        # print(f'len pos output {len(positive_output)}')
        # print(f'neg output {negative_output}')
        # print(f'len neg output {len(negative_output)}')

        # negative sample proportional to the high values
        negative_sampler = WeightedRandomSampler(negative_output - min(negative_output),
                                                 num_samples=self.num_neg_samples * len(positive_output),
                                                 replacement=True)
        negative_sample_output = negative_output[
            torch.tensor(list(BatchSampler(negative_sampler, batch_size=len(positive_output), drop_last=True)),
                         dtype=torch.long).t()]

        # print(f'sample output {negative_sample_output}')
        # print(f'len sample output {len(negative_sample_output)}')
        return -(positive_output.view(-1, 1) - negative_sample_output).sigmoid().log().mean()
