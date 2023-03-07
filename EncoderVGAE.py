import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch.nn import functional as F


class Encoder_VGAE(nn.Module):
    def __init__(self, in_channels, out_channels, isClassificationTask=False):
        super(Encoder_VGAE, self).__init__()
        self.isClassificationTask = isClassificationTask
        self.conv_gene_drug = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_gene_gene = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_bait_gene = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_gene_phenotype = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_drug_phenotype = SAGEConv(in_channels, 2 * out_channels)

        self.bn = nn.BatchNorm1d(5 * 2 * out_channels)
        # variational encoder
        self.conv_mu = SAGEConv(5 * 2 * out_channels, out_channels, )
        self.conv_logvar = SAGEConv(5 * 2 * out_channels, out_channels, )

    def forward(self, x, edge_index, edge_attr):
        # print(self.training)
        x = F.dropout(x, training=self.training)

        index_gene_drug = (edge_attr == 0).nonzero().reshape(1, -1)[0]
        edge_index_gene_drug = edge_index[:, index_gene_drug]

        index_gene_gene = (edge_attr == 1).nonzero().reshape(1, -1)[0]
        edge_index_gene_gene = edge_index[:, index_gene_gene]

        index_bait_gene = (edge_attr == 2).nonzero().reshape(1, -1)[0]
        edge_index_bait_gene = edge_index[:, index_bait_gene]

        index_gene_phenotype = (edge_attr == 3).nonzero().reshape(1, -1)[0]
        edge_index_gene_phenotype = edge_index[:, index_gene_phenotype]

        index_drug_phenotype = (edge_attr == 4).nonzero().reshape(1, -1)[0]
        edge_index_drug_phenotype = edge_index[:, index_drug_phenotype]

        x_gene_drug = F.dropout(F.relu(self.conv_gene_drug(x, edge_index_gene_drug)), p=0.5, training=self.training)
        x_gene_gene = F.dropout(F.relu(self.conv_gene_gene(x, edge_index_gene_gene)), p=0.5, training=self.training)
        x_bait_gene = F.dropout(F.relu(self.conv_bait_gene(x, edge_index_bait_gene)), p=0.1, training=self.training)
        x_gene_phenotype = F.dropout(F.relu(self.conv_gene_phenotype(x, edge_index_gene_phenotype)),
                                     training=self.training)
        x_drug_phenotype = F.dropout(F.relu(self.conv_drug_phenotype(x, edge_index_drug_phenotype)),
                                     training=self.training)

        x = self.bn(torch.cat([x_gene_drug, x_gene_gene, x_bait_gene, x_gene_phenotype, x_drug_phenotype], dim=1))

        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
