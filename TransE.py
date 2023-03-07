import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm


class TransE(nn.Module):
    def __init__(self, kg, num_of_dimension: int, batch_size=100, learning_rate=0.01, num_of_epochs=10, margin=1,
                 norm1: int = 1, norm2: int = 2):
        super().__init__()
        self.num_of_entities = kg.n_entity
        self.num_of_relations = kg.n_relation
        self.num_of_dimension = num_of_dimension
        self.triples = kg.triples_set
        self.norm1 = norm1
        self.norm2 = norm2
        self.kg = kg
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        self.margin = margin
        self.corr_triples = self.corrupt_triples()
        self.array_loss = []

        with torch.no_grad():
            self.entity_embeddings = nn.Embedding(self.num_of_entities, self.num_of_dimension)
            self.entity_embeddings.weight.data.uniform_(-6 / self.num_of_dimension ** 0.5,
                                                        6 / self.num_of_dimension ** 0.5)

            self.relation_embeddings = nn.Embedding(self.num_of_relations, self.num_of_dimension)
            self.relation_embeddings.weight.data.uniform_(-6 / self.num_of_dimension ** 0.5,
                                                          6 / self.num_of_dimension ** 0.5)
            # article TRANSLATING EMBEDDINGS FOR MODELING MULTIRELATIONAL DATA
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)

    def corrupt_triples(self):
        corr_triples = []
        for triple in self.triples:
            if torch.rand(1).uniform_(0, 1).item() >= 0.5:
                head_id = self.kg.entities[torch.randint(self.num_of_entities, (1,)).item()]
                tail_id = triple[1]
                relation_id = triple[2]
            else:
                head_id = triple[0]
                tail_id = self.kg.entities[torch.randint(self.num_of_entities, (1,)).item()]
                relation_id = triple[2]
            corr_triple = (head_id, tail_id, relation_id)
            corr_triples.append(corr_triple)
        return corr_triples

    def forward(self, batch: torch.tensor, corrupted_batch: torch.tensor):
        # normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)

        # destructure batch into head_ids, relation_ids, tail_ids
        batch_head_ids = batch[0]
        batch_tail_ids = batch[1]
        batch_relation_ids = batch[2]

        corr_batch_head_ids = corrupted_batch[0]
        corr_batch_tail_ids = corrupted_batch[1]
        corr_batch_relation_ids = corrupted_batch[2]

        # get corresponding embeddings
        batch_head_embeddings = self.entity_embeddings(batch_head_ids)
        batch_relation_embeddings = self.relation_embeddings(batch_relation_ids)
        batch_tail_embeddings = self.entity_embeddings(batch_tail_ids)

        corr_batch_head_embeddings = self.entity_embeddings(corr_batch_head_ids)
        corr_batch_relation_embeddings = self.relation_embeddings(corr_batch_relation_ids)
        corr_batch_tail_embeddings = self.entity_embeddings(corr_batch_tail_ids)

        batch_energies = batch_head_embeddings + batch_relation_embeddings - batch_tail_embeddings
        corr_batch_energies = corr_batch_head_embeddings + corr_batch_relation_embeddings - corr_batch_tail_embeddings

        return batch_energies, corr_batch_energies

    def train(self, filtered_corrupted_batch=False):
        train_dl = DataLoader(self.triples, batch_size=self.batch_size)
        corr_train_dl = DataLoader(self.corr_triples, batch_size=self.batch_size)
        params = list(self.entity_embeddings.parameters()) + list(self.relation_embeddings.parameters())
        optimizer = optim.SGD(params, lr=self.learning_rate)

        for epoch in range(self.num_of_epochs):
            epoch_loss = 0
            for batch, corr_batch in zip(tqdm(train_dl), tqdm(corr_train_dl)):
                # mini_batch = batch[0]
                mini_batch = batch
                # corr_mini_batch = corr_batch[0]
                corr_mini_batch = corr_batch

                batch_loss, corr_batch_loss = self.forward(mini_batch, corr_mini_batch)
                loss = F.relu(self.margin + batch_loss.norm(p=self.norm1, dim=1)
                              - corr_batch_loss.norm(p=self.norm1, dim=1)).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss
            self.array_loss.append(epoch_loss.tolist())
            print('\nEpoch ', epoch, ' ', epoch_loss.tolist())
        return self.entity_embeddings.weight.data.numpy(), self.relation_embeddings.weight.data.numpy(), self.array_loss
