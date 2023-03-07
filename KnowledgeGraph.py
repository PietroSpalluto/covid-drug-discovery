from collections import defaultdict
from typing import List

import pandas as pd
import torch


class KnowledgeGraph:
    def __init__(self, edge_index, edge_attr_dict):
        self.edge_index = edge_index
        self.edge_attr_dict = edge_attr_dict
        self.entity_dict = {}
        self.entity_dict_id = {}
        self.entities = []
        self.relation_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.triples_set = []  # list of triples in the form of (h, t, r)
        self.corr_triples = []

    # builds the triples
    def triples(self):
        self.entity_dict = dict(zip(list(pd.concat([self.edge_index['node1'], self.edge_index['node2']])),
                                    list(pd.concat([self.edge_index['node1id'], self.edge_index['node2id']]))))
        self.entity_dict_id = dict(zip(list(pd.concat([self.edge_index['node1id'], self.edge_index['node2id']])),
                                       list(pd.concat([self.edge_index['node1'], self.edge_index['node2']]))))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())

        self.relation_dict = self.edge_attr_dict
        self.n_relation = len(self.relation_dict)

        print('#entity: {}'.format(self.n_entity))
        print('#relation: {}'.format(self.n_relation))

        triples_ = self.edge_index[['node1', 'node2', 'type']]

        self.triples_set = list(zip([self.entity_dict[h] for h in triples_['node1']],
                                    [self.entity_dict[t] for t in triples_['node2']],
                                    [self.relation_dict[r] for r in triples_['type']]))

    def load_lookup_dictionaries(self, triples: List[List]):
        head2tail_lookup = defaultdict(lambda: defaultdict(set))
        tail2head_lookup = defaultdict(lambda: defaultdict(set))

        for head_id, relation_id, tail_id in triples:
            head2tail_lookup[head_id][relation_id].add(tail_id)
            tail2head_lookup[tail_id][relation_id].add(head_id)

    def get_corrupted_training_triples(self, triples: torch.tensor) -> torch.tensor:
        return torch.tensor(
            list(map(lambda x: self.corrupt_training_triple(x[0].item(), x[1].item(), x[2].item()), triples)))

    def corrupt_training_triple(self, head_id: int, relation_id: int, tail_id: int) -> List[int]:
        if torch.rand(1).uniform_(0, 1).item() >= 0.5:
            initial_head_id = head_id
            while head_id in self.triples.tail2head_lookup[tail_id][relation_id] or head_id == initial_head_id:
                head_id = torch.randint(self.num_of_entities, (1,)).item()

        else:
            initial_tail_id = tail_id
            while tail_id in self.triples.head2tail_lookup[head_id][relation_id] or tail_id == initial_tail_id:
                tail_id = torch.randint(self.num_of_entities, (1,)).item()

        return [head_id, relation_id, tail_id]
