# -*- coding: utf-8 -*-
import os
import json
import torch
import random
import nxmetis
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from divea.misc import sub_alignment_with_head
from divea.components_base import G1Partitioner
from divea.dataload import UniformData
from divea.components_base import G2Partitioner
from divea.misc import get_neighbours

import pdb

"""Partition G with metis"""
class DivG1Metis(G1Partitioner):
    def __init__(self, data:UniformData, data_dir, kgids, part_n, balance=False):
        super(DivG1Metis, self).__init__(data, data_dir, kgids, part_n)
        self.balance = balance

    def balanced_partitions_list(self, partitions):
        num_partitions = [len(partition) for partition in partitions]
        idx = torch.argsort(torch.tensor(num_partitions).float(), dim=0, descending=True)

        assert self.part_n % 2 == 0, "part_n must be an even number"

        partitions_list = []
        for i in range(self.part_n // 2):
            data = partitions[idx[i]] + partitions[idx[self.part_n - 1 - i]]
            partitions_list.append(data[:len(data) // 2])
            partitions_list.append(data[len(data) // 2:])

        assert sum([len(partition) for partition in partitions]) == sum([len(partition) for partition in partitions_list])       
        return partitions_list

    # use metis to divide the large graph into part_n sub-graphs
    def _divide_entities(self):
        print("index {0}".format(3))
        # graph = nx.Graph()
        # graph.add_nodes_from(self.data.kg1_entities)
        # edges = [(h, t) for h, r, t in self.data.kg1_triples]
        # graph.add_edges_from(edges)

        graph = nx.Graph()
        # here, the type of data must be int, bacause 'nxmetis.partition' only accpet the int type.
        for node in self.data.kg1_entities:
            graph.add_node(node, weight=int(1 / self.data.node_weights1[node] * 100 + 1))
        for h, r, t in self.data.kg1_triples:
            graph.add_edge(h, t, weight=int(self.data.edge_weights1[r] * 100))

        cut, partitions = nxmetis.partition(graph, nparts=self.part_n)
        if self.balance:
            partitions = self.balanced_partitions_list(partitions)

        for i in range(self.part_n):
            with open(os.path.join(self.data_dir, f"partition_{i+1}/tmp_part.json"), "w+") as file:
                # print(len(partitions[i]))
                file.write(json.dumps(partitions[i]))

        test_alignment = self.data.load_test_alignment() # achieve the alignments for test
        # based on part_n sub-graphs, dividing the test alignments into part_n. 
        # 'unmatch_entities_list' only contains the source part of test alignments, and the form of it is [[id0, id1, ...], [], ...]
        unmatch_entities_list = [] 
        for part in partitions:
            sub_align = sub_alignment_with_head(part, test_alignment)
            test_entities = [e1 for e1, e2 in sub_align]
            unmatch_entities_list.append(test_entities)
        return unmatch_entities_list

    def partition_g1_entities(self):
        print("index {0}".format(2))
        part_entities_list = self._divide_entities()
        for idx in range(len(part_entities_list)):
            part_data_dir = os.path.join(self.data_dir, f"partition_{idx+1}")
            obj = {
                "kg1_partition": {
                    "entities": part_entities_list[idx]
                }
            }
            with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
                file.write(json.dumps(obj))

        return part_entities_list

"""Partition G with random algorithm"""
class DivG1Random(DivG1Metis):
    def __init__(self, data:UniformData, data_dir, kgids, part_n):
        super(DivG1Random, self).__init__(data, data_dir, kgids, part_n)

    def _divide_entities(self):
        kg1_entities = self.data.kg1_entities
        # partitioning
        random.shuffle(kg1_entities)
        n_per_part = int(len(kg1_entities) / self.part_n)
        partitions = []
        for i in range(0, self.part_n):
            start_idx = n_per_part * i
            if i == (self.part_n - 1):
                partitions.append(kg1_entities[start_idx:])
            else:
                partitions.append(kg1_entities[start_idx: start_idx + n_per_part])

        for i in range(self.part_n):
            with open(os.path.join(self.data_dir, f"partition_{i+1}/tmp_part.json"), "w+") as file:
                file.write(json.dumps(partitions[i]))

        test_alignment = self.data.load_test_alignment()
        unmatch_entities_list = []
        for part in partitions:
            sub_align = sub_alignment_with_head(part, test_alignment)
            test_entities = [e1 for e1, e2 in sub_align]
            unmatch_entities_list.append(test_entities)

        return unmatch_entities_list

"""==== Counterpart Discovery ===="""
class CounterpartDiscovery(G2Partitioner):
    def __init__(self, data, data_dir, kgids, subtask_num, ctx_g2_size, max_hop_k, out_dir=None, alpha=0.5, beta=1.0, topK=10, ablation="full"):
        super(CounterpartDiscovery, self).__init__(data, data_dir, kgids, subtask_num)
        self.beta = beta
        self.alpha = alpha
        self.ablation = ablation  # full, locality
        self.topK = topK

        self.ctx_g2_size = ctx_g2_size
        self.out_dir = out_dir
        self.max_hop_k = max_hop_k

    def _get_dist_weight(self, neighbours_list):
        ent_to_dist_map = dict()
        neighbours_list.reverse()
        for idx, neis in enumerate(neighbours_list):
            hopk = self.max_hop_k - idx
            for e in neis:
                if e in ent_to_dist_map:
                    ent_to_dist_map[e] = (ent_to_dist_map[e] + hopk) * (1 + 1 / self.data.node_weights2[e])
                else:
                    ent_to_dist_map[e] = hopk * (1 + 1 / self.data.node_weights2[e])

        if len(ent_to_dist_map) == 0:
            return dict()

        dist_list = [v for k,v in ent_to_dist_map.items()]
        max_dist = np.max(dist_list)
        ent_to_dist_weight_map = dict()
        for k, v in ent_to_dist_map.items():
            ent_to_dist_weight_map[k] = max_dist - v
        # "ent_to_dist_weight_map" is a dictionary, and the keys are the entity id, the values are float. Larger means,closer neighbors.
        return ent_to_dist_weight_map

    def get_anchors(self, part_idx):
        all_train_alignment = self.all_train_alignment
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/tmp_part.json")) as file:
            part_entities = json.loads(file.read())
        ent2exist_map = {ent: True for ent in part_entities}
        anchors = set([e2 for e1, e2 in all_train_alignment if e1 in ent2exist_map])
        return anchors

    def _select_g2_candidates(self, part_idx):
        print("index {0}".format(14))
        # cache
        if part_idx == 1:
            conn_arr = np.array(self.data.kg2_triples)[:, [0,2]]
            self.g2_conn_df_cache = pd.DataFrame(data=conn_arr, columns=["h", "t"])
            train_alignment = self.data.load_train_alignment()
            pseudo_alignment = self.load_pseudo_seeds()
            self.all_train_alignment = train_alignment + pseudo_alignment
        anchors = self.get_anchors(part_idx)
        if len(anchors) == 0:
            for i in range(100): # add some training data if not
                anchors.add(self.all_train_alignment[random.randint(0, len(self.all_train_alignment))][1])
        neighbours_list = get_neighbours(self.g2_conn_df_cache, list(anchors), max_hop_k=self.max_hop_k)
        # neighbours_list contain max_hop_k lists, and each list contains a list of neighbors. The deeper list denotes more distant neighbors.
        ent2dist_weight_map = self._get_dist_weight(neighbours_list)
        # print(ent2dist_weight_map)

        ent2weight_map = ent2dist_weight_map

        if "full" == self.ablation:
            if os.path.exists(os.path.join(self.data_dir, f"partition_{part_idx}/tmp_fea.txt")):
                ent2simi_weight_map = self._get_sim_feature(part_idx)
                for k,v in ent2weight_map.items():
                    if k in ent2simi_weight_map:
                        ent2weight_map[k] += (ent2simi_weight_map[k]-self.alpha)*self.beta
        sorted_ent_weight_pairs = sorted(ent2weight_map.items(), key=lambda item: -item[1])
        sorted_nei_entities = [e for e, n in sorted_ent_weight_pairs]
        selected_entities = sorted_nei_entities
        selected_entities = selected_entities[:self.ctx_g2_size]
        # pdb.set_trace()
        tmp_fn = os.path.join(self.data_dir, f"partition_{part_idx}/tmp.txt")
        with open(tmp_fn, "a+") as file:
            file.write(json.dumps(selected_entities) + "\n")
        return selected_entities

    def _get_sim_feature(self, part_idx):
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/tmp_fea.txt")) as file:
            tmp_fea_lines = file.read().strip().split("\n")

        fea_map_list = []
        for idx in range(len(tmp_fea_lines)):
            fea_line = tmp_fea_lines[idx]
            fea_obj = json.loads(fea_line)
            scores = [v[f"score{self.topK}"] for k, v in fea_obj.items()]
            scaler = MinMaxScaler()
            scaler.fit(np.reshape(scores, newshape=(-1, 1)))
            fea_obj = {int(k): scaler.transform([[v[f"score{self.topK}"]]])[0][0] for k, v in fea_obj.items()}
            fea_map_list.append(fea_obj)
        ent2fea_map = dict()
        for fea_obj in fea_map_list:
            for k, v in fea_obj.items():
                ent2fea_map[k] = v  # use latest sim feature
        return ent2fea_map
