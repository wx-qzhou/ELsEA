# -*- coding: utf-8 -*-
import os
import gc
import json
import random
import torch
import numpy as np
from Unsuper.unsupervisedSeeds import obtain_medium_degree_entity, \
    obtain_embed, visual_pivot_induction_mini_batch, place_triplets

def read_tab_lines(fn):
    with open(fn) as file:
        cont = file.read().strip()
        if cont == "":
            return []
        lines = cont.split("\n")
        tuple_list = []
        for line in lines:
            t = line.split("\t")
            tuple_list.append(t)
    return tuple_list

def write_tab_lines(tuple_list, fn, modal="w+"):
    with open(fn, modal) as file:
        for tup in tuple_list:
            s_tup = [str(e) for e in tup]
            file.write("\t".join(s_tup) + "\n")

def read_alignment(fn):
    alignment = read_tab_lines(fn)
    alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
    return alignment


class UniformData:
    def __init__(self, data_dir, kgids=None):
        self.data_dir = data_dir
        data_name = os.path.dirname(data_dir)
        if kgids:
            self.kgid1, self.kgid2 = kgids
        else:
            self.kgid1, self.kgid2 = data_name.split("_")

        kg1_ent_id_uri_list, kg1_rel_id_uri_list, self.kg1_triples = self.load_kg(self.kgid1)
        self.kg1_ent_id2uri_map = dict(kg1_ent_id_uri_list)
        self.kg1_rel_id2uri_map = dict(kg1_rel_id_uri_list)

        kg2_ent_id_uri_list, kg2_rel_id_uri_list, self.kg2_triples = self.load_kg(self.kgid2)
        self.kg2_ent_id2uri_map = dict(kg2_ent_id_uri_list)
        self.kg2_rel_id2uri_map = dict(kg2_rel_id_uri_list)

        self.kg1_entities = sorted(list(self.kg1_ent_id2uri_map.keys()))
        self.kg2_entities = sorted(list(self.kg2_ent_id2uri_map.keys()))
        self.kg1_relations = sorted(list(self.kg1_rel_id2uri_map.keys()))
        self.kg2_relations = sorted(list(self.kg2_rel_id2uri_map.keys()))

        #
        self.kg1_triple_arr = np.array(self.kg1_triples)
        self.kg1_head2triples_map = {}
        self.kg1_tail2triples_map = {}
        for idx, (h,r,t) in enumerate(self.kg1_triples):
            if h not in self.kg1_head2triples_map:
                self.kg1_head2triples_map[h] = []
            self.kg1_head2triples_map[h].append(idx)
            if t not in self.kg1_tail2triples_map:
                self.kg1_tail2triples_map[t] = []
            self.kg1_tail2triples_map[t].append(idx)
            
        self.kg2_triple_arr = np.array(self.kg2_triples)
        self.kg2_head2triples_map = {}
        self.kg2_tail2triples_map = {}
        for idx, (h,r,t) in enumerate(self.kg2_triples):
            if h not in self.kg2_head2triples_map:
                self.kg2_head2triples_map[h] = []
            self.kg2_head2triples_map[h].append(idx)
            if t not in self.kg2_tail2triples_map:
                self.kg2_tail2triples_map[t] = []
            self.kg2_tail2triples_map[t].append(idx)
            
    def load_kg(self, kgid):
        ent_id_uri_list = read_tab_lines(os.path.join(self.data_dir, f"{kgid}_entity_id2uri.txt"))
        rel_id_uri_list = read_tab_lines(os.path.join(self.data_dir, f"{kgid}_relation_id2uri.txt"))
        triple_rel_list = read_tab_lines(os.path.join(self.data_dir, f"{kgid}_triple_rel.txt"))

        ent_id_uri_list = [(int(id), uri) for id, uri in ent_id_uri_list]
        rel_id_uri_list = [(int(id), uri) for id, uri in rel_id_uri_list]
        triple_rel_list = [(int(ent1_id), int(ent2_id), int(rel_id)) for ent1_id, ent2_id, rel_id in triple_rel_list]
        return ent_id_uri_list, rel_id_uri_list, triple_rel_list

    def load_all_alignment(self):
        alignment = read_tab_lines(os.path.join(self.data_dir, "alignment_of_entity.txt"))
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
        return alignment

    def load_train_alignment(self):
        alignment = read_tab_lines(os.path.join(self.data_dir, "train_alignment.txt"))
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
        return alignment

    def load_test_alignment(self):
        alignment = read_tab_lines(os.path.join(self.data_dir, "test_alignment.txt"))
        alignment = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in alignment]
        return alignment

    def kg1_sub_triples(self, entities):
        print("index {0}".format(20))
        head2idxes = []
        tail2idxes = []
        for e in entities:
            head2idxes.extend(self.kg1_head2triples_map.get(e, []))
            tail2idxes.extend(self.kg1_tail2triples_map.get(e, []))
        inter_idxes = list(set(head2idxes).intersection(set(tail2idxes)))
        sub_triples = self.kg1_triple_arr[inter_idxes].tolist()
        return sub_triples

    def kg2_sub_triples(self, entities):
        head2idxes = []
        tail2idxes = []
        for e in entities:
            head2idxes.extend(self.kg2_head2triples_map.get(e, []))
            tail2idxes.extend(self.kg2_tail2triples_map.get(e, []))
        inter_idxes = list(set(head2idxes).intersection(set(tail2idxes)))
        sub_triples = self.kg2_triple_arr[inter_idxes].tolist()
        return sub_triples

    def remove_unlinked_triples(self, triples, linked_entities):
        print("before removing unlinked triples:", len(triples))
        node2batch = {}
        for i, nodes in enumerate(linked_entities):
            node2batch[nodes] = i

        linked_triples = set()
        entities = set()
        relations = set()
        for h, r, t in triples:
            h_batch = node2batch.get(h, -1)
            t_batch = node2batch.get(t, -1)

            # Check if either node belongs to a batch
            if h_batch != -1 and t_batch != -1:
                linked_triples.add((h, r, t))
                entities.add(h)
                entities.add(t)
                relations.add(r)
        print("after removing unlinked triples:", len(linked_triples))
        del triples, linked_entities
        return linked_triples, entities, relations

    def reset_IDs(self, kg1_triples, kg2_triples, kg1_entities, kg2_entities, kg1_relations, kg2_relations):
        # entity
        kg1_ent_uri_tuples = list(enumerate(kg1_entities))
        kg1_ent_num = len(kg1_entities)
        kg2_ent_uri_tuples = [(idx + kg1_ent_num, ent2) for idx, ent2 in enumerate(kg2_entities)]
        kg1_ent_old2new_id_map = {oldid: newid for newid, oldid in kg1_ent_uri_tuples}
        kg2_ent_old2new_id_map = {oldid: newid for newid, oldid in kg2_ent_uri_tuples}

        # relation
        kg1_rel_uri_tuples = list(enumerate(kg1_relations))
        kg1_rel_num = len(kg1_relations)
        kg2_rel_uri_tuples = [(idx + kg1_rel_num, rel2) for idx, rel2 in enumerate(kg2_relations)]
        kg1_rel_old2new_id_map = {oldid: newid for newid, oldid in kg1_rel_uri_tuples}
        kg2_rel_old2new_id_map = {oldid: newid for newid, oldid in kg2_rel_uri_tuples}

        # triples
        new_kg1_triples = [(kg1_ent_old2new_id_map[h], kg1_rel_old2new_id_map[r], kg1_ent_old2new_id_map[t]) for h, r, t in kg1_triples]
        new_kg2_triples = [(kg2_ent_old2new_id_map[h], kg2_rel_old2new_id_map[r], kg2_ent_old2new_id_map[t]) for h, r, t in kg2_triples]

        del kg1_rel_old2new_id_map, kg2_rel_old2new_id_map
        del kg1_ent_old2new_id_map, kg2_ent_old2new_id_map
        del kg1_rel_uri_tuples, kg2_rel_uri_tuples

        return kg1_ent_uri_tuples, kg2_ent_uri_tuples, new_kg1_triples + new_kg2_triples

    def generate_train_alignment(self, num, embed, left_idx, right_idx, kg1_ent_uri_tuples, kg2_ent_uri_tuples, surface=True, ugraph=True, threshold=0.0008, thresholdstr=0.85, \
                                 search_batch_sz=50000, index_batch_sz=500000):
        train_alignment, train_alignment_str = visual_pivot_induction_mini_batch(torch.FloatTensor(embed), left_idx, right_idx, self.data_dir, \
                                                                                 [self.kgid1, self.kgid2], surface=surface, ugraph=ugraph, threshold=threshold, \
                                                                                    thresholdstr=thresholdstr, search_batch_sz=search_batch_sz, index_batch_sz=index_batch_sz)
        if len(train_alignment) < int(num * 0.2) and train_alignment_str != None:
            # train_alignment += random.sample(train_alignment_str, min(int(num * 0.2) - len(train_alignment), len(train_alignment_str)))
            train_alignment += train_alignment_str[:min((int(num * 0.2) - len(train_alignment)), len(train_alignment_str))]
        train_alignment = torch.LongTensor(train_alignment)
        train_alignment = torch.vstack([kg1_ent_uri_tuples[:, 1][train_alignment[:,0]], kg2_ent_uri_tuples[:, 1][train_alignment[:,1]]]).T.tolist()
        train_alignment = [tuple(da) for da in train_alignment]
        gc.collect()
        del embed, left_idx, right_idx, kg1_ent_uri_tuples, kg2_ent_uri_tuples
 
        # train_alignment = list(set(train_alignment))
        train_alignment = list(set(train_alignment))[:int(num * 0.2)]
        return train_alignment

    def divide_train_test(self, train_percent, batch_size=None, surface=True, ugraph=True, thresholdstr=0.85, index_batch_sz=10000):
        all_alignment = self.load_all_alignment()
        num = len(all_alignment)
        if train_percent == 0:
            print("start with unsupervised mode")

            if batch_size != None: 
                print("Here, we use a batch_size to divide some entities.")
                train_alignment = []
                batch_num = (len(all_alignment) // batch_size) + 1
                num = (num // batch_num) + 1
                print("batch_num is {}.".format(batch_num))

                # left_idx_raw, right_idx_raw = torch.LongTensor(obtain_medium_degree_entity(kg1_triples)), \
                # torch.LongTensor(obtain_medium_degree_entity(kg2_triples))
                # dataset is too large, remove unimportant nodes
                kg1_triples_list, left_idx_list  = place_triplets(self.kg1_triples, torch.LongTensor(all_alignment)[:, 0].numpy().tolist(), batch_num, batch_size)
                kg2_triples_list, right_idx_list = place_triplets(self.kg2_triples, torch.LongTensor(all_alignment)[:, 1].numpy().tolist(), batch_num, batch_size)
                batch_num = min(batch_num, len(kg2_triples_list))
                for i in range(batch_num):
                    print("The batch id is {}.".format(i))
                    left_idx = left_idx_list[i]
                    right_idx = right_idx_list[i]
                    if len(kg1_triples_list[i]) > 500000:
                        left_idx, right_idx = torch.LongTensor(obtain_medium_degree_entity(kg1_triples_list[i])), \
                            torch.LongTensor(obtain_medium_degree_entity(kg2_triples_list[i]))
                    left_idx = torch.LongTensor(left_idx)
                    kg1_triples, kg1_entities, kg1_relations = self.remove_unlinked_triples(kg1_triples_list[i], left_idx.numpy().tolist())
                    right_idx = torch.LongTensor(right_idx)
                    kg2_triples, kg2_entities, kg2_relations = self.remove_unlinked_triples(kg2_triples_list[i], right_idx.numpy().tolist())

                    kg1_ent_uri_tuples, kg2_ent_uri_tuples, triples = self.reset_IDs(kg1_triples, kg2_triples, kg1_entities, kg2_entities, \
                                                                                     kg1_relations, kg2_relations)
                    write_tab_lines(kg1_ent_uri_tuples, os.path.join(self.data_dir, "ent_ids_1"), "w")
                    write_tab_lines(kg2_ent_uri_tuples, os.path.join(self.data_dir, "ent_ids_2"), "w")
                    kg1_ent_uri_tuples, kg2_ent_uri_tuples = torch.LongTensor(kg1_ent_uri_tuples), torch.LongTensor(kg2_ent_uri_tuples)
                    left_idx, right_idx = kg1_ent_uri_tuples[:, 0], kg2_ent_uri_tuples[:, 0]

                    if ugraph:
                        embed = obtain_embed(triples, len(kg1_ent_uri_tuples) + len(kg2_ent_uri_tuples), is_Unsuper=True, max_epoch=100)
                    else:
                        embed = [0]
                    train_alignment += self.generate_train_alignment(num, embed, left_idx, right_idx, kg1_ent_uri_tuples, kg2_ent_uri_tuples, \
                                                                     surface=surface, ugraph=ugraph, threshold=0.000008, thresholdstr=thresholdstr,\
                                                                        search_batch_sz=5000, index_batch_sz=index_batch_sz)
                    # exit() 
            else:
                left_idx, right_idx = torch.LongTensor(self.kg1_triples), torch.LongTensor(self.kg2_triples)
                kg1_triples, kg1_entities, kg1_relations = self.kg1_triples, self.kg1_entities, self.kg1_relations
                kg2_triples, kg2_entities, kg2_relations = self.kg2_triples, self.kg2_entities, self.kg2_relations

                kg1_ent_uri_tuples, kg2_ent_uri_tuples, triples = self.reset_IDs(kg1_triples, kg2_triples, kg1_entities, kg2_entities, kg1_relations, kg2_relations)
                write_tab_lines(kg1_ent_uri_tuples, os.path.join(self.data_dir, "ent_ids_1"), "w")
                write_tab_lines(kg2_ent_uri_tuples, os.path.join(self.data_dir, "ent_ids_2"), "w")
                kg1_ent_uri_tuples, kg2_ent_uri_tuples = torch.LongTensor(kg1_ent_uri_tuples), torch.LongTensor(kg2_ent_uri_tuples)
                left_idx, right_idx = kg1_ent_uri_tuples[:, 0], kg2_ent_uri_tuples[:, 0]

                if ugraph:
                    embed = obtain_embed(triples, len(kg1_ent_uri_tuples) + len(kg2_ent_uri_tuples), is_Unsuper=True)
                else:
                    embed = [0]
                train_alignment = self.generate_train_alignment(num, embed, left_idx, right_idx, kg1_ent_uri_tuples, kg2_ent_uri_tuples, surface=surface, \
                                                                ugraph=ugraph, thresholdstr=thresholdstr)
            test_alignment = all_alignment
            gc.collect()
        else:
            train_num = int(num * train_percent)
            random.shuffle(all_alignment)
            train_alignment = all_alignment[:train_num]
            test_alignment = all_alignment[train_num:]
            pseudo_fn = os.path.join(self.data_dir, "name_pseudo_mappings.txt")
            if os.path.exists(pseudo_fn):
                pseudo_mappings = read_tab_lines(pseudo_fn)
                pseudo_mappings = [(int(ent1_id), int(ent2_id)) for ent1_id, ent2_id in pseudo_mappings]
                train_alignment = train_alignment + pseudo_mappings
        write_tab_lines(train_alignment, os.path.join(self.data_dir, "train_alignment.txt"), modal="w")
        write_tab_lines(test_alignment, os.path.join(self.data_dir, "test_alignment.txt"), modal="w")

    def load_weights(self):
        with open(os.path.join(self.data_dir, "KGs_edge_info.json")) as file:
            obj = json.loads(file.read())
        self.edge_weights1 = torch.zeros(len(self.kg1_relations))
        for r in self.kg1_relations:
            self.edge_weights1[r] = obj["KG1"]["r2f"][str(r)] + obj["KG1"]["r2if"][str(r)]
        self.edge_weights2 = torch.zeros(len(self.kg2_relations))
        for r in self.kg2_relations:
            self.edge_weights2[r] = obj["KG2"]["r2f"][str(r)] + obj["KG2"]["r2if"][str(r)]

        with open(os.path.join(self.data_dir, "KGs_node_info.json")) as file:
            obj = json.loads(file.read())
        self.node_weights1 = torch.zeros(len(self.kg1_entities))
        for k in self.kg1_entities:
            self.node_weights1[k] = obj["KG1"]['in_degree'][str(k)] + obj["KG1"]['out_degree'][str(k)] + 1
        self.node_weights2 = torch.zeros(len(self.kg2_entities))
        for k in self.kg2_entities:
            self.node_weights2[k] = obj["KG2"]['in_degree'][str(k)] + obj["KG2"]['out_degree'][str(k)] + 1

def convert_uniform_to_rrea(data_dir, kgids):
    print("index {0}".format(26))
    uni_data = UniformData(data_dir, kgids)

    kg1_entities = uni_data.kg1_entities
    kg2_entities = uni_data.kg2_entities
    kg1_ent_uri_tuples = list(enumerate(kg1_entities))
    kg1_ent_num = len(kg1_entities)
    kg2_ent_uri_tuples = [(idx+kg1_ent_num, ent2) for idx, ent2 in enumerate(kg2_entities)]

    kg1_ent_old2new_id_map = {oldid: newid for newid, oldid in kg1_ent_uri_tuples}
    kg2_ent_old2new_id_map = {oldid: newid for newid, oldid in kg2_ent_uri_tuples}

    kg1_relations = uni_data.kg1_relations
    kg2_relations = uni_data.kg2_relations
    kg1_rel_uri_tuples = list(enumerate(kg1_relations))
    kg1_rel_num = len(kg2_relations)
    kg2_rel_uri_tuples = [(idx + kg1_rel_num, rel2) for idx, rel2 in enumerate(kg2_relations)]

    kg1_rel_old2new_id_map = {oldid: newid for newid, oldid in kg1_rel_uri_tuples}
    kg2_rel_old2new_id_map = {oldid: newid for newid, oldid in kg2_rel_uri_tuples}


    new_kg1_triples = [(kg1_ent_old2new_id_map[h], kg1_rel_old2new_id_map[r], kg1_ent_old2new_id_map[t]) for h,r,t in uni_data.kg1_triples]
    new_kg2_triples = [(kg2_ent_old2new_id_map[h], kg2_rel_old2new_id_map[r], kg2_ent_old2new_id_map[t]) for h, r, t in uni_data.kg2_triples]

    new_all_alignment = [(kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]) for e1, e2 in uni_data.load_all_alignment()]
    new_train_alignment = [(kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]) for e1, e2 in uni_data.load_train_alignment()]
    test_alignment = uni_data.load_test_alignment()
    valid_test_alignment = []
    invalid_test_alignment = []
    new_test_alignment = []
    for e1, e2 in test_alignment:
        if e2 in kg2_ent_old2new_id_map:
            new_test_alignment.append((kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]))
            valid_test_alignment.append((e1, e2))
        else:
            invalid_test_alignment.append((e1, e2))
    # new_test_alignment = [(kg1_ent_old2new_id_map[e1], kg2_ent_old2new_id_map[e2]) for e1, e2 in uni_data.load_test_alignment()]

    write_tab_lines(kg1_ent_uri_tuples, os.path.join(data_dir, "ent_ids_1"))
    write_tab_lines(kg2_ent_uri_tuples, os.path.join(data_dir, "ent_ids_2"))
    write_tab_lines(new_kg1_triples, os.path.join(data_dir, "triples_1"))
    write_tab_lines(new_kg2_triples, os.path.join(data_dir, "triples_2"))
    write_tab_lines(new_all_alignment, os.path.join(data_dir, "ref_ent_ids"))
    write_tab_lines(new_train_alignment, os.path.join(data_dir, "ref_ent_ids_train"))
    write_tab_lines(new_test_alignment, os.path.join(data_dir, "ref_ent_ids_test"))
    write_tab_lines(invalid_test_alignment, os.path.join(data_dir, "test_alignment_invalid.txt"))
    write_tab_lines(valid_test_alignment, os.path.join(data_dir, "test_alignment_valid.txt"))

def convert_uniform_to_openea(data_dir, kgids, out_dir=None):
    uni_data = UniformData(data_dir, kgids)

    ent_links = [(uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]) for e1, e2 in uni_data.load_all_alignment()]
    kg1_rel_triples = [(uni_data.kg1_ent_id2uri_map[h], uni_data.kg1_rel_id2uri_map[r], uni_data.kg1_ent_id2uri_map[t]) for h,r,t in uni_data.kg1_triples]
    kg2_rel_triples = [(uni_data.kg2_ent_id2uri_map[h], uni_data.kg2_rel_id2uri_map[r], uni_data.kg2_ent_id2uri_map[t]) for h,r,t in uni_data.kg2_triples]

    train_ent_links = [(uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]) for e1, e2 in uni_data.load_train_alignment()]
    test_alignment = uni_data.load_test_alignment()
    valid_test_alignment = []
    invalid_test_alignment = []
    new_test_alignment = []
    for e1, e2 in test_alignment:
        if e2 in uni_data.kg2_ent_id2uri_map:
            new_test_alignment.append((uni_data.kg1_ent_id2uri_map[e1], uni_data.kg2_ent_id2uri_map[e2]))
            valid_test_alignment.append((e1, e2))
        else:
            invalid_test_alignment.append((e1, e2))
    if out_dir is None:
        out_dir = os.path.join(data_dir, "openea_format")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    write_tab_lines(kg1_rel_triples, os.path.join(out_dir, "rel_triples_1"))
    write_tab_lines(kg2_rel_triples, os.path.join(out_dir, "rel_triples_2"))
    write_tab_lines(ent_links, os.path.join(out_dir, "ent_links"))
    with open(os.path.join(out_dir, "attr_triples_1"), "w+") as file:
        file.write("")
    with open(os.path.join(out_dir, "attr_triples_2"), "w+") as file:
        file.write("")

    partition_dir = os.path.join(out_dir, "partition")
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    write_tab_lines(train_ent_links, os.path.join(partition_dir, "train_links"))
    write_tab_lines(new_test_alignment, os.path.join(partition_dir, "test_links"))
    write_tab_lines(invalid_test_alignment, os.path.join(partition_dir, "test_alignment_invalid.txt"))
    write_tab_lines(valid_test_alignment, os.path.join(partition_dir, "test_alignment_valid.txt"))
    with open(os.path.join(partition_dir, "valid_links"), "w+") as file:
        cont = "\n".join([])
        file.write(cont)
