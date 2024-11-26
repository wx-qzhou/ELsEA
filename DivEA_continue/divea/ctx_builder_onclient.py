import os
import gc
import json
import torch
import random
import nvidia_smi
from torch import multiprocessing
from divea.components_base import ContextBuilder
from divea.neighborEhance import neighbor_multi_proc_cpus, neighbor_multi_proc, neighbor_nomulpress_multi_proc, \
    Neighbor_Agg_Layer, Neighbor_Ehance, Neighbor_Ehance_generate_split_data, neighbor_nomulpress_multi_proc
from divea.eviPassing import compute_perf_jointly_multi_proc, compute_perf_multi_proc
import pdb


class CtxBuilderV1(ContextBuilder):
    candi_map_list = None
    def __init__(self, data, data_dir, kgids, out_dir, subtask_num, gcn_l, ctx_g1_size, ctx_g2_size, gamma=2, ctx_g2_conn_percent=0.0, torch_devices=["cpu"], \
                 is_mulprocess=False, proc_n=2):
        super().__init__(data, data_dir, kgids, subtask_num, 0.0, g1_ctx_only_once=True)
        self.gcn_l = gcn_l
        self.devices = torch_devices
        self.out_dir = out_dir
        self.gamma = gamma
        self.g1_ctx_size = ctx_g1_size
        self.g2_ctx_size = ctx_g2_size
        self.g2_conn_ent_percent = ctx_g2_conn_percent
        self.g2_conn_ent_num = int(self.g2_ctx_size * self.g2_conn_ent_percent)
        self.is_mulprocess = is_mulprocess
        self.proc_n = proc_n
        
    def _build_g1_context(self, part_idx, neighbor_enhance=False):
        print("index {0}".format(11))
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
            part_obj = json.loads(file.read())
            g1_part_entities = part_obj["kg1_partition"]["entities"]

        if part_idx == 1:
            labelled_alignmennt = self.data.load_train_alignment()
            pseudo_alignment = self.load_pseudo_seeds() # existing return [(id, id), ..], else return []
            self.train_alignment = labelled_alignmennt + pseudo_alignment

        train_alignment = self.train_alignment
        g1_anchors = [e1 for e1, e2 in train_alignment] # source part
        ctx_entities = []
        ctx_entities += self._build_context_for_single_graph1(g1_part_entities, self.data.kg1_entities, self.data.kg1_triples, g1_anchors, self.g1_ctx_size, part_idx)
        # if neighbor_enhance:
        #     g1_part_entities = list((set(g1_part_entities) - set(ctx_entities)) - set(g1_anchors))
        #     ctx_entities += self._build_context_for_neighbor_enhance_graph1(g1_part_entities, self.data.kg1_entities, g1_anchors, \
        #                                                                     self.g1_ctx_size, part_idx)

        #
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
        obj["kg1_partition"]["ctx_entities"] = ctx_entities # here add the key: "ctx_entities" into obj["kg1_partition"]
        ctx_anchors = list(set(g1_anchors).intersection(set(ctx_entities)))
        train_align_map = dict(train_alignment)
        ctx_train_alignment = [(anc, train_align_map[anc]) for anc in ctx_anchors] # The entities are both in the train alignment and in context graph.
        obj["kg1_partition"]["ctx_train_alignment"] = ctx_train_alignment # here add the key: "ctx_train_alignment" into obj["kg1_partition"]
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(obj))

    def _build_g2_context(self, part_idx, neighbor_enhance=True):
        print("index {0}".format(16))
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
            part_obj = json.loads(file.read())
            g2_part_entities = part_obj["kg2_partition"]["entities"]

        if part_idx == 1:
            labelled_alignmennt = self.data.load_train_alignment()
            pseudo_alignment = self.load_pseudo_seeds()
            self.train_alignment = labelled_alignmennt + pseudo_alignment

        train_alignment = self.train_alignment
        g1_entities = part_obj["kg1_partition"]["entities"] + part_obj["kg1_partition"]["ctx_entities"]
        g1_ent2exist_map = {e:True for e in g1_entities}
        train_alignment = [(e1,e2) for e1, e2 in train_alignment if e1 in g1_ent2exist_map]

        g2_anchors = list(set([e2 for e1, e2 in train_alignment]))
        g2_part_entities_ = list(set(g2_part_entities) - set(g2_anchors))
        random.shuffle(g2_part_entities_)
        link_entities = []
        # link_entities += self._build_context_for_single_graph2(g2_part_entities[:(self.g2_ctx_size - self.g2_conn_ent_num - len(g2_anchors))], self.data.kg2_entities, self.data.kg2_triples, \
        #     g2_anchors, self.g2_conn_ent_num)
        if "1m" in self.data_dir and neighbor_enhance and (self.g2_ctx_neighbor_have_done == False):
            print("here")
            link_entities += self._build_context_for_neighbor_enhance_graph2(g2_part_entities[:(self.g2_ctx_size - self.g2_conn_ent_num - len(g2_anchors))], \
                self.data.kg2_triples, g2_anchors, self.g2_ctx_size - self.g2_conn_ent_num - len(g2_anchors))
        elif neighbor_enhance and (self.g2_ctx_neighbor_have_done == False):
            link_entities += self._build_context_for_neighbor_enhance_graph2(g2_part_entities_[:(self.g2_ctx_size - len(g2_anchors)) * 2], \
                self.data.kg2_triples, g2_anchors, self.g2_ctx_size - self.g2_conn_ent_num - len(g2_anchors))
        elif self.g2_ctx_neighbor_have_done:
            link_entities = part_obj["kg2_partition"]["ctx_entities"]
        ctx_entities = list(set(g2_anchors + link_entities))
        # self._save_context(part_idx, g1_ctx_entities=None, g2_ctx_entities=ctx_entities)
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
            obj["kg2_partition"]["ctx_entities"] = ctx_entities
            obj["kg2_partition"]["entities"] = g2_part_entities
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(obj))

    def _save_context(self, part_idx, g1_ctx_entities, g2_ctx_entities):
        part_data_dir = os.path.join(self.data_dir, f"partition_{part_idx}")
        with open(os.path.join(part_data_dir, "kg_partition.json")) as file:
            obj = json.loads(file.read())
        if g1_ctx_entities is not None:
            obj["kg1_partition"]["ctx_entities"] = g1_ctx_entities
        if g2_ctx_entities is not None:
            obj["kg2_partition"]["ctx_entities"] = g2_ctx_entities
        with open(os.path.join(part_data_dir, "kg_partition.json"), "w+") as file:
            file.write(json.dumps(obj))

    def _filter_g1(self, triples, anchors, unmatch_entities, max_hop_k, part_idx):
        with open(os.path.join(self.data_dir, f"partition_{part_idx}/tmp_part.json")) as file:
            graph_partition_entities = json.loads(file.read())
        nei_ent_set = self.get_neighbours4(triples, graph_partition_entities, anchors, max_hop_k=max_hop_k)
        inv_ent_set = set(graph_partition_entities)
        inv_ent_set.update(nei_ent_set)
        candidate_set = inv_ent_set.difference(set(unmatch_entities))
        fil_entities = list(inv_ent_set)
        fil_triples = self.data.kg1_sub_triples(fil_entities)
        fil_anchors = list(set(anchors).intersection(set(fil_entities)))
        return fil_entities, fil_triples, fil_anchors, candidate_set

    def _build_context_for_single_graph1(self, unmatch_entities, entities, triples, anchors, ctx_graph_size, part_idx):
        fil_entities, fil_triples, fil_anchors, all_candidates = self._filter_g1(triples, anchors, unmatch_entities, max_hop_k=1, part_idx=part_idx)

        old2new_entid_map = {e: idx for idx, e in enumerate(fil_entities)}
        new2old_entid_map = {v:k for k,v in old2new_entid_map.items()}
        new_entities = [old2new_entid_map[e] for e in fil_entities]
        new_triples = [(old2new_entid_map[h], r, old2new_entid_map[t]) for h,r,t in fil_triples]
        new_anchors = [old2new_entid_map[e] for e in fil_anchors]
        new_candidates = [old2new_entid_map[e] for e in all_candidates]
        new_unmatch_entities = [old2new_entid_map[e] for e in unmatch_entities]
        new_candi2benefit_map = compute_perf_multi_proc(self.data.edge_weights1, len(new_entities), new_entities, new_triples, self.gcn_l, self.gamma, \
                                                        new_candidates, new_anchors, new_unmatch_entities, self.devices, self.is_mulprocess, self.proc_n)
        candi2benefit_map = {new2old_entid_map[k]:v for k,v in new_candi2benefit_map.items()}

        sel_num = max(0, ctx_graph_size-len(unmatch_entities))
        new_candi2effect_map = candi2benefit_map
        sorted_items = sorted(new_candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones

        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g1 ctx")
        return sel_entities

    def _filter_g2(self, triples, anchors, unmatch_entities, max_hop_k, ignore_more=None):
        nei_list = self.get_neighbours2(triples, anchors, max_hop_k=max_hop_k)
        nei_ent_set = set()
        for neis in nei_list:
            nei_ent_set.update(neis)
        if ignore_more:
            if len(nei_ent_set) > ignore_more:
                nei_ent_set = list(nei_ent_set)
                random.shuffle(nei_ent_set)
                nei_ent_set = set(nei_ent_set[:ignore_more])
        nei_ent_set = nei_ent_set
        inv_ent_set = set(unmatch_entities+anchors)
        inv_ent_set.update(nei_ent_set)
        candidate_set = nei_ent_set.difference(set(unmatch_entities+anchors))
        fil_entities = list(inv_ent_set)
        fil_triples = self.data.kg2_sub_triples(fil_entities)
        fil_anchors = list(set(anchors).intersection(set(fil_entities)))
        # pdb.set_trace()
        return fil_entities, fil_triples, fil_anchors, candidate_set

    def _build_context_for_single_graph2(self, unmatch_entities, entities, triples, anchors, sel_num):
        print("index {0}".format(17))
        if sel_num == 0:
            print("skip building ctx G2")
            return []

        fil_entities, fil_triples, fil_anchors, all_candidates = self._filter_g2(triples, anchors, unmatch_entities, max_hop_k=1, ignore_more=4000)

        old2new_entid_map = {e: idx for idx, e in enumerate(fil_entities)}
        new2old_entid_map = {v:k for k,v in old2new_entid_map.items()}
        new_entities = [old2new_entid_map[e] for e in fil_entities]
        new_triples = [(old2new_entid_map[h], r, old2new_entid_map[t]) for h,r,t in fil_triples]
        new_anchors = [old2new_entid_map[e] for e in fil_anchors]
        new_candidates = [old2new_entid_map[e] for e in all_candidates]
        new_unmatch_entities = [old2new_entid_map[e] for e in unmatch_entities]
        new_candi2effect_map = compute_perf_multi_proc(self.data.edge_weights2, len(new_entities), new_entities, new_triples, self.gcn_l, self.gamma, \
                                                       new_candidates, new_anchors, new_unmatch_entities, self.devices, self.is_mulprocess, self.proc_n)
        candi2effect_map = {new2old_entid_map[k]:v for k,v in new_candi2effect_map.items()}

        sorted_items = sorted(candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones
        # pdb.set_trace()
        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g2 ctx")
        return sel_entities

    def _cache_g2_ents(self):
        print("cache g2 ent perf")
        gpu_no = int(self.devices[0][-1])
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_no)
        info_before = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        agg_model = Neighbor_Agg_Layer(self.data.edge_weights2, 1 / self.data.node_weights2, len(self.data.kg2_entities), self.data.kg2_entities, self.data.kg2_triples)
        
        labelled_alignmennt = self.data.load_train_alignment()
        pseudo_alignment = self.load_pseudo_seeds() # in 1-th epoch is []
        train_alignment = labelled_alignmennt + pseudo_alignment
        g2_anchors = [e2 for e1, e2 in train_alignment] # This list only contains the source part of train alignments.
        
        h_o, x = agg_model(g2_anchors, self.devices[0])
        h_o, x = h_o.cpu().reshape(-1, 1), x.cpu().reshape(-1, 1)

        info_after = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        with open(os.path.join(self.out_dir, "tmp_running.log"), "a+") as file:
            msg1 = {"msg_type": "ctx1_gpu_mem_usage_before", "value": info_before.used/1024/1024}
            file.write(json.dumps(msg1)+"\n")
            msg2 = {"msg_type": "ctx1_gpu_mem_usage_after", "value": info_after.used/1024/1024}
            file.write(json.dumps(msg2)+"\n")
        del agg_model
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cat((h_o, x), dim=-1)

    def _build_context_for_neighbor_enhance_graph1(self, unmatch_entities, triples, anchors, ctx_graph_size, part_idx):
        fil_entities, fil_triples, fil_anchors, all_candidates = self._filter_g1(triples, anchors, unmatch_entities, max_hop_k=1, part_idx=part_idx)

        old2new_entid_map = {e: idx for idx, e in enumerate(fil_entities)}
        new2old_entid_map = {v:k for k,v in old2new_entid_map.items()}
        new_entities = [old2new_entid_map[e] for e in fil_entities]
        new_triples = [(old2new_entid_map[h], r, old2new_entid_map[t]) for h,r,t in fil_triples]
        new_anchors = [old2new_entid_map[e] for e in fil_anchors]
        new_candidates = [old2new_entid_map[e] for e in all_candidates]
        new_unmatch_entities = [old2new_entid_map[e] for e in unmatch_entities]

        if self.is_mulprocess:
            new_candi2benefit_map = neighbor_nomulpress_multi_proc(self.data.edge_weights1, \
                                                        1 / self.data.node_weights1[torch.tensor(list(old2new_entid_map.keys()), dtype=torch.long)], \
                                                            new_entities, new_triples, new_candidates, new_anchors, new_unmatch_entities, self.devices, self.proc_n)
        else:
            agg_model = Neighbor_Agg_Layer(self.data.edge_weights1, 1 / self.data.node_weights1[torch.tensor(list(old2new_entid_map.keys()), dtype=torch.long)], \
                        len(new_entities), new_entities, new_triples)
            h_o, x = agg_model(new_anchors, self.devices[0])
            h_o, x = h_o.cpu(), x.cpu()
            del agg_model
            new_candi2benefit_map = dict()
            self.proc_n = min(multiprocessing.cpu_count() // 2, 20, self.proc_n)
            batch_size = int(len(new_candidates) / self.proc_n) + 1
            for i in range(self.proc_n):
                device = self.devices[int(i % len(self.devices))]
                st = batch_size * i
                ed = batch_size * (i+1)
                sub_candi_entities = new_candidates[st:ed]

                all_candidates3, unmatch_entities3 = Neighbor_Ehance_generate_split_data(sub_candi_entities, new_unmatch_entities, h_o, x)
                model= Neighbor_Ehance(device)
                candi2benefit_map = model.aggregate_nodes(sub_candi_entities, all_candidates3, unmatch_entities3)

                new_candi2benefit_map.update(candi2benefit_map)
                del sub_candi_entities, all_candidates3, unmatch_entities3, model
        del h_o, x   

        candi2benefit_map = {new2old_entid_map[k]:v for k,v in new_candi2benefit_map.items()}

        sel_num = max(0, ctx_graph_size-len(unmatch_entities))
        sorted_items = sorted(candi2benefit_map.items(), key=lambda item: item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones

        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g1 ctx")
        return sel_entities

    def _build_context_for_neighbor_enhance_graph2(self, unmatch_entities, triples, anchors, sel_num, is_GPU=False):
        print("index {0}".format(17))
        if sel_num == 0:
            print("skip building ctx G2")
            return []

        ignore_more=8000
        # if "1m" in self.data_dir:
        #     print("here")
        #     ignore_more=None

        fil_entities, fil_triples, fil_anchors, all_candidates = self._filter_g2(triples, anchors, unmatch_entities, max_hop_k=1, ignore_more=ignore_more)
        
        old2new_entid_map = {e: idx for idx, e in enumerate(fil_entities)}
        new2old_entid_map = {v:k for k,v in old2new_entid_map.items()}
        # new_entities = [old2new_entid_map[e] for e in fil_entities]
        # new_triples = [(old2new_entid_map[h], r, old2new_entid_map[t]) for h,r,t in fil_triples]
        # new_anchors = [old2new_entid_map[e] for e in fil_anchors]
        new_candidates = [old2new_entid_map[e] for e in all_candidates]
        new_unmatch_entities = [old2new_entid_map[e] for e in unmatch_entities]

        if self.candi_map_list is None:
            if os.path.exists(os.path.join(self.out_dir, "g2context.pt")):
                print("load before g2context")
                self.candi_map_list = torch.load(os.path.join(self.out_dir, "g2context.pt"))
            else:
                self.candi_map_list = self._cache_g2_ents()
                torch.save(self.candi_map_list, os.path.join(self.out_dir, "g2context.pt"))

        h_o, x = self.candi_map_list[:, 0][torch.tensor(list(old2new_entid_map.keys()), dtype=torch.long)], \
            self.candi_map_list[:, 1][torch.tensor(list(old2new_entid_map.keys()), dtype=torch.long)]
    
        if self.is_mulprocess and is_GPU:
            new_candi2benefit_map = neighbor_multi_proc(h_o, x, new_candidates, new_unmatch_entities, self.devices, self.proc_n)
        elif self.is_mulprocess and (is_GPU == False):
            new_candi2benefit_map = neighbor_multi_proc_cpus(h_o, x, new_candidates, new_unmatch_entities, self.devices, self.proc_n)
        else:
            new_candi2benefit_map = dict()
            self.proc_n = min(multiprocessing.cpu_count() // 2, 20, self.proc_n)
            batch_size = int(len(new_candidates) / self.proc_n) + 1
            for i in range(self.proc_n):
                device = self.devices[int(i % len(self.devices))]
                st = batch_size * i
                ed = batch_size * (i+1)
                sub_candi_entities = new_candidates[st:ed]

                all_candidates3, unmatch_entities3 = Neighbor_Ehance_generate_split_data(sub_candi_entities, new_unmatch_entities, h_o, x)
                model= Neighbor_Ehance(device)
                candi2benefit_map = model.aggregate_nodes(sub_candi_entities, all_candidates3, unmatch_entities3)

                new_candi2benefit_map.update(candi2benefit_map) 
                del all_candidates3, unmatch_entities3, sub_candi_entities, model

        del h_o, x


        candi2effect_map = {new2old_entid_map[k]:v for k,v in new_candi2benefit_map.items()}

        sorted_items = sorted(candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones
        # pdb.set_trace()
        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g2 ctx")
        return sel_entities


class CtxBuilderV2(CtxBuilderV1):
    candi2benefit_map_list = None

    def __init__(self, data, data_dir, kgids, out_dir, subtask_num, gcn_l, gamma, ctx_g1_size=None, ctx_g2_size=None, ctx_g2_conn_percent=0.0, \
                 torch_devices=["cpu"], is_mulprocess=False, proc_n=2):
        super().__init__(data, data_dir, kgids, out_dir, subtask_num, gcn_l, ctx_g1_size, ctx_g2_size, gamma=gamma, ctx_g2_conn_percent=ctx_g2_conn_percent, \
                         torch_devices=torch_devices, is_mulprocess=is_mulprocess, proc_n=proc_n)

    def cache_g1_ent_effect(self):
        print("index {0}".format(7))
        print("cache g1 ent perf")
        gpu_no = int(self.devices[0][-1])
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_no)
        info_before = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        unmatched_entities_list = [] # This list only contains the source part of test alignments.
        for part_idx in range(1, self.part_n+1):
            with open(os.path.join(self.data_dir, f"partition_{part_idx}/kg_partition.json")) as file:
                part_obj = json.loads(file.read())
                g1_part_entities = part_obj["kg1_partition"]["entities"]
                unmatched_entities_list.append(g1_part_entities)

        labelled_alignmennt = self.data.load_train_alignment()
        pseudo_alignment = self.load_pseudo_seeds() # in 1-th epoch is []
        train_alignment = labelled_alignmennt + pseudo_alignment
        g1_anchors = [e1 for e1, e2 in train_alignment] # This list only contains the source part of train alignments.
        candi2benefit_map_list = compute_perf_jointly_multi_proc(self.data.edge_weights1, len(self.data.kg1_entities), self.data.kg1_entities, \
                                                                 self.data.kg1_triples, gcn_l=self.gcn_l, gamma=self.gamma, candi_entities=self.data.kg1_entities, \
                                                                    anc_entities=g1_anchors, unmatch_entities_list=unmatched_entities_list, \
                                                                        device_list=self.devices, is_mulprocess=self.is_mulprocess, proc_n=self.proc_n)

        info_after = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        with open(os.path.join(self.out_dir, "tmp_running.log"), "a+") as file:
            msg1 = {"msg_type": "ctx1_gpu_mem_usage_before", "value": info_before.used/1024/1024}
            file.write(json.dumps(msg1)+"\n")
            msg2 = {"msg_type": "ctx1_gpu_mem_usage_after", "value": info_after.used/1024/1024}
            file.write(json.dumps(msg2)+"\n")
        return candi2benefit_map_list

    def _build_g1_context(self, part_idx):
        print("index {0}".format(6))
        if CtxBuilderV2.candi2benefit_map_list is None:
            if os.path.exists(os.path.join(self.out_dir, "g1context.json")):
                print("load before g1context")
                with open(os.path.join(self.out_dir, "g1context.json")) as file:
                    CtxBuilderV2.candi2benefit_map_list = json.loads(file.read())
            else:
                CtxBuilderV2.candi2benefit_map_list = self.cache_g1_ent_effect()
                with open(os.path.join(self.out_dir, "g1context.json"), "w") as file:
                    file.write(json.dumps(CtxBuilderV2.candi2benefit_map_list)+"\n")

        super()._build_g1_context(part_idx)

    def _build_g2_context(self, part_idx):
        print("index {0}".format(6))
        if CtxBuilderV2.candi_map_list is None:
            if os.path.exists(os.path.join(self.out_dir, "g2context.pt")):
                print("load before g2context")
                CtxBuilderV2.candi_map_list = torch.load(os.path.join(self.out_dir, "g2context.pt"))
            else:
                CtxBuilderV2.candi_map_list = super()._cache_g2_ents()
                torch.save(CtxBuilderV2.candi_map_list, os.path.join(self.out_dir, "g2context.pt"))

        super()._build_g2_context(part_idx)

    def _build_context_for_single_graph1(self, unmatch_entities, entities, triples, anchors, ctx_graph_size, part_idx):
        print("index {0}".format(12))
        all_candidates = list(set(entities).difference(set(unmatch_entities)))
        candi2benefit_map = CtxBuilderV2.candi2benefit_map_list[part_idx-1] # {ent_id : score}
        candi2benefit_map = {e: candi2benefit_map[e] for e in all_candidates}

        sel_num = max(0, ctx_graph_size-len(unmatch_entities))
        new_candi2effect_map = candi2benefit_map
        sorted_items = sorted(new_candi2effect_map.items(), key=lambda item: - item[1])
        sel_entities = [c for c,v in sorted_items[:sel_num]]  # drop front ones

        # del perfmodel
        torch.cuda.empty_cache()
        print("end of building g1 ctx")
        return sel_entities
