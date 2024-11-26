# -*- coding: utf-8 -*-
import os
import gc
import dgl
import sys
import torch
import nvidia_smi
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import dgl.function as fn
from torch import multiprocessing

nvidia_smi.nvmlInit()
multiprocessing.get_context("spawn")
sys.path.append(os.getcwd())


class EviPassingLayer(nn.Module):
    def __init__(self):
        super(EviPassingLayer, self).__init__()

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'o'))
            """
            g.update_all:
            Send messages along all the edges of the specified type and update all the nodes of the corresponding destination type.
            fn.copy_u is equal to :
            def message_func(edges):
                return {'m': edges.src['h']}
            fn.sum is equal to  :
            def reduce_func(nodes):
                return {'h': torch.sum(nodes.mailbox['m'], dim=1)}
            """
            h_o = g.ndata["o"]
            del g, h
            return h_o


class EviPassingModel(nn.Module):
    def __init__(self, layer_num):
        super(EviPassingModel, self).__init__()
        self.layer_num = layer_num
        self.layer = EviPassingLayer()

    def forward(self, g, h0, e):
        # print("index {0}".format(10))
        h = h0
        for i in range(self.layer_num):
            h = h * e
            h = self.layer(g, h)
        h = h * e
        del g, h0, e
        return h


class PerfModel():
    def __init__(self, edge_weights, ent_num, kg_entities, kg_triples, device, gcn_l, gamma=2.0):
        triple_arr = np.array(kg_triples)
        self.gamma = gamma
        self.gcn_l = gcn_l

        self.graph = dgl.DGLGraph()
        self.graph.add_edges(triple_arr[:,0], triple_arr[:,2])
        self.graph.add_edges(triple_arr[:,2], triple_arr[:,0])
        self.graph.add_edges(np.array(kg_entities), np.array(kg_entities))

        e_weights = []
        e_weights += edge_weights[triple_arr[:, 1]].tolist()
        e_weights = e_weights + e_weights
        max_weight = torch.max(edge_weights)
        max_weight = max_weight + 0.5
        e_weights += max_weight.repeat(len(kg_entities)).tolist()

        self.graph.edata['w'] = torch.tensor(e_weights)
        self.ent_num = ent_num

    def compute_perf_jointly(self, candi_entities, anc_entities, unmatch_entities_list, device):
        print("index {0}".format(9))
        device = torch.device(device)
        model = EviPassingModel(layer_num=self.gcn_l).to(device)
        # entity existence
        e = torch.ones(size=(self.ent_num,), device=device) # (N, ) h_e^0
        # graph
        graph = self.graph.to(device)
        # compute he_to_inanchorpath_num
        h0 = torch.zeros(size=(self.ent_num,), device=device) # (N, )
        h0[anc_entities] = 1 # h^0_e
        init_ent_inanchorpath_nums = model(graph, h0, e)  # + 1e-8
        ent_inanchorpath_ratios = init_ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8) # let values != 0 be 1.
        init_weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1 # h^{0,L}_e
        # compute he_to_inpath_num
        init_ent_inpath_nums = model(graph, init_weight_perf, e) # + 1e-8
        ent_inpath_ratios = init_ent_inpath_nums / (init_ent_inpath_nums+1e-8)
        emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1 # h^{0,2L}_e

        ori_perf_list = []
        for unmatch_entities in unmatch_entities_list:
            ori_perf = torch.sum(emb_perf[unmatch_entities])
            ori_perf_list.append(ori_perf)

        del h0, ent_inanchorpath_ratios, init_weight_perf, ent_inpath_ratios, emb_perf, ori_perf
        # print(len(ori_perf_list)) is n_part.
        candi2benefit_map_list = [dict() for _ in range(len(unmatch_entities_list))]
        with torch.no_grad():
            h0 = torch.zeros(size=(self.ent_num,), device=device)
            h0[anc_entities] = 1
            for candi in tqdm(candi_entities):
                e[candi] = 0 # When removing this id entity. 
                ent_inanchorpath_nums = model(graph, h0, e)
                ent_inanchorpath_ratios = ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
                weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
                ent_inpath_nums = model(graph, weight_perf, e)
                ent_inpath_ratios = ent_inpath_nums / (init_ent_inpath_nums+1e-8)
                emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1
                e[candi] = 1
                for idx, unmatch_entities in enumerate(unmatch_entities_list):
                    perf = torch.sum(emb_perf[unmatch_entities])
                    # candi2benefit_map_list[idx][candi] = (perf - ori_perf_list[idx]).cpu().item()
                    candi2benefit_map_list[idx][candi] = (ori_perf_list[idx] - perf).cpu().item() # {part_id : {ent_id : score}}
            del h0, ent_inanchorpath_ratios, ent_inpath_ratios, emb_perf, ori_perf_list, ent_inpath_nums, weight_perf, e, \
                ent_inanchorpath_nums, graph, model
        gc.collect()
        # pdb.set_trace()
        return candi2benefit_map_list

    def compute_perf(self, candi_entities, anc_entities, unmatch_entities, device):
        device = torch.device(device)
        model = EviPassingModel(layer_num=self.gcn_l).to(device)
        # entity existence
        e = torch.ones(size=(self.ent_num,), device=device)
        # graph
        graph = self.graph.to(device)
        # compute he_to_inanchorpath_num
        h0 = torch.zeros(size=(self.ent_num,), device=device)
        h0[anc_entities] = 1
        init_ent_inanchorpath_nums = model(graph, h0, e)  # + 1e-8
        ent_inanchorpath_ratios = init_ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
        init_weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
        # compute he_to_inpath_num
        init_ent_inpath_nums = model(graph, init_weight_perf, e) # + 1e-8

        ent_inpath_ratios = init_ent_inpath_nums / (init_ent_inpath_nums+1e-8)
        emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1
        ori_perf = torch.sum(emb_perf[unmatch_entities])

        candi2benefit_map = dict()
        with torch.no_grad():
            h0 = torch.zeros(size=(self.ent_num,), device=device)
            h0[anc_entities] = 1
            for candi in tqdm(candi_entities):
                e[candi] = 0
                ent_inanchorpath_nums = model(graph, h0, e)
                ent_inanchorpath_ratios = ent_inanchorpath_nums / (init_ent_inanchorpath_nums+1e-8)
                weight_perf = 2 / (1+torch.exp(-self.gamma*ent_inanchorpath_ratios)) - 1
                ent_inpath_nums = model(graph, weight_perf, e)
                ent_inpath_ratios = ent_inpath_nums / (init_ent_inpath_nums+1e-8)
                emb_perf = 2 / (1+torch.exp(-self.gamma*ent_inpath_ratios)) - 1
                e[candi] = 1
                perf = torch.sum(emb_perf[unmatch_entities])
                # candi2benefit_map[candi] = (perf - ori_perf).cpu().item()  # perf change of dropping entity
                candi2benefit_map[candi] = (ori_perf - perf).cpu().item()  # effect
        return candi2benefit_map

def compute_perf_jointly_multi_proc(edge_weights, ent_num, kg_entities, kg_triples, gcn_l, gamma, candi_entities, anc_entities, \
                                    unmatch_entities_list, device_list, is_mulprocess=False, proc_n=2):
    print("index {0}".format(8))
    if is_mulprocess:
        global single_proc
        def single_proc(i):
            device = device_list[int(i % len(device_list))]
            if device == "cpu":
                torch.set_num_threads(20)
            model = PerfModel(edge_weights, ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
            st = batch_size * i
            ed = batch_size * (i+1)
            sub_candi_entities = candi_entities[st:ed]
            candi2benefit_map_list = model.compute_perf_jointly(sub_candi_entities, anc_entities, unmatch_entities_list, device)
            return candi2benefit_map_list

        proc_n = min(multiprocessing.cpu_count() // 2, 20, proc_n)
        print("process num:", proc_n)
        batch_size = int(len(candi_entities) / proc_n) + 1
        with multiprocessing.Pool(processes=proc_n) as pool:
            results = pool.map(single_proc, list(range(proc_n)))
        # # 关闭进程池
        # pool.close()
        # # 等待所有任务完成
        # pool.join()

        all_candi2benefit_map_list = [dict() for _ in range(len(unmatch_entities_list))]
        for res_list in results:
            for idx, res in enumerate(res_list):
                all_candi2benefit_map_list[idx].update(res)
    else:
        def single_proc():
            device = device_list[0]
            model = PerfModel(edge_weights, ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
            candi2benefit_map_list = model.compute_perf_jointly(candi_entities, anc_entities, unmatch_entities_list, device)
            return candi2benefit_map_list

        res_list = single_proc()
        all_candi2benefit_map_list = [dict() for _ in range(len(unmatch_entities_list))]
        for idx, res in enumerate(res_list):
            all_candi2benefit_map_list[idx].update(res)

    return all_candi2benefit_map_list

def compute_perf_multi_proc(edge_weights, ent_num, kg_entities, kg_triples, gcn_l, gamma, candi_entities, anc_entities, unmatch_entities, \
                            device_list, is_mulprocess=False, proc_n=2):
    if is_mulprocess:
        global single_proc
        def single_proc(i):
            device = device_list[int(i % len(device_list))]
            if device == "cpu":
                torch.set_num_threads(20)
            model = PerfModel(edge_weights, ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
            st = batch_size * i
            ed = batch_size * (i+1)
            sub_candi_entities = candi_entities[st:ed]
            candi2benefit_map = model.compute_perf(sub_candi_entities, anc_entities, unmatch_entities, device)
            return candi2benefit_map

        proc_n = min(multiprocessing.cpu_count() // 2, 20, proc_n)
        print("process num:", proc_n)
        batch_size = int(len(candi_entities) / proc_n) + 1
        with multiprocessing.Pool(processes=proc_n) as pool:
            results = pool.map(single_proc, list(range(proc_n)))
        # # 关闭进程池
        # pool.close()
        # # 等待所有任务完成
        # pool.join()

        all_candi2benefit_map = dict()
        for res in results:
            all_candi2benefit_map.update(res)
    else:
        def single_proc():
            device = device_list[0]
            if device == "cpu":
                torch.set_num_threads(20)
            model = PerfModel(edge_weights, ent_num, kg_entities, kg_triples, device=device, gcn_l=gcn_l, gamma=gamma)
            candi2benefit_map = model.compute_perf(candi_entities, anc_entities, unmatch_entities, device)
            return candi2benefit_map

        res = single_proc()
        all_candi2benefit_map = dict()
        all_candi2benefit_map.update(res)

    return all_candi2benefit_map
