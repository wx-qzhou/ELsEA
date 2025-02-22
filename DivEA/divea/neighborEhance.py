# -*- coding: utf-8 -*-
import gc
import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
from torch import multiprocessing as torch_multiprocessing

import multiprocessing
import math
import time

"the following is the details of Neighbor Enrichment (NE)"
class Neighbor_Agg_Layer(nn.Module):
    def __init__(self, edge_weights, node_weights, ent_num, kg_entities, kg_triples):
        super(Neighbor_Agg_Layer, self).__init__()
        triple_arr = np.array(kg_triples)

        self.graph = dgl.DGLGraph()
        self.graph.add_edges(triple_arr[:,0], triple_arr[:,2])
        self.graph.add_edges(np.array(kg_entities), np.array(kg_entities))
        
        e_weights = []
        e_weights += edge_weights[triple_arr[:, 1]].tolist()

        max_weight = torch.max(edge_weights)
        max_weight = max_weight + 0.5
        e_weights += max_weight.repeat(len(kg_entities)).tolist()

        n_weights = node_weights[np.array(kg_entities)].tolist()

        self.graph.ndata['x'] = torch.tensor(n_weights).float()
        self.graph.edata['w'] = torch.tensor(e_weights)
        self.ent_num = ent_num

    def forward(self, anchors, device):
        device = torch.device(device)
        g = self.graph.to(device)
        h = torch.zeros(size=(self.ent_num,), device=device)
        h[anchors] = 1
        h[anchors] += g.ndata['x'][anchors]
        
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.mean('m', 'o'))
            h_o = g.ndata["o"]
            x = g.ndata['x']
            del g, h
            return h_o, x

class Neighbor_Ehance_layer(nn.Module):
    def __init__(self):
        super(Neighbor_Ehance_layer, self).__init__()

    def z_score(self, embed):
        mean = torch.mean(embed, dim=0)
        std = torch.std(embed, dim=0)
        embed = (embed - mean) / (std + 1e-20)
        del mean, std
        gc.collect()
        return embed

    def forward(self, unmatch_entities3, all_candidates3):
        score = torch.matmul(unmatch_entities3, all_candidates3) # (N, M)
        # att_score = F.sigmoid(score)
        # score = att_score * score
        # del att_score
        del unmatch_entities3, all_candidates3
        # 
        try:
            k = min(score.shape[1], 100)
            score = torch.topk(score, k=k, dim=1)[1].float().T
        except:
            score = torch.argsort(score, dim=1, descending=True).float().T # (M, N)  
        score = self.z_score(score)

        score = torch.sum(score, dim=-1) # (M, )

        gc.collect()
        torch.cuda.empty_cache()
        return score

class Neighbor_Ehance():
    def __init__(self, device) -> None:
        self.device = torch.device(device)

    def aggregate_nodes(self, all_candidates, all_candidates3, unmatch_entities3):
        model = Neighbor_Ehance_layer().to(self.device)

        all_candidates3, unmatch_entities3 = all_candidates3.to(self.device), unmatch_entities3.to(self.device)
        all_candidates3 = model.z_score(all_candidates3.T).T # (3, M)
        all_candidates3 = all_candidates3.T

        # att_score = model.z_score((torch.matmul(unmatch_entities1, all_candidates1.T) + torch.matmul(unmatch_entities2, all_candidates2.T)))
        # att_score = F.sigmoid(att_score)

        unmatch_entities3 = model.z_score(unmatch_entities3.T).T # (N, 3)
        
        score = model(unmatch_entities3, all_candidates3)
        
        del all_candidates3, unmatch_entities3, model
        torch.cuda.empty_cache()

        return dict(zip(all_candidates, score.cpu().numpy().tolist())) # 

# split data into small batches for the preparations of Multi-process data processing
def Neighbor_Ehance_generate_split_data(all_candidates, unmatch_entities, h_o, x):
    with torch.no_grad():
        all_candidates1 = h_o[all_candidates].unsqueeze(1) # (M, 1)
        all_candidates2 = x[all_candidates].unsqueeze(1) # (M, 1)
        all_candidates3 = all_candidates1 + all_candidates2 # (M, 1)
        all_candidates3 = torch.cat((all_candidates1, all_candidates2, all_candidates3), dim=-1) # (M, 3)

        unmatch_entities1 = h_o[unmatch_entities].unsqueeze(1) # (N, 1)
        unmatch_entities2 = x[unmatch_entities].unsqueeze(1) # (N, 1)
        unmatch_entities3 = unmatch_entities1 + unmatch_entities2 # (N, 1)
        unmatch_entities3 = torch.cat((unmatch_entities1, unmatch_entities2, unmatch_entities3), dim=-1) # (N, 3)
        
        del h_o, x
        del all_candidates, unmatch_entities
        del all_candidates1, all_candidates2
        del unmatch_entities1, unmatch_entities2

    gc.collect()
    torch.cuda.empty_cache()
    return (all_candidates3, unmatch_entities3)

def single_pro_(args):
    device, sub_candi, sub_candi_entities, new_unmatch_entities = args
    model= Neighbor_Ehance(device)
    candi2benefit_map = model.aggregate_nodes(sub_candi, sub_candi_entities, new_unmatch_entities) 
    del sub_candi, sub_candi_entities, new_unmatch_entities, args
    gc.collect()
    return candi2benefit_map

def neighbor_multi_proc(h_o, x, new_candidates, new_unmatch_entities, device_list, proc_n):
    proc_n = min(multiprocessing.cpu_count() // 2, 6, proc_n)
    print("The neighbor enhancing mul-GPU process num:", proc_n)
    args_list = []
    batch_size = int(len(new_candidates) / proc_n) + 1
    for i in range(proc_n):
        device = device_list[int(i % len(device_list))]

        st = batch_size * i
        ed = batch_size * (i+1)
        sub_candi_entities = new_candidates[st:ed]
        data_tuple = (device, sub_candi_entities)
        data_tuple += Neighbor_Ehance_generate_split_data(sub_candi_entities, new_unmatch_entities, h_o, x)
        args_list.append(data_tuple)
        del sub_candi_entities, data_tuple

    ctx = torch_multiprocessing.get_context("spawn")
    with ctx.Pool(processes=proc_n) as pool:
        results = pool.map(single_pro_, args_list)
    # # 关闭进程池
    # pool.close()
    # # 等待所有任务完成
    # pool.join()
        
    del h_o, x, args_list
    del new_candidates, new_unmatch_entities
    gc.collect()

    new_candi2benefit_map = dict()
    for res in results:
        new_candi2benefit_map.update(res)
    
    return new_candi2benefit_map

def neighbor_nomulpress_multi_proc(edge_weights, node_weights, new_entities, new_triples, new_candidates, new_anchors, new_unmatch_entities, device_list, proc_n):
    agg_model = Neighbor_Agg_Layer(edge_weights, node_weights, len(new_entities), new_entities, new_triples)
    h_o, x = agg_model(new_anchors, device_list[0])
    h_o, x = h_o.cpu(), x.cpu()
    del agg_model
    proc_n = len(device_list)
    proc_n = min(multiprocessing.cpu_count() // 2, 6, proc_n)
    print("The neighbor enhancing nomul process num:", proc_n)
    args_list = []
    batch_size = int(len(new_candidates) / proc_n) + 1
    for i in range(proc_n):
        device = device_list[int(i % len(device_list))]

        st = batch_size * i
        ed = batch_size * (i+1)
        sub_candi_entities = new_candidates[st:ed]
        data_tuple = (device, sub_candi_entities)
        data_tuple += Neighbor_Ehance_generate_split_data(sub_candi_entities, new_unmatch_entities, h_o, x)
        args_list.append(data_tuple)
        del sub_candi_entities, data_tuple

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=proc_n) as pool:
        results = pool.map(single_pro_, args_list)
    # # 关闭进程池
    # pool.close()
    # # 等待所有任务完成
    # pool.join()

    del h_o, x, args_list
    del edge_weights, node_weights, new_entities, new_triples, new_candidates, new_anchors, new_unmatch_entities
    gc.collect()

    new_candi2benefit_map = dict()
    for res in results:
        new_candi2benefit_map.update(res)
    
    return new_candi2benefit_map

# Multi-process data processing
class DataDealProcess(multiprocessing.Process):
    def __init__(self, in_queue=None, out_queue=None):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            batch = self.in_queue.get()
            try:
                self.process_data(batch)
            except Exception as ex:
                print(ex)
                time.sleep(2)
                self.process_data(batch)
            self.in_queue.task_done()

    def process_data(self, batch):
        for args in batch:
            result = single_pro_(args)
            self.out_queue.put(result)

"""generate the data with multi-processing"""
def neighbor_multi_proc_cpus(h_o, x, new_candidates, new_unmatch_entities, device_list, proc_n=multiprocessing.cpu_count()):
    proc_n = min(multiprocessing.cpu_count() // 2, 6, proc_n)
    print("The neighbor enhancing mul-CPU process num:", proc_n)
    batch_size = int(len(new_candidates) / proc_n) + 1

    with multiprocessing.Manager() as manager:  
        in_queue = multiprocessing.JoinableQueue() 
        out_queue = multiprocessing.JoinableQueue() 
        workerList = []

        for i in range(proc_n):
            worker = DataDealProcess(in_queue=in_queue, out_queue=out_queue)
            workerList.append(worker)
            worker.daemon = True 
            worker.start() 
   
        for i in range(proc_n):
            device = "cpu"

            st = batch_size * i
            ed = batch_size * (i+1)
            sub_candi_entities = new_candidates[st:ed]
            data_tuple = (device, sub_candi_entities)
            data_tuple += Neighbor_Ehance_generate_split_data(sub_candi_entities, new_unmatch_entities, h_o, x)
            in_queue.put([data_tuple])
            del sub_candi_entities, data_tuple

        in_queue.join()

        result_list = []
        # 读取输出队列中的结果
        while not out_queue.empty():
            result_list.append(out_queue.get())

        for worker in workerList:
            worker.terminate()

        new_candi2benefit_map = dict()
        for res in result_list:
            new_candi2benefit_map.update(res)
    
    return new_candi2benefit_map