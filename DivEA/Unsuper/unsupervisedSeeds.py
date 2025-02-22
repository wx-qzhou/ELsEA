import gc
import torch
import numpy as np
import networkx as nx
from os.path import join
from collections import defaultdict
from divea.graph_metrics import ht_func
from Unsuper.text_sim import global_level_semantic_sim, sparse_string_sim
from Unsuper.TranslatetoEN.translate_data import load_json
from Unsuper.struc2Vec import Struc2Vec
from Unsuper.TransE import train_transe
from Unsuper.line import train_line
from node2vec import Node2Vec

def emb_Node2Vec(triples):
    G = nx.DiGraph()

    for h, r, t in triples:
        G.add_edge(h, t, relation=r)

    # 使用 node2vec 生成随机游走
    node2vec = Node2Vec(G, dimensions=16, walk_length=20, num_walks=50, workers=4)

    # 训练嵌入模型
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 获取所有节点的嵌入
    embeddings = {node: model.wv[node] for node in G.nodes()}

    return embeddings

def emb_Struc2Vec(triples):
    G = nx.DiGraph()

    for h, r, t in triples:
        G.add_edge(h, t, relation=r)

    model = Struc2Vec(G, dimensions=16, walk_length=10, num_walks=50, workers=4)
    model.train()
    embeddings = {node: model.get_embedding(node) for node in G.nodes}

    return embeddings

def emb_TransE2Vec(triples, max_epoch=50):
    entity_set, relation_set, neighbor = set(), set(), dict()

    for h, r, t in triples:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)

        if h not in neighbor:
            neighbor[h] = set()
        if t not in neighbor:
            neighbor[t] = set()
        neighbor[h].add(t)
        neighbor[t].add(h)

    embeddings = train_transe(entity_set, relation_set, neighbor, triples, dimensions=16, max_epoch=max_epoch, margin=2, learning_rate=0.001)
    return embeddings

def emb_Line2Vec(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)

    # 训练 LINE 模型
    model = train_line(G, embedding_dim=16, num_epochs=10, batch_size=128)

    embeddings = model.node_embeddings.weight.data.numpy()
    return {node: embeddings[node] for node in G.nodes}

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

def place_triplets(triplets, linked_entities, batch_num, batch_size): # after divide the nodes, place the triples!!
    # divide the entities into multiple batches
    batch_list = [linked_entities[i * batch_size : (i + 1) * batch_size] for i in range(batch_num)]
    node2batch = {}
    for batch_idx, batch in enumerate(batch_list):
        if len(batch) < batch_size - 1000:
            continue
        print(batch_idx, len(batch))
        for node in batch:
            node2batch[node] = batch_idx

    # Initialize lists for batches and associated data
    kg_triples_list = [set() for _ in range(batch_num)]
    nodes_set = [set() for _ in range(batch_num)]
    nodes_has_set = set()
    no_removed = 0

    for h, r, t in triplets:
        h_idx, t_idx = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_idx == t_idx:
            kg_triples_list[h_idx].add((h, r, t))
            nodes_set[h_idx].add(h)
            nodes_set[h_idx].add(t)
            nodes_has_set.add(h)
            nodes_has_set.add(t)
            no_removed += 1
    for h, r, t in triplets:
        flag = False
        h_idx, t_idx = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_idx >= 0 and t_idx == -1 and (h not in nodes_has_set):
            kg_triples_list[h_idx].add((h, r, t))
            nodes_set[h_idx].add(h)
            nodes_set[h_idx].add(t)
            flag = True
        elif t_idx >= 0 and h_idx == -1 and (t not in nodes_has_set):
            kg_triples_list[t_idx].add((h, r, t))
            nodes_set[t_idx].add(h)
            nodes_set[t_idx].add(t)
            flag = True
        if flag:
            no_removed += 1

    print('split triplets complete, total {} triplets removed'.format(len(triplets) - no_removed))

    kg_triples_list, nodes_set = [list(triples) for triples in kg_triples_list], [list(nodes) for nodes in nodes_set]

    return kg_triples_list, nodes_set

def obtain_medium_degree_entity(triples, threshold_deg=2):
    triplets = [(h, t) for h, r, t in triples]
    graph = nx.DiGraph()
    graph.add_edges_from(triplets)

    # sort based on degrees
    sorted_by_degree = dict(sorted(dict(graph.degree()).items(), key=lambda item: item[1], reverse=True)) # {entity : degrees} sort by degrees, max to min

    # clear entities with small degrees
    degrees = list(sorted_by_degree.values())
    index = degrees.index(threshold_deg)
    sorted_by_degree = dict(list(sorted_by_degree.items())[:index])

    # clear entities with large degrees
    threshold = (np.max(degrees) + np.mean(degrees)) * 1 / 3

    print(f"设定的度阈值: {threshold}")

    sorted_degree = []
    for node, degree in list(sorted_by_degree.items()): # min to max
        if degree <= threshold:
            sorted_degree.append(node)

    return sorted_degree

def z_score(embed):
    mean = torch.mean(embed, dim=0)
    std = torch.std(embed, dim=0)
    embed = (embed - mean) / (std + 1e-20)
    del mean, std
    gc.collect()
    return embed

# obtain the pretrained seeds based on entity names
def seedbyname(data_dir, kgids, threshold=0.8, ugraph=True):
    if ugraph:
        kg1_ent_id2uri_map, kg2_ent_id2uri_map = read_tab_lines(join(data_dir, kgids[0] + "_entity_id2uri.txt")), \
            read_tab_lines(join(data_dir, kgids[1] + "_entity_id2uri.txt"))
    else:
        kg1_ent_id2uri_map, kg2_ent_id2uri_map = list(load_json(join(data_dir, kgids[0] + "_entity_txt.json")).items()), \
            list(load_json(join(data_dir, kgids[1] + "_entity_txt.json")).items())
    kg1_new2oldid, kg2_new2oldid = dict(read_tab_lines(join(data_dir, "ent_ids_1"))), dict(read_tab_lines(join(data_dir, "ent_ids_2")))
    kg1_new2oldid, kg2_new2oldid = {value: key for key, value in kg1_new2oldid.items()}, {value: key for key, value in kg2_new2oldid.items()}
    kg1_ent_id2uri_map = [(int(kg1_new2oldid[idx2uri[0]]), idx2uri[1]) for idx2uri in kg1_ent_id2uri_map if idx2uri[0] in kg1_new2oldid]
    kg2_ent_id2uri_map = [(int(kg2_new2oldid[idx2uri[0]]), idx2uri[1]) for idx2uri in kg2_ent_id2uri_map if idx2uri[0] in kg2_new2oldid]

    assert len(kg1_ent_id2uri_map) == len(kg1_new2oldid)
    assert len(kg2_ent_id2uri_map) == len(kg2_new2oldid)

    sparse_sim, train_alignment = sparse_string_sim(dict(kg1_ent_id2uri_map), dict(kg2_ent_id2uri_map), threshold=threshold)
    train_alignment = train_alignment.tolist() 
    return sparse_sim, train_alignment

def batch_val_ind(l_graph_f, r_graph_f, num_batch, batchsize, topk, search_batch_sz=5000, \
                  index_batch_sz=10000):
    inds = []
    vals =  []

    for i in range(num_batch):
        l_graph = l_graph_f[i * batchsize : (i + 1) * batchsize]
        if l_graph.shape[0] == 0:
            print("here is empty.")
            continue
        r_graph = r_graph_f

        graph_sim, _ = global_level_semantic_sim([l_graph, r_graph], search_batch_sz=search_batch_sz, \
                                                 index_batch_sz=index_batch_sz, k=topk)
        graph_sim = graph_sim.cpu()
        ind = graph_sim.indices().T  # 索引 (行和列)
        val = graph_sim.values()   # 对应的值

        ind[:,0] += i * batchsize

        inds.append(ind)
        vals.append(val)

    del val, ind
    del l_graph_f, r_graph_f
    inds = torch.cat(inds, dim=0)
    vals = torch.cat(vals, dim=-1)
    return inds, vals

def get_indices_vals(M, threshold=0.5):
    inds = M.indices().T  # 索引 (行和列)
    vals = M.values()   # 对应的值

    # 将相似度矩阵转换为稀疏格式
    mask = vals >= (threshold * threshold)
    vals = vals[mask]
    inds = inds[mask]
    del M, mask
    print("highest sim:", torch.max(vals), "lowest sim:", torch.min(vals))

    return inds, vals

# generate the pretrained seeds based on structural information
def visual_pivot_induction_mini_batch(graph_features, left_idx, right_idx, data_dir=None, \
                                      kgids=None, surface=True, ugraph=True, search_batch_sz=50000, \
                                        index_batch_sz=500000, batchsize=10000, topk=2000, \
                                            threshold=0.008, thresholdstr=0.8):
    # if unsupervised? use image to obtain links
    if ugraph:
        l_graph_f = graph_features[left_idx]  # left images (M, d)
        r_graph_f = graph_features[right_idx]  # right images (N, d)
        M, N = l_graph_f.shape[0], r_graph_f.shape[0]

        num_batch = min(left_idx.shape[0] // batchsize, right_idx.shape[0] // batchsize)
        if num_batch - 1 < 1:
            num_batch = 4
            batchsize = min(left_idx.shape[0] // num_batch, right_idx.shape[0] // num_batch)

        if surface == False:
            topk = 50
            threshold = thresholdstr

        print("Start r2l and l2r.")
        inds1, vals1 = batch_val_ind(l_graph_f, r_graph_f, num_batch, batchsize, topk, search_batch_sz=search_batch_sz, index_batch_sz=index_batch_sz)
        inds2, vals2 = batch_val_ind(r_graph_f, l_graph_f, num_batch, batchsize, topk, search_batch_sz=search_batch_sz, index_batch_sz=index_batch_sz)
        inds2[:, [0, 1]] = inds2[:, [1, 0]]
        # graph_sim = torch.sparse_coo_tensor(inds1.T, vals1, (M, N)).mul(torch.sparse_coo_tensor(inds2.T, vals2, (M, N)))
        graph_sim = (torch.sparse_coo_tensor(inds1.T, vals1, (M, N)) + torch.sparse_coo_tensor(inds2.T, vals2, (M, N))) / 2
        inds, vals = get_indices_vals(graph_sim.coalesce(), threshold=threshold)
        del inds1, vals1, inds2, vals2
        del graph_features, left_idx, right_idx
        del l_graph_f, r_graph_f

    train_alignment = None
    if surface == True:
        print("Consider to use surfaces.")
        sparse_str_sim, train_alignment = seedbyname(data_dir, kgids, threshold=thresholdstr, ugraph=ugraph)
        if ugraph:
            print("Here is graphs.")
            inds = inds.T
            graph_sim = torch.sparse_coo_tensor(inds, vals, (M, N))
            sparse_str_sim = sparse_str_sim * graph_sim
            del graph_sim
        sparse_str_sim = sparse_str_sim.coalesce()
        inds = sparse_str_sim.indices().T  # 索引 (行和列)
        vals = sparse_str_sim.values()   # 对应的值
        if ugraph:
            vals, indices = vals.topk(min(6000, vals.shape[0]))
            inds = inds[indices]
        del sparse_str_sim, indices
    else:
        print("Consider to not use surfaces.")
        vals, indices = vals.topk(min(10000, vals.shape[0]))
        inds = inds[indices]
        del indices

    inds = inds.tolist()
    
    assert vals.shape[0] == len(inds)

    print("The number of pseudo seeds is {}.".format(len(inds)))

    del vals 

    return inds, train_alignment

# obtain the embeddings of nodes
def obtain_embed(triples, ent_size, is_Unsuper=False, max_epoch=50):
    r2f, r2if = ht_func(triples)

    triplets = []
    en_inrel = defaultdict(set)
    en_ourel = defaultdict(set)
    for h, r, t in triples:
        triplets.append((h, t))
        en_inrel[h].add(r)
        en_ourel[t].add(r)
    graph = nx.DiGraph()
    graph.add_edges_from(triplets)

    matrix = torch.zeros(ent_size, 4)

    for node in graph.nodes():
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        matrix[node][0] = in_deg
        matrix[node][1] = out_deg
        for key in en_inrel[node]:
            matrix[node][2] += r2f[key]
        for key in en_ourel[node]:
            matrix[node][3] += r2if[key]

    if is_Unsuper:
        matrix = torch.cat((matrix, z_score(torch.FloatTensor(list(emb_TransE2Vec(triples, max_epoch=max_epoch).values())))), dim=-1)

    print("The shape of matrix is {}.".format(matrix.shape))
    # matrix = F.normalize(matrix, 1, 0)
    # matrix = z_score(matrix)
    return matrix
