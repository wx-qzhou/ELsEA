import os
import json
import numpy as np
import networkx as nx
from collections import Counter

# the information of adjs
def adj_info(KG, r2f, r2if):
    "Based on common relations of entity"
    def get_e_adjr1(etriples, r2f, r2if):

        M = {}
        for tri in etriples:
            if tri[0] == tri[2]:
                continue
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
            else:
                M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
            if (tri[2], tri[0]) not in M:
                M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
            else:
                M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)

        return M

    "Based on common neighbors of entity"
    def get_e_adjr2(etriples):
        M = {}
        for tri in etriples:
            if tri[0] == tri[2]:
                continue
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = 0
            M[(tri[0], tri[2])] += 1
        return M
    
    ent_adj = get_e_adjr1(KG, r2f, r2if)
    ent_adj_ = get_e_adjr2(KG)
    ent_adj, ent_adj_ = Counter(ent_adj), Counter(ent_adj_)
    ent_adj = dict(ent_adj + ent_adj_)

    ent_adj_temp = {}
    for k in ent_adj:
        for k1 in k:
            if k1 in ent_adj_temp:
                ent_adj_temp[k1] += [ent_adj[k]]
            else:
                ent_adj_temp.update({k1 : [ent_adj[k]]})
    for k in ent_adj_temp:
        ent_adj_temp[k] = np.mean(ent_adj_temp[k])
    return ent_adj_temp

# to compute the weights of entities
def in_degree(graph, kg1_entity_dict, kg2_entity_dict):
    ent_in_degree = {}
    for node in graph.nodes():
        in_deg = graph.in_degree(node)
        ent_in_degree.update({node : in_deg})
    
    ent_in_degree1 = {}
    ent_in_degree2 = {}
    for k in ent_in_degree:
        if k in kg1_entity_dict:
            ent_in_degree1.update({kg1_entity_dict[k] : ent_in_degree[k]})
        else:
            ent_in_degree2.update({kg2_entity_dict[k] : ent_in_degree[k]})
    return ent_in_degree1, ent_in_degree2

# to compute the weights of entities
def out_degree(graph, kg1_entity_dict, kg2_entity_dict):
    ent_out_degree = {}
    for node in graph.nodes():
        out_deg = graph.out_degree(node)
        ent_out_degree.update({node : out_deg})

    ent_out_degree1 = {}
    ent_out_degree2 = {}
    for k in ent_out_degree:
        if k in kg1_entity_dict:
            ent_out_degree1.update({kg1_entity_dict[k] : ent_out_degree[k]})
        else:
            ent_out_degree2.update({kg2_entity_dict[k] : ent_out_degree[k]})
    return ent_out_degree1, ent_out_degree2

def cluster_degree(graph, kg1_entity_dict, kg2_entity_dict):
    ent_cluster_degree = {}
    for node in graph.nodes():
        cluster_deg = nx.clustering(graph, node)
        ent_cluster_degree.update({node : cluster_deg})

    ent_cluster_degree1 = {}
    ent_cluster_degree2 = {}
    for k in ent_cluster_degree:
        if k in kg1_entity_dict:
            ent_cluster_degree1.update({kg1_entity_dict[k] : ent_cluster_degree[k]})
        else:
            ent_cluster_degree2.update({kg2_entity_dict[k] : ent_cluster_degree[k]})
    return ent_cluster_degree1, ent_cluster_degree2

def graph_metrics_main(data, flags=["in_degree", "out_degree"]):
    kg1_entities, kg2_entities, kg1_triples, kg2_triples, match_ent = \
        data.kg1_entities, data.kg2_entities, data.kg1_triples, data.kg2_triples, data.load_train_alignment()
    ent_num = len(kg1_entities)

    kg1_entity_dict = dict(enumerate(kg1_entities)) # {id : name1}
    kg2_entity_dict = {k + ent_num : v for k, v in enumerate(kg2_entities)} # {id : name2}

    kg1_entity_dict_ = dict(zip(kg1_entity_dict.values(), kg1_entity_dict.keys())) # {name1 : id}
    kg2_entity_dict_ = dict(zip(kg2_entity_dict.values(), kg2_entity_dict.keys())) # {name2 : id}

    entities = list(kg1_entity_dict_.values()) + list(kg2_entity_dict_.values()) # [ids, ...]
    match_ent = dict(match_ent) # {name1 : name2}
    match_ent_ = dict(zip(match_ent.values(), match_ent.keys())) # {name2 : name1}

    graph = nx.DiGraph()
    graph.add_nodes_from(entities)
    
    edges = []
    edges += [(kg1_entity_dict_[h], kg2_entity_dict_[t]) for h, t in match_ent.items()]
    edges += [(kg2_entity_dict_[h], kg1_entity_dict_[t]) for t, h in match_ent.items()]
    for h, r, t in kg1_triples:
        edges.append((kg1_entity_dict_[h], kg1_entity_dict_[t]))
        if h in match_ent and t in match_ent:
            edges.append((kg2_entity_dict_[match_ent[h]], kg2_entity_dict_[match_ent[t]]))
        elif h in match_ent:
            edges.append((kg2_entity_dict_[match_ent[h]], kg1_entity_dict_[t]))
        elif t in match_ent:
            edges.append((kg1_entity_dict_[h], kg2_entity_dict_[match_ent[t]]))

    for h, r, t in kg2_triples:
        edges.append((kg2_entity_dict_[h], kg2_entity_dict_[t]))
        if h in match_ent_ and t in match_ent_:
            edges.append((kg1_entity_dict_[match_ent_[h]], kg1_entity_dict_[match_ent_[t]]))
        elif h in match_ent_:
            edges.append((kg1_entity_dict_[match_ent_[h]], kg2_entity_dict_[t]))
        elif t in match_ent_:
            edges.append((kg2_entity_dict_[h], kg1_entity_dict_[match_ent_[t]]))

    graph.add_edges_from(edges)

    KGs_edge = {}
    r2f1, r2if1 = ht_func(kg1_triples)
    KGs_edge.update({"KG1" : {"r2f":r2f1, "r2if":r2if1}})
    r2f2, r2if2 = ht_func(kg2_triples)
    KGs_edge.update({"KG2" : {"r2f":r2f2, "r2if":r2if2}})
    with open(os.path.join(data.data_dir, "KGs_edge_info.json"), "w") as file:
        file.write(json.dumps(KGs_edge))

    KGs_node = {"KG1" : {}, "KG2" : {}}
    for flag in flags:
        for key in KGs_node:
            KGs_node[key].update({flag : {}})
        if flag == "in_degree":
            ent_in_degree1, ent_in_degree2 = in_degree(graph, kg1_entity_dict, kg2_entity_dict)
            KGs_node["KG1"]["in_degree"] = ent_in_degree1
            KGs_node["KG2"]["in_degree"] = ent_in_degree2
        elif flag == "out_degree":
            ent_out_degree1, ent_out_degree2 = out_degree(graph, kg1_entity_dict, kg2_entity_dict)
            KGs_node["KG1"]["out_degree"] = ent_out_degree1
            KGs_node["KG2"]["out_degree"] = ent_out_degree2
        elif flag == "cluster_degree":
            ent_cluster_degree1, ent_cluster_degree2 = cluster_degree(graph, kg1_entity_dict, kg2_entity_dict)
            KGs_node["KG1"]["cluster_degree"] = ent_cluster_degree1
            KGs_node["KG2"]["cluster_degree"] = ent_cluster_degree2
        elif flag == "adj_info":
            KGs_node["KG1"]["adj_info"] = adj_info(kg1_triples, r2f1, r2if1)
            KGs_node["KG2"]["adj_info"] = adj_info(kg2_triples, r2f2, r2if2)
    with open(os.path.join(data.data_dir, "KGs_node_info.json"), "w") as file:
        file.write(json.dumps(KGs_node))

    # exit()

# to compute the weights of relations    
def ht_func(KG):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    r2f = {}
    r2if = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
        r2if[r] = len(tail[r]) / cnt[r]
    return r2f, r2if
