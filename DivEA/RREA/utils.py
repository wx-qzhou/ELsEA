import numpy as np
import scipy.sparse as sp
import os
import gc
import torch
from Unsuper.unsupervisedSeeds import obtain_embed, visual_pivot_induction_mini_batch

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name,'r'):
        head,r,tail = [int(item) for item in line.split()]
        entity.add(head); entity.add(tail); rel.add(r+1)
        triples.append((head,r+1,tail))
    return entity,rel,triples

def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name,'r'):
        e1,e2 = line.split()
        alignment_pair.append((int(e1),int(e2)))
    return alignment_pair

def load_entities(file_name):
    entity = set()
    for line in open(file_name,'r'):
        newid, oldid = [int(item) for item in line.split()]
        entity.add(newid)
    return entity

def get_matrix(triples,entity,rel):
    ent_size = max(entity)+1
    rel_size = (max(rel) + 1)
    print(ent_size,rel_size)


    # === added by bing
    iso_map = {}
    for h,r,t in triples:
        iso_map[h] = False
        iso_map[t] = False
    selfloop_triples = [(e, rel_size, e) for e in range(ent_size) if iso_map.get(e, True)]
    if len(selfloop_triples):
        rel_size += 1
        triples += selfloop_triples
    # ===


    adj_matrix = sp.lil_matrix((ent_size,ent_size))
    adj_features = sp.lil_matrix((ent_size,ent_size))
    radj = []
    rel_in = np.zeros((ent_size,rel_size))
    rel_out = np.zeros((ent_size,rel_size))
    
    for i in range(max(entity)+1):
        adj_features[i,i] = 1

    for h,r,t in triples:        
        adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
        adj_features[h,t] = 1; adj_features[t,h] = 1;
        radj.append([h,t,r]); radj.append([t,h,r+rel_size]); 
        rel_out[h][r] += 1; rel_in[t][r] += 1
        
    count = -1
    s = set()
    d = {}
    r_index,r_val = [],[]
    for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
        if ' '.join([str(h),str(t)]) in s:
            r_index.append([count,r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h),str(t)]))
            r_index.append([count,r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]
    
    rel_features = np.concatenate([rel_in,rel_out],axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))    
    return adj_matrix,r_index,r_val,adj_features,rel_features      

def load_data(data_dir):
    entity1,rel1,triples1 = load_triples(os.path.join(data_dir, 'triples_1'))
    entity2,rel2,triples2 = load_triples(os.path.join(data_dir, 'triples_2'))
    entity1 = load_entities(os.path.join(data_dir, 'ent_ids_1'))
    entity2 = load_entities(os.path.join(data_dir, 'ent_ids_2'))
    train_alignment_pair = load_alignment_pair(os.path.join(data_dir, 'ref_ent_ids_train'))
    dev_alignment_pair = load_alignment_pair(os.path.join(data_dir, 'ref_ent_ids_test'))
    train_pair, dev_pair = train_alignment_pair, dev_alignment_pair
    adj_matrix,r_index,r_val,adj_features,rel_features = get_matrix(triples1+triples2,entity1.union(entity2),rel1.union(rel2))
    raw_train_len = len(train_pair)

    if raw_train_len < 500:
        print("start with unsupervised mode")
        left_idx, right_idx = torch.LongTensor(list(entity1)), torch.LongTensor(list(entity2))
        embed = obtain_embed(triples1 + triples2, len(entity1) + len(entity2))
        try:
            train_pair += visual_pivot_induction_mini_batch(torch.FloatTensor(embed), left_idx, right_idx, surface=False, ugraph=True)[0]
        except:
            pass
        train_pair = [tuple(da) for da in train_pair]
        train_pair = list(set(train_pair[:max(len(dev_pair), 510)]))
        gc.collect()
    print("entity num1", len(entity1), "entity num2", len(entity2))
    print("train num", len(train_pair))
    return np.array(train_pair),np.array(dev_pair),adj_matrix,np.array(r_index),np.array(r_val),adj_features,rel_features