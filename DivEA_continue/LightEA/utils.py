import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import faiss
import pickle
import torch
import json
from tqdm import tqdm
import tensorflow.keras.backend as K
import networkx as nx
import torch.nn.functional as F
import gc

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Unsuper.unsupervisedSeeds import obtain_embed, visual_pivot_induction_mini_batch

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def load_graph(path):
    # if os.path.exists(path+"graph_cache.pkl"):
    #     return pickle.load(open(path+"graph_cache.pkl","rb"))
    triples = []
    with open(os.path.join(path, "triples_1")) as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
    with open(os.path.join(path, "triples_2")) as f:
        for line in f.readlines():
            h,r,t = [int(x) for x in line.strip().split("\t")]
            triples.append([h,t,2*r])
            triples.append([t,h,2*r+1])
    triples = np.unique(triples,axis=0)
    rel_size = np.max(triples[:,2])+1
    node_size = len(set(loadfile(os.path.join(path, "ent_ids_1"), 1)) | set(loadfile(os.path.join(path, "ent_ids_2"), 1)))
    
    ent_tuple,triples_idx = [],[]
    ent_ent_s,rel_ent_s,ent_rel_s = {},set(),set()
    last,index = (-1,-1), -1

    for i in range(node_size):
        ent_ent_s[(i,i)] = 0

    for h,t,r in triples:
        ent_ent_s[(h,h)] += 1
        ent_ent_s[(t,t)] += 1

        if (h,t) != last:
            last = (h,t)
            index += 1
            ent_tuple.append([h,t])
            ent_ent_s[(h,t)] = 0

        triples_idx.append([index,r])
        ent_ent_s[(h,t)] += 1
        rel_ent_s.add((r,h))
        ent_rel_s.add((t,r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx),axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())),axis=0)
    ent_ent_val = np.array([ent_ent_s[(x,y)] for x,y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)),axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)),axis=0)
    
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    # pickle.dump(graph_data, open(path+"graph_cache.pkl","wb"))
    return graph_data

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
    c = 0
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

def load_aligned_pair(data_dir):
    entity1,rel1,triples1 = load_triples(os.path.join(data_dir, 'triples_1'))
    entity2,rel2,triples2 = load_triples(os.path.join(data_dir, 'triples_2'))
    entity1 = load_entities(os.path.join(data_dir, 'ent_ids_1'))
    entity2 = load_entities(os.path.join(data_dir, 'ent_ids_2'))
    train_alignment_pair = load_alignment_pair(os.path.join(data_dir, 'ref_ent_ids_train'))
    dev_alignment_pair = load_alignment_pair(os.path.join(data_dir, 'ref_ent_ids_test'))
    train_pair, dev_pair = train_alignment_pair, dev_alignment_pair
    raw_train_len = len(train_pair)

    if raw_train_len < 500:
        print("start with unsupervised mode")
        left_idx, right_idx = torch.LongTensor(list(entity1)), torch.LongTensor(list(entity2))
        embed = obtain_embed(triples1 + triples2, len(entity1) + len(entity2))
        train_pair += visual_pivot_induction_mini_batch(torch.FloatTensor(embed), left_idx, right_idx, surface=False, ugraph=True)[0]
        train_pair = [tuple(da) for da in train_pair]
        train_pair = list(set(train_pair[:max(len(dev_pair), 510)]))
        gc.collect()
    print("entity num1", len(entity1), "entity num2", len(entity2))
    print("train num", len(train_pair))
    return np.array(train_pair),np.array(dev_pair)

def load_name_features(dataset,vector_path,mode = "word-level"):
    
    try:
        word_vecs = pickle.load(open("./word_vectors.pkl","rb"))
    except:
        word_vecs = {}
        with open(vector_path, encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                word_vecs[line[0]] = [float(x) for x in line[1:]]
        pickle.dump(word_vecs,open("./word_vectors.pkl","wb"))

    if "EN" in dataset:
        ent_names = json.load(open("translated_ent_name/%s.json"%dataset[:-1].lower(),"r"))

    d = {}
    count = 0
    for _,name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word)-1):
                if word[idx:idx+2] not in d:
                    d[word[idx:idx+2]] = count
                    count += 1

    ent_vec = np.zeros((len(ent_names),300),"float32")
    char_vec = np.zeros((len(ent_names),len(d)),"float32")
    for i,name in tqdm(ent_names):
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word)-1):
                char_vec[i,d[word[idx:idx+2]]] += 1
        if k:
            ent_vec[i]/=k
        else:
            ent_vec[i] = np.random.random(300)-0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(d))-0.5
    
    faiss.normalize_L2(ent_vec)
    faiss.normalize_L2(char_vec)

    if mode == "word-level":
        name_feature = ent_vec
    if mode == "char-level":
        name_feature = char_vec
    if mode == "hybrid-level": 
        name_feature = np.concatenate([ent_vec,char_vec],-1)
        
    return name_feature

def sparse_sinkhorn_sims(left,right,features,top_k=500,iteration=15,mode = "test"):
    features_l = features[left]
    features_r = features[right]

    faiss.normalize_L2(features_l); faiss.normalize_L2(features_r)

    res = faiss.StandardGpuResources()
    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    if len(gpus):
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = len(left)
    row_index = K.transpose(([K.arange(size*top_k)//top_k,index,K.arange(size*top_k)]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
        
    return K.reshape(row_index[:,1],(-1,top_k)), K.reshape(row_sims,(-1,top_k))

def test(test_pair,features,topk=[1,3,5,10,50],top_k=500,iteration=15):
    left, right = test_pair[:,0], np.unique(test_pair[:,1])
    index,sims = sparse_sinkhorn_sims(left, right,features,top_k,iteration,"test")
    ranks = tf.argsort(-sims,-1).numpy()
    index = index.numpy()
    
    mrr, mr = 0, 0
    t_num = [0 for k in topk]
    pos = np.zeros(np.max(right)+1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == index[i,ranks[i]])[0]
        if len(rank) != 0:
            for i in range(len(topk)):
                if rank[0] == 0 and i == 0:
                    t_num[i] += 1
                else:
                    if rank[0] < topk[i]:
                        t_num[i] += 1
            mrr += 1/(rank[0]+1) 
            mr += rank[0]+1
    
    metrics = {"mr": mr / len(test_pair), "mrr": mrr / len(test_pair)}
    for idx in range(len(topk)):
        metrics[str(f"hit@{topk[idx]}")] = float(t_num[idx]) / len(test_pair)
    
    for idx in range(len(topk)):
        print(f"hit@{topk[idx]}: %.3f"%(metrics[str(f"hit@{topk[idx]}")]))
        
    print("MR: %.3f\nMRR: %.3f\n"%(mr/len(test_pair), mrr/len(test_pair)))
    return metrics