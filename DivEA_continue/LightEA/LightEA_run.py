#choose the GPU, "-1" represents using the CPU

import os 
import random
import nvidia_smi
import argparse
# import all the requirements
import faiss 
from utils import *
import tensorflow as tf
import tensorflow.keras.backend as K
from CSLS import eval_alignment_by_sim_mat
# main functions of LightEA

def random_projection(x,out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1],out_dim)),axis=-1)
    return K.dot(x,random_vec)

def batch_sparse_matmul(sparse_tensor,dense_tensor,batch_size = 128,save_mem = False):
    results = []
    for i in range(dense_tensor.shape[-1]//batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor,dense_tensor[:, i*batch_size:(i+1)*batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results,-1)
    else:
        return K.concatenate(results,-1)

def get_features(train_pair,extra_feature = None):
    if extra_feature is not None:
        ent_feature = extra_feature
    else:
        random_vec = K.l2_normalize(tf.random.normal((len(train_pair),ent_dim)),axis=-1)
        ent_feature = tf.tensor_scatter_nd_update(tf.zeros((node_size,ent_dim)),train_pair.reshape((-1,1)),tf.repeat(random_vec,2,axis=0))
    rel_feature = tf.zeros((rel_size,ent_feature.shape[-1]))
    
    ent_ent_graph = tf.SparseTensor(indices=ent_ent,values=ent_ent_val,dense_shape=(node_size,node_size))
    rel_ent_graph = tf.SparseTensor(indices=rel_ent,values=K.ones(rel_ent.shape[0]),dense_shape=(rel_size,node_size))
    ent_rel_graph = tf.SparseTensor(indices=ent_rel,values=K.ones(ent_rel.shape[0]),dense_shape=(node_size,rel_size))
    
    ent_list,rel_list = [ent_feature],[rel_feature]
    for i in range(2):
        new_rel_feature = batch_sparse_matmul(rel_ent_graph,ent_feature)
        new_rel_feature = tf.nn.l2_normalize(new_rel_feature,axis=-1)
        
        new_ent_feature = batch_sparse_matmul(ent_ent_graph,ent_feature)
        new_ent_feature += batch_sparse_matmul(ent_rel_graph,rel_feature)
        new_ent_feature = tf.nn.l2_normalize(new_ent_feature,axis=-1)
        
        ent_feature = new_ent_feature; rel_feature = new_rel_feature
        ent_list.append(ent_feature); rel_list.append(rel_feature)
    
    ent_feature = K.l2_normalize(K.concatenate(ent_list,1),-1)
    rel_feature = K.l2_normalize(K.concatenate(rel_list,1),-1)
    rel_feature = random_projection(rel_feature,rel_dim)
    
    
    batch_size = ent_feature.shape[-1]//mini_dim
    sparse_graph = tf.SparseTensor(indices=triples_idx,values=K.ones(triples_idx.shape[0]),dense_shape=(np.max(triples_idx)+1,rel_size))
    adj_value = batch_sparse_matmul(sparse_graph,rel_feature)
    
    features_list = []
    for batch in range(rel_dim//batch_size + 1):
        temp_list = []
        for head in range(batch_size):
            if batch*batch_size+head>=rel_dim:
                break
            sparse_graph = tf.SparseTensor(indices=ent_tuple,values=adj_value[:,batch*batch_size+head],dense_shape=(node_size,node_size))
            feature = batch_sparse_matmul(sparse_graph,random_projection(ent_feature,mini_dim))
            temp_list.append(feature)
        if len(temp_list):
            features_list.append(K.concatenate(temp_list,-1).numpy())
    features = np.concatenate(features_list,axis=-1)
    
    faiss.normalize_L2(features)
    if extra_feature is not None:
        features = np.concatenate([ent_feature,features],axis=-1)
    return features

def save_gpu_mem(tf_gpu_no, output_dir, txt="gpu_mem_usage_before"):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(tf_gpu_no)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    with open(os.path.join(output_dir, "running.log"), "a+") as file:
        msg = {"msg_type": txt, "value": info.used/1024/1024}
        file.write(json.dumps(msg)+"\n")

def seed_everything(seed=1011):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def save():
    with open(os.path.join(data_dir, "ent_ids_1")) as file:
        lines = file.read().strip().split("\n")
        ent1_id_list = [int(line.split()[0]) for line in lines]

    with open(os.path.join(data_dir, "ent_ids_2")) as file:
        lines = file.read().strip().split("\n")
        ent2_id_list = [int(line.split()[0]) for line in lines]

    ent1_ids = np.array(ent1_id_list)
    ent2_ids = np.array(ent2_id_list)


    # Lvec = np.array([features[e1] for e1, e2 in test_pair])
    # Rvec = np.array([features[e2] for e1, e2 in test_pair])
    # # Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    # # Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    # csls_test_alignment, _, csls_test_metrics, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, csls=10, accurate=True)
    # cos_test_alignment, _, cos_test_metrics, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, csls=0, accurate=True)
    # metrics_csls = test(test_pair,features,[1, 5, 10],top_k)

    # metrics_obj = {"metrics_csls": metrics_csls, "metrics_cos": cos_test_metrics}
    # csls_test_alignment = np.array(list(csls_test_alignment), dtype=int)
    # cos_test_alignment = np.array(list(cos_test_alignment))
    # pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment.tolist(), "pred_alignment_cos": cos_test_alignment.tolist()}

    np.savez(os.path.join(output_dir, "emb.npz"), embs=features, ent1_ids=ent1_ids, ent2_ids=ent2_ids)
    # with open(os.path.join(output_dir, "metrics.json"), "w+") as file:
    #     file.write(json.dumps(metrics_obj))
    # with open(os.path.join(output_dir, "pred_alignment.json"), "w+") as file:
    #     file.write(json.dumps(pred_alignment_obj))

# obtain the literal features of entities, only work on DBP15K & SRPRS
# for the first run, you need to download the pre-train word embeddings from "http://nlp.stanford.edu/data/glove.6B.zip"
# unzip this file and put "glove.6B.300d.txt" into the root of LightEA

if __name__ == "__main__":
    nvidia_smi.nvmlInit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/DBP15K/zh_en/", required=False, help="input dataset file directory")
    parser.add_argument('--output_dir', type=str,  default="./results/", required=False, )
    parser.add_argument('--layer_num', type=int, default=2, help="number of GCN layers")
    parser.add_argument('--tf_gpu_no', type=int, default=0, help="number of GCN layers")
    parser.add_argument('--max_train_epoch', type=int, default=50, help="max epoch of training EA model")
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    save_gpu_mem(args.tf_gpu_no, output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.tf_gpu_no)
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    # choose the dataset and set the random seed
    # the first run may be slow because the graph needs to be preprocessed into binary cache
    np.random.seed(12306)


    # set hyper-parameters, load graphs and pre-aligned entity pairs
    # if your GPU is out of memory, try to reduce the ent_dim

    ent_dim, depth, top_k = 128, 2, 500
    if "en" in data_dir:
        rel_dim, mini_dim = ent_dim//2, 16
    else:
        rel_dim, mini_dim = ent_dim//3, 16
        
    node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(data_dir)

    train_pair,test_pair = load_aligned_pair(data_dir)
    candidates_x,candidates_y = set([x for x,y in test_pair]), set([y for x,y in test_pair])
    save_gpu_mem(args.tf_gpu_no, output_dir)
    using_name_features = False
    if using_name_features and "EN" in data_dir: 
        name_features = load_name_features(data_dir,"./glove.6B.300d.txt",mode = "hybrid-level")
        l_features = get_features(train_pair,extra_feature = name_features)

    # Obtain the structural features and iteratively generate Semi-supervised data
    # "epoch = 1" represents removing the iterative strategy

    epochs = 3
    for epoch in range(epochs):
        print("Round %d start:"%(epoch+1))
        s_features = get_features(train_pair)  
        if using_name_features and "EN" in data_dir:
            features = np.concatenate([s_features,l_features],-1)
        else:
            features = s_features
        if epoch < epochs-1:
            if len(candidates_x) == 0 or len(candidates_y) == 0:
                continue
            left,right = list(candidates_x),list(candidates_y)
            top_k = min(len(left), len(right), top_k)
            index,sims = sparse_sinkhorn_sims(left,right,features,top_k)
            ranks = tf.argsort(-sims,-1).numpy()
            sims = sims.numpy(); index = index.numpy()

            temp_pair = []
            x_list,y_list= list(candidates_x),list(candidates_y)
            for i in range(ranks.shape[0]):
                if sims[i,ranks[i,0]] > 0.5:
                    x = x_list[i]
                    y = y_list[index[i,ranks[i,0]]]
                    temp_pair.append((x,y))

            for x,y in temp_pair:
                if x in candidates_x:
                    candidates_x.remove(x)
                if y in candidates_y:
                    candidates_y.remove(y)
            
            print("new generated pairs = %d"%(len(temp_pair)))
            print("rest pairs = %d"%(len(candidates_x)))
            
            if not len(temp_pair):
                break
            train_pair = np.concatenate([train_pair,np.array(temp_pair)])

    save()
    save_gpu_mem(args.tf_gpu_no, output_dir, "gpu_mem_usage_after")

# python LightEA_run.py --data_dir=/public/home/qzhou0/qzhou20194227007/Larger_EA/HugeEA/data/IDS15K_V2/en_fr/partition_3 --output_dir=/public/home/qzhou0/qzhou20194227007/Larger_EA/HugeEA/results/DivEA/lightea/cosine/IDS15K_V2/en_fr/partition_3 --max_train_epoch=50 --tf_gpu_no=0