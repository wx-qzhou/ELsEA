import warnings
warnings.filterwarnings('ignore')

import os
import random
import keras
import json
import argparse
from tqdm import *
import numpy as np
from utils import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer import NR_GraphAttention
from os import makedirs
from CSLS import *
import nvidia_smi


def get_embedding(index_a,index_b,vec = None):
    if vec is None:
        inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]
        vec = get_emb.predict_on_batch(inputs)
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec,axis=-1,keepdims=True)+1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec,axis=-1,keepdims=True)+1e-5)
    return Lvec,Rvec

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings
    
def get_trgat(node_hidden, rel_hidden, triple_size, node_size, rel_size, dropout_rate = 0, gamma = 3, lr = 0.005, depth = 2):
    adj_input = Input(shape=(None,2))
    index_input = Input(shape=(None,2),dtype='int64')
    val_input = Input(shape = (None,))
    rel_adj = Input(shape=(None,2))
    ent_adj = Input(shape=(None,2))
    
    ent_emb = TokenEmbedding(node_size,node_hidden,trainable = True)(val_input) 
    rel_emb = TokenEmbedding(rel_size,node_hidden,trainable = True)(val_input)
    
    def avg(tensor,size):
        adj = K.cast(K.squeeze(tensor[0],axis = 0),dtype = "int64")   
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:,0],dtype = 'float32'), dense_shape=(node_size,size)) 
        adj = tf.compat.v1.sparse_softmax(adj) 
        return tf.compat.v1.sparse_tensor_dense_matmul(adj,tensor[1])
    
    opt = [rel_emb,adj_input,index_input,val_input]
    ent_feature = Lambda(avg,arguments={'size':node_size})([ent_adj,ent_emb])
    rel_feature = Lambda(avg,arguments={'size':rel_size})([rel_adj,rel_emb])
    
    e_encoder = NR_GraphAttention(node_size,activation="tanh",
                                       rel_size = rel_size,
                                       use_bias = True,
                                       depth = depth,
                                       triple_size = triple_size)
    
    r_encoder = NR_GraphAttention(node_size,activation="tanh",
                                       rel_size = rel_size,
                                       use_bias = True,
                                       depth = depth,
                                       triple_size = triple_size)
    
    out_feature = Concatenate(-1)([e_encoder([ent_feature]+opt),r_encoder([rel_feature]+opt)])
    out_feature = Dropout(dropout_rate)(out_feature)
    
    alignment_input = Input(shape=(None,2))
    
    def align_loss(tensor): 
        
        def squared_dist(x):
            A,B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            x = row_norms_A + row_norms_B - 2 * tf.matmul(A, B,transpose_b=True) 
            del A, B, row_norms_A, row_norms_B
            gc.collect()
            return x
        
        def infoNce1(ll, rr):
            temp_node = 0.1
            normalize_user_emb1 = tf.nn.l2_normalize(ll, 1)
            normalize_user_emb2 = tf.nn.l2_normalize(rr, 1)
            normalize_all_user_emb2 = normalize_user_emb2
            pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
            ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
            pos_score_user = tf.exp(pos_score_user / temp_node)
            ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / temp_node), axis=1)
            cl_loss_user = -tf.reduce_mean(tf.compat.v1.log(pos_score_user / ttl_score_user))
            del normalize_user_emb1, normalize_user_emb2, pos_score_user, ttl_score_user, ll, rr
            return cl_loss_user

        weight = (e_encoder.weight + r_encoder.weight) / 2
        weight_norm = tf.nn.softmax(weight, axis=0)
        weight_norm = weight_norm / tf.reduce_sum(weight_norm)

        emb = tensor[1]
        l,r = K.cast(tensor[0][0,:,0],'int32'),K.cast(tensor[0][0,:,1],'int32')
        l_emb,r_emb = K.gather(reference=emb,indices=l),K.gather(reference=emb,indices=r)
        
        del tensor, weight

        pos_dis = K.sum(K.square(l_emb-r_emb),axis=-1,keepdims=True)
        r_neg_dis = squared_dist([r_emb,emb])
        l_neg_dis = squared_dist([l_emb,emb])
        
        l_loss = pos_dis - l_neg_dis + gamma
        l_loss = l_loss *(1 - K.one_hot(indices=l,num_classes=node_size) - K.one_hot(indices=r,num_classes=node_size))
        
        r_loss = pos_dis - r_neg_dis + gamma
        r_loss = r_loss *(1 - K.one_hot(indices=l,num_classes=node_size) - K.one_hot(indices=r,num_classes=node_size))
        
        del emb, pos_dis, r_neg_dis, l_neg_dis, l, r

        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(r_loss,axis=-1,keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss,axis=-1,keepdims=True))) / K.stop_gradient(K.std(l_loss,axis=-1,keepdims=True))
        
        lamb,tau = 30, 10
        l_loss = K.logsumexp(lamb*l_loss+tau,axis=-1)
        r_loss = K.logsumexp(lamb*r_loss+tau,axis=-1)
        if node_size > 40000:
            loss = weight_norm[0] + weight_norm[1] * K.mean(l_loss + r_loss) 
        else:
            loss = weight_norm[0] * infoNce1(l_emb, r_emb) + weight_norm[1] * K.mean(l_loss + r_loss) 
        del l_emb, r_emb, weight_norm, l_loss, r_loss
        gc.collect()
        return loss

    loss = Lambda(align_loss)([alignment_input,out_feature])

    inputs = [adj_input,index_input,val_input,rel_adj,ent_adj]
    train_model = keras.Model(inputs = inputs + [alignment_input],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=keras.optimizers.RMSprop(lr))
    
    feature_model = keras.Model(inputs = inputs,outputs = out_feature)
    
    return train_model,feature_model

def CSLS_test(thread_number=16, csls=10, accurate=True):
    Lvec, Rvec = get_embedding(dev_pair[:,0],dev_pair[:,1])
    _, _, metrics, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    return metrics

def save():
    inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix]
    inputs = [np.expand_dims(item,axis=0) for item in inputs]
    vec = get_emb.predict_on_batch(inputs)

    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    with open(os.path.join(data_dir, "ent_ids_1")) as file:
        lines = file.read().strip().split("\n")
        ent1_id_list = [int(line.split()[0]) for line in lines]

    with open(os.path.join(data_dir, "ent_ids_2")) as file:
        lines = file.read().strip().split("\n")
        ent2_id_list = [int(line.split()[0]) for line in lines]

    ent1_ids = np.array(ent1_id_list)
    ent2_ids = np.array(ent2_id_list)


    # Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    # Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    # # Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    # # Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    # csls_test_alignment, _, csls_test_metrics, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, csls=10, accurate=True)
    # cos_test_alignment, _, cos_test_metrics, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, csls=0, accurate=True)

    # metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
    # csls_test_alignment = np.array(list(csls_test_alignment), dtype=int)
    # cos_test_alignment = np.array(list(cos_test_alignment))
    # pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment.tolist(), "pred_alignment_cos": cos_test_alignment.tolist()}

    np.savez(os.path.join(output_dir, "emb.npz"), embs=vec, ent1_ids=ent1_ids, ent2_ids=ent2_ids)
    # with open(os.path.join(output_dir, "metrics.json"), "w+") as file:
    #     file.write(json.dumps(metrics_obj))
    # with open(os.path.join(output_dir, "pred_alignment.json"), "w+") as file:
    #     file.write(json.dumps(pred_alignment_obj))

    # model.save(os.path.join(output_dir, "model.ckpt"))
    # get_emb.save(os.path.join(output_dir, "get_emb.ckpt"))

def make_dirs(path):
    if not os.path.exists(path):
        makedirs(path)

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

if __name__ == "__main__":
    nvidia_smi.nvmlInit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/DBP15K/zh_en/", required=False, help="input dataset file directory")
    parser.add_argument('--output_dir', type=str,  default="../results/Dual_AMN/", required=False, )
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

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True  
    sess = tf.compat.v1.Session(config=config)  

    seed = 12306
    seed_everything(seed)

    train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data(data_dir)
    adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
    rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
    ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    triple_size = len(adj_matrix)
    node_hidden = 128
    rel_hidden = 128
    batch_size = 1024
    dropout_rate = 0.3
    lr = 0.005
    gamma = 1
    depth = args.layer_num

    model,get_emb = get_trgat(dropout_rate=dropout_rate,
                              node_size=node_size,
                              rel_size=rel_size,
                              triple_size=triple_size,
                              depth=depth,
                              gamma=gamma,
                              node_hidden=node_hidden,
                              rel_hidden=rel_hidden,
                              lr=lr)

    rest_set_1 = [e1 for e1, e2 in dev_pair]
    rest_set_2 = [e2 for e1, e2 in dev_pair]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)

    epoch = args.max_train_epoch
    for turn in range(5):
        for i in trange(epoch):
            np.random.shuffle(train_pair)
            for pairs in [train_pair[i*batch_size:(i+1)*batch_size] for i in range(len(train_pair)//batch_size + 1)]:
                if len(pairs) == 0:
                    continue
                inputs = [adj_matrix,r_index,r_val,rel_matrix,ent_matrix,pairs]
                inputs = [np.expand_dims(item,axis=0) for item in inputs]
                model.train_on_batch(inputs,np.zeros((1,1)))
            # if i==epoch-1:
            #     CSLS_test()
        new_pair = []   
        if len(rest_set_1) == 0 or len(rest_set_2) == 0:
            continue
        Lvec, Rvec = get_embedding(rest_set_1, rest_set_2)
        A, _, _, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        B, _, _, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        A = sorted(list(A))
        B = sorted(list(B))
        for a, b in A:
            if B[b][1] == a:
                new_pair.append([rest_set_1[a],rest_set_2[b]])
        
        train_pair = np.concatenate([train_pair,np.array(new_pair)],axis = 0)
        for e1,e2 in new_pair:
            if e1 in rest_set_1:
                rest_set_1.remove(e1) 
            
        for e1,e2 in new_pair:
            if e2 in rest_set_2:
                rest_set_2.remove(e2)
        epoch = 5
    
    save_gpu_mem(args.tf_gpu_no, output_dir, "gpu_mem_usage_after")
    
    save()

# python run_DualA.py --data_dir=/public/home/qzhou0/qzhou20194227007/Larger_EA/HugeEA/data/IDS15K_V2/en_de/partition_2 --output_dir=./results/ --max_train_epoch=50 --tf_gpu_no=0 