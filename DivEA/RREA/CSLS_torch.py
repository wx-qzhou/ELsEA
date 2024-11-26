import gc
import time
import torch
import faiss
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import trange
import pdb

"""
https://github.com/ZJU-DAILY/LargeEA
"""
def sparse_sinkhorn_sims(features_l,features_r,top_k=500,iteration=15,mode="test"):
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

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as ex:
            print(ex)

    if len(gpus):
        index = faiss.index_cpu_to_gpu(res, len(gpus) - 1, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = features_l.shape[0]
    row_index = K.transpose(([K.arange(size*top_k)//top_k,index,K.arange(size*top_k)]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
    del features_l, features_r, sims, index, covert_idx
    gc.collect()
    return K.reshape(row_index[:,1],(-1,top_k)).numpy(), K.reshape(row_sims,(-1,top_k)).numpy()

class Evaluator():
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.batch_size = 512

    def evaluate_csls(self, embed1: np.ndarray, embed2: np.ndarray, eval_alignment, top_k=(1, 3, 5, 10, 50), target_candidates=None):
        print("index {0}".format(32))
        t1 = time.time()
        csls_sim_mat = self.csls_sim(embed1, embed2, k=10)
        csls_metrics, pred_alignment = self.compute_metrics(csls_sim_mat, eval_alignment, top_k, target_candidates)
        del csls_sim_mat
        gc.collect()
        t2 = time.time()
        print(f"evaluation speeds: {t2 - t1}s")
        return csls_metrics, pred_alignment

    def evaluate_cosine(self, embed1: np.ndarray, embed2: np.ndarray, eval_alignment, top_k=(1, 3, 5, 10, 50), target_candidates=None):
        print("index {0}".format(29))
        t1 = time.time()
        cos_sim_mat = self.cosine_sim(embed1, embed2)
        cos_metrics, pred_alignment = self.compute_metrics(cos_sim_mat, eval_alignment, top_k, target_candidates)
        # "cos_metrics" contains the results of all metrics, 'pred_alignment' stores the predicted aligned seeds.
        del cos_sim_mat
        gc.collect()
        t2 = time.time()
        print(f"evaluation speeds: {t2 - t1}s")
        return cos_metrics, pred_alignment

    def evaluate_sinkhorn(self, embed1: np.ndarray, embed2: np.ndarray, eval_alignment, top_k=(1, 3, 5, 10, 50), target_candidates=None):
        print("index {0}".format(32))
        t1 = time.time()
        index, sims = sparse_sinkhorn_sims(embed1, embed2)
        cos_metrics, pred_alignment = self.compute_metrics(-sims, eval_alignment, top_k, target_candidates)
        # "cos_metrics" contains the results of all metrics, 'pred_alignment' stores the predicted aligned seeds.
        del index,sims
        gc.collect()
        t2 = time.time()
        print(f"evaluation speeds: {t2 - t1}s")
        return cos_metrics, pred_alignment
    
    def predict_alignment(self, embed1: np.ndarray, embed2: np.ndarray, eval_way="sinkhorn"):
        print("index {0}".format(35))
        if eval_way == "sinkhorn":
            index, sim_mtx = sparse_sinkhorn_sims(embed1, embed2)
        if eval_way == "csls":
            sim_mtx = self.csls_sim(embed1, embed2, k=10)
        if eval_way == "cosine":
            sim_mtx = self.cosine_sim(embed1, embed2)

        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            pred_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="predict alignment"):
                if isinstance(sim_mtx, np.ndarray):
                    sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                    sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                else:
                    sub_sim_mtx = sim_mtx[cursor: cursor + self.batch_size].to(self.device)

                pred_ranking = torch.argsort(sub_sim_mtx, dim=1, descending=True)
                pred_list.append(pred_ranking[:, 0].cpu().numpy())
        pred_arr = np.concatenate(pred_list, axis=0)
        pred_alignment = np.stack([np.arange(len(embed1)), pred_arr], axis=1)
        return pred_alignment

    def csls_sim(self, embed1: np.ndarray, embed2: np.ndarray, k=10):
        print("index {0}".format(33))
        t1 = time.time()
        sim_mat = self.cosine_sim(embed1, embed2)
        if k <= 0:
            print("k = 0")
            return sim_mat
        csls1 = self.CSLS_thr(sim_mat, k)
        csls2 = self.CSLS_thr(sim_mat.T, k)

        # csls_sim_mat = 2 * sim_mat.T - csls1
        # csls_sim_mat = csls_sim_mat.T - csls2
        csls_sim_mat = self.compute_csls(sim_mat, csls1, csls2)
        # del sim_mat
        # gc.collect()
        t2 = time.time()
        print("shape", sim_mat.shape)
        return csls_sim_mat

    def compute_csls(self, sim_mtx: np.ndarray, row_thr:np.ndarray, col_thr:np.ndarray):
        # sp-CSLS
        print("index {0}".format(34))
        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            col_thr = torch.tensor(col_thr, device=self.device).unsqueeze(dim=0)
            for cursor in trange(0, total_size, self.batch_size, desc="csls metrix"):
                sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                sub_row_thr = row_thr[cursor: cursor+self.batch_size]
                sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                sub_row_thr = torch.tensor(sub_row_thr, device=self.device)
                sub_sim_mtx = 2 * sub_sim_mtx - sub_row_thr.unsqueeze(dim=1) - col_thr
                sim_mtx[cursor: cursor+self.batch_size] = sub_sim_mtx.cpu().numpy()
        return sim_mtx

    def cosine_sim(self, embed1: np.ndarray, embed2: np.ndarray):
        print("index {0}".format(30))
        with torch.no_grad():
            total_size = embed1.shape[0]
            embed2 = torch.tensor(embed2, device=self.device).t()
            sim_mtx= np.empty(shape=(embed1.shape[0], embed2.shape[1]), dtype=np.float32)
            for cursor in trange(0, total_size, self.batch_size, desc="cosine matrix"):
                sub_embed1 = embed1[cursor: cursor + self.batch_size]
                sub_embed1 = torch.tensor(sub_embed1, device=self.device)
                sub_sim_mtx = torch.matmul(sub_embed1, embed2)
                sim_mtx[cursor: cursor + self.batch_size] = sub_sim_mtx.cpu().numpy()
        return sim_mtx

    def CSLS_thr(self, sim_mtx: np.ndarray, k=10):
        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            sim_value_list = []
            for cursor in trange(0, total_size, self.batch_size, desc="csls thr"):
                sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)
                nearest_k, _ = torch.topk(sub_sim_mtx, dim=1, k=k, largest=True, sorted=False) # (B, k)
                sim_values = nearest_k.mean(dim=1, keepdim=False)
                sim_value_list.append(sim_values.cpu().numpy())
            sim_values = np.concatenate(sim_value_list, axis=0) # (N, )
        return sim_values

    def compute_metrics(self, sim_mtx, eval_alignment, top_k=(1, 5, 10, 50), target_candidates=None):
        # sim_mtx : (M, ), eval_alignment : list (M)
        print("index {0}".format(31))
        eval_alignment_arr = np.array(eval_alignment, dtype=np.int64) # (N, 2)
        eval_alignment_tensor = torch.tensor(eval_alignment, dtype=torch.long, device=self.device) # (N, 2)

        if target_candidates is None:
            target_candidates = torch.arange(0, sim_mtx.shape[1], device=self.device) # (M, )
        else:
            target_candidates = torch.tensor(target_candidates, device=self.device)

        with torch.no_grad():
            total_size = sim_mtx.shape[0]
            gold_rank_list = [] # This list is used to store the indexs of ture labels.
            pred_list = [] # This list is applied to store the first aligned ids.

            rels = torch.zeros(size=(total_size,), device=self.device, dtype=torch.long) # (M, )
            rels[eval_alignment_tensor[:, 0]] = eval_alignment_tensor[:, 1]
            for cursor in trange(0, total_size, self.batch_size, desc="metrics"):
                if isinstance(sim_mtx, np.ndarray):
                    sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                    sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)[:, target_candidates] # (B, M)
                else:
                    sub_sim_mtx = sim_mtx[cursor: cursor + self.batch_size].to(self.device)[:, target_candidates]

                sorted_idxes = torch.argsort(sub_sim_mtx, dim=1, descending=True) # (B, M)
                pred_ranking = target_candidates[sorted_idxes] # (B, M)
                pred_list.append(pred_ranking[:, 0].cpu().numpy())
                sub_rels = rels[cursor: cursor+self.batch_size].unsqueeze(dim=1)
                k, v = (pred_ranking == sub_rels).nonzero(as_tuple=True) # (B, )
                gold_idxes = torch.zeros(size=(pred_ranking.shape[0],), dtype=torch.long, device=pred_ranking.device) # (B, )
                gold_idxes[k] = v # (B, )
                gold_rank_list.append(gold_idxes.cpu().numpy())

        gold_rank_arr = np.concatenate(gold_rank_list, axis=0) + 1 # (N, )
        gold_rank_arr = gold_rank_arr[eval_alignment_arr[:, 0]] # obtain test alignment corresponding values.
        pred_arr = np.concatenate(pred_list, axis=0)
        pred_arr = pred_arr[eval_alignment_arr[:, 0]] # obtain test alignment corresponding values.
        pred_alignment = np.stack([eval_alignment_arr[:, 0], pred_arr], axis=1)
        mean_rank = np.mean(gold_rank_arr)
        mrr = np.mean(1.0 / gold_rank_arr)
        metrics = {"mr": float(mean_rank), "mrr": float(mrr)}
        for k in top_k:
            recall_k = np.mean((gold_rank_arr <= k).astype(np.float32))
            metrics[f"hit@{k}"] = float(recall_k)
        # pdb.set_trace()
        return metrics, pred_alignment

    def compute_metrics2(self, sim_mtx, eval_alignment, target_candidates, top_k=(1, 5, 10, 50)):  # can filter candidate entities
        eval_alignment_arr = np.array(eval_alignment, dtype=np.int)
        # sim_mtx = sim_mtx[eval_alignment_arr[:, 0]][:, eval_alignment_arr[:, 1]]

        eval_alignment_tensor = torch.tensor(eval_alignment, dtype=torch.long, device=self.device)
        # target_candidates = torch.arange(0, sim_mtx.shape[1], device=self.device)
        target_candidates = torch.tensor(target_candidates, device=self.device)
        with torch.no_grad():
            # total_size = eval_alignment_arr.shape[0]
            total_size = sim_mtx.shape[0]
            gold_rank_list = []
            pred_list = []
            # rels = eval_alignment_tensor[:, 1]
            rels = torch.zeros(size=(total_size,), device=self.device, dtype=torch.long)
            rels[eval_alignment_tensor[:, 0]] = eval_alignment_tensor[:, 1]
            for cursor in trange(0, total_size, self.batch_size, desc="metrics"):
                # sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                # sub_sim_mtx = sim_mtx[eval_alignment_arr[:, 0][cursor: cursor+self.batch_size]][:, eval_alignment_arr[:,1]]
                # sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size][:, eval_alignment_arr[:,1]]
                # sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)

                if isinstance(sim_mtx, np.ndarray):
                    sub_sim_mtx = sim_mtx[cursor: cursor+self.batch_size]
                    sub_sim_mtx = torch.tensor(sub_sim_mtx, device=self.device)[:, target_candidates]
                else:
                    sub_sim_mtx = sim_mtx[cursor: cursor + self.batch_size].to(self.device)[:, target_candidates]

                sorted_idxes = torch.argsort(sub_sim_mtx, dim=1, descending=True)
                pred_ranking = target_candidates[sorted_idxes]
                pred_list.append(pred_ranking[:, 0].cpu().numpy())
                sub_rels = rels[cursor: cursor+self.batch_size].unsqueeze(dim=1)
                k, v = (pred_ranking == sub_rels).nonzero(as_tuple=True)
                gold_idxes = torch.zeros(size=(pred_ranking.shape[0],), dtype=torch.long, device=pred_ranking.device)
                gold_idxes[k] = v
                gold_rank_list.append(gold_idxes.cpu().numpy())
        gold_rank_arr = np.concatenate(gold_rank_list, axis=0) + 1
        gold_rank_arr = gold_rank_arr[eval_alignment_arr[:, 0]]
        pred_arr = np.concatenate(pred_list, axis=0)
        pred_arr = pred_arr[eval_alignment_arr[:, 0]]
        pred_alignment = np.stack([eval_alignment_arr[:, 0], pred_arr], axis=1)
        mean_rank = np.mean(gold_rank_arr)
        mrr = np.mean(1.0 / gold_rank_arr)
        metrics = {"mr": float(mean_rank), "mrr": float(mrr)}
        for k in top_k:
            recall_k = np.mean((gold_rank_arr <= k).astype(np.float32))
            metrics[f"hit@{k}"] = float(recall_k)
        return metrics, pred_alignment
