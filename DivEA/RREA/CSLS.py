import gc
import time
import numpy as np
import multiprocessing

def cal_csls_sim(sim_mat, k):
    pk = min(k-1, sim_mat.shape[1]-1)
    sorted_mat = -np.partition(-sim_mat, pk, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values

def CSLS_sim(sim_mat1, k, nums_threads):
    # sorted_mat = -np.partition(-sim_mat1, k, axis=1) # -np.sort(-sim_mat1)
    # nearest_k = sorted_mat[:, 0:k]
    # sim_values = np.mean(nearest_k, axis=1)

    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values

def sim_handler(embed1, embed2, k, nums_threads):
    t1 = time.time()
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    t2 = time.time()
    print(f"sim handler spends time: {t2-t1}s")
    return csls_sim_mat

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    ranks = np.argsort(-sim, axis=1)
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            # rank = (-sim[i, :]).argsort()
            rank = ranks[i]
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set

def eval_alignment_by_sim_mat(embed1, embed2, top_k, nums_threads, csls=0, accurate=False, output=True):
    t = time.time()
    print("handle similarity")
    sim_mat = sim_handler(embed1, embed2, csls, nums_threads)
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    print("divide tasks")
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()
    
    print("computing metrics")
    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                       t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    # del sim_mat
    gc.collect()
    metrics = {"mr": t_mean, "mrr": t_mrr}
    for idx in range(len(top_k)):
        metrics[f"hit@{top_k[idx]}"] = float(acc[idx])
    return t_prec_set, hits1, metrics, sim_mat
