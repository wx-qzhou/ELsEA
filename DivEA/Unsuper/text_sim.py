import gc
import time
import faiss
import torch
import torch_sparse
from typing import *
from tqdm import tqdm
from Unsuper.text_utils import apply, remove_prefix_to_list, PUNC, \
    selected_edit_distance, ind2sparse, topk2spmat, norm_process
from torch import Tensor
from datasketch import MinHashLSH, MinHash

def get_iv(sps: List[Tensor]):
    iv = []
    for sp in sps:
        a = sp.coalesce()
        iv += [a._indices(), a._values()]

    return tuple(iv)

def spspmm(a, b, separate=False):
    n = a.size(0)
    m = a.size(1)
    assert m == b.size(0)
    k = b.size(1)
    ai, av, bi, bv = get_iv([a, b])
    print('n,m,k=', n, m, k)
    print('ai, av, bi, bv=', apply(lambda x: x.numel(), ai, av, bi, bv))
    i, v = torch_sparse.spspmm(ai, av, bi, bv, n, m, k)
    if separate:
        nonzero_mask = v != 0.
        return i[:, nonzero_mask], v[nonzero_mask], [n, k]

    return torch.sparse_coo_tensor(i, v, [n, k])

def spmm_ds(d: Tensor, s: Tensor) -> Tensor:
    return spmm(s.t(), d.t()).t()

def spmm_sd(s: Tensor, d: Tensor) -> Tensor:
    s = s.coalesce()
    i, v, s, t = s._indices(), s._values(), s.size(0), s.size(1)
    return torch_sparse.spmm(i, v, s, t, d)

def spmm(s: Tensor, d: Tensor) -> Tensor:
    if s.is_sparse and d.is_sparse:
        return spspmm(s, d)
    elif s.is_sparse:
        return spmm_sd(s, d)
    elif d.is_sparse:
        return spmm_ds(s, d)
    else:
        return s.mm(d)

def dense_to_sparse(x):
    if x.is_sparse:
        return x
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size()).coalesce()

def makeset(ent_list, num_perm):
    sets = [set(ei.split('_')) for ei in ent_list]
    print('build MinHash')
    for s in tqdm(sets):
        m = MinHash(num_perm)
        for d in s:
            m.update(d.encode('utf-8'))
        yield m

def minhash_select_pairs(e1: Iterable[MinHash], e2: Iterable[str],
                         begin_with=0, threshold=0.85, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    print('build LSH index')
    for i, e in enumerate(makeset(e2, num_perm)):
        lsh.insert(str(i + begin_with), e)

    query_result = []
    print('query LSH')
    for i, e in enumerate(tqdm(e1)):
        query_result.append([x for x in map(int, lsh.query(e))])

    print('total pairs', sum(map(len, query_result)))
    print('expand result into oo pairs')
    pairs = []
    for i, r in enumerate(tqdm(query_result)):
        pairs += [(i, v) for v in r]

    time.sleep(1)
    return pairs

def faiss_search_impl(emb_q, emb_id, emb_size, shift, k=50, search_batch_sz=50000, gpu=True):
    index = faiss.IndexFlat(emb_size)
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(emb_id)
    print('Total index =', index.ntotal)
    vals, inds = [], []
    for i_batch in tqdm(range(0, len(emb_q), search_batch_sz)):
        val, ind = index.search(emb_q[i_batch:min(i_batch + search_batch_sz, len(emb_q))], k)
        val = torch.from_numpy(val)
        val = 1 - val
        vals.append(val)
        inds.append(torch.from_numpy(ind) + shift)
    del index, emb_id, emb_q
    vals, inds = torch.cat(vals), torch.cat(inds)
    return vals, inds

@torch.no_grad()
def global_level_semantic_sim(embs, k=50, search_batch_sz=50000, index_batch_sz=500000, split=False, norm=True, gpu=True):
    # embs: [[N, d], [M, d]]
    print('FAISS number of GPUs=', faiss.get_num_gpus())
    size = [embs[0].size(0), embs[1].size(0)] # [N, M]
    emb_size = embs[0].size(1) # d
    if norm:
        embs = apply(norm_process, *embs)
    emb_q, emb_id = apply(lambda x: x.cpu().numpy(), *embs)
    del embs
    gc.collect()
    vals, inds = [], []
    total_size = emb_id.shape[0] # M
    for i_batch in range(0, total_size, index_batch_sz):
        i_end = min(total_size, i_batch + index_batch_sz)
        val, ind = faiss_search_impl(emb_q, emb_id[i_batch:i_end], emb_size, i_batch, k, search_batch_sz, gpu)
        vals.append(val)
        inds.append(ind)

    vals, inds = torch.cat(vals, dim=1), torch.cat(inds, dim=1)

    return topk2spmat(vals, inds, size, 0, torch.device('cpu'), split), inds

def sparse_string_sim(ent1, ent2, batch_size=1000000, num_perm=128, threshold=0.85) -> Tensor:
    e1, e2 = remove_prefix_to_list(ent1, punc=PUNC), remove_prefix_to_list(ent2, punc=PUNC)
    len_all = len(e2)
    edit_dists = []
    idxs = []
    minhashs = [x for x in tqdm(makeset(e1, num_perm))]
    for i_batch in range(0, len_all, batch_size):
        batch_end = min(len_all, i_batch + batch_size)
        pairs = minhash_select_pairs(minhashs, e2[i_batch:batch_end], i_batch, threshold=threshold, num_perm=num_perm)
        edit_dist, idx = selected_edit_distance(e1, e2, pairs)
        edit_dists.append(torch.from_numpy(edit_dist))
        idxs.append(torch.tensor(idx))
    idxs = torch.cat(idxs)
    return ind2sparse(idxs.t(), [len(e1), len(e2)], values=torch.cat(edit_dists)), idxs
