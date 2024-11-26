
import torch
import regex
import string
import numpy as np
from tqdm import tqdm
import multiprocessing
from torch import Tensor
from functools import partial
from multiprocessing import Pool
from typing import *
try:
    from Levenshtein import ratio
except:
    print('holy shit')

PREFIX = r'http(s)?://[a-z\.]+/[^/]+/'

def apply(func, *args):
    if func is None:
        func = lambda x: x
    lst = []
    for arg in args:
        lst.append(func(arg))
    return tuple(lst)

def norm_process(embed: torch.Tensor, eps=1e-5) -> torch.Tensor:
    n = embed.norm(dim=1, p=2, keepdim=True)
    embed = embed / (n + eps)
    return embed

def ind2sparse(indices: Tensor, size, size2=None, dtype=torch.float, values=None):
    device = indices.device
    if isinstance(size, int):
        size = (size, size if size2 is None else size2)

    assert indices.dim() == 2 and len(size) == indices.size(0)
    if values is None:
        values = torch.ones([indices.size(1)], device=device, dtype=dtype)
    else:
        assert values.dim() == 1 and values.size(0) == indices.size(1)
    return torch.sparse_coo_tensor(indices, values, size).coalesce()

def topk2spmat(val0, ind0, size, dim=0, device: torch.device = 'cuda', split=False):
    if isinstance(val0, np.ndarray):
        val0, ind0 = torch.from_numpy(val0).to(device), \
                     torch.from_numpy(ind0).to(device)

    if split:
        return val0, ind0, size

    ind_x = torch.arange(size[dim]).to(device)
    ind_x = ind_x.view(-1, 1).expand_as(ind0).reshape(-1)
    ind_y = ind0.reshape(-1)
    ind = torch.stack([ind_x, ind_y])
    val0 = val0.reshape(-1)
    filter_invalid = torch.logical_and(ind[0] >= 0, ind[1] >= 0)
    ind = ind[:, filter_invalid]
    val0 = val0[filter_invalid]
    return ind2sparse(ind, list(size), values=val0)

def get_punctuations():
    en = string.punctuation
    zh = ""
    puncs = set()
    for i in (zh + en):
        puncs.add(i)
    puncs.remove('_')
    return puncs

PUNC = get_punctuations()

def remove_punc(str, punc=None):
    if punc is None:
        punc = PUNC
    if punc == '':
        return str
    return ''.join([' ' if i in punc else i for i in str])

def remove_prefix_to_list(entity_dict: Dict[int, str], prefix=PREFIX, punc='') -> []:
    import random
    import string

    def generate_random_string(length=10):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=length))
        return random_string

    entity_dict_temp = {}
    for key, value in entity_dict.items():
        if value not in entity_dict_temp:
            entity_dict_temp.update({value : key})
        else:
            print(value)
            entity_dict_temp.update({value + generate_random_string(len(str(key)) * 4) : key})
    entity_dict = entity_dict_temp
    del entity_dict_temp
    tmp_dict = {}
    entity_list = []
    p = regex.compile(prefix)
    for ent in entity_dict.keys():
        res = p.search(ent)
        if res is None:
            entity_list.append(remove_punc(ent, punc))
        else:
            _, end = res.span()
            entity_list.append(remove_punc(ent[end:], punc))

        tmp_dict[entity_list[-1]] = entity_dict[ent]
    entity_list = sorted(entity_list, key=lambda x: entity_dict[x] if x in entity_dict else tmp_dict[x])
    return entity_list

def mp2list(mp, assoc=None):
    if assoc is None:
        return sorted(list(mp.keys()), key=lambda x: mp[x])
    if isinstance(assoc, Tensor):
        assoc = assoc.cpu().numpy()
    return mp2list({k: assoc[v] for k, v in mp.items()}, None)

def edit_dist_of(sent0, sent1, item):
    x, y = item
    return ratio(sent0[x], sent1[y])

def selected_edit_distance(sent0, sent1, needed, batch_size=100000):
    x = np.empty([len(needed)])
    cpu = multiprocessing.cpu_count()
    print('cpu has', cpu, 'cores')
    pool = Pool(processes=cpu)
    for i in tqdm(range(0, len(needed), batch_size)):
        x[i: i + batch_size] = pool.map(partial(edit_dist_of, sent0, sent1), needed[i:i + batch_size])
    return x, needed

def pairwise_edit_distance(sent0, sent1, to_tensor=True):
    x = np.empty([len(sent0), len(sent1)], np.float)
    print(multiprocessing.cpu_count())
    pool = Pool(processes=multiprocessing.cpu_count())
    for i, s0 in enumerate(sent0):
        if i % 5000 == 0:
            print("edit distance --", i, "complete")
        x[i, :] = pool.map(partial(ratio, s0), sent1)
    if to_tensor:
        return (torch.from_numpy(x).to(torch.float))
