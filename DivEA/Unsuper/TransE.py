import pdb
import math
import torch
import random
import numpy as np
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

def task_divide(idx, n):
    """
    devide tasks, each task consists of batches
    Input:
    idx : list(range(relation_triple_steps)), n : number
    Output:
    [[batch1, batch2, ...], ...]
    """
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks

def generate_relation_triple_batch_queue(triple_set, entity_set, batch_size, steps, out_queue, neg_triples_num, is_fixed_size=False, neighbor=None):
    """
    generate the triples' queue based on relations
    Input:
    triple_set : {(h_id, r_id, t_id), ...}
    entity_set : [id, ...]
    batch_size : number
    out_queue : type is queue
    neg_triples_num : number
    Output:
    None
    """
    for step in steps: # steps : [batch1, batch2, ...]
        pos_batch = generate_pos_triples(triple_set, batch_size, step, is_fixed_size=is_fixed_size)
        neg_batch = generate_neg_triples_fast(pos_batch, entity_set, neg_triples_num, neighbor=neighbor)
        out_queue.put((pos_batch, neg_batch))
    exit(0)

def generate_pos_triples(triples, batch_size, step, is_fixed_size=False):
    """
    generate positive triples
    Input:
    triples : [(h_id, r_id, t_id), ...]
    batch_size : number 
    step : number and is a batch index
    Output:
    pos_batch : [(h_id, r_id, t_id), ...]
    """
    start = step * batch_size
    end = start + batch_size
    if end > len(triples):
        end = len(triples)

    pos_batch = triples[start: end]

    if is_fixed_size and len(pos_batch) < batch_size:
        pos_batch += triples[len(pos_batch) - batch_size : ]
        assert len(pos_batch) == batch_size
    return pos_batch

def generate_neg_triples_fast(pos_batch, entities_list, neg_triples_num, neighbor=None):
    """
    generate negative triples
    Input:
    pos_batch : [(h_id, r_id, t_id), ...]
    entities_list : [id, ...] 
    neg_triples_num : number
    Output:
    neg_batch : [(h_id, r_id, t_id), ...]
    """
    if neighbor is None:
        neighbor = dict()

    pos_batch_ = np.array(pos_batch)
    entities = set(pos_batch_[:, 0]) | set(pos_batch_[:, 2])
    del pos_batch_
    candidates_cache = {entity: random.sample(entities_list - neighbor.get(entity, set([])), neg_triples_num * 100) for entity in entities}
    neg_batch = list()
    for head, relation, tail in pos_batch:
        if np.random.binomial(1, 0.5):
            neg_heads = random.sample(candidates_cache[head], neg_triples_num)
            neg_triples = [(h2, relation, tail) for h2 in neg_heads]
        else:
            neg_tails = random.sample(candidates_cache[tail], neg_triples_num)
            neg_triples = [(head, relation, t2) for t2 in neg_tails]

        neg_batch.extend(neg_triples)
    
    del entities, candidates_cache, neg_triples, neg_heads, neg_tails, neighbor
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch

def process_loaded_data(device, triples, entity_set, relation_triple_steps=100, batch_threads_num=4, neg_triples_num=1, is_fixed_size=False, neighbor=None):
    print("load data.")
    manager = mp.Manager()
    relation_batch_queue = manager.Queue()
    batch_size = int(math.ceil(len(triples) / relation_triple_steps))
    
    relation_step_tasks = task_divide(list(range(relation_triple_steps)), batch_threads_num)
    for idx, steps_task in enumerate(relation_step_tasks):
        print("The id is {}.".format(idx))
        mp.Process(target=generate_relation_triple_batch_queue,
                    args=(triples, entity_set, batch_size, steps_task, relation_batch_queue, neg_triples_num, is_fixed_size, neighbor)).start()
    
    pos_neg_triples_list = []
    for _ in range(relation_triple_steps):
        batch_pos, batch_neg = relation_batch_queue.get()
        rel_p_h = torch.LongTensor([x[0] for x in batch_pos]).to(device)
        rel_p_r = torch.LongTensor([x[1] for x in batch_pos]).to(device)
        rel_p_t = torch.LongTensor([x[2] for x in batch_pos]).to(device)
        rel_n_h = torch.LongTensor([x[0] for x in batch_neg]).to(device)
        rel_n_r = torch.LongTensor([x[1] for x in batch_neg]).to(device)
        rel_n_t = torch.LongTensor([x[2] for x in batch_neg]).to(device)
        pos_neg_triples_list.append([rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t])

    return pos_neg_triples_list

class TransE(nn.Module):
    "https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"
    def __init__(self, margin, ent_num, rel_num, hidden_dim, norm=1, reduction='mean') -> None:
        super(TransE, self).__init__()
        self.margin = margin
        self.norm = norm
        self.reduction = reduction  

        'relation'
        self.ent_embed = nn.Embedding(ent_num, hidden_dim)
        self.rel_embed = nn.Embedding(rel_num, hidden_dim)
        nn.init.xavier_normal_(self.ent_embed.weight.data)
        nn.init.xavier_normal_(self.rel_embed.weight.data)

    def e_rep(self, e):
        return self.ent_embed(e)
    
    def r_rep(self, e):
        return self.rel_embed(e)

    def _distance(self, heads, relations, tails):
        score = (heads + relations - tails).norm(p=self.norm, dim=1) # (N)
        return score
    
    def forward(self, r_p_h, r_p_r, r_p_t, r_n_h, r_n_r, r_n_t):
        r_p_h = self.e_rep(r_p_h)
        r_p_r = self.r_rep(r_p_r)
        r_p_t = self.e_rep(r_p_t)
        r_n_h = self.e_rep(r_n_h)
        r_n_r = self.r_rep(r_n_r)
        r_n_t = self.e_rep(r_n_t)

        # pos_score = -self._distance(r_p_h, r_p_r, r_p_t)  # 正样本得分
        # neg_score = -self._distance(r_n_h, r_n_r, r_n_t)  # 负样本得分
        # del r_p_h, r_p_t, r_p_r, r_n_h, r_n_t, r_n_r

        # # 计算InfoNCE损失
        # logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)  # (N, 2)
        # labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)  # 正样本标签为0

        # # 使用温度参数进行缩放
        # logits /= 2

        # relation_loss = F.cross_entropy(logits, labels)

        pos_score_r = self._distance(r_p_h, r_p_r, r_p_t) # h + r -t = 0 Transe
        neg_score_r = self._distance(r_n_h, r_n_r, r_n_t) # h + r -t = 0
        del r_p_h, r_p_t, r_p_r, r_n_h, r_n_t, r_n_r

        neg_score_r = torch.mean(neg_score_r.reshape(pos_score_r.shape[0], -1), dim=-1)
        if self.reduction == 'sum':
            relation_loss = torch.sum(F.relu(self.margin + pos_score_r - neg_score_r))
        else:
            relation_loss = torch.mean(F.relu(self.margin + pos_score_r - neg_score_r))
        del pos_score_r, neg_score_r

        return relation_loss

    def get_all_embeddings(self):
        all_embeddings = self.ent_embed.weight.data  # 获取所有嵌入向量
        return {i: all_embeddings[i].cpu().numpy() for i in range(all_embeddings.size(0))}

'TransE'
def train_transe(entity_set, relation_set, neighbor, triples, is_GPU=True, dimensions=16, max_epoch=10, margin=2, learning_rate=0.001):
    device = torch.device('cuda:0' if torch.cuda.is_available() and is_GPU else 'cpu')
    ent_num = len(entity_set)
    rel_num = len(relation_set)
    tri_num = len(triples)
    print("The number of triples is {}.".format(tri_num))
    print("The number of entities is {}.".format(ent_num))
    print("The number of relations is {}.".format(rel_num))

    batch_threads_num = min(mp.cpu_count(), 8)

    if tri_num >= 200000:
        relation_triple_steps = tri_num // 60000
    else:
        relation_triple_steps = 40

    print("relation_triple_steps is {}.".format(relation_triple_steps))

    pos_neg_triples_list = process_loaded_data(device, triples, entity_set, relation_triple_steps=relation_triple_steps, batch_threads_num=batch_threads_num, neg_triples_num=1, neighbor=neighbor)
    model = TransE(margin, ent_num, rel_num, dimensions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)

    for i in range(1, max_epoch + 1):
        loss = 0
        for pos_neg_triples in pos_neg_triples_list:
            rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t = pos_neg_triples
            loss += model(rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t)
            del rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t

        loss /= len(pos_neg_triples_list)

        'loss backward'
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # 梯度裁剪
        optimizer.step()

        scheduler1.step()   
        torch.cuda.empty_cache()

        print("The {} loss of validing is {:.4f}".format(i, loss.item()))

    return model.get_all_embeddings()