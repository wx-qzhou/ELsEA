import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute the RMS (Root Mean Square)
        rms = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** (-0.5))
        
        # Normalize the input
        x_normalized = x / (rms + self.eps)
        
        # Apply the learned scale (weight)
        return self.weight * x_normalized

class LINE(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LINE, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.rms_norm = RMSNorm(embedding_dim)
        # 初始化节点嵌入
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, pos_edges, neg_edges):
        pos_src, pos_dst = pos_edges
        neg_src, neg_dst = neg_edges
        
        # 获取节点嵌入
        emb_src = self.node_embeddings(pos_src)
        emb_dst = self.node_embeddings(pos_dst)
        # emb_src = F.normalize(emb_src, 2, -1)
        # emb_dst = F.normalize(emb_dst, 2, -1)
        emb_src = self.rms_norm(emb_src)
        emb_dst = self.rms_norm(emb_dst)


        emb_neg_src = self.node_embeddings(neg_src)
        emb_neg_dst = self.node_embeddings(neg_dst)
        emb_neg_src = torch.mean(emb_neg_src, dim=1)
        emb_neg_dst = torch.mean(emb_neg_dst, dim=1)
        # emb_neg_src = F.normalize(emb_neg_src, 2, -1)
        # emb_neg_dst = F.normalize(emb_neg_dst, 2, -1)
        emb_neg_src = self.rms_norm(emb_neg_src)
        emb_neg_dst = self.rms_norm(emb_neg_dst)


        epsilon = 1e-7  # 一个小常数，用于避免log(0)

        # 一阶损失
        pos_score = torch.mean(emb_src * emb_dst, dim=1)
        neg_score = torch.mean(emb_neg_src * emb_neg_dst, dim=1)

        one_order_loss = -torch.mean(torch.log(torch.sigmoid(pos_score) + epsilon) +
                                     torch.log(1 - torch.sigmoid(neg_score) + epsilon))

        # 二阶损失
        pos_edges_score = torch.matmul(emb_src, emb_dst.t()).diagonal()
        neg_edges_score = torch.matmul(emb_neg_src, emb_neg_dst.t()).diagonal()

        # 二阶损失
        two_order_loss = -torch.mean(torch.log(torch.sigmoid(pos_edges_score) + epsilon) +
                                      torch.log(1 - torch.sigmoid(neg_edges_score) + epsilon))

        return one_order_loss + two_order_loss

# generate the negative samples
def generate_negative_samples(num_nodes, num_samples):
    """生成负样本"""
    neg_dst = torch.randint(0, num_nodes, (num_samples,))
    return neg_dst

# generate the graphs using nextworks
def prepare_graph_data(G):
    """从 networkx 图 G 生成训练数据"""
    edges = list(G.edges())
    num_nodes = len(G.nodes())
    pos_edges = [(torch.tensor(src), torch.tensor(dst)) for src, dst in edges]
    return pos_edges, num_nodes

# train the Line model
def train_line(G, embedding_dim, num_epochs, batch_size, num_samples=5):
    pos_edges, num_nodes = prepare_graph_data(G)
    
    model = LINE(num_nodes, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 按批处理训练
        for i in range(0, len(pos_edges), batch_size):
            # 提取批次边
            batch_edges = pos_edges[i:i + batch_size]
            pos_src = torch.tensor([src for src, _ in batch_edges])
            pos_dst = torch.tensor([dst for _, dst in batch_edges])
            
            # 生成负样本
            neg_dst = generate_negative_samples(num_nodes, len(batch_edges) * num_samples)
            neg_dst = neg_dst.reshape(-1, num_samples)
            neg_src = pos_src.unsqueeze(1).repeat(1, num_samples)
            neg_src = neg_src.reshape(-1, num_samples)
            optimizer.zero_grad()
            loss = model((pos_src, pos_dst), (neg_src, neg_dst))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')
    
    return model

# obtain the embeddings of nodes
def get_node_embeddings(model):
    """获取节点嵌入"""
    return model.node_embeddings.weight.data.numpy()
