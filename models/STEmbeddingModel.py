import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.init_emb()
    
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, u_pos, v_pos, v_neg, batch_size):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)
        
        score = torch.mul(embed_u, embed_v)
        score = torch.mean(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()
        
        neg_embed_v = self.v_embeddings(v_neg)
        
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.mean(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()
        
        loss = log_target + sum_log_sampled
        
        return -1 * loss.sum() / batch_size

    def get_embedding(self, id2word):
        embeds = self.u_embeddings.weight.data
        id_embeds = dict()
        for idx in range(len(embeds)):
            id_embeds[idx] = embeds[idx].tolist()
    
        return pd.DataFrame(id_embeds)

