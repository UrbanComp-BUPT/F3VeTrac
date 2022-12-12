import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.optim as optim
from models.STEmbeddingModel import skipgram


def train_skipgram(data, road_intersections_num, embedding_dim,
                   epoch_num, batch_size, iter_count, id_2_traj):
    model = skipgram(road_intersections_num, embedding_dim)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    context_data = data[0][0]
    label_data = data[0][1]
    neg_data = data[1]
    running_loss = 0
    current_ite = 0
    for epoch in range(epoch_num):
        # print('epoch: ' + str(epoch))
        for index in range(iter_count):
            
            pos_u, pos_v, neg_v = context_data[index], label_data[index], neg_data[index]
            
            pos_u = torch.LongTensor(pos_u)
            pos_v = torch.LongTensor(pos_v)
            neg_v = torch.LongTensor(neg_v)
            
            if torch.cuda.is_available():
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()
            
            optimizer.zero_grad()
            
            loss = model(pos_u, pos_v, neg_v, batch_size)
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            current_ite += 1
        
        print('\tloss: ' + str(running_loss/current_ite))
        running_loss = 0
    
    print("Optimization Finished!")
    return model.get_embedding(id_2_traj)