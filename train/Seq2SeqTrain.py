import copy
import random
import torch
import numpy as np
import copy

import torch.nn as nn
import torch.nn.functional as F

# set random seed
SEED = 44

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def init_weights(model):
	"""
	Here we reproduce Keras default initialization weights for consistency with Keras version
	Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
	"""
	w = (param.data for name, param in model.named_parameters() if 'weight' in name)
	b = (param.data for name, param in model.named_parameters() if 'bias' in name)
	
	for t in w:
		nn.init.xavier_uniform_(t)
	for t in b:
		nn.init.constant_(t, 0.1)


def train(model, iterator, optimizer, parameters,
          neighbors, is_fine_tune=False):
	model.train()
	
	batch_num = 0
	total_tokens = 0
	total_loss = 0
	MSE_loss = 0
	cross_loss = 0
	
	total_num = 1
	total_acc = 0
	
	
	neighbor_criterion = nn.NLLLoss()
	time_criterion = nn.MSELoss()
	
	out_x, out_y, acc = None, None, None
	for i, batch in enumerate(iterator):
		src_spatial_seqs, src_temporal_seqs, src_mask, \
		trg_spatial_seqs, trg_temporal_seqs, trg_mask, \
		trg_lengths, \
		candidate_spatial_seqs, candidate_temporal_seqs, candidate_mask, \
		trg_y, ntokens, miss_index, miss_count = batch
		
		batch_num += 1
		
		src_spatial_seqs = src_spatial_seqs.to(device)
		src_temporal_seqs = src_temporal_seqs.to(device)
		
		trg_spatial_seqs = trg_spatial_seqs.to(device)
		trg_temporal_seqs = trg_temporal_seqs.to(device)
		
		candidate_spatial_seqs = candidate_spatial_seqs.to(device)
		candidate_temporal_seqs = candidate_temporal_seqs.to(device)
		
		
		src_mask = src_mask.to(device)
		trg_mask = trg_mask.to(device)
		candidate_mask = candidate_mask.to(device)
		
		trg_y = trg_y.to(device)
		
		trg_y_temp = copy.deepcopy(trg_y)
		
		node_size = neighbors.shape[0]
		neighbor_size = neighbors.shape[1]
		batch_size = trg_spatial_seqs.shape[0]
		seq_size = trg_spatial_seqs.shape[1]
		
		trg_seqs_repeat = trg_spatial_seqs.detach().view(batch_size, seq_size, 1).repeat(1, 1, neighbor_size)
		
		neighbors_batch_repeat = neighbors.detach().view(1, node_size, neighbor_size).repeat(batch_size, 1, 1)
		tgt_neighbors = neighbors_batch_repeat.gather(1, trg_seqs_repeat).long()
		trg_y_temp = trg_y_temp.view(trg_y_temp.shape[0], trg_y_temp.shape[1], 1).repeat(1, 1, neighbor_size)
		
		trg_neighbors_non_zero_index = torch.nonzero(trg_y_temp == tgt_neighbors)
		last_item = trg_neighbors_non_zero_index[0]
		
		trg_neighbors_index = trg_neighbors_non_zero_index[:, -1].to(device)
		flatten_miss_index = [[x - 1 for x_list in row for x in x_list] for row in miss_index]
		
		optimizer.zero_grad()
		output_road_id, _ = model(src_spatial_seqs, trg_spatial_seqs, candidate_spatial_seqs,
		                                            src_temporal_seqs, trg_temporal_seqs, candidate_temporal_seqs,
		                                            src_mask, trg_mask, candidate_mask,
		                                            parameters, trg_lengths,
		                                            trg_neighbors=tgt_neighbors)

		output_mask = torch.zeros(batch_size, seq_size).to(device)
		
		instead_val_2d = torch.zeros((batch_size * seq_size, output_road_id.shape[1])).to(device)
		instead_val_long = torch.zeros(batch_size * seq_size).long().to(device)
		instead_val_float = torch.zeros(batch_size * seq_size).float().to(device)
		
		for row, col in enumerate(flatten_miss_index):
			output_mask[row][col] = 1
		output_mask = output_mask.reshape(-1, 1)
		output_road_id = torch.where(output_mask > 0.5, output_road_id, instead_val_2d)
		
		output_mask = output_mask.reshape(-1)
		trg_neighbors_index = torch.where(output_mask > 0.5, trg_neighbors_index, instead_val_long)

		L1 = neighbor_criterion(output_road_id, trg_neighbors_index)

		neighbor_val = torch.argmax(output_road_id, dim=1)
		L1.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
		
		optimizer.step()
		total_loss += L1.detach().cpu().numpy()
		cross_loss += L1.detach().cpu().numpy()
		
		if i % 600 == 0:
			optimizer.get_rate()
			print('iteration {} loss: {}'.format(i, L1.detach().cpu().numpy()))
