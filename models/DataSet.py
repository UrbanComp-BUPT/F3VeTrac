import copy

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
	def __init__(self, data_ori, parameters, train=True):
		self.seqs_id = []
		self.seqs_len = []
		self.src_spatial_seqs, self.src_temporal_seqs = [], []
		self.trg_spatial_seqs, self.trg_temporal_seqs = [], []
		self.trg_temporal_diffs, self.src_temporal_diffs, self.candidate_temporal_diffs = [], [], []
		self.candidate_spatial_seqs, self.candidate_temporal_seqs = [], []
		self.gps_spatial_seqs, self.gps_temporal_seqs = [], []
		self.miss_index, self.miss_count = [], []
		self.trg_y = []
		
		# above should be [num_seq, len_seq(unpadded)]
		self.generate_dataset(data_ori, parameters['auxi_traj_num'], train)
	
	def __len__(self):
		"""Denotes the total number of samples"""
		return len(self.src_spatial_seqs)
	
	def __getitem__(self, index):
		"""Generate one sample of data"""
		src_spatial_seq = self.src_spatial_seqs[index]
		src_temporal_seq = self.src_temporal_seqs[index]
		
		trg_spatial_seq = self.trg_spatial_seqs[index]
		trg_temporal_seq = self.trg_temporal_seqs[index]
		
		candidate_spatial_seq = self.candidate_spatial_seqs[index]
		candidate_temporal_seq = self.candidate_temporal_seqs[index]
		
		miss_index = self.miss_index[index]
		miss_count = self.miss_count[index]
		
		trg_y = self.trg_y[index]
		
		return src_spatial_seq, src_temporal_seq, \
		       trg_spatial_seq, trg_temporal_seq, \
		       candidate_spatial_seq, candidate_temporal_seq, \
		       trg_y, miss_index, miss_count
		
	def generate_dataset(self, data_ori, auxi_traj_num, train=True):


		# self.seqs_id = data_ori['id'].values.astype('str')
		seqs_data = data_ori['session'].values
		
		for i, each_seq in enumerate(seqs_data):
			
			self.src_spatial_seqs.append(torch.tensor(each_seq['src_traj']))
			# self.src_temporal_seqs.append(torch.tensor([torch.tensor(x) for x in each_seq['src_time']]))
			self.src_temporal_seqs.append(torch.tensor(each_seq['src_time']))
			src_time_diff = each_seq['src_time_diff']
			self.src_temporal_diffs.append(torch.tensor(src_time_diff, dtype=torch.float32))
			miss_index_temp = []
			miss_count_temp = []
			
			self.miss_index.append(each_seq['static_miss_index'])
			self.miss_count.append(each_seq['static_miss_count'])

			candidate_trajs = each_seq['candidates'][:auxi_traj_num]

			lengths = [len(seq[1]) for seq in candidate_trajs]
			if len(lengths) == 0:
				lengths.append(1)

			candidate_spatial_temp = np.zeros((auxi_traj_num_temp, max(lengths)))
			candidate_temporal_temp = np.zeros((auxi_traj_num_temp, max(lengths), 4))
			candidate_diffs_temp = np.zeros((auxi_traj_num_temp, max(lengths)))
			if candidate_trajs:
				for i, each_candidate_traj in enumerate(candidate_trajs):
					if i >= auxi_traj_num:
						break
					candidate_spatial_temp[i, :len(each_candidate_traj[1])] = each_candidate_traj[1]
					candidate_temporal_temp[i, :len(each_candidate_traj[2])] = each_candidate_traj[2]
					candidate_diffs_temp[i, 1:len(each_candidate_traj[3])] = each_candidate_traj[3][1:]

			self.candidate_spatial_seqs.append(torch.from_numpy(candidate_spatial_temp).long())
			self.candidate_temporal_seqs.append(torch.from_numpy(candidate_temporal_temp).long())
			self.candidate_temporal_diffs.append(torch.from_numpy(candidate_diffs_temp).float())
			
			s_traj = each_seq['s_traj'][:-1]
			t_traj = each_seq['t_traj'][:-1]
			time_diff = each_seq['time_diff'][1:]
			
			self.trg_spatial_seqs.append(torch.tensor(s_traj))
			self.trg_temporal_seqs.append(torch.tensor(t_traj))
			self.trg_temporal_diffs.append(torch.tensor(time_diff, dtype=torch.float32))
			trg_y_traj = copy.deepcopy(each_seq['s_traj'])
			self.trg_y.append(torch.tensor(trg_y_traj[1:]))



def subsequent_mask(size):
	"""
    mask后续的位置，返回[size, size]尺寸下三角Tensor
    对角线及其左下角全是1，右上角全是0
    """
	attn_shape = (1, size, size)
	subs_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subs_mask) == 0


def make_std_mask(tgt, pad):
	"""Create a mask to hide padding and future words."""
	tgt_mask = (tgt != pad).unsqueeze(-2)
	tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data).clone().detach()
	# 得到考虑单词长度和下三角矩阵的decoder的mask
	return tgt_mask


def merge(sequences, is_auxi_seq=False, is_time_seq=False, data_type=torch.int64):
	if is_auxi_seq:
		lengths = [[len(each_seq) for each_seq in seq] for seq in sequences]
		max_length = max([max(each_len) for each_len in lengths])
		traj_num = len(sequences[0])
		if is_time_seq:
			padded_seqs = torch.zeros((len(sequences), traj_num, max_length, 4), dtype=data_type)
			for i, seq in enumerate(sequences):
				for traj_index in range(traj_num):
					end = lengths[i][traj_index]
					padded_seqs[i, traj_index, :end, :] = seq[traj_index][:end, :]
		else:
			padded_seqs = torch.zeros((len(sequences), traj_num, max_length), dtype=data_type)
			for i, seq in enumerate(sequences):
				for traj_index in range(traj_num):
					end = lengths[i][traj_index]
					padded_seqs[i, traj_index, :end] = seq[traj_index][:end]
	else:
		lengths = [len(seq) for seq in sequences]
		if is_time_seq:
			padded_seqs = torch.zeros((len(sequences), max(lengths), 4), dtype=data_type)
			for i, seq in enumerate(sequences):
				end = lengths[i]
				padded_seqs[i, :end, :] = seq[:end, :]
		else:
			padded_seqs = torch.zeros((len(sequences), max(lengths)), dtype=data_type)
			for i, seq in enumerate(sequences):
				end = lengths[i]
				padded_seqs[i, :end] = seq[:end]
	
	return padded_seqs, lengths


def collate_fn(data):
	# sort a list by source sequence length (descending order) to use pack_padded_sequence
	data.sort(key=lambda x: len(x[0]), reverse=True)
	
	# seperate source and target sequences
	src_spatial_seqs, src_temporal_seqs, \
	trg_spatial_seqs, trg_temporal_seqs, \
	candidate_spatial_seqs, candidate_temporal_seqs, \
	trg_y, miss_index, miss_count = zip(*data)  # unzip data
	
	# merge sequences (from tuple of 1D tensor to 2D tensor)
	src_spatial_seqs, src_lengths = merge(src_spatial_seqs)
	src_temporal_seqs, _ = merge(src_temporal_seqs,
	                             is_time_seq=True)
	
	trg_spatial_seqs, _ = merge(trg_spatial_seqs)
	trg_temporal_seqs, _ = merge(trg_temporal_seqs,
	                             is_time_seq=True)
	
	candidate_spatial_seqs, hist_lengths = merge(candidate_spatial_seqs,
	                                             is_auxi_seq=True)
	candidate_temporal_seqs, _ = merge(candidate_temporal_seqs,
	                                   is_auxi_seq=True,
	                                   is_time_seq=True)
	
	src_mask = (src_spatial_seqs != 0).unsqueeze(-2)
	candidate_mask = (candidate_spatial_seqs != 0).unsqueeze(-2)
	trg_mask = make_std_mask(trg_spatial_seqs, 0)
	
	# ########################################################
	# 后期删除
	
	trg_y, trg_lengths = merge(trg_y)
	ntokens = (trg_y != 0).data.sum()
	
	# 后期删除：结尾
	# ########################################################
	
	return src_spatial_seqs, src_temporal_seqs, src_mask, \
	       trg_spatial_seqs, trg_temporal_seqs, trg_mask, \
	       trg_lengths, \
	       candidate_spatial_seqs, candidate_temporal_seqs, candidate_mask, \
	       trg_y, ntokens, miss_index, miss_count
