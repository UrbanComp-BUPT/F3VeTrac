import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def stop_after_eos(traj_prob, trg_len, is_time=False):
	batch_size = traj_prob.shape[0]
	processed_traj_prob = torch.full((traj_prob.shape[0], traj_prob.shape[1], traj_prob.shape[-1]), 1e-9).to(device)

	for i in range(batch_size):
		processed_traj_prob[i][:trg_len[i]] = traj_prob[i][:trg_len[i]]
		if is_time:
			processed_traj_prob[i][trg_len[i]:, -1] = 0
		else:
			processed_traj_prob[i][trg_len[i]:, -1] = 1
			# make sure argmax will return eid0
	return processed_traj_prob


class EncoderDecoder(nn.Module):  # encoder+decoder
	"""标准的Encoder-Decoder架构"""

	def __init__(self, encoder, decoder, pretrained_embed, spatial_embed, temporal_embed,
	             generator, traj_gen, self_attn, cross_attn):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.pretrained_embed = pretrained_embed
		self.spatial_embed = spatial_embed      # 空间序列embedding
		self.temporal_embed = temporal_embed    # 时间序列embedding
		
		self.time_diff_linear = nn.Linear(1, 64)
		
		self.generator = generator  # 生成目标单词的概率
	
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.traj_gen = traj_gen
		
		self.src_init_attn = copy.deepcopy(self_attn)
		self.gps_init_attn = copy.deepcopy(self_attn)
		
		self.src_gps_attn = copy.deepcopy(cross_attn)
		
		self.fuse_attn = copy.deepcopy(cross_attn)
		
		self.out_attn = copy.deepcopy(cross_attn)
		
		self.fine_tune_linear = nn.Linear(64, 64)
		
	def forward(self, spatial_src, spatial_trg, spatial_candidate,
	            temporal_src, temporal_trg, temporal_candidate,
	            src_mask, trg_mask, candidate_mask,
	            train_parameters, trg_lengths,
	            trg_neighbors=None, neighbors=None, is_train=True,
	            flatten_miss_index=None, miss_count=None):

		src_init_rec, candidate_init_rec = self.init_rec(spatial_src, temporal_src,
		                                                 spatial_candidate, temporal_candidate,
		                                                 src_mask, candidate_mask)
		src_gps_rec = self.src_fuse_auxi_rec(src_init_rec, candidate_init_rec, candidate_mask)
		fused_rec = self.fuse_traj_info(src_gps_rec, src_init_rec)
		
		max_trg_len = spatial_trg.shape[1]
		batch_size = spatial_trg.shape[0]
		node_size = train_parameters['node_size']
		neighbor_size = train_parameters['neighbor_size']
		
		is_ground_truth = torch.ones(batch_size, max_trg_len)

		neighbor_time = temporal_trg.unsqueeze(-2).repeat(1, 1, neighbor_size, 1)

		trg_neighbors_embd = self.spatial_embed(trg_neighbors)

		output_states = self.decode(fused_rec, src_mask, spatial_trg, temporal_trg, trg_mask)  # , trg_temporal_diffs)

		output_states_t = self.leakyrelu(output_states.view(output_states.shape[0], output_states.shape[1],
		                                                    -1, output_states.shape[2]))

		output_neighbor_prob = self.traj_gen(output_states_t, trg_neighbors_embd)

		# 轨迹点id生成
		trg_neighbor_mask = ((trg_neighbors != 0) & (trg_neighbors != train_parameters['node_size']))

		output_road_id = F.softmax(output_neighbor_prob, dim=2)
		output_road_id = stop_after_eos(output_road_id, trg_lengths)
		output_road_id = torch.log(output_road_id)
		output_road_id = output_road_id.view(-1, output_road_id.shape[2])

		return output_road_id, is_ground_truth

	def init_rec(self, s_src, t_src,
	             s_candidate, t_candidate,
	             src_mask, candidate_mask):
		src_embed = self.spatial_embed(s_src) + self.temporal_embed(t_src)
		
		src_init_rec = self.src_init_attn(src_embed, src_mask)

		candidate_embed = self.spatial_embed(s_candidate) + self.temporal_embed(t_candidate)
		candidate_init_rec = self.gps_init_attn(candidate_embed, candidate_mask)
		
		return src_init_rec, candidate_init_rec
	
	def src_fuse_auxi_rec(self, src_init_rec, gps_init_rec, gps_mask):
		src_gps_rec_mean = self.src_gps_attn(src_init_rec, gps_init_rec, gps_mask)
		
		return src_gps_rec_mean
	
	def fuse_traj_info(self, src_gps_rec, src_init_rec):
		fused_rec = torch.cat((src_gps_rec, src_init_rec), 2)
		return fused_rec
	
	def encode(self, src, src_mask):
		embd = self.src_embed(src)
		
		return self.encoder(embd, src_mask)

	def decode(self, memory, src_mask, s_trg, t_trg,  trg_mask):
		trg_embed = self.spatial_embed(s_trg)
		
		return self.decoder(trg_embed, memory, src_mask, trg_mask)
	
	def init_spatial_embed(self, node_features, adj):
		return self.pretrained_embed(node_features, adj)
		

class TrajGenerator(nn.Module):
	"""根据空间约束生成轨迹点"""
	def __init__(self, d_model, dropout):
		super(TrajGenerator, self).__init__()
		layer = nn.Linear(d_model, d_model)
		self.layers = clones(layer, 2)
		self.neighbor_score = nn.Linear(d_model, 1)
		self.time_cal = nn.Linear(2 * d_model, 1)
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.dropout = nn.Dropout(dropout)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, source_embed, neighbor_embed):
		batch_size = source_embed.shape[0]
		seq_size = source_embed.shape[1]
		neighbor_size = neighbor_embed.shape[2]
		
		source_embed = self.dropout(self.leakyrelu(self.layers[0](source_embed)))
		neighbor_embed = self.dropout(self.leakyrelu(self.layers[1](neighbor_embed)))
		neighbor_score = torch.matmul(source_embed, neighbor_embed.transpose(2, 3))
		neighbor_score = neighbor_score.view(batch_size, seq_size, neighbor_size)

		return neighbor_score #, time_diff

# Encoder部分
def clones(module, N):
	# 产生N个相同的层
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):  # norm
	"""构造一个layernorm模块"""

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # 加个极小数


class SublayerConnection(nn.Module):  # add & norm
	"""Add+Norm"""

	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))


# Attention
def attention(query, key, value, mask=None, dropout=None):
	# 单头attention
	"""计算Attention即点乘V"""
	# 每个头的head
	d_k = query.size(-1)
	# [B, h, L, L]
	# 单头自注意力公式实现
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	# 多头attention
	def __init__(self, h, d_model, dropout=0.1):
		"""Take in model size and number of heads."""
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0  # h= heads
		self.d_k = d_model // h
		# a/b 向下取整  #d_k 每个头负责多少维
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		# 复制四个  3个给qkv，1个给最后一层
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
	
	def forward(self, query, key, value, mask=None):
		"""
		实现MultiHeadedAttention。
		   输入的q，k，v是形状 [batch, L, d_model]。  #L 是注意力W矩阵的列数，这个长度是矩阵的超参数，看图可以理解
		   输出的x 的形状同上。
		"""
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
			
		# 1) 这一步qkv变化:[batch, L, d_model] ->[batch, head, L, d_model/h]
		# 输入的q，k，v是形状 [batch, head, L, d_model]
		if len(query.shape) == 3 and len(key.shape) == 3:
			nbatches = key.shape[0]
			query, key, value = \
				[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
				 for l, x in zip(self.linears, (query, key, value))]
			# 2) 计算注意力attn 得到attn*v 与attn
			# qkv :[batch, head, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
			x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
			# 3) 上一步的结果合并在一起还原成原始输入序列的形状
			x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		elif len(query.shape) == 4 and len(key.shape) == 4:
			nbatches = key.shape[0]
			seq_len = key.shape[2]
			query, key, value = \
				[l(x).view(nbatches, -1, seq_len, self.h, self.d_k).permute(0, 3, 1, 2, 4).contiguous()
				 for l, x in zip(self.linears, (query, key, value))]
			# 2) 计算注意力attn 得到attn*v 与attn
			# qkv :[batch, head, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
			x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
			# 3) 上一步的结果合并在一起还原成原始输入序列的形状
			x = x.permute(0, 2, 3, 1, 4).contiguous().view(nbatches, -1, seq_len, self.h * self.d_k)
		elif len(query.shape) == 3 and len(key.shape) == 4:
			nbatches = key.shape[0]
			key_seq_len = key.shape[2]
			query_seq_len = query.shape[1]
			query = self.linears[0](query).view(nbatches, 1, query_seq_len, self.h, self.d_k).\
				permute(0, 3, 1, 2, 4).contiguous()
			
			key, value = \
				[l(x).view(nbatches, -1, key_seq_len, self.h, self.d_k).permute(0, 3, 1, 2, 4).contiguous()
				 for l, x in zip(self.linears[1:], (key, value))]
			# 2) 计算注意力attn 得到attn*v 与attn
			# qkv :[batch, head, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
			x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
			# 3) 上一步的结果合并在一起还原成原始输入序列的形状
			x = x.permute(0, 2, 3, 1, 4).contiguous().view(nbatches, -1, query_seq_len, self.h * self.d_k)
			# 4) 对候选轨迹代表的维度求和，得到和输入相同的维度
			x = torch.sum(x, dim=1)
			
		# 最后再过一个线性层
		return self.linears[-1](x)  # 再过一层linear


# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
	# 实现FFN函数
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
	"""N层堆叠的Encoder"""

	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"""每层layer依次通过输入序列与mask"""
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)


class SelfAttn(nn.Module):
	def __init__(self, layer, N):
		super(SelfAttn, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.dim_size)
	
	def forward(self, x, mask):
		"""每层layer依次通过输入序列与mask"""
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)
	

class SelfAttnLayer(nn.Module):
	"""Encoder分为两层Self-Attn和Feed Forward"""
	def __init__(self, dim_size, self_attn, feed_forward, dropout):
		super(SelfAttnLayer, self).__init__()
		self.dim_size = dim_size
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(self.dim_size, dropout), 2)
	
	def forward(self, x, mask):
		# 先过self-attn
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		# 再过feedforward
		return self.sublayer[1](x, self.feed_forward)


class CrossAttn(nn.Module):
	def __init__(self, layer, N):
		super(CrossAttn, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.dim_size)
	
	def forward(self, x, memory, src_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask)
		return self.norm(x)


class CrossAttnLayer(nn.Module):
	"""Decoder is made of self-attn, src-attn, and feed forward"""
	def __init__(self, dim_size, cross_attn, feed_forward, dropout):
		super(CrossAttnLayer, self).__init__()
		self.dim_size = dim_size
		self.cross_attn = cross_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(self.dim_size, dropout), 2)
	
	def forward(self, x, memory, src_mask):
		"""将decoder的三个Sublayer串联起来"""
		m = memory
		# corss attention
		x = self.sublayer[0](x, lambda x: self.cross_attn(x, m, m, src_mask))  # x m m  m是encoder过来的 x来自上一个 DecoderLayer
		return self.sublayer[1](x, self.feed_forward)


class EncoderLayer(nn.Module):   # 多层注意力block + ffn block
	"""Encoder分为两层Self-Attn和Feed Forward"""

	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

	def forward(self, x, mask):
		# 先过self-attn
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		# 再过feedforward
		return self.sublayer[1](x, self.feed_forward)
		

class Decoder(nn.Module):  # decoder 和encoder差不多 但是传进来的layer不一样
	"""带mask功能的通用Decoder结构"""

	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, trg_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, trg_mask)
		return self.norm(x)

	'''
	（1）Decoder SubLayer-1 使用的是 “Masked” Multi-Headed Attention 机制，防止为了模型看到要预测的数据，防止泄露。
	
	（2）SubLayer-2 是一个 Encoder-Decoder Multi-head Attention。

	（3）LinearLayer 和 SoftmaxLayer 作用于 SubLayer-3 的输出后面，来预测对应的 word 的 probabilities 。
	'''
	
	
class DecoderLayer(nn.Module):
	"""Decoder is made of self-attn, src-attn, and feed forward"""

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(self.size, dropout), 2)
		self.cross_sublayer = SublayerConnection(self.size * 2, dropout)
		self.transform_linear = nn.Linear(self.size * 2, self.size)

	def forward(self, x, memory, src_mask, trg_mask):
		"""将decoder的三个Sublayer串联起来"""
		m = memory
		# self attention
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))  # sequence mask
		# corss attention
		x = self.cross_sublayer(x.repeat(1, 1, 2), lambda x: self.src_attn(x, m, m, src_mask))  # x m m  m是encoder过来的 x来自上一个 DecoderLayer
		# x = self.cross_sublayer(x.repeat(1, 1, 1), lambda x: self.src_attn(x, m, m, src_mask))  # x m m  m是encoder过来的 x来自上一个 DecoderLayer
		
		x = self.transform_linear(x)
		return self.sublayer[1](x, self.feed_forward)


class TemporalEncoding(nn.Module):
	# 时间四元组嵌入
	def __init__(self, d_model, dropout):
		super(TemporalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# self.day_embed = nn.Embedding(7, int(d_model/4))
		# self.hour_embed = nn.Embedding(24, int(d_model/4))
		# self.minute_embed = nn.Embedding(60, int(d_model/4))
		# self.second_embed = nn.Embedding(60, int(d_model/4))
		self.day_embed = nn.Embedding(8, 3)
		self.hour_embed = nn.Embedding(24, 12)
		self.minute_embed = nn.Embedding(60, 30)
		self.second_embed = nn.Embedding(60, 19)
		
	def forward(self, time_tuple):
		day_e = self.day_embed(time_tuple[..., 0])
		hour_e = self.hour_embed(time_tuple[..., 1])
		minute_e = self.minute_embed(time_tuple[..., 2])
		second_e = self.second_embed(time_tuple[..., 3])
		time_e = torch.cat([day_e, hour_e, minute_e, second_e], -1)
		return self.dropout(time_e)
