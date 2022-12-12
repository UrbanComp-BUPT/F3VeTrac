import time
from tqdm import tqdm
import argparse
import sys
import copy
import random
from math import floor
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_processers.get_graph_input import get_adj_from_json, get_node_features
from data_processers.get_data_sessions import load_session_data_from_diff_files, load_session_data_from_diff_pickle_files
from models.GATModel import GATModule
from models.DataSet import Dataset, collate_fn
from models import Seq2SeqModel
from train.Seq2SeqTrain import init_weights, train, evaluate, NoamOpt
from models.save_and_load import save_model, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_statement(parameters):
	c = copy.deepcopy
	attn = Seq2SeqModel.MultiHeadedAttention(parameters['head_num'], parameters['d_model'], parameters['dropout'])
	ff = Seq2SeqModel.PositionwiseFeedForward(parameters['d_model'], parameters['d_ff'], parameters['dropout'])
	
	attn_dec = Seq2SeqModel.MultiHeadedAttention(parameters['head_num'], parameters['d_model'] * 2, parameters['dropout'])
	ff_dec = Seq2SeqModel.PositionwiseFeedForward(parameters['d_model'] * 2, parameters['d_ff'] * 2, parameters['dropout'])

	each_head_dim = int(parameters['d_model'] / parameters['head_num'])
	
	pretrained_spatial_embed = GATModule(parameters['embedding_size'], each_head_dim, parameters['dropout'],
	                                     args['alpha'], parameters['head_num'])
	# spatial_embedding_module = nn.Embedding(parameters['node_size'], parameters['embedding_size'])
	spatial_embedding_module = load_model(parameters['embedding_path'], True)
	temporal_embedding_module = Seq2SeqModel.TemporalEncoding(parameters['d_model'], parameters['dropout'])
	
	enc_layer = Seq2SeqModel.EncoderLayer(parameters['d_model'], c(attn), c(ff), parameters['dropout'])
	enc = Seq2SeqModel.Encoder(enc_layer, parameters['stacked_N'])
	
	self_attn_layer = Seq2SeqModel.SelfAttnLayer(parameters['d_model'], c(attn), c(ff), parameters['dropout'])
	self_attn = Seq2SeqModel.SelfAttn(self_attn_layer, parameters['stacked_N'])
	
	cross_attn_layer = Seq2SeqModel.CrossAttnLayer(parameters['d_model'], c(attn), c(ff), parameters['dropout'])
	cross_attn = Seq2SeqModel.CrossAttn(cross_attn_layer, parameters['stacked_N'])
	
	dec_layer = Seq2SeqModel.DecoderLayer(parameters['d_model'], c(attn), c(attn_dec), c(ff), parameters['dropout'])
	# dec_layer = Seq2SeqModel.DecoderLayer(parameters['d_model'], c(attn), c(attn), c(ff), parameters['dropout'])
	dec = Seq2SeqModel.Decoder(dec_layer, parameters['stacked_N'])
	spatial_embed = c(spatial_embedding_module)
	temporal_embed = c(temporal_embedding_module)
	
	generator = Seq2SeqModel.Generator(parameters['d_model'], parameters['node_size'])
	traj_gen = Seq2SeqModel.TrajGenerator(parameters['d_model'], parameters['dropout'])
	return Seq2SeqModel.EncoderDecoder(enc, dec, pretrained_spatial_embed, spatial_embed, temporal_embed,
	                                   generator, traj_gen, self_attn, cross_attn)


if __name__ == '__main__':
	# ################################################################################
	# 参数设置
	# ################################################################################
	
	args = dict()
	args_dict = {
		'device': device,

		'node_size': 245 + 2,

		'neighbor_size': 6,
		'embedding_size': 64,

		'd_model': 64,
		'd_ff': 256,
		'head_num': 4,
		'auxi_traj_num': 4,

		'embedding_path': '',
		
		# input data params
		'shuffle': True,
		# model params
		'stacked_N': 4,
		# 'stacked_N': 4,
		'dropout': 0.5,
		'alpha': 0.2,
		'n_epochs': 5,
		'batch_size': 256,
		'learning_rate': 1e-4,
		'tf_ratio': 0.5,
		'clip': 1,
		'log_step': 1,
		
		# optimizer
		'factor': 1,
		'warmup': 2000,
		'device_ID': 0
	}
	print(args_dict)
	args.update(args_dict)
	
	# ################################################################################
	# 指定GPU
	# ################################################################################
	torch.cuda.set_device(args_dict['device_ID'])
	
	# ################################################################################
	# 文件路径设置
	# ################################################################################

	adj_file_path = ''

	data_file_path = ''
	train_file_name = ''
	# ################################################################################
	# 读入数据
	# ################################################################################
	data_windows_size = 2
	train_data = load_session_data_from_diff_pickle_files(data_file_path,
	                                                      train_file_name)

	train_dataset = Dataset(train_data, args)

	train_data = None


	adj, neighbor_pad = get_adj_from_json(adj_file_path, args['node_size'] - 1)
	args_dict['neighbor_size'] = neighbor_pad.shape[1]

	neighbor_pad = neighbor_pad.to(device)

	# ################################################################################
	# 设置log文件
	# ################################################################################

	# ################################################################################
	# 声明模型
	# ################################################################################
	model = model_statement(args).to(device)
	init_weights(model)
	# ################################################################################
	# 开始训练
	# ################################################################################

	train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'],
	                                             shuffle=args['shuffle'], collate_fn=collate_fn,
	                                             num_workers=36, pin_memory=True)

	# for epoch in tqdm(range(args['n_epochs'])):
	for epoch in tqdm(range(15)):
		start_time = time.time()

		print('')
		optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

		train(model, train_iterator, optimizer, args, neighbor_pad)
		print('')

		end_time = time.time()
		elapsed_time = end_time - start_time
		epoch_mins = int(elapsed_time / 60)
		epoch_secs = int(elapsed_time - (epoch_mins * 60))

	print('end')
	del train_iterator
	current_date = datetime.datetime.now()
	current_time = datetime.datetime.strftime(current_date,'%Y-%m-%d %H:%M:%S')
	model_file_name = current_time + '_d_model_' + str(args['d_model']) + \
	                  '_stack_' + str(args['stacked_N']) + \
	                  '_dropout_' + str(args['dropout']) + \
	                  '_batch_size_' + str(args['batch_size']) + \
	                  '_lr_' + str(args['learning_rate']) + '.pth'
