import torch
import os


def save_model(model, file_path, file_name, is_full_model):
	if not os.path.exists(file_path):
		os.mkdir(file_path)
	if is_full_model:
		torch.save(model, file_path + '/' + file_name)
	else:
		torch.save(model.state_dict(),  file_path + '/' + file_name)
	print('save mode in file:' + file_path + '/' + file_name)


def load_model(file_name, is_full_model, model=None):
	if is_full_model:
		model = torch.load(file_name)
	else:
		model.load_state_dict(torch.load(file_name))
	return model