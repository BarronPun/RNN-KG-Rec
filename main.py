# -*- coding: utf-8 -*-
# Created by Barron Pun

import numpy as np
import argparse
import time
import torch
from classes import SVAE, VAELoss, DataReader
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data(args, device):
	data_train, data_test = [], []
	
	# get the number of users/items from user/item_list.txt, the first line is a header
	num_users, num_items = 0, 0
	with open(args['user_list_path'], encoding='utf-8') as f:
		content = f.readlines()
		num_users = len(content) - 1
	with open(args['item_list_path'], encoding='utf-8') as f:
		content = f.readlines()
		num_items = len(content) - 1

	with open(args['train_path'], encoding='utf-8') as f:
		content = f.readlines()
		for line in content:
			line_tuple = line.strip().split(' ')
			line_tuple = np.array(line_tuple, dtype=int)
			data_train.append(line_tuple[1:])

	with open(args['test_path'], encoding='utf-8') as f:
		content = f.readlines()
		for line in content:
			line_tuple = line.strip().split(' ')
			line_tuple = np.array(line_tuple, dtype=int)
			data_test.append(line_tuple[1:])

	train_reader = DataReader(args, data_train, None, num_items, True, device)
	test_reader = DataReader(args, data_train, data_test, num_items, False, device)

	return train_reader, test_reader, num_users, num_items


def train(epoch, reader, hyper_params, model, optimizer, criterion):
	model.train()
	total_loss = 0
	start_time = time.time()
	batch = 0
	batch_limit = int(reader.num_b)
	total_anneal_steps = 200000
	anneal = 0.0
	update_count = 0.0
	anneal_cap = 0.2

	for x, y_s in reader.iter():
		batch += 1
		
		# Empty the gradients
		model.zero_grad()
		optimizer.zero_grad()
	
		# Forward pass
		decoder_output, z_mean, z_log_sigma = model(x)
		
		# Backward pass
		loss = criterion(decoder_output, z_mean, z_log_sigma, y_s, anneal)
		loss.backward()
		optimizer.step()

		total_loss += loss.data
		
		# Anneal logic
		if total_anneal_steps > 0:
			anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
		else:
			anneal = anneal_cap
		update_count += 1.0
		
		# Logging mechanism
		if (batch % hyper_params['batch_log_interval'] == 0 and batch > 0) or batch == batch_limit:
			div = hyper_params['batch_log_interval']
			if batch == batch_limit: div = (batch_limit % hyper_params['batch_log_interval']) - 1
			if div <= 0: div = 1

			cur_loss = (total_loss.item() / div)
			elapsed = time.time() - start_time
			
			ss = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
					epoch, batch, batch_limit, (elapsed * 1000) / div, cur_loss
			)
			
			# file_write(hyper_params['log_file'], ss)
			print(ss)

			total_loss = 0
			start_time = time.time()


def evaluate(model, criterion, reader, hyper_params, is_train_set):
	model.eval()

	metrics = {}
	metrics['loss'] = 0.0
	Ks = [10, 100]
	for k in Ks: 
		metrics['NDCG@' + str(k)] = 0.0
		metrics['Rec@' + str(k)] = 0.0
		metrics['Prec@' + str(k)] = 0.0

	batch = 0
	total_users = 0.0
	
	# For plotting the results (seq length vs. NDCG@100)
	len_to_ndcg_at_100_map = {}

	for x, y_s, test_movies, test_movies_r in reader.iter_eval():
		batch += 1
		if is_train_set == True and batch > hyper_params['train_cp_users']: break

		decoder_output, z_mean, z_log_sigma = model(x)
		
		metrics['loss'] += criterion(decoder_output, z_mean, z_log_sigma, y_s, 0.2).data[0]
		
		# Making the logits of previous items in the sequence to be "- infinity"
		decoder_output = decoder_output.data
		x_scattered = torch.zeros(decoder_output.shape[0], decoder_output.shape[2])
		if is_cuda_available: x_scattered = x_scattered.cuda()
		x_scattered[0, :].scatter_(0, x[0].data, 1.0)
		last_predictions = decoder_output[:, -1, :] - (torch.abs(decoder_output[:, -1, :] * x_scattered) * 100000000)
		
		for batch_num in range(last_predictions.shape[0]): # batch_num is ideally only 0, since batch_size is enforced to be always 1
			predicted_scores = last_predictions[batch_num]
			actual_movies_watched = test_movies[batch_num]
			actual_movies_ratings = test_movies_r[batch_num]
					
			# Calculate NDCG
			_, argsorted = torch.sort(-1.0 * predicted_scores)
			for k in Ks:
				best, now_at, dcg, hits = 0.0, 0.0, 0.0, 0.0
				
				rec_list = list(argsorted[:k].cpu().numpy())
				for m in range(len(actual_movies_watched)):
					movie = actual_movies_watched[m]
					now_at += 1.0
					if now_at <= k: best += 1.0 / float(np.log2(now_at + 1))
					
					if movie not in rec_list: continue
					hits += 1.0
					dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))
				
				metrics['NDCG@' + str(k)] += float(dcg) / float(best)
				metrics['Rec@' + str(k)] += float(hits) / float(len(actual_movies_watched))
				metrics['Prec@' + str(k)] += float(hits) / float(k)
				
				# Only for plotting the graph (seq length vs. NDCG@100)
				if k == 100:
					seq_len = int(len(actual_movies_watched)) + int(x[batch_num].shape[0]) + 1
					if seq_len not in len_to_ndcg_at_100_map: len_to_ndcg_at_100_map[seq_len] = []
					len_to_ndcg_at_100_map[seq_len].append(float(dcg) / float(best))
				
			total_users += 1.0
	
	metrics['loss'] = float(metrics['loss']) / float(batch)
	metrics['loss'] = round(metrics['loss'], 4)
	
	for k in Ks:
		metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_users), 4)
		metrics['Rec@' + str(k)] = round((100.0 * metrics['Rec@' + str(k)]) / float(total_users), 4)
		metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_users), 4)
		
	return metrics, len_to_ndcg_at_100_map


def args_parse():
	parser = argparse.ArgumentParser(description='Running the model')
	parser.add_argument('--train_path', type=str, default='./data/train.txt')
	parser.add_argument('--test_path', type=str, default='./data/test.txt')
	parser.add_argument('--item_list_path', type=str, default='./data/item_list.txt')
	parser.add_argument('--user_list_path', type=str, default='./data/user_list.txt')
	parser.add_argument('--item_embed_size', type=int, default=64)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_epoches', type=int, default=20)
	parser.add_argument('--rnn_size', type=int, default=128)
	parser.add_argument('--hidden_size', type=int, default=128, help='dim of the hidden layer in vae')
	parser.add_argument('--latent_size', type=int, default=64, help='dim of the latent variables in vae')
	parser.add_argument('--optimizer', type=str, default='adam')
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=5e-3)
	parser.add_argument('--loss_type', type=str, default='next_k')
	parser.add_argument('--next_k', type=int, default=20)
	parser.add_argument('--number_users_to_keep', type=int, default=1000000000)
	parser.add_argument('--history_split_test', type=str, default='[0.8, 0.2]')
	parser.add_argument('--batch_log_interval', type=int, default=100)
	parser.add_argument('--train_cp_users', type=int, default=200)
	parser.add_argument('--cuda', type=str, default='2')

	return parser.parse_args()



def main():
	parser = args_parse()
	train_path = parser.train_path
	test_path = parser.test_path
	item_list_path = parser.item_list_path
	user_list_path = parser.user_list_path
	item_embed_size = parser.item_embed_size
	batch_size = parser.batch_size
	num_epoches = parser.num_epoches
	rnn_size = parser.rnn_size
	hidden_size = parser.hidden_size
	latent_size = parser.latent_size
	lr = parser.lr
	weight_decay = parser.weight_decay
	loss_type = parser.loss_type
	next_k = parser.next_k
	number_users_to_keep = parser.number_users_to_keep
	history_split_test = eval(parser.history_split_test)
	batch_log_interval = parser.batch_log_interval
	train_cp_users = parser.train_cp_users
	cuda = parser.cuda

	device = torch.device("cuda:"+cuda if torch.cuda.is_available() else "cpu")

	args = {
		'train_path': train_path,
		'test_path': test_path,
		'item_list_path': item_list_path,
		'user_list_path': user_list_path,
		'item_embed_size': item_embed_size,
		'batch_size': batch_size,
		'num_epoches': num_epoches,
		'rnn_size': rnn_size,
		'hidden_size': hidden_size,
		'latent_size': latent_size,
		'loss_type': loss_type,
		'next_k': next_k,
		'number_users_to_keep': number_users_to_keep,
		'history_split_test': history_split_test,
		'batch_log_interval': batch_log_interval,
		'train_cp_users': train_cp_users,
		'device': device
	}

	train_reader, test_reader, num_users, num_items = load_data(args, device)


	model = SVAE(rnn_size, hidden_size, latent_size, num_items, item_embed_size, device)
	model.to(device)

	criterion = VAELoss(args)

	optimizer = None

	if parser.optimizer == 'adagrad':
		optimizer = torch.optim.Adagrad(
			model.parameters(), weight_decay=weight_decay, lr = lr
		)
	elif parser.optimizer == 'adadelta':
		optimizer = torch.optim.Adadelta(
			model.parameters(), weight_decay=weight_decay
		)
	elif parser.optimizer == 'adam':
		optimizer = torch.optim.Adam(
			model.parameters(), weight_decay=weight_decay
		)
	elif parser.optimizer == 'rmsprop':
		optimizer = torch.optim.RMSprop(
			model.parameters(), weight_decay=weight_decay
		)


	### Training...
	# For the training set, split 20% for validation
	try:
		for epoch in range(num_epoches):
			epoch_start_time = time.time()

			train(epoch, train_reader, args, model, optimizer, criterion)

			# metrics, _ = evaluate()




	except KeyboardInterrupt:
		print('Exiting from training early!')


	### Testing
	# Combine the split validation for testing the test set


if __name__ == '__main__':
	main()
	